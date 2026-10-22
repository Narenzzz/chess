#!/usr/bin/env python3
"""
Chess Middlegame Strength & Progress Tracker — Streamlit Web App
================================================================

Features in this version:
1. Transposition-Aware Popular Opening Classifier (Moves 5–8):
   Inspects the actual board position (pawn structure + minor piece placement)
   between moves 5 and 8 so games that start with unusual move orders and
   transpose into well-known openings (Italian Game, Ruy Lopez, Sicilian,
   Caro-Kann, French, Queen's Gambit, London System, King's Indian, etc.) are
   grouped cleanly into recognizable Popular Opening Families & Variations.
2. Interactive Full-Game Board Reviewer:
   Select any analyzed game (filtered by Popular Opening Family), step through
   every move on a visual chessboard (Prev / Next / Jump to Book Exit / Jump to
   Middlegame Mistakes), see green/red arrows comparing your move with the
   recommended plan, and read the live pawn-structure coaching panel.
3. All 3 High-Impact Middlegame Modules:
   - Module 1: Weekly Middlegame Accuracy (Moves 11–30)
   - Module 2: Post-Opening Transition Audit (grouped by Popular Opening Family
     or Transposed Variation)
   - Module 3: Personal "Plan Finder" Flashcards + Lichess Study PGN export
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import shutil
import statistics
import subprocess
import sys
import urllib.parse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Auto-Bootstrap Dependencies (Helps when deploying a single file from Mobile)
# ---------------------------------------------------------------------------
try:
    import chess
    import chess.engine
    import chess.pgn
    import chess.polyglot
    import chess.svg
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "chess", "requests"])
    import chess
    import chess.engine
    import chess.pgn
    import chess.polyglot
    import chess.svg

try:
    import requests
except ImportError:
    requests = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import streamlit as st
except ImportError:
    st = None


# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------

DEFAULT_MIDDLEGAME_START_MOVE = 11
DEFAULT_MIDDLEGAME_END_MOVE = 30
DEFAULT_BOOK_EXIT_FALLBACK_MOVE = 10
TRANSITION_WINDOW_MOVES = 5

INACCURACY_THRESHOLD_CP = 50
MISTAKE_THRESHOLD_CP = 100
BLUNDER_THRESHOLD_CP = 300

FLASHCARD_MIN_CP_LOSS = 80
FLASHCARD_MAX_CP_LOSS = 260
MIN_PIECES_FOR_MIDDLEGAME = 7
MATE_SCORE_CAP_CP = 1000


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class MoveEvalRecord:
    """Stores evaluation details for a single player move."""
    move_number: int
    ply: int
    fen_before: str
    played_move_uci: str
    played_move_san: str
    best_move_uci: str
    best_move_san: str
    eval_before_cp: int
    eval_after_cp: int
    cp_loss: int
    is_best_move_quiet: bool
    pv_san: List[str] = field(default_factory=list)
    pawn_structure_hint: str = ""


@dataclass
class PlySnapshot:
    """Stores the board state at every half-move for the Interactive Game Reviewer."""
    ply: int
    move_number: int
    turn_before: chess.Color
    move_uci: Optional[str]
    move_san: str
    fen_after: str
    fen_before: str
    pawn_structure_hint: str
    eval_record: Optional[MoveEvalRecord] = None


@dataclass
class GameAnalysisSummary:
    """Aggregated middlegame, transposition opening, and board review data for a game."""
    game_index: int
    game_id: str
    date_str: str
    iso_week: str
    player_color: chess.Color
    white_name: str
    black_name: str
    opponent_name: str
    eco_code: str
    raw_pgn_opening: str
    popular_family: str          # e.g. "Italian Game", "Caro-Kann Defense", "London System"
    popular_variation: str       # e.g. "Italian Game — Giuoco Piano / Pianissimo"
    result_str: str
    book_exit_move: int
    middlegame_moves_count: int
    middlegame_acpl: float
    inaccuracies: int
    mistakes: int
    blunders: int
    transition_drop_cp: float
    flashcard_candidates: List[MoveEvalRecord] = field(default_factory=list)
    all_eval_records: List[MoveEvalRecord] = field(default_factory=list)
    ply_snapshots: List[PlySnapshot] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Transposition-Aware Popular Opening Classifier (Inspects Board at Moves 5–8)
# ---------------------------------------------------------------------------

def classify_transposed_opening(
    game: chess.pgn.Game,
    raw_eco: str = "",
    raw_opening: str = "",
) -> Tuple[str, str]:
    """
    Solves the 'unfamiliar opening name' problem by inspecting the actual board
    state between moves 5 and 8 (plies 9 to 16) to detect transpositions into
    popular, recognizable openings and variations.

    Returns:
        (popular_family, popular_family_and_variation)
        e.g. ("Italian Game", "Italian Game — Giuoco Pianissimo (c3 + d3)")
    """
    board = game.board()
    snapshots_5_to_8: List[chess.Board] = []
    ever_had_b5_or_a4_bishop = False

    for ply_idx, node in enumerate(game.mainline(), start=1):
        board.push(node.move)
        wb_now = board.pieces(chess.BISHOP, chess.WHITE)
        if chess.B5 in wb_now or chess.A4 in wb_now:
            ever_had_b5_or_a4_bishop = True
        if 8 <= ply_idx <= 16:
            snapshots_5_to_8.append(board.copy())
        if ply_idx >= 16:
            break

    if not snapshots_5_to_8:
        snapshots_5_to_8 = [board.copy()]

    for b in reversed(snapshots_5_to_8):
        wp = b.pieces(chess.PAWN, chess.WHITE)
        bp = b.pieces(chess.PAWN, chess.BLACK)
        wb = b.pieces(chess.BISHOP, chess.WHITE)
        bb = b.pieces(chess.BISHOP, chess.BLACK)
        wn = b.pieces(chess.KNIGHT, chess.WHITE)
        bn = b.pieces(chess.KNIGHT, chess.BLACK)

        wp_files = {chess.square_file(sq) for sq in wp}
        bp_files = {chess.square_file(sq) for sq in bp}

        white_e_pawn_passive = (chess.E2 in wp or chess.E3 in wp)
        white_e_pawn_active_or_traded = (chess.E4 in wp or chess.E5 in wp or 4 not in wp_files)

        # 1. Ruy Lopez / Spanish (White Bishop visited b5/a4 against e5)
        if (chess.E4 in wp and (chess.E5 in bp or chess.D6 in bp) and ever_had_b5_or_a4_bishop):
            if chess.A6 in bp:
                return ("Ruy Lopez (Spanish)", "Ruy Lopez — Morphy Defense (...a6)")
            if chess.F6 in bn:
                return ("Ruy Lopez (Spanish)", "Ruy Lopez — Berlin Defense (...Nf6)")
            return ("Ruy Lopez (Spanish)", "Ruy Lopez — Classical / Main Line")

        # 2. Italian Game (White e4, Bc4/Bb3, Nf3 vs Black e5, Nc6)
        if (chess.E4 in wp and chess.E5 in bp
                and (chess.C4 in wb or chess.B3 in wb)
                and (chess.C6 in bn or chess.D5 in bp)):
            if chess.C5 in bb or chess.B6 in bb or chess.A7 in bb:
                if chess.C3 in wp and chess.D3 in wp:
                    return ("Italian Game", "Italian Game — Giuoco Pianissimo (c3 + d3)")
                return ("Italian Game", "Italian Game — Giuoco Piano (...Bc5)")
            if chess.F6 in bn:
                return ("Italian Game", "Italian Game — Two Knights Defense (...Nf6)")
            return ("Italian Game", "Italian Game — Classical Setup (Bc4)")

        # 3. Scotch Game / Four Knights (e4 + e5 where White d-pawn and Black e-pawn were traded early)
        if (chess.E4 in wp and 3 not in wp_files and 4 not in bp_files
                and (chess.D4 in wn or chess.F3 in wn) and (chess.C6 in bn or chess.F6 in bn)):
            if chess.C3 in wn and chess.F3 in wn and chess.C6 in bn and chess.F6 in bn:
                return ("Four Knights / Scotch Game", "Four Knights — Scotch Variation")
            return ("Four Knights / Scotch Game", "Scotch Game — Open Center (d4 exd4)")

        # 4. Petrov / Four Knights Symmetrical (e4 + e5, Nf3 + Nf6, no Bc4/Bb5)
        if (chess.E4 in wp and chess.E5 in bp and chess.F3 in wn and chess.F6 in bn):
            if chess.C3 in wn and chess.C6 in bn:
                return ("Four Knights / Scotch Game", "Four Knights Game — Symmetrical")
            return ("King's Pawn (1.e4 e5)", "Petrov's Defense / Russian Game")

        # 5. Sicilian Defense (White e4 vs Black c5 or traded c-pawn)
        if white_e_pawn_active_or_traded and (
            chess.C5 in bp or (chess.E4 in wp and 2 not in bp_files and (chess.D6 in bp or chess.E6 in bp))
        ):
            if chess.C3 in wp and chess.D4 in wp:
                return ("Sicilian Defense", "Sicilian Defense — Alapin Variation (c3)")
            if 3 not in wp_files and 2 not in bp_files:
                if chess.G6 in bp:
                    return ("Sicilian Defense", "Sicilian Defense — Dragon / Accelerated Dragon")
                if chess.A6 in bp and chess.D6 in bp:
                    return ("Sicilian Defense", "Sicilian Defense — Najdorf Variation")
                return ("Sicilian Defense", "Sicilian Defense — Open Sicilian")
            if chess.B5 in wb:
                return ("Sicilian Defense", "Sicilian Defense — Rossolimo / Canal Attack (Bb5)")
            if chess.C3 in wn and (chess.G3 in wp or chess.F4 in wp):
                return ("Sicilian Defense", "Sicilian Defense — Closed / Grand Prix Attack")
            return ("Sicilian Defense", "Sicilian Defense — Flexible / Closed Setup")

        # 6. London System / Colle-Zukertort (White d4 + Bf4/Bg3 + e3 + c3/Nbd2)
        if (chess.D4 in wp and chess.E3 in wp
                and (chess.F4 in wb or chess.G3 in wb)
                and (chess.C3 in wp or chess.D2 in wn)
                and chess.C4 not in wp):
            if chess.G6 in bp:
                return ("London System", "London System — vs King's Indian Fianchetto (...g6)")
            return ("London System", "London System — Pyramid Setup (d4 + Bf4 + e3 + c3)")

        # 7. Queen's Gambit (White d4 + e2/e3 + c4 or c-pawn traded vs Black d5)
        if chess.D4 in wp and white_e_pawn_passive and (chess.C4 in wp or 2 not in wp_files) and (chess.D5 in bp or 3 not in bp_files):
            if 2 not in wp_files and 4 not in bp_files and chess.D5 in bp:
                return ("Queen's Gambit", "Queen's Gambit Declined — Exchange (Carlsbad)")
            if chess.C6 in bp and chess.E6 in bp:
                return ("Queen's Gambit", "Queen's Gambit — Semi-Slav Defense (...c6 + ...e6)")
            if chess.C6 in bp:
                return ("Queen's Gambit", "Queen's Gambit — Slav Defense (...c6)")
            if chess.E6 in bp:
                return ("Queen's Gambit", "Queen's Gambit Declined — Orthodox (...e6)")
            if 3 not in bp_files and 2 not in wp_files:
                return ("Queen's Gambit", "Queen's Gambit Accepted (...dxc4)")

        # 8. Caro-Kann Defense (White e4/e5 or e-pawn traded vs Black c6 + d5)
        if white_e_pawn_active_or_traded and (
            (chess.C6 in bp and (chess.D5 in bp or chess.E5 in wp))
            or (4 not in wp_files and 2 not in bp_files and chess.D4 in wp and chess.D5 in bp)
            or raw_eco.startswith("B1")
        ):
            if chess.E5 in wp and chess.D4 in wp and chess.D5 in bp:
                return ("Caro-Kann Defense", "Caro-Kann Defense — Advance Variation (e5)")
            if 4 not in wp_files and 2 not in bp_files and chess.D4 in wp and chess.D5 in bp:
                if chess.C4 in wp:
                    return ("Caro-Kann Defense", "Caro-Kann Defense — Panov-Botvinnik Attack")
                return ("Caro-Kann Defense", "Caro-Kann Defense — Exchange Variation")
            return ("Caro-Kann Defense", "Caro-Kann Defense — Classical / Main Line")

        # 9. French Defense (e6 + d5 against e4/e5)
        if white_e_pawn_active_or_traded and (
            (chess.E6 in bp and chess.D5 in bp and (chess.E4 in wp or chess.E5 in wp))
            or raw_eco.startswith("C0") or raw_eco.startswith("C1")
        ):
            if chess.E5 in wp and chess.D4 in wp:
                return ("French Defense", "French Defense — Advance Variation (e5)")
            if 4 not in wp_files and 4 not in bp_files and chess.D4 in wp and chess.D5 in bp:
                return ("French Defense", "French Defense — Exchange Variation")
            if chess.C3 in wn or chess.D2 in wn:
                return ("French Defense", "French Defense — Classical / Tarrasch / Winawer")
            return ("French Defense", "French Defense — Standard Setup")

        # 10. Scandinavian Defense (e4 vs early ...d5, d-file open for Black)
        if (chess.E4 in wp or 4 not in wp_files) and 3 not in bp_files and 2 in bp_files and 4 in bp_files:
            if raw_eco.startswith("B01") or "scandinavian" in raw_opening.lower():
                return ("Scandinavian Defense", "Scandinavian Defense — Center Counter (1.e4 d5)")

        # 11. Pirc / Modern / Philidor Defense (White e4 + d4 vs Black d6 + g6/Nf6/e5)
        if chess.E4 in wp and chess.D4 in wp and chess.D6 in bp:
            if chess.G6 in bp:
                return ("Pirc / Modern Defense", "Pirc / Modern Defense — Kingside Fianchetto (...g6)")
            if chess.E5 in bp:
                return ("Philidor Defense", "Philidor Defense — Solid Center (...d6 + ...e5)")

        # 12. King's Indian / Grünfeld / Benoni (White d4 + c4 vs Black Nf6 + g6/e6/c5)
        if chess.D4 in wp and chess.F6 in bn:
            if chess.G6 in bp or chess.G7 in bb:
                if chess.D5 in bp or 3 not in bp_files:
                    return ("King's Indian / Grünfeld", "Grünfeld Defense (...Nf6 + ...g6 + ...d5)")
                return ("King's Indian / Grünfeld", "King's Indian Defense (...Nf6 + ...g6 + ...d6)")
            if chess.C5 in bp and chess.D5 in wp:
                return ("Benoni / Benko Defense", "Modern Benoni (d5 vs ...c5)")
            if chess.E6 in bp and (chess.B4 in bb or chess.B6 in bp):
                return ("Nimzo / Queen's Indian", "Nimzo-Indian / Queen's Indian (...Nf6 + ...e6)")

    # Fallback: Map raw PGN opening name / ECO code into a clean Popular Family
    low_op = raw_opening.lower()
    keyword_map = [
        ("italian", "Italian Game"),
        ("giuoco", "Italian Game"),
        ("two knights", "Italian Game"),
        ("ruy lopez", "Ruy Lopez (Spanish)"),
        ("spanish", "Ruy Lopez (Spanish)"),
        ("scotch", "Four Knights / Scotch Game"),
        ("four knights", "Four Knights / Scotch Game"),
        ("petrov", "King's Pawn (1.e4 e5)"),
        ("philidor", "Philidor Defense"),
        ("king's gambit", "King's Gambit"),
        ("vienna", "Vienna Game"),
        ("bishop's opening", "Italian Game"),
        ("sicilian", "Sicilian Defense"),
        ("caro-kann", "Caro-Kann Defense"),
        ("french", "French Defense"),
        ("scandinavian", "Scandinavian Defense"),
        ("pirc", "Pirc / Modern Defense"),
        ("modern defense", "Pirc / Modern Defense"),
        ("alekhine", "Alekhine's Defense"),
        ("london", "London System"),
        ("queen's gambit", "Queen's Gambit"),
        ("slav", "Queen's Gambit"),
        ("king's indian", "King's Indian / Grünfeld"),
        ("grunfeld", "King's Indian / Grünfeld"),
        ("grünfeld", "King's Indian / Grünfeld"),
        ("nimzo", "Nimzo / Queen's Indian"),
        ("queen's indian", "Nimzo / Queen's Indian"),
        ("english", "English / Réti Opening"),
        ("reti", "English / Réti Opening"),
        ("réti", "English / Réti Opening"),
        ("dutch", "Dutch Defense"),
        ("benoni", "Benoni / Benko Defense"),
    ]
    for kw, fam in keyword_map:
        if kw in low_op:
            short_var = raw_opening.split(":")[1].strip() if ":" in raw_opening else "Standard Line"
            return (fam, f"{fam} — {short_var}")

    if raw_eco.startswith("C5"):
        return ("Italian Game", f"Italian Game ({raw_opening})")
    if raw_eco.startswith("C6") or raw_eco.startswith("C7") or raw_eco.startswith("C8") or raw_eco.startswith("C9"):
        return ("Ruy Lopez (Spanish)", f"Ruy Lopez ({raw_opening})")
    if raw_eco.startswith(("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9")):
        return ("Sicilian Defense", f"Sicilian Defense ({raw_opening})")
    if raw_eco.startswith("B1"):
        return ("Caro-Kann Defense", f"Caro-Kann Defense ({raw_opening})")
    if raw_eco.startswith(("C0", "C1")):
        return ("French Defense", f"French Defense ({raw_opening})")
    if raw_eco.startswith("D"):
        return ("Queen's Gambit", f"Queen's Pawn ({raw_opening})")
    if raw_eco.startswith("E"):
        return ("King's Indian / Grünfeld", f"Indian Defense ({raw_opening})")

    return ("King's / Queen's Pawn Setup", f"Transposed Setup ({raw_opening})")


# ---------------------------------------------------------------------------
# Pawn Structure Classifier (Links Opening Knowledge to Middlegame Plans)
# ---------------------------------------------------------------------------

def classify_pawn_structure(board: chess.Board) -> str:
    """Identifies common pawn structures and returns the core strategic plan."""
    white_pawns = board.pieces(chess.PAWN, chess.WHITE)
    black_pawns = board.pieces(chess.PAWN, chess.BLACK)

    wp_files = {chess.square_file(sq) for sq in white_pawns}
    bp_files = {chess.square_file(sq) for sq in black_pawns}

    d_file = 3
    c_file, e_file = 2, 4

    if chess.D4 in white_pawns and c_file not in wp_files and e_file not in wp_files:
        return (
            "Isolated Queen's Pawn (White IQP on d4) — Plan: Use e5/c5 outposts, "
            "build a kingside attack, avoid minor-piece trades, or time the d4-d5 break."
        )
    if chess.D5 in black_pawns and c_file not in bp_files and e_file not in bp_files:
        return (
            "Isolated Queen's Pawn (Black IQP on d5) — Plan: Blockade on d4 with a "
            "knight, trade minor pieces to enter a favorable endgame, or play ...d5-d4."
        )

    if (chess.D4 in white_pawns and chess.D5 in black_pawns
            and ((c_file not in wp_files and e_file not in bp_files)
                 or (e_file not in wp_files and c_file not in bp_files))):
        return (
            "Carlsbad Pawn Structure (QGD / Caro-Kann Exchange) — Plan: Launch a "
            "minority attack (b4-b5 or ...b5-b4) to create a backward queenside pawn, "
            "or prepare a central pawn break with piece support."
        )

    if chess.C4 in white_pawns and chess.E4 in white_pawns and d_file not in wp_files:
        return (
            "Maroczy Bind (c4 + e4 Space Advantage) — Plan: White restricts Black's "
            "...b5 and ...d5 breaks and maneuvers knights to d5/f5; Black prepares "
            "...b5 or ...f5 to challenge White's grip."
        )

    if chess.D4 in white_pawns and chess.E5 in white_pawns and chess.D5 in black_pawns:
        return (
            "Locked Pawn Chain (d4-e5 vs d5-e6) — Plan: White attacks the kingside "
            "(f4-f5, Bd3, Qg4) and fights for the c5 square; Black undermines the chain with ...c5 and ...f6."
        )

    if chess.D5 in white_pawns and chess.E4 in white_pawns and chess.D6 in black_pawns:
        return (
            "Closed Center (d5-e4 vs d6-e5) — Plan: White expands on the queenside "
            "via c4-c5 and invades the c-file; Black prepares the kingside ...f5 break."
        )

    if chess.C6 in black_pawns and chess.D5 in black_pawns and chess.D4 in white_pawns:
        return (
            "Caro-Slav Pawn Triangle (c6-d5) — Plan: Black develops the light-squared "
            "bishop and prepares ...c5 or ...e5; White fights for central e4/e5 control."
        )

    if c_file not in bp_files and d_file not in wp_files and chess.E4 in white_pawns:
        return (
            "Asymmetrical Open Sicilian (Half-Open c-file vs d-file) — Plan: White "
            "pressures d6/d5 and attacks via f4/g4; Black uses the half-open c-file "
            "and prepares the liberating ...d5 break."
        )

    if chess.E4 in white_pawns and chess.E5 in black_pawns:
        return (
            "Symmetrical / Classical King's Pawn Center (e4 vs e5) — Plan: Prepare the "
            "d4 (or ...d5) central break with c3/Re1 support, fight for the d5/f5 outposts, "
            "and maneuver the queenside knight via d2-f1-g3/e3."
        )

    return (
        "Flexible Middlegame Structure — Plan: Identify your worst-placed minor piece, "
        "reroute it to an active outpost, and prepare your primary central pawn break."
    )


# ---------------------------------------------------------------------------
# Engine & Heuristic Evaluation Helpers
# ---------------------------------------------------------------------------

class PositionEvaluator:
    """
    Wraps either a local/cloud Stockfish UCI engine (if installed) or a fast
    built-in 2-ply positional heuristic so Streamlit Cloud never hangs.
    """

    def __init__(
        self,
        stockfish_path: Optional[str] = None,
        depth: int = 12,
        time_limit_sec: float = 0.08,
        use_fast_heuristic: bool = True,
    ) -> None:
        self.depth = depth
        self.time_limit_sec = time_limit_sec
        self.use_fast_heuristic = use_fast_heuristic
        self.engine: Optional[chess.engine.SimpleEngine] = None

        if not use_fast_heuristic:
            resolved_path = self.resolve_stockfish(stockfish_path)
            if resolved_path:
                try:
                    self.engine = chess.engine.SimpleEngine.popen_uci(resolved_path)
                except Exception:
                    self.engine = None
                    self.use_fast_heuristic = True
            else:
                self.use_fast_heuristic = True

    @staticmethod
    def resolve_stockfish(custom_path: Optional[str] = None) -> Optional[str]:
        if custom_path:
            if Path(custom_path).is_file():
                return custom_path
            found = shutil.which(custom_path)
            if found:
                return found
        for candidate in ("stockfish", "/usr/games/stockfish", "/usr/local/bin/stockfish", "/opt/homebrew/bin/stockfish"):
            found = shutil.which(candidate)
            if found:
                return found
        return None

    def analyse_position(self, board: chess.Board) -> Tuple[int, chess.Move, List[chess.Move]]:
        if self.engine is not None:
            limit = chess.engine.Limit(depth=self.depth, time=self.time_limit_sec)
            info = self.engine.analyse(board, limit)
            score_obj = info["score"].pov(board.turn)
            eval_cp = score_obj.score(mate_score=MATE_SCORE_CAP_CP)
            eval_cp = max(-MATE_SCORE_CAP_CP, min(MATE_SCORE_CAP_CP, int(eval_cp)))
            pv = info.get("pv", [])
            best_move = pv[0] if pv else next(iter(board.legal_moves))
            return eval_cp, best_move, pv[:5]

        return self._heuristic_analyse(board)

    def evaluate_after_move(self, board_after: chess.Board, mover_color: chess.Color) -> int:
        if self.engine is not None:
            limit = chess.engine.Limit(depth=max(8, self.depth - 2), time=self.time_limit_sec)
            info = self.engine.analyse(board_after, limit)
            score_obj = info["score"].pov(mover_color)
            eval_cp = score_obj.score(mate_score=MATE_SCORE_CAP_CP)
            return max(-MATE_SCORE_CAP_CP, min(MATE_SCORE_CAP_CP, int(eval_cp)))

        return self._heuristic_eval_after_move(board_after, mover_color)

    @staticmethod
    def _static_eval(board: chess.Board, color: chess.Color) -> int:
        if board.is_checkmate():
            return -MATE_SCORE_CAP_CP if board.turn == color else MATE_SCORE_CAP_CP
        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 335,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0,
        }
        center_squares = {chess.D4, chess.E4, chess.D5, chess.E5}
        extended_center = {
            chess.C3, chess.D3, chess.E3, chess.F3,
            chess.C4, chess.F4, chess.C5, chess.F5,
            chess.C6, chess.D6, chess.E6, chess.F6,
        }
        rim_files = {0, 7}

        total = 0
        for sq, piece in board.piece_map().items():
            sign = 1 if piece.color == color else -1
            val = piece_values[piece.piece_type]
            f_idx = chess.square_file(sq)
            r_idx = chess.square_rank(sq)

            if sq in center_squares:
                val += 35 if piece.piece_type in (chess.PAWN, chess.KNIGHT, chess.BISHOP) else 15
            elif sq in extended_center and piece.piece_type in (chess.KNIGHT, chess.BISHOP, chess.PAWN):
                val += 18

            if piece.piece_type == chess.KNIGHT and f_idx in rim_files:
                val -= 35

            back_rank = 0 if piece.color == chess.WHITE else 7
            if piece.piece_type in (chess.KNIGHT, chess.BISHOP) and r_idx == back_rank:
                val -= 25

            if piece.piece_type == chess.ROOK and f_idx in (2, 3, 4):
                val += 20

            total += sign * val
        return total

    def _heuristic_eval_after_move(self, board_after: chess.Board, mover_color: chess.Color) -> int:
        if not any(board_after.legal_moves):
            return self._static_eval(board_after, mover_color)

        worst_for_mover = 99999
        for reply in board_after.legal_moves:
            board_after.push(reply)
            score = self._static_eval(board_after, mover_color)
            board_after.pop()
            if score < worst_for_mover:
                worst_for_mover = score
        return worst_for_mover

    def _heuristic_analyse(self, board: chess.Board) -> Tuple[int, chess.Move, List[chess.Move]]:
        mover_color = board.turn
        best_move: Optional[chess.Move] = None
        best_score = -99999
        best_reply_for_best_move: Optional[chess.Move] = None

        for move in board.legal_moves:
            board.push(move)
            worst_for_mover = 99999
            best_reply: Optional[chess.Move] = None
            for reply in board.legal_moves:
                board.push(reply)
                score = self._static_eval(board, mover_color)
                board.pop()
                if score < worst_for_mover:
                    worst_for_mover = score
                    best_reply = reply
            if worst_for_mover == 99999:
                worst_for_mover = self._static_eval(board, mover_color)
            board.pop()

            if worst_for_mover > best_score or best_move is None:
                best_score = worst_for_mover
                best_move = move
                best_reply_for_best_move = best_reply

        assert best_move is not None
        pv = [best_move]
        if best_reply_for_best_move is not None:
            pv.append(best_reply_for_best_move)
        return best_score, best_move, pv

    def close(self) -> None:
        if self.engine is not None:
            try:
                self.engine.quit()
            except Exception:
                pass
            self.engine = None


# ---------------------------------------------------------------------------
# Lichess API Downloader, Direct Link Builder & PGN Auto-Detection
# ---------------------------------------------------------------------------

def clean_lichess_username(raw_input: str) -> str:
    cleaned = raw_input.strip()
    if "lichess.org" in cleaned:
        parsed = urllib.parse.urlparse(cleaned)
        parts = [p for p in parsed.path.split("/") if p and p != "@"]
        if parts:
            cleaned = parts[-1]
    return cleaned.strip()


def build_direct_lichess_download_url(username: str, max_games: int = 15, perf_type: str = "rapid") -> str:
    user_clean = clean_lichess_username(username) or "Narenzxy"
    params = urllib.parse.urlencode({
        "max": str(max_games),
        "perfType": perf_type,
        "opening": "true",
        "moves": "true",
        "pgnInJson": "false",
    })
    return f"https://lichess.org/api/games/user/{user_clean}?{params}"


def fetch_lichess_games(
    username: str,
    max_games: int = 15,
    perf_type: str = "rapid",
    api_token: str = "",
) -> str:
    if requests is None:
        raise RuntimeError("The 'requests' library is required to fetch games from Lichess.")

    user_clean = clean_lichess_username(username)
    params = {
        "max": str(max_games),
        "perfType": perf_type,
        "opening": "true",
        "moves": "true",
        "pgnInJson": "false",
    }
    headers = {
        "Accept": "application/x-chess-pgn",
        "User-Agent": f"MiddlegameStrengthTracker/3.0 (Streamlit; user={user_clean})",
    }
    if api_token.strip():
        headers["Authorization"] = f"Bearer {api_token.strip()}"

    endpoints = [
        f"https://lichess.org/api/games/user/{user_clean}",
        f"https://lichess.org/games/export/{user_clean}",
    ]

    last_exc: Optional[Exception] = None
    for url in endpoints:
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            if response.status_code == 200 and "[Event " in response.text:
                return response.text
            response.raise_for_status()
        except Exception as exc:
            last_exc = exc

    raise RuntimeError(
        f"Lichess blocked the unauthenticated cloud request (HTTP 404 — see Lichess API Issue #667). "
        f"Original error: {last_exc}"
    )


def load_pgn_games(pgn_text: str) -> List[chess.pgn.Game]:
    games: List[chess.pgn.Game] = []
    stream = io.StringIO(pgn_text)
    while True:
        game = chess.pgn.read_game(stream)
        if game is None:
            break
        games.append(game)
    return games


def detect_primary_player(games: List[chess.pgn.Game]) -> str:
    counts: Counter[str] = Counter()
    original_casing: Dict[str, str] = {}
    for g in games:
        for tag in ("White", "Black"):
            name = g.headers.get(tag, "").strip()
            if name and name != "?":
                key = name.lower()
                counts[key] += 1
                original_casing[key] = name
    if not counts:
        return ""
    best_key, _ = counts.most_common(1)[0]
    return original_casing[best_key]


# ---------------------------------------------------------------------------
# Core Analysis Pipeline (Modules 1, 2, 3 + Full-Game Board Snapshots)
# ---------------------------------------------------------------------------

def count_major_minor_pieces(board: chess.Board) -> int:
    count = 0
    for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        count += len(board.pieces(pt, chess.WHITE)) + len(board.pieces(pt, chess.BLACK))
    return count


def parse_iso_week(date_header: str) -> str:
    try:
        cleaned = date_header.replace("-", ".").strip()
        parts = [int(p) for p in cleaned.split(".") if p.isdigit()]
        if len(parts) == 3:
            d = dt.date(parts[0], parts[1], parts[2])
            iso_yr, iso_wk, _ = d.isocalendar()
            return f"{iso_yr}-W{iso_wk:02d}"
    except Exception:
        pass
    return "Unknown-Week"


def analyze_single_game(
    game_index: int,
    game: chess.pgn.Game,
    player_username: str,
    evaluator: PositionEvaluator,
    mg_start: int = DEFAULT_MIDDLEGAME_START_MOVE,
    mg_end: int = DEFAULT_MIDDLEGAME_END_MOVE,
    fallback_book_exit: int = DEFAULT_BOOK_EXIT_FALLBACK_MOVE,
) -> Optional[GameAnalysisSummary]:
    headers = game.headers
    white_name = headers.get("White", "White").strip()
    black_name = headers.get("Black", "Black").strip()

    target = clean_lichess_username(player_username).lower()
    if target and white_name.lower() == target:
        player_color = chess.WHITE
        opponent_name = black_name
    elif target and black_name.lower() == target:
        player_color = chess.BLACK
        opponent_name = white_name
    elif not target:
        player_color = chess.WHITE
        opponent_name = black_name or "Opponent"
    else:
        return None

    date_str = headers.get("Date", headers.get("UTCDate", "Unknown"))
    iso_week = parse_iso_week(date_str)
    eco_code = headers.get("ECO", "A00")
    raw_pgn_opening = headers.get("Opening", "Unclassified Opening")
    result_str = headers.get("Result", "*")
    game_id = headers.get("Site", f"Game-{game_index}-{date_str}")

    # Detect Transposed Popular Opening Family & Variation (at moves 5–8)
    popular_family, popular_variation = classify_transposed_opening(
        game, raw_eco=eco_code, raw_opening=raw_pgn_opening
    )

    board = game.board()
    book_exit_move: Optional[int] = None

    middlegame_records: List[MoveEvalRecord] = []
    transition_records: List[MoveEvalRecord] = []
    flashcard_candidates: List[MoveEvalRecord] = []
    all_eval_records: List[MoveEvalRecord] = []
    ply_snapshots: List[PlySnapshot] = []

    # Record initial starting position (ply 0)
    ply_snapshots.append(
        PlySnapshot(
            ply=0,
            move_number=1,
            turn_before=chess.WHITE,
            move_uci=None,
            move_san="Start Position",
            fen_before=board.fen(),
            fen_after=board.fen(),
            pawn_structure_hint=classify_pawn_structure(board),
            eval_record=None,
        )
    )

    for node in game.mainline():
        move = node.move
        fullmove = board.fullmove_number
        turn = board.turn
        fen_before = board.fen()
        played_san = board.san(move)

        if book_exit_move is None and fullmove >= fallback_book_exit:
            book_exit_move = fullmove

        in_middlegame_window = (
            mg_start <= fullmove <= mg_end
            and count_major_minor_pieces(board) >= MIN_PIECES_FOR_MIDDLEGAME
        )
        in_transition_window = (
            book_exit_move is not None
            and book_exit_move <= fullmove < book_exit_move + TRANSITION_WINDOW_MOVES
        )

        eval_rec: Optional[MoveEvalRecord] = None

        if turn == player_color and (in_middlegame_window or in_transition_window):
            eval_before_cp, best_move, pv_moves = evaluator.analyse_position(board)
            best_san = board.san(best_move)
            is_best_quiet = (not board.is_capture(best_move)) and (not board.gives_check(best_move))

            pv_san: List[str] = []
            temp_board = board.copy()
            for pv_m in pv_moves:
                if pv_m in temp_board.legal_moves:
                    pv_san.append(temp_board.san(pv_m))
                    temp_board.push(pv_m)
                else:
                    break

            structure_hint = classify_pawn_structure(board)

            board.push(move)
            if move == best_move:
                eval_after_cp = eval_before_cp
                cp_loss = 0
            else:
                eval_after_cp = evaluator.evaluate_after_move(board, player_color)
                cp_loss = max(0, eval_before_cp - eval_after_cp)

            eval_rec = MoveEvalRecord(
                move_number=fullmove,
                ply=board.ply(),
                fen_before=fen_before,
                played_move_uci=move.uci(),
                played_move_san=played_san,
                best_move_uci=best_move.uci(),
                best_move_san=best_san,
                eval_before_cp=eval_before_cp,
                eval_after_cp=eval_after_cp,
                cp_loss=cp_loss,
                is_best_move_quiet=is_best_quiet,
                pv_san=pv_san,
                pawn_structure_hint=structure_hint,
            )
            all_eval_records.append(eval_rec)

            if in_middlegame_window:
                middlegame_records.append(eval_rec)
                if (
                    FLASHCARD_MIN_CP_LOSS <= cp_loss <= FLASHCARD_MAX_CP_LOSS
                    and is_best_quiet
                    and abs(eval_before_cp) <= 280
                ):
                    flashcard_candidates.append(eval_rec)

            if in_transition_window:
                transition_records.append(eval_rec)
        else:
            board.push(move)

        ply_snapshots.append(
            PlySnapshot(
                ply=board.ply(),
                move_number=fullmove,
                turn_before=turn,
                move_uci=move.uci(),
                move_san=played_san,
                fen_before=fen_before,
                fen_after=board.fen(),
                pawn_structure_hint=classify_pawn_structure(board),
                eval_record=eval_rec,
            )
        )

    if not middlegame_records and not transition_records:
        return None

    mg_losses = [r.cp_loss for r in middlegame_records]
    mg_acpl = statistics.mean(mg_losses) if mg_losses else 0.0
    inaccuracies = sum(1 for loss in mg_losses if INACCURACY_THRESHOLD_CP <= loss < MISTAKE_THRESHOLD_CP)
    mistakes = sum(1 for loss in mg_losses if MISTAKE_THRESHOLD_CP <= loss < BLUNDER_THRESHOLD_CP)
    blunders = sum(1 for loss in mg_losses if loss >= BLUNDER_THRESHOLD_CP)
    transition_drop_cp = float(sum(r.cp_loss for r in transition_records))

    return GameAnalysisSummary(
        game_index=game_index,
        game_id=game_id,
        date_str=date_str,
        iso_week=iso_week,
        player_color=player_color,
        white_name=white_name,
        black_name=black_name,
        opponent_name=opponent_name,
        eco_code=eco_code,
        raw_pgn_opening=raw_pgn_opening,
        popular_family=popular_family,
        popular_variation=popular_variation,
        result_str=result_str,
        book_exit_move=book_exit_move or fallback_book_exit,
        middlegame_moves_count=len(middlegame_records),
        middlegame_acpl=round(mg_acpl, 1),
        inaccuracies=inaccuracies,
        mistakes=mistakes,
        blunders=blunders,
        transition_drop_cp=round(transition_drop_cp, 1),
        flashcard_candidates=flashcard_candidates,
        all_eval_records=all_eval_records,
        ply_snapshots=ply_snapshots,
    )


def build_flashcards_pgn_string(summaries: List[GameAnalysisSummary], player_username: str) -> str:
    buf = io.StringIO()
    count = 0
    for summary in summaries:
        for rec in summary.flashcard_candidates:
            count += 1
            puzzle_game = chess.pgn.Game()
            puzzle_game.setup(rec.fen_before)
            puzzle_game.headers["Event"] = f"Middlegame Plan Finder #{count} ({summary.popular_variation})"
            puzzle_game.headers["Site"] = summary.game_id
            puzzle_game.headers["Date"] = summary.date_str
            puzzle_game.headers["White"] = summary.white_name
            puzzle_game.headers["Black"] = summary.black_name
            puzzle_game.headers["ECO"] = summary.eco_code
            puzzle_game.headers["Opening"] = summary.popular_variation

            puzzle_game.comment = (
                f"[Move {rec.move_number}] {rec.pawn_structure_hint} | "
                f"Question: What is the best positional plan here? "
                f"(In the game you played {rec.played_move_san}, losing {rec.cp_loss} cp.)"
            )

            board = chess.Board(rec.fen_before)
            best_move = chess.Move.from_uci(rec.best_move_uci)
            played_move = chess.Move.from_uci(rec.played_move_uci)

            main_node = puzzle_game.add_variation(best_move)
            pv_text = " -> ".join(rec.pv_san) if rec.pv_san else rec.best_move_san
            main_node.comment = (
                f"Best Plan ({rec.best_move_san}, Eval {rec.eval_before_cp:+d} cp): "
                f"Recommended continuation: {pv_text}"
            )

            if played_move != best_move and played_move in board.legal_moves:
                side_node = puzzle_game.add_variation(played_move)
                side_node.comment = (
                    f"Game Move ({rec.played_move_san}, Eval {rec.eval_after_cp:+d} cp, "
                    f"Loss: -{rec.cp_loss} cp). Review why {rec.best_move_san} better serves the pawn structure."
                )

            exporter = chess.pgn.FileExporter(buf)
            puzzle_game.accept(exporter)
            buf.write("\n\n")

    return buf.getvalue()


def build_csv_string(summaries: List[GameAnalysisSummary]) -> str:
    buf = io.StringIO()
    fieldnames = [
        "date", "iso_week", "color", "opponent", "popular_family",
        "popular_variation", "raw_pgn_opening", "eco",
        "book_exit_move", "mg_moves_evaluated", "mg_acpl",
        "mg_inaccuracies", "mg_mistakes", "mg_blunders",
        "post_book_5move_drop_cp", "flashcards_generated", "game_id",
    ]
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for s in summaries:
        writer.writerow({
            "date": s.date_str,
            "iso_week": s.iso_week,
            "color": "White" if s.player_color == chess.WHITE else "Black",
            "opponent": s.opponent_name,
            "popular_family": s.popular_family,
            "popular_variation": s.popular_variation,
            "raw_pgn_opening": s.raw_pgn_opening,
            "eco": s.eco_code,
            "book_exit_move": s.book_exit_move,
            "mg_moves_evaluated": s.middlegame_moves_count,
            "mg_acpl": s.middlegame_acpl,
            "mg_inaccuracies": s.inaccuracies,
            "mg_mistakes": s.mistakes,
            "mg_blunders": s.blunders,
            "post_book_5move_drop_cp": s.transition_drop_cp,
            "flashcards_generated": len(s.flashcard_candidates),
            "game_id": s.game_id,
        })
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Embedded Demo Games (Including Transposed Opening Examples)
# ---------------------------------------------------------------------------

DEMO_PGN_TEXT = """
[Event "Rapid 15+10 Training Game 1"]
[Site "https://lichess.org/demo0001"]
[Date "2026.10.05"]
[White "Narenzxy"]
[Black "ClubOpponent1"]
[Result "0-1"]
[ECO "A00"]
[Opening "Van't Kruijs Opening"]

1. e3 c6 2. e4 d5 3. exd5 cxd5 4. d4 Nc6 5. c3 Nf6 6. Bd3 Bg4 7. Qb3 Qd7
8. Bf4 e6 9. Nd2 Be7 10. Ngf3 Bxf3 11. Nxf3 O-O 12. O-O Rfc8 13. Nh4 Na5
14. Qd1 Nc4 15. Qe2 b5 16. g3 b4 17. Ng2 bxc3 18. bxc3 Ba3 0-1

[Event "Rapid 15+10 Training Game 2"]
[Site "https://lichess.org/demo0002"]
[Date "2026.10.14"]
[White "ClubOpponent2"]
[Black "Narenzxy"]
[Result "1-0"]
[ECO "A04"]
[Opening "Zukertort Opening: Queen's Gambit Invitation"]

1. Nf3 d5 2. d4 e6 3. c4 Nf6 4. Nc3 c6 5. cxd5 exd5 6. Bg5 Be7 7. e3 Nbd7
8. Bd3 O-O 9. Qc2 Re8 10. O-O Nf8 11. Rab1 a5 12. a3 Nh5 13. Bxe7 Qxe7
14. b4 axb4 15. axb4 Ng6 16. b5 Nh4 17. Nxh4 Qxh4 18. bxc6 bxc6 1-0

[Event "Rapid 15+10 Training Game 3"]
[Site "https://lichess.org/demo0003"]
[Date "2026.10.20"]
[White "Narenzxy"]
[Black "ClubOpponent3"]
[Result "1-0"]
[ECO "C50"]
[Opening "King's Pawn Game: Leonardis Variation"]

1. e4 e5 2. Bc4 Nf6 3. d3 Nc6 4. Nf3 Bc5 5. c3 d6 6. O-O a6 7. Bb3 Ba7
8. Nbd2 O-O 9. h3 h6 10. Re1 Re8 11. Nf1 Be6 12. Ng3 Bxb3 13. axb3 Qd7
14. Nh4 Qe6 15. Ngf5 Ne7 16. Qf3 Nxf5 17. Nxf5 Kh7 18. b4 Bb6 1-0
"""


# ---------------------------------------------------------------------------
# Streamlit Web Application UI
# ---------------------------------------------------------------------------

def run_streamlit_app() -> None:
    st.set_page_config(
        page_title="Chess Middlegame Strength Tracker",
        page_icon="♟️",
        layout="wide",
    )

    st.title("♟️ Chess Middlegame Strength & Transition Tracker")
    st.caption(
        "**Transposition-Aware Popular Opening Grouping (Moves 5–8)** • "
        "**Interactive Full-Game Board Reviewer** • **Modules 1, 2 & 3**"
    )

    stockfish_bin = PositionEvaluator.resolve_stockfish()

    with st.expander("⚙️ Step 1: Choose Games & Analysis Settings", expanded=True):
        col1, col2 = st.columns([2, 1])
        with col1:
            source_mode = st.radio(
                "Select Game Source:",
                options=[
                    "1. Load Your Lichess Games (Direct Phone Download or API)",
                    "2. Upload or Paste Any PGN File",
                    "3. Instant Demo Games (Shows Transposition Detection)",
                ],
                index=0,
            )
        with col2:
            engine_options = ["Fast Built-In Positional Evaluator (Instant on Cloud)"]
            if stockfish_bin:
                engine_options.insert(0, f"Stockfish UCI Engine ({stockfish_bin})")
            engine_choice = st.selectbox("Evaluation Engine:", options=engine_options)
            use_fast_heuristic = not engine_choice.startswith("Stockfish")

        player_username = "Narenzxy"
        lichess_count = 15
        lichess_perf = "rapid"
        lichess_token = ""
        uploaded_pgn_text = ""

        if source_mode.startswith("1."):
            c_user, c_perf, c_num = st.columns([2, 1, 1])
            with c_user:
                player_username = st.text_input(
                    "Your Lichess Username:",
                    value="Narenzxy",
                    placeholder="e.g. Narenzxy",
                ).strip()
            with c_perf:
                lichess_perf = st.selectbox("Time Control:", ["rapid", "classical", "blitz", "bullet"], index=0)
            with c_num:
                lichess_count = st.slider("Number of Games:", min_value=5, max_value=40, value=15, step=5)

            direct_url = build_direct_lichess_download_url(player_username, lichess_count, lichess_perf)

            dl_col, up_col = st.columns([1, 1.3])
            with dl_col:
                st.link_button(
                    f"📥 1. Download {clean_lichess_username(player_username) or 'Your'}'s {lichess_count} {lichess_perf.title()} Games (.pgn)",
                    url=direct_url,
                    use_container_width=True,
                )
                lichess_token = st.text_input(
                    "Optional: Lichess Personal API Token (for direct cloud fetch):",
                    value="",
                    type="password",
                    help="Create a free read-only token at https://lichess.org/account/oauth/token",
                )
            with up_col:
                lichess_up_file = st.file_uploader(
                    "📂 2. Select Your Downloaded Lichess `.pgn` File Here:",
                    type=["pgn", "txt"],
                    key="lichess_pgn_uploader",
                )
                if lichess_up_file is not None:
                    uploaded_pgn_text = lichess_up_file.getvalue().decode("utf-8", errors="replace")

        elif source_mode.startswith("2."):
            player_username = st.text_input(
                "Your Username (leave as Narenzxy or blank to auto-detect from PGN):",
                value="Narenzxy",
            ).strip()
            up_file = st.file_uploader("Upload a `.pgn` file:", type=["pgn", "txt"], key="general_pgn_uploader")
            if up_file is not None:
                uploaded_pgn_text = up_file.getvalue().decode("utf-8", errors="replace")
            else:
                uploaded_pgn_text = st.text_area(
                    "Or paste PGN text directly here:",
                    height=140,
                    placeholder='[Event "Rated Rapid game"]\n[White "Narenzxy"]\n1. e4 c6 2. d4 d5 ...',
                )
        else:
            player_username = "Narenzxy"
            st.info(
                "Loaded 3 sample games for **`Narenzxy`** where obscure move-1 names "
                "(`Van't Kruijs Opening`, `Zukertort Opening`, `Leonardis Variation`) automatically "
                "transpose by move 6 into **Caro-Kann Exchange**, **Queen's Gambit Declined**, and **Italian Game — Giuoco Pianissimo**!"
            )

        c_m1, c_m2, c_m3 = st.columns(3)
        with c_m1:
            mg_start = st.number_input("Middlegame Start Move:", min_value=6, max_value=20, value=11)
        with c_m2:
            mg_end = st.number_input("Middlegame End Move:", min_value=20, max_value=50, value=30)
        with c_m3:
            book_exit_default = st.number_input("Default Book Exit Move:", min_value=6, max_value=18, value=10)

        run_clicked = st.button("🚀 Run Middlegame Analysis", type="primary", use_container_width=True)

    if "summaries" not in st.session_state:
        st.session_state["summaries"] = None
        st.session_state["analyzed_player"] = "Narenzxy"

    should_run = (
        run_clicked
        or bool(uploaded_pgn_text.strip() and st.session_state.get("last_uploaded_len") != len(uploaded_pgn_text))
        or (st.session_state["summaries"] is None and source_mode.startswith("3."))
    )

    if should_run:
        if uploaded_pgn_text.strip():
            st.session_state["last_uploaded_len"] = len(uploaded_pgn_text)
            raw_pgn = uploaded_pgn_text
            active_player = clean_lichess_username(player_username)
        elif source_mode.startswith("3."):
            raw_pgn = DEMO_PGN_TEXT
            active_player = "Narenzxy"
        elif source_mode.startswith("1."):
            if not player_username:
                st.warning("Please enter your Lichess username first.")
                return
            with st.spinner(f"Trying direct cloud fetch for `{player_username}` from Lichess..."):
                try:
                    raw_pgn = fetch_lichess_games(
                        player_username,
                        max_games=lichess_count,
                        perf_type=lichess_perf,
                        api_token=lichess_token,
                    )
                    active_player = clean_lichess_username(player_username)
                except Exception as exc:
                    direct_url = build_direct_lichess_download_url(player_username, lichess_count, lichess_perf)
                    st.warning(
                        f"**Cloud Fetch Note:** {exc}\n\n"
                        f"👉 **Easy Fix on Mobile:** Tap **📥 1. Download {clean_lichess_username(player_username)}'s Games (.pgn)** "
                        f"above (or long-press [this direct download link]({direct_url}) and choose **Download link**), "
                        f"then select that file in **📂 2. Select Your Downloaded Lichess `.pgn` File Here**!"
                    )
                    return
        else:
            st.warning("Please upload a `.pgn` file or paste PGN text first.")
            return

        games = load_pgn_games(raw_pgn)
        if not games:
            st.error("No valid PGN games were found in the input.")
            return

        detected_player = detect_primary_player(games)
        if not active_player or (
            detected_player
            and active_player.lower() not in {
                g.headers.get("White", "").strip().lower() for g in games
            } | {
                g.headers.get("Black", "").strip().lower() for g in games
            }
        ):
            active_player = detected_player

        progress_bar = st.progress(0, text=f"Analyzing {len(games)} game(s) for {active_player}...")
        evaluator = PositionEvaluator(use_fast_heuristic=use_fast_heuristic)
        results: List[GameAnalysisSummary] = []
        try:
            for idx, g in enumerate(games, start=1):
                summary = analyze_single_game(
                    game_index=idx,
                    game=g,
                    player_username=active_player,
                    evaluator=evaluator,
                    mg_start=int(mg_start),
                    mg_end=int(mg_end),
                    fallback_book_exit=int(book_exit_default),
                )
                if summary is not None:
                    results.append(summary)
                progress_bar.progress(
                    idx / len(games),
                    text=f"Analyzed {idx}/{len(games)} game(s) for {active_player}...",
                )
        finally:
            evaluator.close()
            progress_bar.empty()

        if not results:
            st.error(
                f"No games in the PGN reached move {mg_start} with 7+ pieces for `{active_player}`. "
                "Try lowering 'Middlegame Start Move' to 8 or uploading more games."
            )
            return

        st.session_state["summaries"] = results
        st.session_state["analyzed_player"] = active_player or "Player"
        st.session_state["review_ply"] = 20

    summaries: Optional[List[GameAnalysisSummary]] = st.session_state.get("summaries")
    active_player_label: str = st.session_state.get("analyzed_player", "Narenzxy")

    if not summaries:
        return

    # Global Popular Opening Family filter
    all_families = sorted({s.popular_family for s in summaries})
    filter_col1, filter_col2 = st.columns([2, 1])
    with filter_col1:
        selected_families = st.multiselect(
            "🔍 Filter Dashboard by Transposed Popular Opening Family (Moves 5–8):",
            options=all_families,
            default=all_families,
        )
    with filter_col2:
        grouping_mode = st.radio(
            "Group Openings By:",
            options=["Popular Opening Family", "Popular Opening + Variation", "Raw PGN Header Name"],
            index=1,
        )

    filtered_summaries = [s for s in summaries if s.popular_family in selected_families]
    if not filtered_summaries:
        st.warning("Select at least one Popular Opening Family above to view metrics.")
        return

    # -----------------------------------------------------------------------
    # Top-Level KPI Metrics
    # -----------------------------------------------------------------------
    overall_acpl = statistics.mean(s.middlegame_acpl for s in filtered_summaries)
    overall_drop = statistics.mean(s.transition_drop_cp for s in filtered_summaries)
    total_flashcards = sum(len(s.flashcard_candidates) for s in filtered_summaries)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Games Shown", f"{len(filtered_summaries)} / {len(summaries)}")
    m2.metric("Middlegame ACPL (Moves 11–30)", f"{overall_acpl:.1f} cp")
    m3.metric("Avg 5-Move Post-Book Drop", f"{overall_drop:.1f} cp")
    m4.metric("Strategic Plan Flashcards", f"{total_flashcards}")

    # Download buttons
    dl1, dl2 = st.columns(2)
    pgn_export_str = build_flashcards_pgn_string(filtered_summaries, active_player_label)
    csv_export_str = build_csv_string(filtered_summaries)
    with dl1:
        st.download_button(
            "📥 Download Study Flashcards (.pgn for Lichess Study)",
            data=pgn_export_str or "[Event \"No flashcards flagged\"]\n\n*\n",
            file_name=f"{active_player_label}_middlegame_flashcards.pgn",
            mime="application/x-chess-pgn",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            "📊 Download Full Middlegame Metrics (.csv)",
            data=csv_export_str,
            file_name=f"{active_player_label}_middlegame_metrics.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # -----------------------------------------------------------------------
    # Four Interactive Tabs (Including Interactive Game Board Reviewer!)
    # -----------------------------------------------------------------------
    tab_board, tab_m2, tab_m1, tab_m3 = st.tabs([
        "♟️ Game Board Reviewer",
        "🧭 Module 2: Popular Opening Transitions",
        "📈 Module 1: Weekly Accuracy",
        "🧩 Module 3: Plan Finder Flashcards",
    ])

    # --- TAB 1: INTERACTIVE FULL-GAME BOARD REVIEWER ---
    with tab_board:
        st.subheader("♟️ Interactive Game & Middlegame Transition Board Reviewer")
        st.write(
            "Select any of your analyzed games below to step through the board move by move. "
            "**Green arrows** show the recommended positional plan; **red arrows** show a move that lost centipawns."
        )

        game_labels = [
            (
                f"Game #{s.game_index} ({s.date_str}) — {s.popular_variation} | "
                f"{'White' if s.player_color == chess.WHITE else 'Black'} vs {s.opponent_name} ({s.result_str}) | "
                f"MG ACPL: {s.middlegame_acpl:.1f} cp"
            )
            for s in filtered_summaries
        ]
        selected_idx = st.selectbox(
            "Select a Game to Review on the Board:",
            options=list(range(len(filtered_summaries))),
            format_func=lambda i: game_labels[i],
            key="board_game_selector",
        )
        active_game = filtered_summaries[selected_idx]
        max_ply = len(active_game.ply_snapshots) - 1

        state_key = f"ply_game_{active_game.game_index}"
        if state_key not in st.session_state:
            default_ply = min(max_ply, max(0, (active_game.book_exit_move - 1) * 2 + 1))
            st.session_state[state_key] = default_ply

        btn_c1, btn_c2, btn_c3, btn_c4, btn_c5, btn_c6 = st.columns(6)
        with btn_c1:
            if st.button("⏮️ Start", use_container_width=True):
                st.session_state[state_key] = 0
        with btn_c2:
            if st.button("◀️ Prev", use_container_width=True):
                st.session_state[state_key] = max(0, st.session_state[state_key] - 1)
        with btn_c3:
            if st.button("▶️ Next", use_container_width=True):
                st.session_state[state_key] = min(max_ply, st.session_state[state_key] + 1)
        with btn_c4:
            if st.button("⏭️ End", use_container_width=True):
                st.session_state[state_key] = max_ply
        with btn_c5:
            if st.button("🧭 Move 6 (Opening)", use_container_width=True):
                st.session_state[state_key] = min(max_ply, 12)
        with btn_c6:
            if st.button(f"🎯 Book Exit (Move {active_game.book_exit_move})", use_container_width=True):
                st.session_state[state_key] = min(max_ply, max(1, (active_game.book_exit_move - 1) * 2 + 1))

        mistake_records = [r for r in active_game.all_eval_records if r.cp_loss >= INACCURACY_THRESHOLD_CP]
        if mistake_records:
            st.caption("⚡ **Quick-Jump to Your Middlegame Inaccuracies / Mistakes in This Game:**")
            m_cols = st.columns(min(6, len(mistake_records)))
            for m_idx, rec in enumerate(mistake_records[:6]):
                icon = "💥" if rec.cp_loss >= BLUNDER_THRESHOLD_CP else ("❌" if rec.cp_loss >= MISTAKE_THRESHOLD_CP else "❓")
                with m_cols[m_idx]:
                    if st.button(
                        f"{icon} Mv {rec.move_number} ({rec.played_move_san}, -{rec.cp_loss}cp)",
                        key=f"jump_{active_game.game_index}_{rec.ply}",
                        use_container_width=True,
                    ):
                        st.session_state[state_key] = min(max_ply, rec.ply)

        current_ply = st.slider(
            "Move Timeline (Half-Moves / Plies):",
            min_value=0,
            max_value=max_ply,
            value=int(st.session_state[state_key]),
            key=f"slider_{active_game.game_index}",
        )
        st.session_state[state_key] = current_ply
        snap = active_game.ply_snapshots[current_ply]

        board_col, coach_col = st.columns([1.05, 1.35])
        with board_col:
            flip_board = st.checkbox("Flip Board Orientation", value=False, key=f"flip_{active_game.game_index}")
            orientation = (not active_game.player_color) if flip_board else active_game.player_color

            board_to_draw = chess.Board(snap.fen_after)
            arrows = []
            if snap.eval_record is not None:
                rec = snap.eval_record
                played_m = chess.Move.from_uci(rec.played_move_uci)
                best_m = chess.Move.from_uci(rec.best_move_uci)
                if played_m != best_m:
                    arrows.append(chess.svg.Arrow(played_m.from_square, played_m.to_square, color="#cc0000cc"))
                arrows.append(chess.svg.Arrow(best_m.from_square, best_m.to_square, color="#009933cc"))
            elif snap.move_uci:
                last_m = chess.Move.from_uci(snap.move_uci)
                arrows.append(chess.svg.Arrow(last_m.from_square, last_m.to_square, color="#3366ccaa"))

            svg_board = chess.svg.board(
                board=board_to_draw,
                orientation=orientation,
                arrows=arrows,
                size=390,
            )
            st.markdown(svg_board, unsafe_allow_html=True)

        with coach_col:
            mover_str = "White" if snap.turn_before == chess.WHITE else "Black"
            if snap.ply == 0:
                st.markdown("### 📍 Starting Position")
            else:
                dots = "." if snap.turn_before == chess.WHITE else "..."
                st.markdown(f"### 📍 Move `{snap.move_number}{dots} {snap.move_san}` ({mover_str})")

            st.markdown(
                f"- **Transposed Popular Opening (Moves 5–8):** **{active_game.popular_variation}**\n"
                f"- **Original PGN Tag:** `{active_game.eco_code} — {active_game.raw_pgn_opening}`\n"
                f"- **Matchup:** `{active_game.white_name}` (White) vs `{active_game.black_name}` (Black) — Result: `{active_game.result_str}`"
            )

            st.info(f"**Pawn Structure & Strategic Plan Here:**\n\n{snap.pawn_structure_hint}")

            if snap.eval_record is not None:
                rec = snap.eval_record
                pv_str = " → ".join(rec.pv_san) if rec.pv_san else rec.best_move_san
                if rec.cp_loss >= BLUNDER_THRESHOLD_CP:
                    badge = f"💥 **Blunder (-{rec.cp_loss} cp)**"
                elif rec.cp_loss >= MISTAKE_THRESHOLD_CP:
                    badge = f"❌ **Mistake (-{rec.cp_loss} cp)**"
                elif rec.cp_loss >= INACCURACY_THRESHOLD_CP:
                    badge = f"❓ **Inaccuracy (-{rec.cp_loss} cp)**"
                else:
                    badge = f"✅ **Good Move (-{rec.cp_loss} cp)**"

                st.markdown(
                    f"#### Middlegame Engine Audit for Move {rec.move_number}\n"
                    f"- **Verdict:** {badge}\n"
                    f"- **Your Move (Red Arrow):** `{rec.played_move_san}` (Eval after: `{rec.eval_after_cp:+d} cp`)\n"
                    f"- **Recommended Move (Green Arrow):** **`{rec.best_move_san}`** (Eval before: `{rec.eval_before_cp:+d} cp`)\n"
                    f"- **Recommended Plan / Continuation:** `{pv_str}`"
                )
            else:
                st.caption(
                    "*(Move is outside your evaluated middlegame/transition window or is your opponent's move. "
                    "Use the Quick-Jump buttons above to inspect your evaluated middlegame moves.)*"
                )

            lichess_fen_url = "https://lichess.org/analysis/" + urllib.parse.quote(snap.fen_after)
            st.link_button("🔬 Open This Exact Position on Lichess Analysis Board", url=lichess_fen_url)

    # --- TAB 2: MODULE 2 (TRANSPOSED POPULAR OPENINGS) ---
    with tab_m2:
        st.subheader("Module 2: Opening-to-Middlegame Transition Audit (Transposition-Aware)")
        st.write(
            "Instead of obscure move-1 names, games are categorized by their **board position at moves 5–8** "
            "into popular openings and variations. Switch **Group Openings By** at the top of the page to compare "
            "broad **Popular Opening Families** vs **Specific Variations**."
        )
        by_opening: Dict[str, List[GameAnalysisSummary]] = defaultdict(list)
        for s in filtered_summaries:
            if grouping_mode == "Popular Opening Family":
                key = s.popular_family
            elif grouping_mode == "Popular Opening + Variation":
                key = s.popular_variation
            else:
                key = f"{s.eco_code} — {s.raw_pgn_opening}"
            by_opening[key].append(s)

        opening_rows = []
        for op_name, op_games in by_opening.items():
            raw_examples = ", ".join(sorted({g.raw_pgn_opening for g in op_games})[:2])
            opening_rows.append({
                "Opening Group": op_name,
                "Games": len(op_games),
                "Avg 5-Move Post-Book Drop (cp)": round(statistics.mean(g.transition_drop_cp for g in op_games), 1),
                "Avg Middlegame ACPL (cp)": round(statistics.mean(g.middlegame_acpl for g in op_games), 1),
                "Inaccuracies / Mistakes / Blunders": (
                    f"{sum(g.inaccuracies for g in op_games)} / "
                    f"{sum(g.mistakes for g in op_games)} / "
                    f"{sum(g.blunders for g in op_games)}"
                ),
                "Original PGN Tags Merged": raw_examples,
            })
        opening_rows.sort(key=lambda r: r["Avg 5-Move Post-Book Drop (cp)"], reverse=True)

        if pd is not None and opening_rows:
            df_op = pd.DataFrame(opening_rows)
            st.bar_chart(df_op.set_index("Opening Group")["Avg 5-Move Post-Book Drop (cp)"])
            st.dataframe(df_op, use_container_width=True, hide_index=True)
        else:
            st.table(opening_rows)

    # --- TAB 3: MODULE 1 (WEEKLY ACCURACY) ---
    with tab_m1:
        st.subheader("Module 1: Middlegame Phase Accuracy Tracker (Moves 11–30)")
        st.write(
            "Tracks your **Average Centipawn Loss (ACPL)** and error counts exclusively in the "
            "middlegame (moves 11–30 with 7+ pieces on the board)."
        )
        by_week: Dict[str, List[GameAnalysisSummary]] = defaultdict(list)
        for s in filtered_summaries:
            by_week[s.iso_week].append(s)

        weekly_rows = []
        for week in sorted(by_week.keys()):
            w_games = by_week[week]
            weekly_rows.append({
                "ISO Week": week,
                "Games": len(w_games),
                "Middlegame ACPL (cp)": round(statistics.mean(g.middlegame_acpl for g in w_games), 1),
                "5-Move Post-Book Drop (cp)": round(statistics.mean(g.transition_drop_cp for g in w_games), 1),
                "Inaccuracies": sum(g.inaccuracies for g in w_games),
                "Mistakes": sum(g.mistakes for g in w_games),
                "Blunders": sum(g.blunders for g in w_games),
            })

        if pd is not None and weekly_rows:
            df_week = pd.DataFrame(weekly_rows)
            if len(df_week) > 1:
                st.line_chart(
                    df_week.set_index("ISO Week")[["Middlegame ACPL (cp)", "5-Move Post-Book Drop (cp)"]]
                )
            st.dataframe(df_week, use_container_width=True, hide_index=True)
        else:
            st.table(weekly_rows)

    # --- TAB 4: MODULE 3 (FLASHCARDS) ---
    with tab_m3:
        st.subheader("Module 3: Personal Middlegame 'Plan Finder' Flashcards")
        st.write(
            "Quiet middlegame positions from your games where the best move was a **positional plan, "
            "pawn break, or piece reroute** (not a capture or check) and your move lost 80–260 centipawns."
        )
        all_cards: List[Tuple[GameAnalysisSummary, MoveEvalRecord]] = []
        for s in filtered_summaries:
            for rec in s.flashcard_candidates:
                all_cards.append((s, rec))

        if not all_cards:
            st.success("No quiet strategic planning mistakes (80–260 cp loss) were flagged in these games!")
        else:
            for idx, (s, rec) in enumerate(all_cards, start=1):
                with st.container(border=True):
                    c_board, c_info = st.columns([1, 1.4])
                    with c_board:
                        board_obj = chess.Board(rec.fen_before)
                        played_m = chess.Move.from_uci(rec.played_move_uci)
                        svg_str = chess.svg.board(
                            board=board_obj,
                            orientation=s.player_color,
                            lastmove=board_obj.peek() if board_obj.move_stack else None,
                            arrows=[chess.svg.Arrow(played_m.from_square, played_m.to_square, color="#cc0000cc")],
                            size=320,
                        )
                        st.markdown(svg_str, unsafe_allow_html=True)
                    with c_info:
                        side_str = "White" if s.player_color == chess.WHITE else "Black"
                        st.markdown(f"#### Flashcard #{idx} — Move {rec.move_number} ({side_str} to move)")
                        st.markdown(
                            f"**Transposed Opening:** **{s.popular_variation}** "
                            f"*(PGN Tag: `{s.raw_pgn_opening}`, {s.date_str})*"
                        )
                        st.markdown(f"**Pawn Structure & Textbook Plan:**\n> {rec.pawn_structure_hint}")
                        st.markdown(
                            f"**What happened in your game:** You played **`{rec.played_move_san}`** "
                            f"(red arrow), which dropped **`-{rec.cp_loss} cp`**."
                        )
                        with st.expander("💡 Tap to Reveal Recommended Plan & Continuation"):
                            pv_str = " → ".join(rec.pv_san) if rec.pv_san else rec.best_move_san
                            st.markdown(
                                f"- **Best Positional Move:** **`{rec.best_move_san}`** "
                                f"(Eval `{rec.eval_before_cp:+d} cp`)\n"
                                f"- **Principal Variation:** `{pv_str}`\n"
                                f"- **FEN:** `{rec.fen_before}`"
                            )


# ---------------------------------------------------------------------------
# Entry Point: Detects Streamlit vs Command-Line Execution
# ---------------------------------------------------------------------------

def _is_running_in_streamlit() -> bool:
    if st is None:
        return False
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if _is_running_in_streamlit():
    run_streamlit_app()
elif __name__ == "__main__":
    evaluator = PositionEvaluator(use_fast_heuristic=True)
    demo_games = load_pgn_games(DEMO_PGN_TEXT)
    detected = detect_primary_player(demo_games)
    print(f"[*] CLI Self-Test: Auto-detected primary player '{detected}' across {len(demo_games)} games...")
    for idx, g in enumerate(demo_games, start=1):
        res = analyze_single_game(idx, g, detected, evaluator)
        if res:
            print(
                f"    [{idx}/{len(demo_games)}] Raw PGN: '{res.raw_pgn_opening}' "
                f"-> Transposed: '{res.popular_variation}' | "
                f"MG ACPL: {res.middlegame_acpl:.1f} cp | "
                f"Post-Book 5-Move Drop: {res.transition_drop_cp:.1f} cp"
            )
    evaluator.close()
    print("[+] Tip: Run `streamlit run middlegame_trainer.py` to launch the interactive web app!")