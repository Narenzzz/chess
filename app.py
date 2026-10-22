#!/usr/bin/env python3
"""
Chess Middlegame Strength & Progress Tracker (middlegame_trainer.py)
====================================================================

A complete command-line application that bridges opening knowledge into
middlegame planning and tracks your personal progress across three modules:

1. Module 1 — Middlegame Phase Accuracy Tracker (Moves 11–30):
   Calculates your Average Centipawn Loss (ACPL), Inaccuracies, Mistakes, and
   Blunders exclusively in the middlegame phase, grouped by week and by opening.

2. Module 2 — Post-Opening Transition Audit (5-Move Post-Book Drop):
   Detects where you leave opening book (via a Polyglot .bin book or move
   threshold) and measures how much evaluation you lose over the next 5 moves
   of independent planning.

3. Module 3 — Personal Middlegame "Plan Finder" Flashcard Generator:
   Extracts positions where you lost 80–260 centipawns in a quiet/strategic
   middlegame position (where the best move was a pawn break or piece reroute,
   not an obvious capture/check) and exports an annotated PGN file ready to
   import into a Lichess Study (https://lichess.org/study).

Requirements:
    pip install chess requests

Also requires the Stockfish UCI binary installed on your system:
    - Windows: Download from https://stockfishchess.org/download/
    - macOS:   brew install stockfish
    - Linux:   sudo apt install stockfish

Usage Examples:
    # Analyze a local PGN file downloaded from Lichess or Chess.com:
    python middlegame_trainer.py --player YourUsername --pgn my_games.pgn --stockfish /usr/games/stockfish

    # Automatically fetch your last 30 rapid games from the Lichess API:
    python middlegame_trainer.py --player YourLichessUsername --fetch-lichess 30 --stockfish stockfish

    # Include a Polyglot opening book (.bin) for exact book-exit detection:
    python middlegame_trainer.py --player YourUsername --pgn my_games.pgn --book gm2001.bin --stockfish stockfish

    # Run a quick self-test demo without Stockfish (uses a lightweight positional heuristic):
    python middlegame_trainer.py --demo
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import shutil
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chess
import chess.engine
import chess.pgn
import chess.polyglot

try:
    import requests
except ImportError:
    requests = None  # Optional unless --fetch-lichess is used


# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------

DEFAULT_MIDDLEGAME_START_MOVE = 11
DEFAULT_MIDDLEGAME_END_MOVE = 30
DEFAULT_BOOK_EXIT_FALLBACK_MOVE = 10
TRANSITION_WINDOW_MOVES = 5

# Centipawn loss thresholds (from the player's perspective)
INACCURACY_THRESHOLD_CP = 50
MISTAKE_THRESHOLD_CP = 100
BLUNDER_THRESHOLD_CP = 300

# Strategic "Plan Finder" flashcard criteria
FLASHCARD_MIN_CP_LOSS = 80
FLASHCARD_MAX_CP_LOSS = 260
MIN_PIECES_FOR_MIDDLEGAME = 7  # Total major + minor pieces on board (both sides)
MATE_SCORE_CAP_CP = 1000       # Cap extreme evaluations to avoid skewing ACPL


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
    eval_before_cp: int          # From the player's perspective
    eval_after_cp: int           # From the player's perspective
    cp_loss: int                 # max(0, eval_before_cp - eval_after_cp)
    is_best_move_quiet: bool     # True if best move is neither capture nor check
    pv_san: List[str] = field(default_factory=list)
    pawn_structure_hint: str = ""


@dataclass
class GameAnalysisSummary:
    """Aggregated middlegame and transition metrics for a single game."""
    game_id: str
    date_str: str
    iso_week: str
    player_color: chess.Color
    opponent_name: str
    eco_code: str
    opening_name: str
    result_str: str
    book_exit_move: int
    middlegame_moves_count: int
    middlegame_acpl: float
    inaccuracies: int
    mistakes: int
    blunders: int
    transition_drop_cp: float    # Total CP loss across first 5 moves out of book
    flashcard_candidates: List[MoveEvalRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pawn Structure Classifier (Links Opening Knowledge to Middlegame Plans)
# ---------------------------------------------------------------------------

def classify_pawn_structure(board: chess.Board) -> str:
    """
    Identifies common pawn structures from the current board state and returns
    both the structure name and the core strategic plan to guide study.
    """
    white_pawns = board.pieces(chess.PAWN, chess.WHITE)
    black_pawns = board.pieces(chess.PAWN, chess.BLACK)

    wp_files = {chess.square_file(sq) for sq in white_pawns}
    bp_files = {chess.square_file(sq) for sq in black_pawns}

    d_file = 3  # 0=a, 1=b, 2=c, 3=d, 4=e, 5=f, 6=g, 7=h
    c_file, e_file = 2, 4

    if (chess.D4 in white_pawns and c_file not in wp_files and e_file not in wp_files):
        return (
            "Isolated Queen's Pawn (White IQP on d4) — Plan: Use e5/c5 outposts, "
            "build a kingside attack, avoid minor-piece trades, or time the d4-d5 break."
        )
    if (chess.D5 in black_pawns and c_file not in bp_files and e_file not in bp_files):
        return (
            "Isolated Queen's Pawn (Black IQP on d5) — Plan: Blockade on d4 with a "
            "knight, trade minor pieces to enter a favorable endgame, or play ...d5-d4."
        )

    # Carlsbad Structure (White d4, no c-pawn vs Black d5, no e-pawn, OR reversed Caro-Kann Exchange)
    if (chess.D4 in white_pawns and chess.D5 in black_pawns
            and ((c_file not in wp_files and e_file not in bp_files)
                 or (e_file not in wp_files and c_file not in bp_files))):
        return (
            "Carlsbad Pawn Structure (QGD / Caro-Kann Exchange) — Plan: Launch a "
            "minority attack (b4-b5 or ...b5-b4) to create a backward queenside pawn, "
            "or prepare a central pawn break with piece support."
        )

    # Maroczy Bind (White pawns on c4 and e4, d-file open)
    if (chess.C4 in white_pawns and chess.E4 in white_pawns and d_file not in wp_files):
        return (
            "Maroczy Bind (c4 + e4 Space Advantage) — Plan: White restricts Black's "
            "...b5 and ...d5 breaks and maneuvers knights to d5/f5; Black prepares "
            "...b5 or ...f5 to challenge White's grip."
        )

    # French / Locked Center (White d4+e5 vs Black d5+e6)
    if (chess.D4 in white_pawns and chess.E5 in white_pawns and chess.D5 in black_pawns):
        return (
            "Locked Pawn Chain (d4-e5 vs d5-e6) — Plan: White attacks the kingside "
            "(f4-f5, Bd3, Qg4) and fights for the c5 square; Black undermines the chain with ...c5 and ...f6."
        )

    # King's Indian / Benoni Locked Center (White d5+e4 vs Black d6+e5)
    if (chess.D5 in white_pawns and chess.E4 in white_pawns and chess.D6 in black_pawns):
        return (
            "Closed Center (d5-e4 vs d6-e5) — Plan: White expands on the queenside "
            "via c4-c5 and invades the c-file; Black prepares the kingside ...f5 break."
        )

    # Slav / Caro-Kann Triangle (Black c6+d5 vs White d4)
    if (chess.C6 in black_pawns and chess.D5 in black_pawns and chess.D4 in white_pawns):
        return (
            "Caro-Slav Pawn Triangle (c6-d5) — Plan: Black develops the light-squared "
            "bishop and prepares ...c5 or ...e5; White fights for central e4/e5 control."
        )

    # Open Sicilian (White d4 traded, Black c-pawn traded, Black d6/e6)
    if (c_file not in bp_files and d_file not in wp_files and chess.E4 in white_pawns):
        return (
            "Asymmetrical Open Sicilian (Half-Open c-file vs d-file) — Plan: White "
            "pressures d6/d5 and attacks via f4/g4; Black uses the half-open c-file "
            "and prepares the liberating ...d5 break."
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
    Wraps either a real Stockfish UCI engine or a built-in positional heuristic
    (used when running with --demo so you can test the pipeline without Stockfish).
    """

    def __init__(
        self,
        stockfish_path: Optional[str] = None,
        depth: int = 14,
        time_limit_sec: float = 0.15,
        use_demo_heuristic: bool = False,
    ) -> None:
        self.depth = depth
        self.time_limit_sec = time_limit_sec
        self.use_demo_heuristic = use_demo_heuristic
        self.engine: Optional[chess.engine.SimpleEngine] = None

        if not use_demo_heuristic:
            resolved_path = self._resolve_stockfish(stockfish_path)
            if not resolved_path:
                raise FileNotFoundError(
                    "Could not find the Stockfish binary. Please install Stockfish "
                    "and pass --stockfish /path/to/stockfish, or run with --demo "
                    "to test the application without an external engine."
                )
            self.engine = chess.engine.SimpleEngine.popen_uci(resolved_path)

    @staticmethod
    def _resolve_stockfish(custom_path: Optional[str]) -> Optional[str]:
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
        """
        Returns (eval_cp_from_side_to_move_perspective, best_move, pv_moves).
        """
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
        """
        Evaluates the board position immediately after `mover_color` has made a move,
        returning centipawns from `mover_color`'s perspective.
        """
        if self.engine is not None:
            limit = chess.engine.Limit(depth=max(10, self.depth - 2), time=self.time_limit_sec)
            info = self.engine.analyse(board_after, limit)
            score_obj = info["score"].pov(mover_color)
            eval_cp = score_obj.score(mate_score=MATE_SCORE_CAP_CP)
            return max(-MATE_SCORE_CAP_CP, min(MATE_SCORE_CAP_CP, int(eval_cp)))

        return self._heuristic_eval_after_move(board_after, mover_color)

    @staticmethod
    def _static_eval(board: chess.Board, color: chess.Color) -> int:
        """Static material + positional piece-square evaluation from `color`'s perspective."""
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
        rim_files = {0, 7}  # a-file and h-file

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

            # Knights on the rim are dim
            if piece.piece_type == chess.KNIGHT and f_idx in rim_files:
                val -= 35

            # Penalize undeveloped minor pieces on back rank
            back_rank = 0 if piece.color == chess.WHITE else 7
            if piece.piece_type in (chess.KNIGHT, chess.BISHOP) and r_idx == back_rank:
                val -= 25

            # Reward rooks on central files (c, d, e)
            if piece.piece_type == chess.ROOK and f_idx in (2, 3, 4):
                val += 20

            total += sign * val
        return total

    def _heuristic_eval_after_move(self, board_after: chess.Board, mover_color: chess.Color) -> int:
        """
        Computes `mover_color`'s score after making a move by finding the opponent's
        strongest 1-ply response (matching the 2-ply minimax depth of `_heuristic_analyse`).
        """
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
        """
        2-ply minimax evaluator used strictly for `--demo` mode so that `eval_before_cp`
        and `eval_after_cp` are measured on the exact same scale.
        """
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
            self.engine.quit()
            self.engine = None


# ---------------------------------------------------------------------------
# Lichess API Downloader & PGN Loading
# ---------------------------------------------------------------------------

def fetch_lichess_games(username: str, max_games: int = 30, perf_type: str = "rapid") -> str:
    """
    Downloads a user's recent games in PGN format directly from the Lichess API
    (https://lichess.org/api).
    """
    if requests is None:
        raise RuntimeError("The 'requests' library is required for --fetch-lichess. Run: pip install requests")

    url = f"https://lichess.org/api/games/user/{username}"
    params = {
        "max": str(max_games),
        "perfType": perf_type,
        "opening": "true",
        "moves": "true",
        "pgnInJson": "false",
    }
    headers = {"Accept": "application/x-chess-pgn"}
    print(f"[*] Fetching up to {max_games} '{perf_type}' games for '{username}' from Lichess API...")
    response = requests.get(url, params=params, headers=headers, timeout=30)
    response.raise_for_status()
    return response.text


def load_pgn_games(pgn_text: str) -> List[chess.pgn.Game]:
    """Parses a raw PGN string into a list of `chess.pgn.Game` objects."""
    games: List[chess.pgn.Game] = []
    stream = io.StringIO(pgn_text)
    while True:
        game = chess.pgn.read_game(stream)
        if game is None:
            break
        games.append(game)
    return games


# ---------------------------------------------------------------------------
# Core Analysis Pipeline (Modules 1, 2, and 3)
# ---------------------------------------------------------------------------

def count_major_minor_pieces(board: chess.Board) -> int:
    """Counts total knights, bishops, rooks, and queens on the board."""
    count = 0
    for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        count += len(board.pieces(pt, chess.WHITE)) + len(board.pieces(pt, chess.BLACK))
    return count


def parse_iso_week(date_header: str) -> str:
    """Converts a PGN Date tag ('YYYY.MM.DD') into an ISO week string ('YYYY-Www')."""
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
    game: chess.pgn.Game,
    player_username: str,
    evaluator: PositionEvaluator,
    book_reader: Optional[chess.polyglot.MemoryMappedReader] = None,
    mg_start: int = DEFAULT_MIDDLEGAME_START_MOVE,
    mg_end: int = DEFAULT_MIDDLEGAME_END_MOVE,
    fallback_book_exit: int = DEFAULT_BOOK_EXIT_FALLBACK_MOVE,
) -> Optional[GameAnalysisSummary]:
    """
    Runs Module 1 (Middlegame ACPL), Module 2 (Post-Book Transition Audit),
    and Module 3 (Strategic Plan Finder Flashcards) on a single PGN game.
    """
    headers = game.headers
    white_name = headers.get("White", "").strip()
    black_name = headers.get("Black", "").strip()

    if white_name.lower() == player_username.lower():
        player_color = chess.WHITE
        opponent_name = black_name
    elif black_name.lower() == player_username.lower():
        player_color = chess.BLACK
        opponent_name = white_name
    else:
        return None  # Player not in this game

    date_str = headers.get("Date", headers.get("UTCDate", "Unknown"))
    iso_week = parse_iso_week(date_str)
    eco_code = headers.get("ECO", "A00")
    opening_name = headers.get("Opening", "Unclassified Opening")
    result_str = headers.get("Result", "*")
    game_id = headers.get("Site", f"Game-{date_str}-vs-{opponent_name}")

    board = game.board()
    book_exit_move: Optional[int] = None

    middlegame_records: List[MoveEvalRecord] = []
    transition_records: List[MoveEvalRecord] = []
    flashcard_candidates: List[MoveEvalRecord] = []

    for node in game.mainline():
        move = node.move
        fullmove = board.fullmove_number
        turn = board.turn

        # Detect book exit using Polyglot book if provided, else fallback move number
        if book_exit_move is None:
            if book_reader is not None:
                try:
                    entries = list(book_reader.find_all(board))
                    if not entries:
                        book_exit_move = fullmove
                except Exception:
                    book_exit_move = fullmove
            elif fullmove >= fallback_book_exit:
                book_exit_move = fullmove

        # Determine if this ply needs engine analysis for the target player
        in_middlegame_window = (
            mg_start <= fullmove <= mg_end
            and count_major_minor_pieces(board) >= MIN_PIECES_FOR_MIDDLEGAME
        )
        in_transition_window = (
            book_exit_move is not None
            and book_exit_move <= fullmove < book_exit_move + TRANSITION_WINDOW_MOVES
        )

        if turn == player_color and (in_middlegame_window or in_transition_window):
            fen_before = board.fen()
            eval_before_cp, best_move, pv_moves = evaluator.analyse_position(board)

            played_san = board.san(move)
            best_san = board.san(best_move)

            # Check if the engine's best move is a quiet strategic move (not a capture or check)
            is_best_quiet = (not board.is_capture(best_move)) and (not board.gives_check(best_move))

            # Build readable SAN principal variation
            pv_san: List[str] = []
            temp_board = board.copy()
            for pv_m in pv_moves:
                if pv_m in temp_board.legal_moves:
                    pv_san.append(temp_board.san(pv_m))
                    temp_board.push(pv_m)
                else:
                    break

            structure_hint = classify_pawn_structure(board)

            # Advance board with the played move and evaluate resulting position
            board.push(move)
            if move == best_move:
                eval_after_cp = eval_before_cp
                cp_loss = 0
            else:
                eval_after_cp = evaluator.evaluate_after_move(board, player_color)
                cp_loss = max(0, eval_before_cp - eval_after_cp)

            record = MoveEvalRecord(
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

            if in_middlegame_window:
                middlegame_records.append(record)
                # Module 3: Strategic "Plan Finder" Flashcard criteria
                # Position was roughly balanced (-280 to +280 cp), best move was quiet,
                # and the player lost between 80 and 260 cp (strategic/planning error).
                if (
                    FLASHCARD_MIN_CP_LOSS <= cp_loss <= FLASHCARD_MAX_CP_LOSS
                    and is_best_quiet
                    and abs(eval_before_cp) <= 280
                ):
                    flashcard_candidates.append(record)

            if in_transition_window:
                transition_records.append(record)
        else:
            board.push(move)

    if not middlegame_records and not transition_records:
        return None

    mg_losses = [r.cp_loss for r in middlegame_records]
    mg_acpl = statistics.mean(mg_losses) if mg_losses else 0.0
    inaccuracies = sum(1 for loss in mg_losses if INACCURACY_THRESHOLD_CP <= loss < MISTAKE_THRESHOLD_CP)
    mistakes = sum(1 for loss in mg_losses if MISTAKE_THRESHOLD_CP <= loss < BLUNDER_THRESHOLD_CP)
    blunders = sum(1 for loss in mg_losses if loss >= BLUNDER_THRESHOLD_CP)

    transition_drop_cp = float(sum(r.cp_loss for r in transition_records))

    return GameAnalysisSummary(
        game_id=game_id,
        date_str=date_str,
        iso_week=iso_week,
        player_color=player_color,
        opponent_name=opponent_name,
        eco_code=eco_code,
        opening_name=opening_name,
        result_str=result_str,
        book_exit_move=book_exit_move or fallback_book_exit,
        middlegame_moves_count=len(middlegame_records),
        middlegame_acpl=round(mg_acpl, 1),
        inaccuracies=inaccuracies,
        mistakes=mistakes,
        blunders=blunders,
        transition_drop_cp=round(transition_drop_cp, 1),
        flashcard_candidates=flashcard_candidates,
    )


# ---------------------------------------------------------------------------
# Report & Flashcard Exporters
# ---------------------------------------------------------------------------

def export_flashcards_pgn(
    summaries: List[GameAnalysisSummary],
    player_username: str,
    output_pgn_path: Path,
) -> int:
    """
    Exports Module 3 Strategic Plan Finder positions into an annotated PGN file
    that can be imported directly into a Lichess Study (https://lichess.org/study).
    """
    count = 0
    with output_pgn_path.open("w", encoding="utf-8") as f:
        for summary in summaries:
            for rec in summary.flashcard_candidates:
                count += 1
                puzzle_game = chess.pgn.Game()
                puzzle_game.setup(rec.fen_before)
                puzzle_game.headers["Event"] = f"Middlegame Plan Finder #{count} ({summary.opening_name})"
                puzzle_game.headers["Site"] = summary.game_id
                puzzle_game.headers["Date"] = summary.date_str
                puzzle_game.headers["White"] = (
                    player_username if summary.player_color == chess.WHITE else summary.opponent_name
                )
                puzzle_game.headers["Black"] = (
                    player_username if summary.player_color == chess.BLACK else summary.opponent_name
                )
                puzzle_game.headers["ECO"] = summary.eco_code
                puzzle_game.headers["Opening"] = summary.opening_name

                puzzle_game.comment = (
                    f"[Move {rec.move_number}] {rec.pawn_structure_hint} | "
                    f"Question: What is the best positional plan here? "
                    f"(In the game you played {rec.played_move_san}, losing {rec.cp_loss} cp.)"
                )

                board = chess.Board(rec.fen_before)
                best_move = chess.Move.from_uci(rec.best_move_uci)
                played_move = chess.Move.from_uci(rec.played_move_uci)

                # Mainline = Engine's recommended strategic plan
                main_node = puzzle_game.add_variation(best_move)
                pv_text = " -> ".join(rec.pv_san) if rec.pv_san else rec.best_move_san
                main_node.comment = (
                    f"Best Plan ({rec.best_move_san}, Eval {rec.eval_before_cp:+d} cp): "
                    f"Recommended continuation: {pv_text}"
                )

                # Sideline = What was played in the game
                if played_move != best_move and played_move in board.legal_moves:
                    side_node = puzzle_game.add_variation(played_move)
                    side_node.comment = (
                        f"Game Move ({rec.played_move_san}, Eval {rec.eval_after_cp:+d} cp, "
                        f"Loss: -{rec.cp_loss} cp). Review why {rec.best_move_san} better serves the pawn structure."
                    )

                exporter = chess.pgn.FileExporter(f)
                puzzle_game.accept(exporter)
                f.write("\n\n")

    return count


def export_csv_metrics(summaries: List[GameAnalysisSummary], output_csv_path: Path) -> None:
    """Writes per-game middlegame and transition metrics to a CSV file."""
    fieldnames = [
        "date",
        "iso_week",
        "color",
        "opponent",
        "eco",
        "opening",
        "book_exit_move",
        "mg_moves_evaluated",
        "mg_acpl",
        "mg_inaccuracies",
        "mg_mistakes",
        "mg_blunders",
        "post_book_5move_drop_cp",
        "flashcards_generated",
        "game_id",
    ]
    with output_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries:
            writer.writerow({
                "date": s.date_str,
                "iso_week": s.iso_week,
                "color": "White" if s.player_color == chess.WHITE else "Black",
                "opponent": s.opponent_name,
                "eco": s.eco_code,
                "opening": s.opening_name,
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


def generate_markdown_report(
    summaries: List[GameAnalysisSummary],
    player_username: str,
    flashcard_pgn_path: Path,
    csv_path: Path,
) -> str:
    """Builds a structured summary report across all three modules."""
    if not summaries:
        return "No matching games with middlegame positions were found."

    # 1. Weekly Progress Aggregation (Module 1)
    by_week: Dict[str, List[GameAnalysisSummary]] = defaultdict(list)
    for s in summaries:
        by_week[s.iso_week].append(s)

    # 2. Opening Transition Aggregation (Module 2)
    by_opening: Dict[str, List[GameAnalysisSummary]] = defaultdict(list)
    for s in summaries:
        key = f"{s.eco_code} — {s.opening_name}"
        by_opening[key].append(s)

    lines: List[str] = []
    lines.append(f"# Middlegame Strength & Transition Report for `{player_username}`")
    lines.append("")
    lines.append(f"- **Games Analyzed:** {len(summaries)}")
    overall_acpl = statistics.mean(s.middlegame_acpl for s in summaries)
    overall_drop = statistics.mean(s.transition_drop_cp for s in summaries)
    total_flashcards = sum(len(s.flashcard_candidates) for s in summaries)
    lines.append(f"- **Overall Middlegame ACPL (Moves 11–30):** `{overall_acpl:.1f} cp`")
    lines.append(f"- **Average 5-Move Post-Book Evaluation Drop:** `{overall_drop:.1f} cp`")
    lines.append(f"- **Strategic Plan Flashcards Generated:** `{total_flashcards}` (saved to `{flashcard_pgn_path.name}`)")
    lines.append(f"- **Raw Game Metrics CSV:** `{csv_path.name}`")
    lines.append("")

    # Module 1 Table
    lines.append("## Module 1: Weekly Middlegame Accuracy Progress (Moves 11–30)")
    lines.append("")
    lines.append("| ISO Week | Games | Avg MG ACPL (cp) | Avg Post-Book 5-Move Drop (cp) | Inaccuracies | Mistakes | Blunders |")
    lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for week in sorted(by_week.keys()):
        w_games = by_week[week]
        w_acpl = statistics.mean(g.middlegame_acpl for g in w_games)
        w_drop = statistics.mean(g.transition_drop_cp for g in w_games)
        w_inac = sum(g.inaccuracies for g in w_games)
        w_mist = sum(g.mistakes for g in w_games)
        w_blun = sum(g.blunders for g in w_games)
        lines.append(
            f"| {week} | {len(w_games)} | {w_acpl:.1f} | {w_drop:.1f} | {w_inac} | {w_mist} | {w_blun} |"
        )
    lines.append("")

    # Module 2 Table (Sorted by highest transition drop first so you know what to study)
    lines.append("## Module 2: Opening-to-Middlegame Transition Audit (Sorted by Highest Drop)")
    lines.append("")
    lines.append("| ECO & Opening | Games | Avg Book Exit Move | Avg 5-Move Post-Book Drop (cp) | Avg Middlegame ACPL (cp) |")
    lines.append("| :--- | ---: | ---: | ---: | ---: |")
    opening_rows = []
    for op_name, op_games in by_opening.items():
        avg_exit = statistics.mean(g.book_exit_move for g in op_games)
        avg_drop = statistics.mean(g.transition_drop_cp for g in op_games)
        avg_acpl = statistics.mean(g.middlegame_acpl for g in op_games)
        opening_rows.append((avg_drop, op_name, len(op_games), avg_exit, avg_acpl))

    for avg_drop, op_name, count, avg_exit, avg_acpl in sorted(opening_rows, reverse=True):
        lines.append(
            f"| {op_name} | {count} | Move {avg_exit:.1f} | **{avg_drop:.1f} cp** | {avg_acpl:.1f} cp |"
        )
    lines.append("")

    # Module 3 Summary
    lines.append("## Module 3: Personal Middlegame 'Plan Finder' Flashcards")
    lines.append("")
    if total_flashcards == 0:
        lines.append("No quiet strategic planning mistakes (80–260 cp loss) were flagged in this batch.")
    else:
        lines.append(
            f"Import `{flashcard_pgn_path.name}` into a [Lichess Study](https://lichess.org/study) "
            "to practice these exact middlegame positions from your own games:"
        )
        lines.append("")
        lines.append("| # | Date | Opening | Move | Played Move (Loss) | Recommended Plan (PV) | Structure & Strategic Plan |")
        lines.append("| ---: | :--- | :--- | ---: | :--- | :--- | :--- |")
        idx = 0
        for s in summaries:
            for rec in s.flashcard_candidates:
                idx += 1
                pv_str = " ".join(rec.pv_san[:4]) if rec.pv_san else rec.best_move_san
                lines.append(
                    f"| {idx} | {s.date_str} | {s.eco_code} {s.opening_name} | "
                    f"Move {rec.move_number} | `{rec.played_move_san}` (-{rec.cp_loss} cp) | "
                    f"**`{rec.best_move_san}`** (`{pv_str}`) | {rec.pawn_structure_hint} |"
                )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Embedded Demo Games (for `--demo` verification without external files)
# ---------------------------------------------------------------------------

DEMO_PGN_TEXT = """
[Event "Rapid 15+10 Training Game 1"]
[Site "https://lichess.org/demo0001"]
[Date "2026.10.05"]
[White "HeroPlayer"]
[Black "ClubOpponent1"]
[Result "0-1"]
[ECO "B13"]
[Opening "Caro-Kann Defense: Exchange Variation"]

1. e4 c6 2. d4 d5 3. exd5 cxd5 4. Bd3 Nc6 5. c3 Nf6 6. Bf4 Bg4 7. Qb3 Qd7
8. Nd2 e6 9. Ngf3 Bxf3 10. Nxf3 Bd6 11. Bxd6 Qxd6 12. O-O O-O 13. Nh4 Rab8
14. a4 Rfc8 15. Qd1 b6 16. g3 Na5 17. Ng2 Nc4 18. Qe2 b5 0-1

[Event "Rapid 15+10 Training Game 2"]
[Site "https://lichess.org/demo0002"]
[Date "2026.10.14"]
[White "ClubOpponent2"]
[Black "HeroPlayer"]
[Result "1-0"]
[ECO "D35"]
[Opening "Queen's Gambit Declined: Exchange Variation"]

1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. cxd5 exd5 5. Bg5 Be7 6. e3 c6 7. Bd3 Nbd7
8. Nf3 O-O 9. Qc2 Re8 10. O-O Nf8 11. Rab1 a5 12. a3 Nh5 13. Bxe7 Qxe7
14. b4 axb4 15. axb4 Ng6 16. b5 Nh4 17. Nxh4 Qxh4 18. bxc6 bxc6 1-0

[Event "Rapid 15+10 Training Game 3"]
[Site "https://lichess.org/demo0003"]
[Date "2026.10.20"]
[White "HeroPlayer"]
[Black "ClubOpponent3"]
[Result "1-0"]
[ECO "C02"]
[Opening "French Defense: Advance Variation"]

1. e4 e6 2. d4 d5 3. e5 c5 4. c3 Nc6 5. Nf3 Qb6 6. a3 Nh6 7. b4 cxd4
8. cxd4 Nf5 9. Bb2 Bd7 10. Bd3 Be7 11. O-O O-O 12. Bxf5 exf5 13. Nc3 Be6
14. Na4 Qb5 15. Nc3 Qc4 16. Rc1 Rac8 17. Re1 Rfd8 18. Qd2 h6 1-0
"""


# ---------------------------------------------------------------------------
# Command-Line Entry Point
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Track your chess middlegame accuracy, post-opening transitions, and strategic flashcards."
    )
    parser.add_argument(
        "--player",
        type=str,
        default="HeroPlayer",
        help="Your exact username as it appears in the PGN White/Black headers.",
    )
    parser.add_argument(
        "--pgn",
        type=str,
        default=None,
        help="Path to a local .pgn file containing your games.",
    )
    parser.add_argument(
        "--fetch-lichess",
        type=int,
        default=0,
        metavar="N_GAMES",
        help="Automatically download your last N rapid games from Lichess API for --player.",
    )
    parser.add_argument(
        "--perf-type",
        type=str,
        default="rapid",
        help="Lichess time control when using --fetch-lichess (default: rapid).",
    )
    parser.add_argument(
        "--stockfish",
        type=str,
        default=None,
        help="Path to the Stockfish UCI executable (e.g., /usr/games/stockfish or stockfish.exe).",
    )
    parser.add_argument(
        "--book",
        type=str,
        default=None,
        help="Optional path to a Polyglot opening book (.bin) to detect exact book exit moves.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=14,
        help="Stockfish search depth per position (default: 14).",
    )
    parser.add_argument(
        "--mg-start",
        type=int,
        default=DEFAULT_MIDDLEGAME_START_MOVE,
        help=f"Starting move number for Middlegame accuracy tracking (default: {DEFAULT_MIDDLEGAME_START_MOVE}).",
    )
    parser.add_argument(
        "--mg-end",
        type=int,
        default=DEFAULT_MIDDLEGAME_END_MOVE,
        help=f"Ending move number for Middlegame accuracy tracking (default: {DEFAULT_MIDDLEGAME_END_MOVE}).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory where the Markdown report, CSV metrics, and Flashcards PGN will be saved.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a quick self-test using embedded sample games and the built-in heuristic evaluator.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # 1. Load PGN source
    if args.demo:
        print("[*] Running in --demo mode with 3 embedded rapid games for 'HeroPlayer'...")
        pgn_text = DEMO_PGN_TEXT
        player_name = "HeroPlayer"
    elif args.fetch_lichess > 0:
        pgn_text = fetch_lichess_games(args.player, max_games=args.fetch_lichess, perf_type=args.perf_type)
        player_name = args.player
    elif args.pgn:
        pgn_path = Path(args.pgn)
        if not pgn_path.is_file():
            print(f"[!] Error: PGN file not found at {pgn_path}", file=sys.stderr)
            return 1
        pgn_text = pgn_path.read_text(encoding="utf-8", errors="replace")
        player_name = args.player
    else:
        parser.print_help()
        print("\n[!] Please supply --pgn <file.pgn>, --fetch-lichess <N>, or --demo.")
        return 1

    games = load_pgn_games(pgn_text)
    print(f"[*] Loaded {len(games)} game(s) from PGN input.")

    # 2. Initialize Evaluator & Optional Polyglot Opening Book
    evaluator = PositionEvaluator(
        stockfish_path=args.stockfish,
        depth=args.depth,
        use_demo_heuristic=args.demo,
    )

    book_reader: Optional[chess.polyglot.MemoryMappedReader] = None
    if args.book:
        book_path = Path(args.book)
        if book_path.is_file():
            book_reader = chess.polyglot.open_reader(str(book_path))
            print(f"[*] Loaded Polyglot opening book: {book_path}")
        else:
            print(f"[!] Warning: Polyglot book '{book_path}' not found; using move {DEFAULT_BOOK_EXIT_FALLBACK_MOVE} fallback.")

    # 3. Run Analysis across all games
    summaries: List[GameAnalysisSummary] = []
    try:
        for idx, game in enumerate(games, start=1):
            summary = analyze_single_game(
                game=game,
                player_username=player_name,
                evaluator=evaluator,
                book_reader=book_reader,
                mg_start=args.mg_start,
                mg_end=args.mg_end,
            )
            if summary is not None:
                summaries.append(summary)
                print(
                    f"    [{idx}/{len(games)}] {summary.date_str} | {summary.eco_code} {summary.opening_name} | "
                    f"MG ACPL: {summary.middlegame_acpl:.1f} cp | "
                    f"Post-Book 5-Move Drop: {summary.transition_drop_cp:.1f} cp | "
                    f"Flashcards: {len(summary.flashcard_candidates)}"
                )
    finally:
        evaluator.close()
        if book_reader is not None:
            book_reader.close()

    if not summaries:
        print(f"[!] No games found matching player '{player_name}'. Check your --player username.")
        return 1

    # 4. Export Outputs (Flashcard PGN, CSV Metrics, Markdown Progress Report)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    flashcard_pgn_path = out_dir / "middlegame_plan_flashcards.pgn"
    csv_path = out_dir / "middlegame_metrics.csv"
    report_md_path = out_dir / "middlegame_progress_report.md"

    flashcard_count = export_flashcards_pgn(summaries, player_name, flashcard_pgn_path)
    export_csv_metrics(summaries, csv_path)
    report_md = generate_markdown_report(summaries, player_name, flashcard_pgn_path, csv_path)
    report_md_path.write_text(report_md, encoding="utf-8")

    print("\n" + "=" * 76)
    print(report_md)
    print("=" * 76)
    print(f"[+] Saved Markdown Report : {report_md_path}")
    print(f"[+] Saved CSV Metrics     : {csv_path}")
    print(f"[+] Saved Study PGN ({flashcard_count} flashcards): {flashcard_pgn_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())