#!/usr/bin/env python3
"""
Chess Middlegame Strength & Progress Tracker — Streamlit Web App
================================================================

Designed for one-file deployment on Streamlit Community Cloud (including from
the GitHub mobile app).

Why the previous script kept loading continuously on Streamlit Cloud:
1. It was a command-line script that called `sys.exit(main())`, which killed
   the Streamlit server process on startup and left the browser spinning.
2. It had no `streamlit` UI components (`st.title`, `st.tabs`, etc.).
3. If `chess` (`python-chess`) was not in a `requirements.txt` file, Streamlit
   could not import it. This version automatically installs `chess` if missing
   and renders a mobile-friendly Streamlit dashboard with visual chessboards!

Recommended GitHub Repo Files for Fastest Streamlit Cloud Startup:
- `middlegame_trainer.py` (this file)
- `requirements.txt` (optional but recommended):
      streamlit
      chess
      requests
      pandas
- `packages.txt` (optional, only if you want real Stockfish on Streamlit Cloud):
      stockfish
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
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
    transition_drop_cp: float
    flashcard_candidates: List[MoveEvalRecord] = field(default_factory=list)


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
# Lichess API Downloader & PGN Loading
# ---------------------------------------------------------------------------

def fetch_lichess_games(username: str, max_games: int = 15, perf_type: str = "rapid") -> str:
    """Downloads a user's recent games in PGN format from the Lichess API."""
    if requests is None:
        raise RuntimeError("The 'requests' library is required to fetch games from Lichess.")

    url = f"https://lichess.org/api/games/user/{username.strip()}"
    params = {
        "max": str(max_games),
        "perfType": perf_type,
        "opening": "true",
        "moves": "true",
        "pgnInJson": "false",
    }
    headers = {"Accept": "application/x-chess-pgn"}
    response = requests.get(url, params=params, headers=headers, timeout=30)
    response.raise_for_status()
    return response.text


def load_pgn_games(pgn_text: str) -> List[chess.pgn.Game]:
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
    game: chess.pgn.Game,
    player_username: str,
    evaluator: PositionEvaluator,
    mg_start: int = DEFAULT_MIDDLEGAME_START_MOVE,
    mg_end: int = DEFAULT_MIDDLEGAME_END_MOVE,
    fallback_book_exit: int = DEFAULT_BOOK_EXIT_FALLBACK_MOVE,
) -> Optional[GameAnalysisSummary]:
    headers = game.headers
    white_name = headers.get("White", "").strip()
    black_name = headers.get("Black", "").strip()

    target = player_username.strip().lower()
    if target and white_name.lower() == target:
        player_color = chess.WHITE
        opponent_name = black_name
    elif target and black_name.lower() == target:
        player_color = chess.BLACK
        opponent_name = white_name
    elif not target:
        # If username left blank, analyze from White's perspective
        player_color = chess.WHITE
        opponent_name = black_name or "Opponent"
    else:
        return None

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

        if turn == player_color and (in_middlegame_window or in_transition_window):
            fen_before = board.fen()
            eval_before_cp, best_move, pv_moves = evaluator.analyse_position(board)

            played_san = board.san(move)
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


def build_flashcards_pgn_string(summaries: List[GameAnalysisSummary], player_username: str) -> str:
    """Returns an annotated PGN string of all Module 3 flashcards for Lichess Study import."""
    buf = io.StringIO()
    count = 0
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
    """Returns CSV text of all per-game middlegame metrics."""
    buf = io.StringIO()
    fieldnames = [
        "date", "iso_week", "color", "opponent", "eco", "opening",
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
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Embedded Demo Games (Works instantly with 1 tap on Mobile)
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
        "Measure your **Moves 11–30 Middlegame Accuracy (Module 1)**, audit your "
        "**5-Move Post-Opening Transition Drop (Module 2)**, and practice interactive "
        "**Strategic Plan Finder Flashcards (Module 3)**."
    )

    stockfish_bin = PositionEvaluator.resolve_stockfish()

    with st.expander("⚙️ Step 1: Choose Games & Analysis Settings", expanded=True):
        col1, col2 = st.columns([2, 1])
        with col1:
            source_mode = st.radio(
                "Select Game Source:",
                options=[
                    "1. Instant Demo Games (Test Immediately)",
                    "2. Fetch Live Games from Lichess API",
                    "3. Upload or Paste PGN File",
                ],
                index=0,
            )
        with col2:
            engine_options = ["Fast Built-In Positional Evaluator (Instant on Cloud)"]
            if stockfish_bin:
                engine_options.insert(0, f"Stockfish UCI Engine ({stockfish_bin})")
            engine_choice = st.selectbox("Evaluation Engine:", options=engine_options)
            use_fast_heuristic = not engine_choice.startswith("Stockfish")

        player_username = "HeroPlayer"
        lichess_count = 15
        lichess_perf = "rapid"
        uploaded_pgn_text = ""

        if source_mode.startswith("1."):
            st.info(
                "Loaded 3 sample rapid games for **`HeroPlayer`** (Caro-Kann Exchange, "
                "QGD Exchange, and French Advance). Tap **Run Middlegame Analysis** below!"
            )
        elif source_mode.startswith("2."):
            c_user, c_perf, c_num = st.columns([2, 1, 1])
            with c_user:
                player_username = st.text_input(
                    "Your Lichess Username:",
                    value="",
                    placeholder="e.g. DrNykterstein",
                ).strip()
            with c_perf:
                lichess_perf = st.selectbox("Time Control:", ["rapid", "classical", "blitz", "bullet"], index=0)
            with c_num:
                lichess_count = st.slider("Number of Games:", min_value=5, max_value=40, value=15, step=5)
        else:
            player_username = st.text_input(
                "Your Username (exact match to White/Black PGN tag):",
                value="",
                placeholder="Leave blank to analyze White in all games",
            ).strip()
            up_file = st.file_uploader("Upload a `.pgn` file:", type=["pgn", "txt"])
            if up_file is not None:
                uploaded_pgn_text = up_file.getvalue().decode("utf-8", errors="replace")
            else:
                uploaded_pgn_text = st.text_area(
                    "Or paste PGN text directly here:",
                    height=140,
                    placeholder='[Event "Rated Rapid game"]\n[White "YourName"]\n1. e4 c6 2. d4 d5 ...',
                )

        c_m1, c_m2, c_m3 = st.columns(3)
        with c_m1:
            mg_start = st.number_input("Middlegame Start Move:", min_value=6, max_value=20, value=11)
        with c_m2:
            mg_end = st.number_input("Middlegame End Move:", min_value=20, max_value=50, value=30)
        with c_m3:
            book_exit_default = st.number_input("Default Book Exit Move:", min_value=6, max_value=18, value=10)

        run_clicked = st.button("🚀 Run Middlegame Analysis", type="primary", use_container_width=True)

    # Run analysis on click (or automatically once on initial boot in Demo mode)
    if "summaries" not in st.session_state:
        st.session_state["summaries"] = None
        st.session_state["analyzed_player"] = "HeroPlayer"

    if run_clicked or (st.session_state["summaries"] is None and source_mode.startswith("1.")):
        if source_mode.startswith("1."):
            raw_pgn = DEMO_PGN_TEXT
            active_player = "HeroPlayer"
        elif source_mode.startswith("2."):
            if not player_username:
                st.warning("Please enter your Lichess username first.")
                return
            with st.spinner(f"Downloading last {lichess_count} {lichess_perf} games for `{player_username}` from Lichess..."):
                try:
                    raw_pgn = fetch_lichess_games(player_username, max_games=lichess_count, perf_type=lichess_perf)
                    active_player = player_username
                except Exception as exc:
                    st.error(f"Could not fetch games from Lichess API: {exc}")
                    return
        else:
            if not uploaded_pgn_text.strip():
                st.warning("Please upload a `.pgn` file or paste PGN text first.")
                return
            raw_pgn = uploaded_pgn_text
            active_player = player_username

        games = load_pgn_games(raw_pgn)
        if not games:
            st.error("No valid PGN games were found in the input.")
            return

        progress_bar = st.progress(0, text=f"Analyzing {len(games)} game(s)...")
        evaluator = PositionEvaluator(use_fast_heuristic=use_fast_heuristic)
        results: List[GameAnalysisSummary] = []
        try:
            for idx, g in enumerate(games, start=1):
                summary = analyze_single_game(
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
                    text=f"Analyzed {idx}/{len(games)} game(s)...",
                )
        finally:
            evaluator.close()
            progress_bar.empty()

        if not results:
            st.error(
                f"No games matched username `{active_player}` with middlegame moves. "
                "Check the spelling of your username or try more games."
            )
            return

        st.session_state["summaries"] = results
        st.session_state["analyzed_player"] = active_player or "White"

    summaries: Optional[List[GameAnalysisSummary]] = st.session_state.get("summaries")
    active_player_label: str = st.session_state.get("analyzed_player", "HeroPlayer")

    if not summaries:
        return

    # -----------------------------------------------------------------------
    # Top-Level KPI Metrics
    # -----------------------------------------------------------------------
    overall_acpl = statistics.mean(s.middlegame_acpl for s in summaries)
    overall_drop = statistics.mean(s.transition_drop_cp for s in summaries)
    total_flashcards = sum(len(s.flashcard_candidates) for s in summaries)
    total_mistakes = sum(s.mistakes + s.blunders for s in summaries)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Games Analyzed", f"{len(summaries)}")
    m2.metric("Middlegame ACPL (Moves 11–30)", f"{overall_acpl:.1f} cp")
    m3.metric("Avg 5-Move Post-Book Drop", f"{overall_drop:.1f} cp")
    m4.metric("Strategic Plan Flashcards", f"{total_flashcards}")

    # Download buttons
    dl1, dl2 = st.columns(2)
    pgn_export_str = build_flashcards_pgn_string(summaries, active_player_label)
    csv_export_str = build_csv_string(summaries)
    with dl1:
        st.download_button(
            "📥 Download Study Flashcards (.pgn for Lichess Study)",
            data=pgn_export_str or "[Event \"No flashcards flagged\"]\n\n*\n",
            file_name="middlegame_plan_flashcards.pgn",
            mime="application/x-chess-pgn",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            "📊 Download Full Middlegame Metrics (.csv)",
            data=csv_export_str,
            file_name="middlegame_metrics.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # -----------------------------------------------------------------------
    # Three Module Tabs
    # -----------------------------------------------------------------------
    tab1, tab2, tab3 = st.tabs([
        "📈 Module 1: Weekly Accuracy",
        "🧭 Module 2: Opening Transitions",
        "🧩 Module 3: Plan Finder Flashcards",
    ])

    # --- TAB 1: MODULE 1 ---
    with tab1:
        st.subheader("Module 1: Middlegame Phase Accuracy Tracker (Moves 11–30)")
        st.write(
            "Tracks your **Average Centipawn Loss (ACPL)** and error counts exclusively in the "
            "middlegame (moves 11–30 with 7+ pieces on the board). Watch for your weekly ACPL to trend downward."
        )
        by_week: Dict[str, List[GameAnalysisSummary]] = defaultdict(list)
        for s in summaries:
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

    # --- TAB 2: MODULE 2 ---
    with tab2:
        st.subheader("Module 2: Post-Opening Transition Audit (5-Move Post-Book Drop)")
        st.write(
            "Measures how much evaluation you lose in the **first 5 moves immediately after leaving opening theory**. "
            "Openings at the top of this table are where you know the book moves but struggle to find a middlegame plan."
        )
        by_opening: Dict[str, List[GameAnalysisSummary]] = defaultdict(list)
        for s in summaries:
            key = f"{s.eco_code} — {s.opening_name}"
            by_opening[key].append(s)

        opening_rows = []
        for op_name, op_games in by_opening.items():
            opening_rows.append({
                "ECO & Opening": op_name,
                "Games": len(op_games),
                "Avg Book Exit Move": round(statistics.mean(g.book_exit_move for g in op_games), 1),
                "Avg 5-Move Post-Book Drop (cp)": round(statistics.mean(g.transition_drop_cp for g in op_games), 1),
                "Avg Middlegame ACPL (cp)": round(statistics.mean(g.middlegame_acpl for g in op_games), 1),
            })
        opening_rows.sort(key=lambda r: r["Avg 5-Move Post-Book Drop (cp)"], reverse=True)

        if pd is not None and opening_rows:
            df_op = pd.DataFrame(opening_rows)
            st.bar_chart(df_op.set_index("ECO & Opening")["Avg 5-Move Post-Book Drop (cp)"])
            st.dataframe(df_op, use_container_width=True, hide_index=True)
        else:
            st.table(opening_rows)

    # --- TAB 3: MODULE 3 ---
    with tab3:
        st.subheader("Module 3: Personal Middlegame 'Plan Finder' Flashcards")
        st.write(
            "Quiet middlegame positions from your games where the best move was a **positional plan, "
            "pawn break, or piece reroute** (not a capture or check) and your move lost 80–260 centipawns."
        )
        all_cards: List[Tuple[GameAnalysisSummary, MoveEvalRecord]] = []
        for s in summaries:
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
                        st.markdown(f"**Opening:** `{s.eco_code}` {s.opening_name} ({s.date_str})")
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
    # When invoked from a standard terminal (e.g. `python middlegame_trainer.py --demo`)
    evaluator = PositionEvaluator(use_fast_heuristic=True)
    demo_games = load_pgn_games(DEMO_PGN_TEXT)
    print(f"[*] CLI Self-Test: Analyzing {len(demo_games)} embedded demo games...")
    for idx, g in enumerate(demo_games, start=1):
        res = analyze_single_game(g, "HeroPlayer", evaluator)
        if res:
            print(
                f"    [{idx}/{len(demo_games)}] {res.eco_code} {res.opening_name} | "
                f"MG ACPL: {res.middlegame_acpl:.1f} cp | "
                f"Post-Book 5-Move Drop: {res.transition_drop_cp:.1f} cp | "
                f"Flashcards: {len(res.flashcard_candidates)}"
            )
    evaluator.close()
    print("[+] Tip: Run `streamlit run middlegame_trainer.py` to launch the interactive web app!")