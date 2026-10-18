import streamlit as st
import chess
import chess.pgn
import chess.svg
import io
import json
import os
import requests
import pandas as pd

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Chess Thought-Process & Structure Analyzer",
    page_icon="♟️",
    layout="wide"
)

# --- PERSISTENT STORAGE FOR PUZZLES ---
PUZZLE_FILE = "mistake_puzzles.json"

def load_puzzles():
    if os.path.exists(PUZZLE_FILE):
        try:
            with open(PUZZLE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_puzzles(puzzles):
    with open(PUZZLE_FILE, "w") as f:
        json.dump(puzzles, f, indent=2)

# --- SESSION STATE INITIALIZATION ---
if "moves_data" not in st.session_state:
    st.session_state.moves_data = []  # List of dicts per ply
if "current_ply" not in st.session_state:
    st.session_state.current_ply = 0
if "annotations" not in st.session_state:
    st.session_state.annotations = {} # {ply_int: {"mistake": str, "note": str, "critical": str}}
if "puzzles" not in st.session_state:
    st.session_state.puzzles = load_puzzles()
if "flip_board" not in st.session_state:
    st.session_state.flip_board = False
if "puzzle_idx" not in st.session_state:
    st.session_state.puzzle_idx = 0

MISTAKE_CATEGORIES = [
    "Prophylaxis: Missed Opponent's Candidate / Threat",
    "Calculation: Tunnel Vision / Ignored Board Geometry",
    "Calculation: Stopped Calculating Too Early (Quiescence Error)",
    "Blunder Check: Left Piece Loose or Undefended",
    "Positional: Premature Pawn Break",
    "Positional: Bad Minor Piece Trade (Activated Opponent)",
    "Evaluation: Misjudged King Safety / Attack Speed",
    "Time Management: Rushed Move in Critical Position"
]

# --- CORE FUNCTIONS ---
def parse_pgn_string(pgn_text):
    pgn_io = io.StringIO(pgn_text.strip())
    game = chess.pgn.read_game(pgn_io)
    if not game:
        return False
    
    board = game.board()
    moves_list = []
    ply = 0
    for move in game.mainline_moves():
        ply += 1
        fen_before = board.fen()
        san = board.san(move)
        turn = "w" if board.turn == chess.WHITE else "b"
        move_num = board.fullmove_number
        
        # Count available captures before move for tension heuristic
        captures_avail = sum(1 for m in board.legal_moves if board.is_capture(m))
        is_pawn_move = board.piece_type_at(move.from_square) == chess.PAWN
        to_sq_name = chess.square_name(move.to_square)
        
        board.push(move)
        fen_after = board.fen()
        
        moves_list.append({
            "ply": ply,
            "move_num": move_num,
            "turn": turn,
            "san": san,
            "uci": move.uci(),
            "from_sq": move.from_square,
            "to_sq": move.to_square,
            "to_sq_name": to_sq_name,
            "is_pawn_move": is_pawn_move,
            "captures_avail": captures_avail,
            "fen_before": fen_before,
            "fen_after": fen_after
        })
        
    st.session_state.moves_data = moves_list
    st.session_state.current_ply = 1 if moves_list else 0
    st.session_state.annotations = {}
    return True

def render_board_svg(fen, last_move=None, flipped=False, size=400):
    board = chess.Board(fen)
    svg = chess.svg.board(
        board=board,
        lastmove=last_move,
        orientation=chess.BLACK if flipped else chess.WHITE,
        size=size
    )
    st.markdown(f'<div style="display:flex; justify-content:center;">{svg}</div>', unsafe_allow_html=True)

def evaluate_structure_and_plan(fen):
    board = chess.Board(fen)
    advice = []
    
    # 1. Gather Pawn Coordinates
    w_pawns = board.pieces(chess.PAWN, chess.WHITE)
    b_pawns = board.pieces(chess.PAWN, chess.BLACK)
    
    w_sq_names = {chess.square_name(sq) for sq in w_pawns}
    b_sq_names = {chess.square_name(sq) for sq in b_pawns}
    
    # Check Italian / Giuoco Piano Structure
    if "e4" in w_sq_names and "d3" in w_sq_names and "e5" in b_sq_names and "d6" in b_sq_names:
        advice.append((
            "📐 Structure: Giuoco Piano / Closed Italian (e4-d3 vs e5-d6)",
            "**Middlegame Plan:** Avoid rushing swaps in the center. Route the Queen's Knight via `Nd2-Nf1-Ng3` (eyeing `f5` and `d5`). Prepare the central break `c3` and `d3-d4` only after Rooks are centralized on `e1`/`d1`, or gain queenside space with `a4` and `b4`."
        ))
    # Check King's Indian / Locked Center
    elif "d4" in w_sq_names and "e4" in w_sq_names and "d6" in b_sq_names and "e5" in b_sq_names:
        advice.append((
            "📐 Structure: King's Indian / Central Tension",
            "**Middlegame Plan:** Clarify the center. If the center locks (`d4-d5`), White's plan is a queenside pawn break (`c4-c5`) targeting the base of Black's pawn chain (`d6`), while Black attacks on the kingside with `...f5` and `...g5`."
        ))
    elif "d5" in w_sq_names and "e4" in w_sq_names and "d6" in b_sq_names and "e5" in b_sq_names:
        advice.append((
            "📐 Structure: Locked KID Center (d5-e4 vs d6-e5)",
            "**Middlegame Plan:** Opposite-wing attacks! White must push `c4-c5` and infiltrate on the c-file. Black must prepare `...f5` supported by minor pieces aiming at the kingside."
        ))

    # 2. Open Files & IQP Detection
    w_files = {chess.square_file(sq) for sq in w_pawns}
    b_files = {chess.square_file(sq) for sq in b_pawns}
    
    open_files = [chess.FILE_NAMES[f].upper() for f in range(8) if f not in w_files and f not in b_files]
    if open_files:
        advice.append((
            f"🛣️ Open File(s) Detected: {', '.join(open_files)}-file",
            f"**Middlegame Plan:** Fight for control of the **{open_files[0]}-file** by doubling Rooks and looking for 7th/8th rank entry squares. Do not cede the open file without contesting it."
        ))

    # Isolated d-pawn check
    w_iqp = (3 in w_files) and (2 not in w_files) and (4 not in w_files)
    b_iqp = (3 in b_files) and (2 not in b_files) and (4 not in b_files)
    if w_iqp or b_iqp:
        side = "White" if w_iqp else "Black"
        advice.append((
            f"🎯 Isolated Queen's Pawn (IQP) Held by {side}",
            f"**Middlegame Plan:** {side} must play actively, plant a Knight on `e5`/`c5` (or `e4`/`c4`), build a kingside attack, and **avoid trading minor pieces**. The opponent should firmly blockade the square directly in front of the IQP with a Knight and trade pieces toward an endgame."
        ))

    # 3. Minor Piece Imbalances (Bishops vs Knights)
    w_bishops = len(board.pieces(chess.BISHOP, chess.WHITE))
    b_bishops = len(board.pieces(chess.BISHOP, chess.BLACK))
    
    if w_bishops == 2 and b_bishops < 2:
        advice.append((
            "⚖️ Imbalance: White Holds the Bishop Pair",
            "**Strategic Directive:** White should look for pawn breaks to open diagonals for the two bishops. Black should keep the pawn structure closed/fixed and anchor a Knight on a protected central outpost."
        ))
    elif b_bishops == 2 and w_bishops < 2:
        advice.append((
            "⚖️ Imbalance: Black Holds the Bishop Pair",
            "**Strategic Directive:** Black wants to open the center with pawn breaks. White should trade off one of Black's bishops or lock the central pawns on the color of Black's remaining unopposed bishop."
        ))

    return advice

def query_lichess_cloud_eval(fen):
    try:
        url = "https://lichess.org/api/cloud-eval"
        res = requests.get(url, params={"fen": fen, "multiPv": 2}, timeout=4)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return None

# Load default sample game on first launch
if not st.session_state.moves_data:
    default_pgn = "1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. c3 Nf6 5. d3 d6 6. O-O a6 7. Bb3 Ba7 8. Re1 O-O 9. h3 h6 10. Nbd2 Re8 11. Nf1 Be6 12. Bc2 d5 13. exd5 Bxd5 14. Ng3 Qd7 15. Nh4 Rad8"
    parse_pgn_string(default_pgn)

# --- HEADER & TABS ---
st.title("♟️ Chess Thought-Process & Structure Analyzer")
tab_analyze, tab_puzzles = st.tabs([
    "🔍 1. Game Analyzer & PGN Mistake Tagger", 
    f"🧩 2. Personal Mistake Puzzle Trainer ({len(st.session_state.puzzles)} Saved)"
])

# ==========================================
# TAB 1: GAME ANALYZER & PGN TAGGING
# ==========================================
with tab_analyze:
    col_left, col_mid, col_right = st.columns([1.15, 1.35, 1.2])
    
    # --- LEFT COLUMN: BOARD & INPUT ---
    with col_left:
        st.subheader("Interactive Board")
        
        # Determine current FEN and last move
        cur_ply = st.session_state.current_ply
        if cur_ply > 0 and cur_ply <= len(st.session_state.moves_data):
            cur_move_obj = st.session_state.moves_data[cur_ply - 1]
            active_fen = cur_move_obj["fen_after"]
            last_move = chess.Move.from_uci(cur_move_obj["uci"])
            move_label = f"Move {cur_move_obj['move_num']}{'.' if cur_move_obj['turn']=='w' else '...'} {cur_move_obj['san']}"
        else:
            active_fen = chess.STARTING_FEN
            last_move = None
            move_label = "Starting Position"
            
        render_board_svg(active_fen, last_move=last_move, flipped=st.session_state.flip_board, size=390)
        st.caption(f"**Viewing:** {move_label}")
        
        # Navigation Buttons
        b1, b2, b3, b4, b5 = st.columns(5)
        if b1.button("⏮ Start", use_container_width=True):
            st.session_state.current_ply = 0
            st.rerun()
        if b2.button("◀ Prev", use_container_width=True):
            st.session_state.current_ply = max(0, st.session_state.current_ply - 1)
            st.rerun()
        if b3.button("Next ▶", use_container_width=True):
            st.session_state.current_ply = min(len(st.session_state.moves_data), st.session_state.current_ply + 1)
            st.rerun()
        if b4.button("End ⏭", use_container_width=True):
            st.session_state.current_ply = len(st.session_state.moves_data)
            st.rerun()
        if b5.button("🔄 Flip", use_container_width=True):
            st.session_state.flip_board = not st.session_state.flip_board
            st.rerun()

        st.divider()
        st.subheader("Upload PGN or FEN")
        pgn_input = st.text_area(
            "Paste Played Game PGN:",
            value="1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. c3 Nf6 5. d3 d6 6. O-O a6 7. Bb3 Ba7 8. Re1 O-O 9. h3 h6 10. Nbd2 Re8 11. Nf1 Be6 12. Bc2 d5 13. exd5 Bxd5 14. Ng3 Qd7 15. Nh4 Rad8",
            height=110
        )
        c_load1, c_load2 = st.columns(2)
        if c_load1.button("📥 Load PGN Game", type="primary", use_container_width=True):
            if parse_pgn_string(pgn_input):
                st.success("PGN loaded!")
                st.rerun()
            else:
                st.error("Invalid PGN text.")
                
        if c_load2.button("⚡ Scan Critical Moments", use_container_width=True):
            flagged = 0
            for m in st.session_state.moves_data:
                if m["ply"] < 10:
                    continue
                is_central_break = m["is_pawn_move"] and m["to_sq_name"] in ["c4","d4","e4","f4","c5","d5","e5","f5"] and m["ply"] <= 36
                is_high_tension = m["captures_avail"] >= 4
                if (is_central_break or is_high_tension) and flagged < 5:
                    ann = st.session_state.annotations.get(m["ply"], {})
                    ann["critical"] = "Central Pawn Break" if is_central_break else "High Tactical Tension"
                    st.session_state.annotations[m["ply"]] = ann
                    flagged += 1
            st.toast(f"Flagged {flagged} critical moments in PGN!")
            st.rerun()

        fen_custom = st.text_input("Or Load a Single FEN Position:")
        if st.button("Load Single FEN", use_container_width=True) and fen_custom:
            try:
                b_test = chess.Board(fen_custom)
                st.session_state.moves_data = [{
                    "ply": 1, "move_num": 1, "turn": "w" if b_test.turn else "b",
                    "san": "Custom FEN", "uci": "0000", "from_sq": 0, "to_sq": 0,
                    "to_sq_name": "", "is_pawn_move": False, "captures_avail": 0,
                    "fen_before": fen_custom, "fen_after": fen_custom
                }]
                st.session_state.current_ply = 1
                st.rerun()
            except Exception:
                st.error("Invalid FEN string.")

    # --- MIDDLE COLUMN: PGN VIEWER & INLINE MISTAKE TAGGING ---
    with col_mid:
        st.subheader("Annotated PGN & Mistake Tagger")
        
        # Jump to move slider / selector
        if st.session_state.moves_data:
            move_options = {
                m["ply"]: f"{m['move_num']}{'.' if m['turn']=='w' else '...'} {m['san']}" 
                for m in st.session_state.moves_data
            }
            selected_ply = st.selectbox(
                "Select Move to Inspect or Tag:",
                options=list(move_options.keys()),
                format_func=lambda x: move_options[x],
                index=max(0, st.session_state.current_ply - 1) if st.session_state.current_ply > 0 else 0
            )
            if selected_ply != st.session_state.current_ply:
                st.session_state.current_ply = selected_ply
                st.rerun()

        # Render Rich Annotated PGN Box with Inline Mistake Statements
        pgn_html_parts = []
        for m in st.session_state.moves_data:
            p = m["ply"]
            if m["turn"] == "w":
                pgn_html_parts.append(f'<span style="color:#94a3b8; font-weight:bold;">{m["move_num"]}.</span>')
            
            is_cur = (p == st.session_state.current_ply)
            bg_style = "background:#38bdf8; color:#000; font-weight:bold; padding:2px 6px; border-radius:4px;" if is_cur else "padding:2px 4px;"
            pgn_html_parts.append(f'<span style="{bg_style}">{m["san"]}</span>')
            
            ann = st.session_state.annotations.get(p, {})
            if ann.get("critical"):
                pgn_html_parts.append(
                    f'<span style="background:rgba(245,158,11,0.2); color:#fde68a; border:1px solid #f59e0b; font-size:0.78rem; padding:1px 6px; border-radius:4px;">⚡ {ann["critical"]}</span>'
                )
            if ann.get("mistake"):
                note_str = f" — {ann['note']}" if ann.get("note") else ""
                pgn_html_parts.append(
                    f'<span style="background:rgba(244,63,94,0.2); color:#fda4af; border:1px solid #f43f5e; font-size:0.78rem; padding:1px 6px; border-radius:4px;">[Mistake: {ann["mistake"]}{note_str}]</span>'
                )

        pgn_box_html = f"""
        <div style="background:#0f172a; border:1px solid #334155; border-radius:8px; padding:14px; height:250px; overflow-y:auto; line-height:2.0; font-size:0.95rem;">
            {" ".join(pgn_html_parts)}
        </div>
        """
        st.markdown(pgn_box_html, unsafe_allow_html=True)

        # Mistake Tagging Form
        st.markdown("#### 📌 Tag Your Thought-Process Mistake on Current Move")
        with st.container(border=True):
            chosen_mistake = st.selectbox("Pre-Listed Thought-Process Leak:", MISTAKE_CATEGORIES)
            custom_note = st.text_input("Corrective Plan / Calculation Note (visible in PGN & Puzzle):", placeholder="e.g., Missed opponent's ...d5 break; should have played Re1 first")
            
            t_col1, t_col2 = st.columns(2)
            if t_col1.button("🚨 Tag Mistake & Save Puzzle", type="primary", use_container_width=True):
                if st.session_state.current_ply > 0:
                    p = st.session_state.current_ply
                    cur_m = st.session_state.moves_data[p - 1]
                    ann = st.session_state.annotations.get(p, {})
                    ann["mistake"] = chosen_mistake
                    ann["note"] = custom_note
                    st.session_state.annotations[p] = ann
                    
                    # Save FEN *before* the mistake was made into the persistent puzzle bank
                    new_puzzle = {
                        "fen_before": cur_m["fen_before"],
                        "played_move": f"{cur_m['move_num']}{'.' if cur_m['turn']=='w' else '...'} {cur_m['san']}",
                        "turn": cur_m["turn"],
                        "category": chosen_mistake,
                        "note": custom_note or "Check opponent's forcing replies and evaluate pawn structure before moving."
                    }
                    st.session_state.puzzles.append(new_puzzle)
                    save_puzzles(st.session_state.puzzles)
                    st.success("Mistake tagged in PGN & position saved to Puzzle Trainer!")
                    st.rerun()
            if t_col2.button("🧹 Clear Tag on Move", use_container_width=True):
                if st.session_state.current_ply in st.session_state.annotations:
                    del st.session_state.annotations[st.session_state.current_ply]
                    st.rerun()

    # --- RIGHT COLUMN: STRUCTURE & PLAN COACH ---
    with col_right:
        st.subheader("Middlegame Plan & Structure Coach")
        
        if st.button("☁️ Query Lichess Cloud Engine for Position", use_container_width=True):
            cloud_data = query_lichess_cloud_eval(active_fen)
            if cloud_data and "pvs" in cloud_data:
                pv = cloud_data["pvs"][0]
                score = f"{pv['cp']/100:+.2f}" if "cp" in pv else f"Mate in {pv.get('mate')}"
                best_line = " ".join(pv.get("moves", "").split()[:5])
                st.info(f"**Cloud Eval:** `{score}`  \n**Top Engine Line (UCI):** `{best_line}`")
            else:
                st.warning("Position not cached in Lichess Cloud database. Rely on the structural breakdown below.")

        advice_cards = evaluate_structure_and_plan(active_fen)
        for title, body in advice_cards:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.markdown(body)

        with st.container(border=True):
            st.markdown("**🧠 Pre-Move Thought-Process Protocol**")
            st.markdown(
                "1. **Opponent's Intent:** What is my opponent threatening if it were their turn right now?  \n"
                "2. **Candidate Generation:** List 2–3 candidate moves *before* calculating deep variations.  \n"
                "3. **Blunder Check:** Does my chosen move leave a piece or key square undefended?"
            )

# ==========================================
# TAB 2: PERSONAL MISTAKE PUZZLE TRAINER
# ==========================================
with tab_puzzles:
    p_col1, p_col2 = st.columns([1.1, 1.5])
    
    with p_col2:
        st.subheader("Filter Puzzles by Your Mistake Category")
        filter_cat = st.selectbox("Select Mistake Pattern to Drill:", ["ALL CATEGORIES"] + MISTAKE_CATEGORIES)
        
        filtered_list = (
            st.session_state.puzzles if filter_cat == "ALL CATEGORIES"
            else [p for p in st.session_state.puzzles if p["category"] == filter_cat]
        )

    with p_col1:
        st.subheader("Puzzle Board")
        if not filtered_list:
            render_board_svg(chess.STARTING_FEN, size=390)
            st.info("No puzzles saved in this category yet. Tag mistakes in Tab 1 to build your custom puzzle bank!")
        else:
            idx = st.session_state.puzzle_idx % len(filtered_list)
            active_puzzle = filtered_list[idx]
            render_board_svg(
                active_puzzle["fen_before"],
                flipped=(active_puzzle["turn"] == "b"),
                size=390
            )
            
            pc1, pc2 = st.columns(2)
            if pc1.button("◀ Previous Puzzle", use_container_width=True):
                st.session_state.puzzle_idx = (idx - 1) % len(filtered_list)
                st.rerun()
            if pc2.button("Next Puzzle ▶", type="primary", use_container_width=True):
                st.session_state.puzzle_idx = (idx + 1) % len(filtered_list)
                st.rerun()

    with p_col2:
        if filtered_list:
            idx = st.session_state.puzzle_idx % len(filtered_list)
            active_puzzle = filtered_list[idx]
            side_str = "White to Move" if active_puzzle["turn"] == "w" else "Black to Move"
            
            st.warning(
                f"### Puzzle {idx + 1} of {len(filtered_list)} — {side_str}\n"
                f"**Mistake Category Alert:** In your actual game, you committed a **`{active_puzzle['category']}`** in this exact position.\n\n"
                f"*Take 90 seconds: Identify the opponent's threats, evaluate the pawn structure, and choose your best candidate move.*"
            )
            
            with st.expander("💡 Click to Reveal What You Played & Your Corrective Plan"):
                st.markdown(f"**Move Played in Game:** `{active_puzzle['played_move']}`")
                st.markdown(f"**Thought-Process Leak:** `{active_puzzle['category']}`")
                st.markdown(f"**Corrective Plan / Note:** {active_puzzle['note']}")
                
                if st.button("☁️ Check Lichess Cloud Best Move for This Puzzle"):
                    c_eval = query_lichess_cloud_eval(active_puzzle["fen_before"])
                    if c_eval and "pvs" in c_eval:
                        pv = c_eval["pvs"][0]
                        st.success(f"Engine Best Continuation (UCI): `{pv.get('moves', '')[:25]}`")
                    else:
                        st.info("Position not in Lichess Cloud cache.")

        st.divider()
        st.subheader("📊 Your Mistake Distribution Analytics")
        if st.session_state.puzzles:
            df = pd.DataFrame(st.session_state.puzzles)
            counts = df["category"].value_counts().reset_index()
            counts.columns = ["Mistake Category", "Count"]
            st.dataframe(counts, use_container_width=True, hide_index=True)
            
            # Export / Reset Controls
            ex_col1, ex_col2 = st.columns(2)
            ex_col1.download_button(
                "📥 Backup Puzzle Bank (JSON)",
                data=json.dumps(st.session_state.puzzles, indent=2),
                file_name="my_chess_mistakes.json",
                mime="application/json",
                use_container_width=True
            )
            if ex_col2.button("🗑️ Clear All Saved Puzzles", use_container_width=True):
                st.session_state.puzzles = []
                save_puzzles([])
                st.rerun()
        else:
            st.caption("Tag mistakes across your games to see a breakdown of your biggest thought-process leaks.")