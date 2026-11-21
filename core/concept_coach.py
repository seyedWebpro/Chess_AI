"""
concept_coach.py: AI Grandmaster Conceptual & Strategic Voice Coach.
Translates move recommendations into pedagogical Persian grandmaster commentary,
explaining the strategic "WHY" behind every move (Pins, Forks, Skewers, Discovered Attacks,
Center Control, Outposts, King Safety, Pawn Breaks, and Tactical Threats).
"""

from typing import Dict, Any, List, Optional, Tuple, Set
import chess
from loguru import logger


# Persian piece terminology
PIECE_FA = {
    chess.PAWN: "پیاده",
    chess.KNIGHT: "اسب",
    chess.BISHOP: "فیل",
    chess.ROOK: "رخ",
    chess.QUEEN: "وزیر",
    chess.KING: "شاه",
}

# Persian square names in conversational Persian
SQUARE_FA = {
    "a": "آ", "b": "بی", "c": "سی", "d": "دی",
    "e": "ای", "f": "اف", "g": "جی", "h": "اچ"
}


def square_to_fa(sq_name: str) -> str:
    """Converts square e.g. 'e4' to natural Persian 'ای چهار'."""
    if len(sq_name) != 2:
        return sq_name
    col, row = sq_name[0], sq_name[1]
    col_fa = SQUARE_FA.get(col.lower(), col)
    row_map = {
        "1": "یک", "2": "دو", "3": "سه", "4": "چهار",
        "5": "پنج", "6": "شش", "7": "هفت", "8": "هشت"
    }
    return f"{col_fa} {row_map.get(row, row)}"


class TacticalTheme:
    CHECKMATE = "CHECKMATE"
    CHECK = "CHECK"
    CAPTURE = "CAPTURE"
    PIN = "PIN"
    FORK = "FORK"
    SKEWER = "SKEWER"
    DISCOVERED_ATTACK = "DISCOVERED_ATTACK"
    CASTLING = "CASTLING"
    CENTER_CONTROL = "CENTER_CONTROL"
    DEVELOPMENT = "DEVELOPMENT"
    DEFENSE = "DEFENSE"
    OUTPOST = "OUTPOST"
    PAWN_BREAK = "PAWN_BREAK"
    PROMOTION = "PROMOTION"
    TACTICAL_PRESSURE = "TACTICAL_PRESSURE"


class ConceptCoach:
    """
    Analyzes chess positions and generates insightful Persian grandmaster commentary
    explaining the strategic and tactical purpose behind moves.
    """

    def __init__(self):
        self.center_squares = {chess.E4, chess.D4, chess.E5, chess.D5}
        self.extended_center = {
            chess.C3, chess.D3, chess.E3, chess.F3,
            chess.C4, chess.D4, chess.E4, chess.F4,
            chess.C5, chess.D5, chess.E5, chess.F5,
            chess.C6, chess.D6, chess.E6, chess.F6,
        }

    def analyze_move(
        self,
        board: chess.Board,
        move: chess.Move,
        score_cp: Optional[float] = None,
        is_mate: bool = False,
        mate_in: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Deeply inspects board state before and after the move to identify strategic motivations.
        
        Args:
            board: Current board before move.
            move: The recommended chess.Move.
            score_cp: Engine evaluation centipawns.
            is_mate: Whether this line leads to mate.
            mate_in: Number of plies to mate.

        Returns:
            Dictionary containing themes, short concept, detailed pedagogical explanation in Persian.
        """
        if move not in board.legal_moves:
            return {
                "themes": [],
                "short_concept": "حرکت نامعتبر",
                "explanation_fa": "این حرکت طبق قوانین استاندارد شطرنج مجاز نیست.",
                "tactical_weight": 0,
            }

        themes: List[str] = []
        moved_piece = board.piece_at(move.from_square)
        if not moved_piece:
            return {"themes": [], "short_concept": "", "explanation_fa": ""}

        color = moved_piece.color
        opponent_color = not color
        piece_name = PIECE_FA.get(moved_piece.piece_type, "مهره")
        from_sq_name = chess.square_name(move.from_square)
        to_sq_name = chess.square_name(move.to_square)
        to_sq_fa = square_to_fa(to_sq_name)

        # 1. Checkmate or Mate Threat
        if is_mate or board.is_into_check(move):
            pass

        temp_board = board.copy()
        temp_board.push(move)

        if temp_board.is_checkmate():
            themes.append(TacticalTheme.CHECKMATE)
            return {
                "themes": themes,
                "short_concept": "مات قطعی",
                "explanation_fa": f"حرکت تمام‌کننده {piece_name} به {to_sq_fa} که مستقیماً به مات شدن شاه حریف و پیروزی بازی می‌انجامد.",
                "tactical_weight": 100,
            }

        # 2. Check
        is_check = temp_board.is_check()
        if is_check:
            themes.append(TacticalTheme.CHECK)

        # 3. Castling (King Safety)
        if board.is_castling(move):
            themes.append(TacticalTheme.CASTLING)
            side = "کوچک" if chess.square_file(move.to_square) == 6 else "بزرگ"
            return {
                "themes": themes,
                "short_concept": f"قلعه {side}",
                "explanation_fa": f"انجام قلعه {side} جهت تامین امنیت کامل شاه و اتصال رخ‌ها برای کنترل ستون‌های باز میانی.",
                "tactical_weight": 70,
            }

        # 4. Promotion
        if move.promotion:
            themes.append(TacticalTheme.PROMOTION)
            promoted_name = PIECE_FA.get(move.promotion, "وزیر")
            return {
                "themes": themes,
                "short_concept": f"ارتقای پیاده به {promoted_name}",
                "explanation_fa": f"پیشروی موفقیت‌آمیز پیاده تا خانه آخر و تبدیل آن به {promoted_name} جهت پیروزی قاطع در بازی.",
                "tactical_weight": 95,
            }

        # 5. Capture & Removing Defender
        is_capture = board.is_capture(move)
        captured_piece = board.piece_at(move.to_square)
        if is_capture:
            themes.append(TacticalTheme.CAPTURE)
            captured_name = PIECE_FA.get(captured_piece.piece_type, "مهره") if captured_piece else "مهره"

        # 6. Fork Analysis (Attacking 2+ higher/equal value pieces)
        fork_targets = self._detect_fork(temp_board, move.to_square, opponent_color)
        if len(fork_targets) >= 2:
            themes.append(TacticalTheme.FORK)

        # 7. Pin Detection (Pinning an enemy piece to King or Queen)
        pinned_enemy = self._detect_new_pin(temp_board, move.to_square, opponent_color)
        if pinned_enemy:
            themes.append(TacticalTheme.PIN)

        # 8. Discovered Attack
        if self._detect_discovered_attack(board, temp_board, move, opponent_color):
            themes.append(TacticalTheme.DISCOVERED_ATTACK)

        # 9. Center Control & Outpost
        if move.to_square in self.center_squares:
            themes.append(TacticalTheme.CENTER_CONTROL)
        if moved_piece.piece_type == chess.KNIGHT and self._is_outpost(temp_board, move.to_square, color):
            themes.append(TacticalTheme.OUTPOST)

        # 10. Defensive Retreat / Relieving Pressure
        was_attacked = board.is_attacked_by(opponent_color, move.from_square)
        is_attacked_now = temp_board.is_attacked_by(opponent_color, move.to_square)
        if was_attacked and not is_attacked_now:
            themes.append(TacticalTheme.DEFENSE)

        # 11. Opening Development (Plies 1-16)
        is_development = False
        if board.fullmove_number <= 8 and moved_piece.piece_type in {chess.KNIGHT, chess.BISHOP}:
            from_rank = chess.square_rank(move.from_square)
            back_rank = 0 if color == chess.WHITE else 7
            if from_rank == back_rank:
                is_development = True
                themes.append(TacticalTheme.DEVELOPMENT)

        # 12. Pawn Break
        if moved_piece.piece_type == chess.PAWN and not is_capture:
            if move.to_square in self.center_squares or chess.square_rank(move.to_square) in (3, 4):
                if any(temp_board.is_attacked_by(color, sq) for sq in self.center_squares):
                    themes.append(TacticalTheme.PAWN_BREAK)

        # Compose Persian Pedagogical Commentary
        short_concept, explanation_fa = self._generate_persian_text(
            board=board,
            move=move,
            moved_piece=moved_piece,
            themes=themes,
            fork_targets=fork_targets,
            pinned_enemy=pinned_enemy,
            is_check=is_check,
            is_capture=is_capture,
            captured_piece=captured_piece,
            score_cp=score_cp,
        )

        return {
            "themes": themes,
            "short_concept": short_concept,
            "explanation_fa": explanation_fa,
            "tactical_weight": len(themes) * 15 + (40 if is_check else 0) + (30 if is_capture else 0),
        }

    def _detect_fork(self, board_after: chess.Board, piece_sq: chess.Square, opponent_color: chess.Color) -> List[chess.PieceType]:
        """Detects if piece at piece_sq attacks 2 or more opponent pieces."""
        attacks = board_after.attacks(piece_sq)
        valuable_targets: List[chess.PieceType] = []
        for sq in attacks:
            p = board_after.piece_at(sq)
            if p and p.color == opponent_color and p.piece_type != chess.PAWN:
                valuable_targets.append(p.piece_type)
        return valuable_targets

    def _detect_new_pin(self, board_after: chess.Board, attacker_sq: chess.Square, opponent_color: chess.Color) -> Optional[chess.PieceType]:
        """Checks if the move created an absolute pin to the King or relative pin to Queen."""
        # 1. Absolute pin to King
        for sq in chess.SQUARES:
            p = board_after.piece_at(sq)
            if p and p.color == opponent_color and p.piece_type != chess.KING:
                if board_after.is_pinned(opponent_color, sq):
                    return p.piece_type

        # 2. Relative pin to Queen along sliders
        attacker = board_after.piece_at(attacker_sq)
        if attacker and attacker.piece_type in (chess.BISHOP, chess.ROOK, chess.QUEEN):
            queen_sqs = board_after.pieces(chess.QUEEN, opponent_color)
            for q_sq in queen_sqs:
                ray = chess.ray(attacker_sq, q_sq)
                if ray is not None and (chess.BB_SQUARES[attacker_sq] & ray) and (chess.BB_SQUARES[q_sq] & ray):
                    between = chess.between(attacker_sq, q_sq)
                    occupied_between = [sq for sq in chess.scan_forward(between) if board_after.piece_at(sq)]
                    if len(occupied_between) == 1:
                        pinned_sq = occupied_between[0]
                        pinned_p = board_after.piece_at(pinned_sq)
                        if pinned_p and pinned_p.color == opponent_color and pinned_p.piece_type != chess.QUEEN:
                            return pinned_p.piece_type
        return None

    def _detect_discovered_attack(
        self,
        board_before: chess.Board,
        board_after: chess.Board,
        move: chess.Move,
        opponent_color: chess.Color
    ) -> bool:
        """Checks if moving the piece unmasked a line of sight attack on king/queen."""
        from_sq = move.from_square
        # If moving piece exposes an attack from back rank
        for attacker_type in (chess.BISHOP, chess.ROOK, chess.QUEEN):
            for sq in board_before.pieces(attacker_type, board_before.turn):
                if sq != move.from_square:
                    attacks_before = board_before.attacks(sq)
                    attacks_after = board_after.attacks(sq)
                    if from_sq in attacks_after:
                        # Opened line
                        target_sqs = attacks_after - attacks_before
                        for t_sq in target_sqs:
                            tgt = board_after.piece_at(t_sq)
                            if tgt and tgt.color == opponent_color and tgt.piece_type in (chess.KING, chess.QUEEN):
                                return True
        return False

    def _is_outpost(self, board: chess.Board, sq: chess.Square, color: chess.Color) -> bool:
        """Square in 4th/5th/6th rank protected by our pawn and cannot be easily challenged by enemy pawns."""
        rank = chess.square_rank(sq)
        if color == chess.WHITE and rank not in (3, 4, 5):
            return False
        if color == chess.BLACK and rank not in (2, 3, 4):
            return False
        # Protected by pawn
        pawn_defenders = board.attackers(color, sq)
        has_pawn_support = any(board.piece_at(a) and board.piece_at(a).piece_type == chess.PAWN for a in pawn_defenders)
        return has_pawn_support

    def _generate_persian_text(
        self,
        board: chess.Board,
        move: chess.Move,
        moved_piece: chess.Piece,
        themes: List[str],
        fork_targets: List[chess.PieceType],
        pinned_enemy: Optional[chess.PieceType],
        is_check: bool,
        is_capture: bool,
        captured_piece: Optional[chess.Piece],
        score_cp: Optional[float],
    ) -> Tuple[str, str]:
        """Crafts human-like, articulate Persian grandmaster commentary."""
        piece_name = PIECE_FA.get(moved_piece.piece_type, "مهره")
        to_sq_fa = square_to_fa(chess.square_name(move.to_square))
        from_sq_name = chess.square_name(move.from_square)

        # Check + Fork
        if TacticalTheme.FORK in themes and is_check:
            targets_fa = " و ".join(PIECE_FA.get(t, "مهره") for t in fork_targets if t != chess.KING)
            short = "کیش و چنگال دوگانه"
            desc = f"ضربه مرگبار {piece_name} به {to_sq_fa} با کیش مستقیم به شاه و حمله چنگالی به {targets_fa} حریف."
            return short, desc

        # Pin
        if TacticalTheme.PIN in themes:
            pinned_name = PIECE_FA.get(pinned_enemy, "مهره") if pinned_enemy else "مهره کلیدی"
            short = f"آچمز کردن {pinned_name}"
            desc = f"استقرار استراتژیک {piece_name} در {to_sq_fa} جهت آچمز کردن {pinned_name} حریف و خلع سلاح تحرکات دفاعی آن."
            return short, desc

        # Fork
        if TacticalTheme.FORK in themes:
            targets_fa = " و ".join(PIECE_FA.get(t, "مهره") for t in set(fork_targets))
            short = "حمله چنگال دوگانه"
            desc = f"حمله تاکتیکی همزمان {piece_name} از خانه {to_sq_fa} به {targets_fa} حریف برای کسب برتری قطعی متریال."
            return short, desc

        # Capture + Check
        if is_capture and is_check:
            cap_name = PIECE_FA.get(captured_piece.piece_type, "مهره") if captured_piece else "مهره"
            short = f"زدن {cap_name} با کیش"
            desc = f"تصاحب مقتدرانه {cap_name} در {to_sq_fa} همراه با وارد کردن کیش و ایجاد تزلزل در سنگر شاه حریف."
            return short, desc

        # Clean Capture
        if is_capture:
            cap_name = PIECE_FA.get(captured_piece.piece_type, "مهره") if captured_piece else "مهره"
            short = f"تصاحب {cap_name}"
            desc = f"زدن {cap_name} حریف در {to_sq_fa} جهت افزایش برتری قوا و تضعیف ساختار پیاده‌ای دشمن."
            return short, desc

        # Check
        if is_check:
            short = "کیش تاکتیکی به شاه"
            desc = f"وارد کردن کیش با {piece_name} در {to_sq_fa} برای برهم زدن هماهنگی مهره‌ها و محدود کردن پناهگاه شاه."
            return short, desc

        # Center Outpost
        if TacticalTheme.OUTPOST in themes:
            short = "اشغال پایگاه مرکزی"
            desc = f"تثبیت {piece_name} در پایگاه مستحکم {to_sq_fa} با حمایت پیاده، بدون امکان رانده شدن توسط پیاده‌های حریف."
            return short, desc

        # Opening Development
        if TacticalTheme.DEVELOPMENT in themes:
            short = f"گسترش {piece_name}"
            desc = f"توسعه اصولی {piece_name} از ردیف آخر به {to_sq_fa} جهت افزایش فشار روی مرکز و تسریع در قلعه رفتن شاه."
            return short, desc

        # Defense / Escape
        if TacticalTheme.DEFENSE in themes:
            short = f"عقب‌نشینی و فرار {piece_name}"
            desc = f"خارج کردن به موقع {piece_name} از خطر حمله و استقرار در خانه امن {to_sq_fa}."
            return short, desc

        # Center Control
        if TacticalTheme.CENTER_CONTROL in themes:
            short = "تسلط بر مرکز"
            desc = f"پیشروی {piece_name} به خانه مرکزی {to_sq_fa} جهت تسلط بر فضاهای حیاتی صفحه و ممانعت از تحرک حریف."
            return short, desc

        # Pawn Break
        if TacticalTheme.PAWN_BREAK in themes:
            short = "شکست پیاده‌ای"
            desc = f"ضربه پیاده‌ای در {to_sq_fa} برای گشودن خطوط بازی، شکستن زنجیره حریف و فعال‌سازی سوارهای خودی."
            return short, desc

        # General positional default
        short = f"بهبود موقعیت {piece_name}"
        desc = f"انتقال هدفمند {piece_name} به {to_sq_fa} برای ارتقای ساختار هماهنگی مهره‌ها و اعمال فشار موقعیتی."
        return short, desc
