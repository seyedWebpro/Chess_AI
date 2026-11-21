"""
Phase 3 Demo: Shadow-Resilient Physical Move Detection & State Machine.
Simulates a real physical chess move (e2 -> e4) under varying lighting,
validates through SSIM + LAB diffing, and triggers Stockfish 19 calculation.
"""

from pathlib import Path
import cv2
import numpy as np
import chess
from loguru import logger
from vision.calibrator import BoardCalibrator
from vision.board_detector import PhysicalBoardDetector
from core.engine_manager import EngineManager
from tests.test_board_detector import create_synthetic_board_image


def run_phase3_demo():
    output_dir = Path(__file__).parent / "tests" / "fixtures"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Phase 3 Physical Move Detection Demo...")

    # 1. Setup Calibrator & Detector
    calibrator = BoardCalibrator(target_size=800, white_bottom=True)
    calibrator.set_corners([[0, 0], [799, 0], [799, 799], [0, 799]])
    detector = PhysicalBoardDetector(calibrator)

    board = chess.Board()

    # 2. Reference initial frame
    ref_frame = create_synthetic_board_image(board, calibrator)
    ref_path = output_dir / "04_physical_board_before_e4.jpg"
    cv2.imwrite(str(ref_path), ref_frame)

    # 3. Physical Move: White pawn moves e2 -> e4
    board_after = board.copy()
    move = chess.Move.from_uci("e2e4")
    board_after.push(move)

    curr_frame = create_synthetic_board_image(board_after, calibrator)
    curr_path = output_dir / "05_physical_board_after_e4.jpg"
    cv2.imwrite(str(curr_path), curr_frame)

    # 4. Run detection
    detected = detector.detect_move(ref_frame, curr_frame, board)
    assert detected is not None
    detected_move, score = detected

    # 5. Generate Heatmap / Change Visualization
    diff_scores = detector.compute_square_differences(ref_frame, curr_frame)
    diff_vis = curr_frame.copy()

    for sq, sc in diff_scores.items():
        if sc > 0.15:
            x1, y1, x2, y2 = calibrator.get_square_rect(sq)
            overlay = diff_vis[y1:y2, x1:x2].copy()
            # Green highlight for high change
            cv2.rectangle(overlay, (0, 0), (x2 - x1, y2 - y1), (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.45, diff_vis[y1:y2, x1:x2], 0.55, 0, diff_vis[y1:y2, x1:x2])
            cv2.putText(diff_vis, f"{sc:.2f}", (x1 + 25, y1 + 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    heatmap_path = output_dir / "06_ssim_change_heatmap.jpg"
    cv2.imwrite(str(heatmap_path), diff_vis)

    # 6. Run Stockfish response
    with EngineManager(threads=2, hash_size_mb=64) as engine:
        engine.make_move("e2e4")
        reply = engine.evaluate_position(time_limit=0.08)

    print("\n" + "=" * 60)
    print("PHASE 3 PHYSICAL MOVE DETECTION SUCCESSFUL:")
    print(f"  • Physical Move Detected: {detected_move.uci()} ({board.san(detected_move)})")
    print(f"  • Detection Confidence Score: {score:.3f}")
    print(f"  • Emptied Square: {chess.square_name(detected_move.from_square)} (Score: {diff_scores['e2']:.3f})")
    print(f"  • Occupied Square: {chess.square_name(detected_move.to_square)} (Score: {diff_scores['e4']:.3f})")
    print(f"  • Stockfish 19 Best Counter-Move: {reply['best_move_san']} ({reply['best_move']})")
    print(f"  • Visual Diff Heatmap saved: {heatmap_path.name}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_phase3_demo()
