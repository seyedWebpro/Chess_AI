"""
Phase 2 Demo: Camera Pipeline, Perspective Warp, and Grid Calibration.
Demonstrates taking a tilted camera view of a physical chessboard and
transforming it into an 800x800 normalized top-down view with square mapping.
"""

from pathlib import Path
import cv2
import numpy as np
from loguru import logger
from vision.calibrator import BoardCalibrator


def run_phase2_demo():
    output_dir = Path(__file__).parent / "tests" / "fixtures"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running Phase 2 Vision & Perspective Calibration Demo...")

    # 1. Simulate a camera frame (1280x720) with a tilted board
    raw_frame = np.full((720, 1280, 3), 35, dtype=np.uint8)

    # Draw room elements / table
    cv2.putText(raw_frame, "Webcam Live Feed (1280x720 Tilted View)", (40, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)

    # 800x800 master board
    ideal_board = np.zeros((800, 800, 3), dtype=np.uint8)
    for r in range(8):
        for c in range(8):
            color = (235, 236, 208) if (r + c) % 2 == 0 else (119, 149, 86)  # Chess.com theme
            cv2.rectangle(ideal_board, (c * 100, r * 100), ((c + 1) * 100, (r + 1) * 100), color, -1)

    # Simulated physical camera perspective (tilted angle)
    src_pts = np.array([[0, 0], [799, 0], [799, 799], [0, 799]], dtype=np.float32)
    physical_corners = np.array([
        [280, 110],   # Top-Left (a8)
        [1000, 130],  # Top-Right (h8)
        [1160, 660],  # Bottom-Right (h1)
        [140, 630]    # Bottom-Left (a1)
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_pts, physical_corners)
    tilted_board = cv2.warpPerspective(ideal_board, M, (1280, 720))
    mask = cv2.cvtColor(tilted_board, cv2.COLOR_BGR2GRAY) > 0
    raw_frame[mask] = tilted_board[mask]

    # Save original camera view
    raw_cam_path = output_dir / "01_camera_tilted_view.jpg"
    cv2.imwrite(str(raw_cam_path), raw_frame)
    logger.info(f"Saved simulated raw camera frame: {raw_cam_path}")

    # 2. Calibrate and Warp to 800x800
    calibrator = BoardCalibrator(target_size=800, white_bottom=True)
    calibrator.set_corners(physical_corners)

    warped_800 = calibrator.warp_board(raw_frame)
    assert warped_800 is not None

    # 3. Draw grid verification lines and sample square centers on the warped view
    annotated_warped = warped_800.copy()
    for col in range(9):
        x = col * 100
        cv2.line(annotated_warped, (x, 0), (x, 800), (0, 255, 0), 1)
    for row in range(9):
        y = row * 100
        cv2.line(annotated_warped, (0, y), (800, y), (0, 255, 0), 1)

    # Highlight squares e2 and e4 with centers
    for sq in ["e2", "e4", "d2", "d4", "f7"]:
        cx, cy = calibrator.get_square_center(sq)
        cv2.circle(annotated_warped, (cx, cy), 8, (0, 0, 255), -1)
        cv2.putText(annotated_warped, sq.upper(), (cx - 15, cy - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    warped_path = output_dir / "02_warped_800x800_grid.jpg"
    cv2.imwrite(str(warped_path), annotated_warped)
    logger.info(f"Saved normalized 800x800 grid view: {warped_path}")

    # 4. Save calibration to config
    config_path = Path(__file__).parent / "config" / "calibration.json"
    calibrator.save_calibration(config_path)

    # 5. Reverse Projection test (Unwarp arrow from 800x800 onto original camera view)
    annotated_camera = raw_frame.copy()
    e2_center = calibrator.get_square_center("e2")
    e4_center = calibrator.get_square_center("e4")

    # Project to tilted camera coordinates
    cam_e2 = calibrator.unwarp_point(*e2_center)
    cam_e4 = calibrator.unwarp_point(*e4_center)

    # Draw AR arrow on tilted camera view
    cv2.arrowedLine(annotated_camera, cam_e2, cam_e4, (0, 255, 0), 5, tipLength=0.25)
    cv2.putText(annotated_camera, "AR Best Move: e2 -> e4", (cam_e4[0] - 60, cam_e4[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    ar_cam_path = output_dir / "03_ar_overlay_on_camera.jpg"
    cv2.imwrite(str(ar_cam_path), annotated_camera)
    logger.success(f"Saved AR reverse projection on tilted camera: {ar_cam_path}")

    print("\n" + "=" * 60)
    print("PHASE 2 VALIDATION SUCCESSFUL:")
    print(f"  1. Raw tilted camera capture: {raw_cam_path.name}")
    print(f"  2. Normalized 800x800 grid view: {warped_path.name}")
    print(f"  3. AR reverse projection arrow: {ar_cam_path.name}")
    print(f"  4. Saved calibration config: {config_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_phase2_demo()
