"""
Diagnostic tool: Extract frames and test Screen marker (12-15) detection
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "vendor" / "pl-neon-recording-main" / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "vendor" / "pl-marker-mapper-main" / "src"))

import cv2
import numpy as np
import pupil_apriltags
import pupil_labs.neon_recording as nr

# Configuration
RECORDING_PATH = pathlib.Path(r"C:\My Files\THOWL\Gamification\Pupil-labs\Recordings\Gamified")  # UPDATE THIS
FRAMES_TO_TEST = [100, 1000, 5000, 10000, 20000, 30000]  # Sample frames
SCREEN_MARKER_IDS = {12, 13, 14, 15}

def main():
    # Find first recording in Gamified folder
    recordings = list(RECORDING_PATH.glob("*/info.json"))
    if not recordings:
        print(f"No recordings found in {RECORDING_PATH}")
        return

    rec_dir = recordings[0].parent
    print(f"Testing recording: {rec_dir.name}")

    # Load recording
    recording = nr.load(str(rec_dir), load_video=True)
    print(f"Total frames: {len(recording.scene)}")

    # Create detector
    detector = pupil_apriltags.Detector(families="tag36h11", nthreads=4, quad_decimate=1.0)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    output_dir = rec_dir / "diagnostic_frames"
    output_dir.mkdir(exist_ok=True)

    for frame_idx in FRAMES_TO_TEST:
        if frame_idx >= len(recording.scene):
            continue

        # Get frame
        frame_ts = recording.scene.timestamps[frame_idx]
        frame = recording.scene.sample([frame_ts])[0]

        if frame is None:
            print(f"Frame {frame_idx}: Could not load")
            continue

        # Enhance
        gray = frame.gray
        enhanced = clahe.apply(gray)

        # Detect
        detections = detector.detect(enhanced)
        detected_ids = [d.tag_id for d in detections]
        screen_markers = [id for id in detected_ids if id in SCREEN_MARKER_IDS]

        print(f"Frame {frame_idx}: Total markers={len(detected_ids)}, Screen markers={screen_markers}")

        # Draw on frame
        rgb = cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2RGB)
        for det in detections:
            color = (0, 255, 0) if det.tag_id in SCREEN_MARKER_IDS else (255, 0, 0)
            corners = det.corners.astype(int)
            cv2.polylines(rgb, [corners], True, color, 2)
            cx, cy = int(det.center[0]), int(det.center[1])
            cv2.putText(rgb, str(det.tag_id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Save
        output_path = output_dir / f"frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(output_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        print(f"  Saved: {output_path}")

    print(f"\nDiagnostic frames saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Open the diagnostic frames")
    print("2. Check if Screen markers (12-15) are physically visible in the video")
    print("3. If visible but not detected → markers too small or poor lighting")
    print("4. If not visible at all → markers not in camera field of view")

if __name__ == "__main__":
    main()
