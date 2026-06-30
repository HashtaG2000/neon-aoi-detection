"""
Feature-based AOI detection using reference images from assets/
Falls back when AprilTag detection fails
"""
import cv2
import numpy as np
import pathlib
from typing import Optional, Tuple, List

# Assets path
ASSETS_DIR = pathlib.Path(__file__).parent.parent.parent / "assets"

# AOI reference image mapping
AOI_REFERENCE_IMAGES = {
    "Board": ASSETS_DIR / "board.jpeg",
    "Left_Box": ASSETS_DIR / "left_box.jpeg",
    "Middle_Box": ASSETS_DIR / "middle_box.jpeg",
    "Right_Box": ASSETS_DIR / "right_box.jpeg",
    "Screen": ASSETS_DIR / "screen_non_gamified.jpeg",  # Use non-gamified as reference
    "Stream_Deck": None,  # Will handle separately if needed
}

class FeatureBasedDetector:
    """Detects AOIs using ORB feature matching against reference images"""

    def __init__(self):
        # ORB detector (fast, rotation-invariant)
        self.orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)

        # BFMatcher with Hamming distance for ORB
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Per-frame keypoint cache: avoids re-running ORB.detectAndCompute()
        # when multiple AOIs are checked against the same frame (Bug 7 fix).
        self._cached_frame_id: int = -1
        self._cached_kp: list = []
        self._cached_desc: np.ndarray | None = None

        # Load and process reference images
        self.references = {}
        self._load_references()

    def _load_references(self):
        """Load reference images and compute keypoints/descriptors"""
        for aoi_name, img_path in AOI_REFERENCE_IMAGES.items():
            if img_path is None or not img_path.exists():
                continue

            # Load reference image
            ref_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if ref_img is None:
                print(f"[FeatureDetector] Warning: Could not load {img_path}")
                continue

            # Detect keypoints and compute descriptors
            kp, desc = self.orb.detectAndCompute(ref_img, None)

            if desc is None or len(kp) < 10:
                print(f"[FeatureDetector] Warning: Insufficient features in {aoi_name}")
                continue

            self.references[aoi_name] = {
                "image": ref_img,
                "keypoints": kp,
                "descriptors": desc,
                "shape": ref_img.shape  # (height, width)
            }

            print(f"[FeatureDetector] Loaded {aoi_name}: {len(kp)} keypoints")

    def detect_aoi(self, frame_gray: np.ndarray, aoi_name: str,
                   min_matches: int = 15) -> Optional[np.ndarray]:
        """
        Detect AOI in frame using feature matching

        Args:
            frame_gray: Grayscale frame from scene camera
            aoi_name: AOI to detect (e.g., "Screen", "Board")
            min_matches: Minimum good matches required

        Returns:
            4-point polygon (corners) if detected, None otherwise
        """
        if aoi_name not in self.references:
            return None

        ref = self.references[aoi_name]

        # Use cached keypoints if this is the same frame (Bug 7 fix).
        # id() is used as a fast identity check — numpy arrays are not
        # copied between consecutive detect_aoi() calls for the same frame.
        frame_id = id(frame_gray)
        if frame_id != self._cached_frame_id:
            kp_frame, desc_frame = self.orb.detectAndCompute(frame_gray, None)
            self._cached_frame_id = frame_id
            self._cached_kp = kp_frame if kp_frame else []
            self._cached_desc = desc_frame
        else:
            kp_frame = self._cached_kp
            desc_frame = self._cached_desc

        if desc_frame is None or len(kp_frame) < 10:
            return None

        # Match descriptors
        try:
            matches = self.matcher.knnMatch(ref["descriptors"], desc_frame, k=2)
        except cv2.error:
            return None

        # Lowe's ratio test to filter good matches (standard 0.75 threshold).
        # Previously relaxed to 0.85 which caused too many false positives.
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        if len(good_matches) < min_matches:
            return None

        # Extract matched keypoint coordinates
        src_pts = np.float32([ref["keypoints"][m.queryIdx].pt for m in good_matches])
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches])

        # Find homography using RANSAC
        try:
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        except cv2.error:
            return None

        if H is None:
            return None

        # Count inliers — require at least 60% (standard; was 40%)
        inliers = np.sum(mask)
        if inliers < min_matches * 0.6:
            return None

        # Project reference corners to frame
        h, w = ref["shape"]
        ref_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        frame_corners = cv2.perspectiveTransform(ref_corners, H)

        # Validate corners (must form reasonable quadrilateral)
        corners = frame_corners.reshape(-1, 2)
        if not self._validate_quadrilateral(corners, frame_gray.shape):
            return None

        return corners

    def _validate_quadrilateral(self, corners: np.ndarray, frame_shape: Tuple[int, int]) -> bool:
        """Check if detected corners form a valid quadrilateral"""
        h, w = frame_shape

        # All corners must be within frame bounds (with margin)
        if np.any(corners[:, 0] < -50) or np.any(corners[:, 0] > w + 50):
            return False
        if np.any(corners[:, 1] < -50) or np.any(corners[:, 1] > h + 50):
            return False

        # Area must be reasonable (not too small or too large)
        area = cv2.contourArea(corners.astype(np.int32))
        frame_area = w * h
        if area < frame_area * 0.001 or area > frame_area * 0.9:
            return False

        return True

    def detect_all_aois(self, frame_gray: np.ndarray) -> dict:
        """
        Detect all available AOIs in frame

        Returns:
            dict mapping AOI name -> corners (4-point polygon) or None
        """
        results = {}
        for aoi_name in self.references.keys():
            results[aoi_name] = self.detect_aoi(frame_gray, aoi_name)
        return results


# Global detector instance (lazy-loaded)
_detector_instance = None

def get_feature_detector() -> FeatureBasedDetector:
    """Get or create global feature detector instance"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = FeatureBasedDetector()
    return _detector_instance


def detect_screen_fallback(frame_gray: np.ndarray) -> Optional[np.ndarray]:
    """
    Fallback Screen detection using feature matching
    Used when AprilTag detection fails

    Returns:
        4-point polygon (corners) if Screen detected, None otherwise
    """
    detector = get_feature_detector()
    return detector.detect_aoi(frame_gray, "Screen", min_matches=20)


def detect_all_aois_fallback(frame_gray: np.ndarray) -> dict:
    """
    Fallback detection for all AOIs using feature matching

    Returns:
        dict: {aoi_name: corners} where corners is 4-point polygon or None
    """
    detector = get_feature_detector()
    return detector.detect_all_aois(frame_gray)
