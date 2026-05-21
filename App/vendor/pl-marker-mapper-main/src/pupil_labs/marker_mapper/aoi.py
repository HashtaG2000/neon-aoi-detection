"""
AOI (Area of Interest) support for Pupil Labs Marker Mapper
============================================================
Each AOI is defined by a set of AprilTag marker IDs.
The AOI builds its own Surface from those markers and localizes itself
independently every frame — so multiple AOIs can coexist without interfering.

Usage example:
    aoi = AOI("Screen", [20, 21, 22, 23])
    aoi.initialize(detections, camera)           # call until returns True
    localization = aoi.localize(detections, camera)
    if localization:
        img2surface, surface2image = localization
        gaze_surf = aoi.map_gaze(gaze_px, camera, img2surface)
        if aoi.contains_gaze(gaze_surf):
            print("Gaze inside Screen AOI!")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pupil_apriltags
from pupil_labs.camera import Camera, perspective_transform

from .surface import Surface

log = logging.getLogger(__name__)


@dataclass
class AOI:
    """
    An Area of Interest defined by a set of AprilTag marker IDs.

    The AOI has its own Surface which is built the first time enough of its
    markers are detected in a frame. After initialization it localizes itself
    every frame using whichever subset of its markers are currently visible
    (minimum 1 marker, but 2+ gives a more stable homography).

    For AOIs with only 2 boundary markers (e.g. Stream Deck with tags 6 & 7),
    the convex hull of the 8 corner points of both tags forms the bounding
    rectangle automatically — no special handling needed.
    """

    name: str
    marker_ids: list[int]
    _surface: Optional[Surface] = field(default=None, init=False, repr=False)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _filter(self, detections: list[pupil_apriltags.Detection]) -> list[pupil_apriltags.Detection]:
        """Return only detections whose tag_id belongs to this AOI."""
        return [d for d in detections if d.tag_id in self.marker_ids]

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def is_initialized(self) -> bool:
        """True once the AOI surface has been built from at least one frame."""
        return self._surface is not None

    def initialize(
        self,
        detections: list[pupil_apriltags.Detection],
        camera: Camera,
    ) -> bool:
        """
        Try to build this AOI's Surface from the current frame's detections.

        Returns True if the surface was successfully created, False if not
        enough markers were visible. Call this every frame until it returns True.
        Requires at least 1 marker; a single tag's 4 corners form a valid hull
        and are sufficient to bootstrap SIFT-based re-localisation.
        """
        aoi_detections = self._filter(detections)
        if len(aoi_detections) < 1:
            return False
        try:
            self._surface = Surface.from_apriltag_detections(
                self.name, aoi_detections, camera
            )
            log.debug(
                "AOI '%s' initialized from markers %s",
                self.name,
                [d.tag_id for d in aoi_detections],
            )
            return True
        except Exception as exc:
            log.debug("AOI '%s' initialization failed: %s", self.name, exc)
            return False

    def reinitialize(
        self,
        detections: list[pupil_apriltags.Detection],
        camera: Camera,
    ) -> bool:
        """Force re-initialization even if already initialized."""
        self._surface = None
        return self.initialize(detections, camera)

    def localize(
        self,
        detections: list[pupil_apriltags.Detection],
        camera: Camera,
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """
        Localize this AOI in the current frame.

        Returns (img2surface, surface2image) homography pair if successful,
        or None if the AOI is not initialized or none of its markers are visible.
        """
        if self._surface is None:
            return None
        aoi_detections = self._filter(detections)
        if len(aoi_detections) == 0:
            return None
        return self._surface.localize(aoi_detections, camera)

    def map_gaze(
        self,
        gaze_point_px: np.ndarray,
        camera: Camera,
        img2surface: np.ndarray,
    ) -> np.ndarray:
        """
        Map a gaze point from distorted scene-camera pixel space into this
        AOI's normalized surface space ([0,1] × [0,1]).
        """
        pt = np.array([[gaze_point_px[0], gaze_point_px[1]]], dtype=np.float32)
        pt_undist = camera.undistort_points(pt)[:, :2]
        pt_surf = perspective_transform(pt_undist, img2surface)
        return np.array(pt_surf[0], dtype=np.float64)

    def contains_gaze(self, gaze_surf_xy: np.ndarray) -> bool:
        """
        Return True if surface-space coordinates fall within the AOI
        boundary ([0,1] × [0,1]).
        """
        x, y = float(gaze_surf_xy[0]), float(gaze_surf_xy[1])
        return 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
