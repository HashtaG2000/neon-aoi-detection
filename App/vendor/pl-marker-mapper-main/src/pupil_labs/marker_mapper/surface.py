import logging
from collections import OrderedDict
from dataclasses import dataclass

import cv2
import numpy as np
import pupil_apriltags

from pupil_labs.camera import Camera, get_perspective_transform, perspective_transform

logger = logging.getLogger(__name__)


def normalized_corners():
    return np.array(
        [
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1],
        ],
        dtype=np.float32,
    )


def normalized_boundary_points(n=10):
    # returns the coordinates of the boundary points of a 1x1 square
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)

    x0 = np.zeros_like(y)
    x1 = np.ones_like(y)
    y0 = np.zeros_like(x)
    y1 = np.ones_like(x)

    return np.concatenate([
        np.vstack([x, y0]).T[:-1],
        np.vstack([x1, y]).T[:-1],
        np.vstack([x, y1]).T[::-1][:-1],
        np.vstack([x0, y]).T[::-1][:-1],
    ])


@dataclass
class Surface:
    name: str
    markers: OrderedDict[int, np.ndarray]

    @staticmethod
    def from_apriltag_detections(
        name: str,
        detections: list[pupil_apriltags.Detection],
        camera: Camera,
    ) -> "Surface":
        """Create a `Surface` from a list of `pupil_apriltags.Detection`.
        The surface corners will be set to the "convex quadrilateral" formed by the detected markers.
        """
        # TODO: How do we handle marker duplicates?

        vertices_undist = Surface._get_undist_vertices(detections, camera)
        quadrilateral = _get_convex_quadrilateral(vertices_undist)

        img2surface = get_perspective_transform(
            quadrilateral,
            normalized_corners(),
        )
        vertices_surf = perspective_transform(vertices_undist, img2surface)
        vertices_surf = vertices_surf.reshape(-1, 4, 2)
        marker_ids = [marker.tag_id for marker in detections]
        markers = OrderedDict(zip(marker_ids, vertices_surf, strict=False))

        return Surface(name, markers)

    def localize(
        self,
        visible_markers: list[pupil_apriltags.Detection],
        camera: Camera,
    ):
        # TODO: How to handle duplicate markers?

        visible_markers = [
            m for m in visible_markers if m.tag_id in self.markers.keys()
        ]
        sorted(visible_markers, key=lambda m: m.tag_id)

        if len(visible_markers) == 0:
            return None

        visible_vertices_undist = self._get_undist_vertices(visible_markers, camera)

        matching_markers = [self.markers[m.tag_id] for m in visible_markers]
        matching_vertices = np.array(matching_markers).reshape(-1, 2)

        img2surface, surface2image = Surface._find_homographies(
            matching_vertices, visible_vertices_undist
        )
        return img2surface, surface2image

    def add_marker(
        self,
        marker: pupil_apriltags.Detection,
        camera: Camera,
        img2surface: np.ndarray,
    ):
        if marker.tag_id in self.markers:
            raise ValueError(
                f"Marker {marker.tag_id} already exists in surface {self.name}"
            )

        vertices_undist = self._get_undist_vertices([marker], camera)
        vertices_surf = perspective_transform(vertices_undist, img2surface)
        self.markers[marker.tag_id] = vertices_surf

    def remove_marker(self, marker_id: int):
        if marker_id not in self.markers:
            raise ValueError(f"Marker {marker_id} not found in surface {self.name}")
        self.markers.pop(marker_id)

    def rotate(self):
        corners_original = normalized_corners()
        corners_rotated = np.roll(corners_original, -1, axis=0)

        trans = get_perspective_transform(
            corners_rotated, corners_original
        )
        for marker_id, vertices in self.markers.items():
            self.markers[marker_id] = perspective_transform(
                vertices, trans
            )

    def move_corner(
        self,
        corner_idx: int,
        new_pos: tuple[float, float],
        img2surface: np.ndarray,
        camera: Camera,
    ):
        new_pos_undist = camera.undistort_points(np.array(new_pos))
        new_pos_surf = perspective_transform(new_pos_undist, img2surface)
        new_pos_surf = new_pos_surf.flatten()

        corners_original = normalized_corners()
        corners_new = corners_original.copy()
        corners_new[corner_idx] = new_pos_surf

        trans = get_perspective_transform(corners_new, corners_original)
        for marker_id, vertices in self.markers.items():
            self.markers[marker_id] = perspective_transform(
                vertices, trans
            )

    @staticmethod
    def _get_undist_vertices(markers: list[pupil_apriltags.Detection], camera: Camera):
        vertices_dist = np.array([marker.corners for marker in markers])
        vertices_dist = vertices_dist.reshape(-1, 2)
        vertices_undist = camera.undistort_points(vertices_dist)[:, :2]
        return vertices_undist

    @staticmethod
    def _find_homographies(points_A, points_B):
        points_A = points_A.reshape((-1, 1, 2))
        points_B = points_B.reshape((-1, 1, 2))

        B_to_A, mask = cv2.findHomography(points_A, points_B)
        # NOTE: cv2.findHomography(A, B) will not produce the inverse of
        # cv2.findHomography(B, A)! The errors can actually be quite large, resulting in
        # on-screen discrepancies of up to 50 pixels. We try to find the inverse
        # analytically instead with fallbacks.
        try:
            A_to_B = np.linalg.inv(B_to_A)
            return A_to_B, B_to_A
        except np.linalg.LinAlgError:
            logger.debug(
                "Failed to calculate inverse homography with np.inv()! "
                "Trying with np.pinv() instead."
            )

        try:
            A_to_B = np.linalg.pinv(B_to_A)
            return A_to_B, B_to_A
        except np.linalg.LinAlgError:
            logger.warning(
                "Failed to calculate inverse homography with np.pinv()! "
                "Falling back to inaccurate manual computation!"
            )

        A_to_B, mask = cv2.findHomography(points_B, points_A)
        return A_to_B, B_to_A


# From pupil_src/shared_modules/methods.py
def _GetAnglesPolyline(polyline, closed=False):
    """see: http://stackoverflow.com/questions/3486172/angle-between-3-points
    ported to numpy
    returns n-2 signed angles
    """
    points = polyline[:, 0]

    if closed:
        a = np.roll(points, 1, axis=0)
        b = points
        c = np.roll(points, -1, axis=0)
    else:
        a = points[0:-2]  # all "a" points
        b = points[1:-1]  # b
        c = points[2:]  # c points
    # ab =  b.x - a.x, b.y - a.y
    ab = b - a
    # cb =  b.x - c.x, b.y - c.y
    cb = b - c
    # float dot = (ab.x * cb.x + ab.y * cb.y); # dot product
    # print 'ab:',ab
    # print 'cb:',cb

    # float dot = (ab.x * cb.x + ab.y * cb.y) dot product
    # dot  = np.dot(ab,cb.T) # this is a full matrix mulitplication we only need the diagonal \
    # dot = dot.diagonal() #  because all we look for are the dotproducts of corresponding vectors (ab[n] and cb[n])
    dot = np.sum(
        ab * cb, axis=1
    )  # or just do the dot product of the correspoing vectors in the first place!

    # float cross = (ab.x * cb.y - ab.y * cb.x) cross product
    cros = np.cross(ab, cb)

    # float alpha = atan2(cross, dot);
    alpha = np.arctan2(cros, dot)
    return alpha * (180.0 / np.pi)  # degrees
    # return alpha #radians


def _get_convex_quadrilateral(vertices: np.ndarray):
    # According to OpenCV implementation, cv2.convexHull only accepts arrays with
    # 32bit floats (CV_32F) or 32bit signed ints (CV_32S).
    # See: https://github.com/opencv/opencv/blob/3.4/modules/imgproc/src/convhull.cpp#L137
    # See: https://github.com/pupil-labs/pupil/issues/1544
    vertices = np.asarray(vertices, dtype=np.float32)

    hull_points = cv2.convexHull(vertices, clockwise=True)

    # The convex hull of a list of markers must have at least 4 corners, since a
    # single marker already has 4 corners. If the convex hull has more than 4
    # corners we reduce that number with approximations of the hull.
    if len(hull_points) > 4:
        new_hull = cv2.approxPolyDP(hull_points, epsilon=1, closed=True)
        if new_hull.shape[0] >= 4:
            hull_points = new_hull

    if len(hull_points) > 4:
        curvature = abs(_GetAnglesPolyline(hull_points, closed=True))
        most_acute_4_threshold = sorted(curvature)[3]
        hull_points = hull_points[curvature <= most_acute_4_threshold]

    # Vertices space is flipped in y.  We need to change the order of the
    # hull_points vertices
    hull_points = hull_points[[1, 0, 3, 2], :, :]

    # Roll the hull_points vertices until we have the right orientation:
    distance_to_top_left = np.sqrt(
        (hull_points[:, :, 0] + 1) ** 2 + (hull_points[:, :, 1] + 1) ** 2
    )
    bot_left_idx = np.argmin(distance_to_top_left)
    hull_points = np.roll(hull_points, -bot_left_idx, axis=0)

    return hull_points.reshape(-1, 2)
