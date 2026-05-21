from __future__ import annotations

import cv2
import numpy as np

from pupil_labs.camera import perspective_transform
from pupil_labs.marker_mapper import surface


def crop_image(undist_image, surface2image, width=None, height=None):
    surface_corners_norm = surface.normalized_corners()
    surface_corners_undist = perspective_transform(
        surface_corners_norm, surface2image
    )
    crop_size = _calculate_crop_size(
        *surface_corners_undist, width=width, height=height
    )
    crop_corners = surface_corners_norm * np.array(crop_size)
    crop_transform = cv2.getPerspectiveTransform(
        surface_corners_undist.astype(np.float32),
        crop_corners.astype(np.float32),
    )
    crop = cv2.warpPerspective(undist_image, crop_transform, crop_size)
    return crop


def _calculate_crop_size(
    tl: int,
    tr: int,
    br: int,
    bl: int,
    width: int | None,
    height: int | None,
) -> tuple[int, int]:
    # TODO: this is appriximately correct but not perfect. How does this work?

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(width_a, width_b)

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(height_a, height_b)

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order

    if width is not None and height is not None:
        raise ValueError('Expected only "width" OR "height" to be supplied')

    if width is not None:
        ratio = max_width / width
        max_width = width
        max_height /= ratio

    if height is not None:
        ratio = max_height / height
        max_height = height
        max_width /= ratio

    return int(max_width), int(max_height)


def get_surface_boundary(surface2image, distorted=False, camera=None, n=10):
    surface_boundary_norm = surface.normalized_boundary_points(n)
    surface_boundary_undist = perspective_transform(
        surface_boundary_norm, surface2image
    )
    if distorted:
        assert camera is not None
        surface_boundary_dist = camera.distort_points(surface_boundary_undist)
        return surface_boundary_dist
    else:
        return surface_boundary_undist
