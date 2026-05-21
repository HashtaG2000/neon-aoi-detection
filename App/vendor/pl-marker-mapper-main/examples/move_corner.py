import cv2
import numpy as np

# Workaround for https://github.com/opencv/opencv/issues/21952
cv2.imshow("cv/av bug", np.zeros(1))
cv2.destroyAllWindows()

import sys

import helpers

from pupil_labs.camera import perspective_transform
from pupil_labs.marker_mapper import Surface


def main(recording_dir):
    camera, marker_detector, frames = helpers.setup(recording_dir)

    # Define an example surface based on markers detected in the first frame
    frame = next(iter(frames))
    markers = marker_detector.detect(frame.gray)
    surface = Surface.from_apriltag_detections("test surface", markers, camera)

    move_corner(camera, markers, surface)

    for frame in frames:
        markers = marker_detector.detect(frame.gray)
        localization = surface.localize(markers, camera)
        helpers.visualize_results(camera, frame, markers, surface, localization)
        cv2.waitKey(1)


def move_corner(camera, markers, surface):
    img2surface, surface2image = surface.localize(markers, camera)
    current_corner_pos_undist = perspective_transform(
        np.array([1, 0]), surface2image
    )
    current_corner_pos_dist = camera.distort_points(current_corner_pos_undist)
    new_corner_pos = current_corner_pos_dist + 200
    surface.move_corner(1, new_corner_pos, img2surface, camera)


if __name__ == "__main__":
    main(sys.argv[1])
