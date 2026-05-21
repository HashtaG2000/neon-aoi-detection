import cv2
import numpy as np

# Workaround for https://github.com/opencv/opencv/issues/21952
cv2.imshow("cv/av bug", np.zeros(1))
cv2.destroyAllWindows()

import sys

import helpers

from pupil_labs.marker_mapper import Surface


def main(recording_dir):
    camera, marker_detector, frames, gaze = helpers.setup(
        recording_dir, include_gaze=True
    )

    # Define an example surface based on markers detected in the first frame
    frame = next(iter(frames))
    markers = marker_detector.detect(frame.gray)
    surface = Surface.from_apriltag_detections("test surface", markers, camera)

    surface.remove_marker(24)
    img2surface, surface2image = surface.localize(markers, camera)
    surface.add_marker(markers[0], camera, img2surface)

    for frame, g in zip(frames, gaze, strict=False):
        markers = marker_detector.detect(frame.gray)
        localization = surface.localize(markers, camera)
        helpers.visualize_results(camera, frame, markers, surface, localization, g)
        cv2.waitKey(1)


if __name__ == "__main__":
    main(sys.argv[1])
