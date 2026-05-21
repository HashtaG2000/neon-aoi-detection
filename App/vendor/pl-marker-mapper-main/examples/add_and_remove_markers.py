import sys

import cv2
import helpers

from pupil_labs.marker_mapper import Surface


def main(recording_dir):
    camera, marker_detector, frames = helpers.setup(recording_dir)

    # Define an example surface based on markers detected in the first frame
    frame = next(iter(frames))
    markers = marker_detector.detect(frame.gray)
    surface = Surface.from_apriltag_detections("test surface", markers, camera)

    surface.remove_marker(markers[0].tag_id)  # Remove the first marker
    img2surface, surface2image = surface.localize(markers, camera)

    # Add the first marker back to the surface
    surface.add_marker(markers[0], camera, img2surface)

    for frame in frames:
        markers = marker_detector.detect(frame.gray)
        localization = surface.localize(markers, camera)
        helpers.visualize_results(camera, frame, markers, surface, localization)
        cv2.waitKey(1)


if __name__ == "__main__":
    main(sys.argv[1])
