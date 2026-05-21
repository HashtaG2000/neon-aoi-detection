import cv2
import numpy as np

# Workaround for https://github.com/opencv/opencv/issues/21952
cv2.imshow("cv/av bug", np.zeros(1))
cv2.destroyAllWindows()

import helpers
import pupil_apriltags
import pupil_labs.neon_recording as nr
from pupil_labs.camera import Camera, perspective_transform

from pupil_labs.marker_mapper import Surface, utils


def setup(recording_dir, include_gaze=False):
    recording = nr.load(recording_dir)
    camera = helpers.get_cam(recording)

    marker_detector = pupil_apriltags.Detector(
        families="tag36h11",
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )

    frames = recording.scene.sample(recording.scene.time)
    res = camera, marker_detector, frames

    if include_gaze:
        gaze = recording.gaze.sample(recording.scene.time)
        res += (gaze,)
    return res


def visualize_results(camera, frame, markers, surface, localization, gaze=None):
    img_original = frame.bgr
    img_dist = img_original.copy()
    img_undist_original = camera.undistort_image(frame.bgr)
    img_undist = img_undist_original.copy()

    gaze_dist = None
    gaze_undist = None
    if gaze is not None:
        gaze_dist = np.array([gaze.point[0], gaze.point[1]])
        gaze_undist = camera.undistort_points(gaze_dist)

        cv2.circle(img_dist, tuple(gaze_dist.astype(int)), 50, (0, 0, 255), 3)
        cv2.circle(img_undist, tuple(gaze_undist.astype(int)), 50, (0, 0, 255), 3)

    if localization is not None:
        img2surface, surface2image = localization

        vertices_dist = np.array([m.corners for m in markers])
        vertices_undist = Surface._get_undist_vertices(markers, camera).reshape(
            -1, 4, 2
        )

        marker_ids = [m.tag_id for m in markers]
        img_dist = helpers.draw_markers(img_dist, marker_ids, vertices_dist, surface)
        img_undist = helpers.draw_markers(
            img_undist, marker_ids, vertices_undist, surface
        )

        surface_boundary_undist = utils.get_surface_boundary(surface2image)
        helpers.draw_surface(img_undist, surface_boundary_undist)

        surface_boundary_dist = utils.get_surface_boundary(
            surface2image, distorted=True, camera=camera
        )
        helpers.draw_surface(img_dist, surface_boundary_dist)

        img_cropped = utils.crop_image(
            img_undist_original, surface2image, width=500, height=None
        )
        if gaze_undist is not None:
            gaze_surf = perspective_transform(gaze_undist, img2surface)[0]
            gaze_cropped = gaze_surf * img_cropped.shape[:2][::-1]
            cv2.circle(img_cropped, tuple(gaze_cropped.astype(int)), 20, (0, 0, 255), 3)

        cv2.imshow("Cropped Image", img_cropped)
    cv2.imshow("Undistorted Image", img_undist)
    cv2.imshow("Distorted Image", img_dist)


def get_cam(rec: nr.neon_recording.NeonRecording) -> Camera:
    camera_matrix = rec.calibration.scene_camera_matrix
    dist_coeffs = rec.calibration.scene_distortion_coefficients

    return Camera(1600, 1200, camera_matrix, dist_coeffs)


def draw_markers(img, marker_ids, marker_verts, surface):
    included_color = (0, 255, 0)
    excluded_color = (0, 0, 255)

    overlay = img.copy()
    for m_id, vert in zip(marker_ids, marker_verts, strict=False):
        color = included_color if m_id in surface.markers.keys() else excluded_color
        cv2.fillPoly(overlay, [vert.astype(int)], color)

    alpha = 0.3
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    return img


def draw_surface(img, boundary_points):
    cv2.polylines(
        img,
        [boundary_points.astype(int)],
        True,
        (255, 0, 0),
        2,
    )
    cv2.polylines(
        img,
        [boundary_points[:10].astype(int)],
        False,
        (0, 0, 255),
        2,
    )
    top_center = boundary_points[:10].mean(axis=0).astype(int)
    cv2.putText(
        img,
        "Top",
        tuple(top_center),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
