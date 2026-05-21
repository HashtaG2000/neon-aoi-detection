import sys

import cv2
import pupil_apriltags
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from screen_image_widget import SceneImageWidget
from surface_image_widget import SurfaceImageWidget

from pupil_labs.camera import Camera, perspective_transform
from pupil_labs.marker_mapper import Surface, utils
from pupil_labs.realtime_api.simple import Device


class MainWindow(QMainWindow):
    def __init__(self, neon_ip, neon_port):
        super().__init__()
        self.neon_device = self.connect_to_neon(neon_ip, neon_port)

        if not self.neon_device:
            raise RuntimeError("No Neon device found. Please connect a device.")
        calibration = self.neon_device.get_calibration()
        self.camera = Camera(
            1600,
            1200,
            calibration.scene_camera_matrix,
            calibration.scene_distortion_coefficients,
        )

        self.marker_detector = pupil_apriltags.Detector(
            families="tag36h11",
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )
        self.surface = None
        self.detected_markers = []

        self.setWindowTitle("Scene Video")
        self.scene_image_widget = SceneImageWidget()
        self.surface_image_widget = SurfaceImageWidget()
        self.surface_image_widget.setMinimumSize(500, 500)
        self.surface_image_widget.setVisible(True)
        self.surface_image_widget.setWindowTitle("Surface Crop")

        layout = QVBoxLayout()
        layout.addWidget(self.scene_image_widget)
        self.define_surface_button = QPushButton("Define Surface")
        layout.addWidget(self.define_surface_button)
        self.define_surface_button.clicked.connect(self.define_surface)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.scene_image_widget.marker_clicked.connect(self.on_marker_clicked)
        self.scene_image_widget.corner_dragged.connect(self.on_corner_dragged)

        # Timer for auto-update (optional)
        self.timer = QTimer()
        self.timer.timeout.connect(self.poll)
        self.timer.start(30)

    @staticmethod
    def connect_to_neon(neon_ip, neon_port) -> Device:
        print("Connecting to Neon device...", end="")
        device = Device(address=neon_ip, port=neon_port)
        if not device:
            raise RuntimeError("No Neon device found. Please connect a device.")
        print("Connected.")
        return device

    def poll(self):
        matched_data = self.neon_device.receive_matched_scene_video_frame_and_gaze(
            timeout_seconds=0.1
        )

        if matched_data is None:
            return

        scene_img, gaze_scene = matched_data
        gaze_scene = gaze_scene[0:2]
        scene_img = scene_img.bgr_pixels

        scene_gray = cv2.cvtColor(scene_img, cv2.COLOR_BGR2GRAY)
        scene_undist = self.camera.undistort_image(scene_img)
        self.detected_markers = self.marker_detector.detect(scene_gray)

        surface_boundary = None
        surface_corners = None
        surface_markers = []
        non_surface_markers = self.detected_markers
        surface_img = None
        gaze_surface = None
        if self.surface is not None:
            surface_marker_keys = self.surface.markers.keys()
            surface_markers = [
                m for m in self.detected_markers if m.tag_id in surface_marker_keys
            ]
            non_surface_markers = [
                m for m in self.detected_markers if m.tag_id not in surface_marker_keys
            ]

            self.localization = self.surface.localize(
                self.detected_markers, self.camera
            )
            if self.localization is not None:
                img2surface, surface2image = self.localization
                surface_boundary = utils.get_surface_boundary(
                    surface2image, distorted=True, camera=self.camera
                )

                surface_corners = utils.get_surface_boundary(
                    surface2image, distorted=True, camera=self.camera, n=2
                )
                surface_img = utils.crop_image(
                    scene_undist, surface2image, width=500, height=None
                )

                gaze_undist = self.camera.undistort_points(gaze_scene)
                gaze_surface_norm = perspective_transform(
                    gaze_undist, img2surface
                )[0]
                gaze_surface = gaze_surface_norm * surface_img.shape[:2][::-1]

        self.scene_image_widget.update_data(
            scene_img,
            surface_markers,
            non_surface_markers,
            surface_boundary,
            surface_corners,
            gaze_scene,
        )
        self.surface_image_widget.update_data(surface_img, gaze_surface)

    def define_surface(self):
        self.surface = Surface.from_apriltag_detections(
            "test surface", self.detected_markers, self.camera
        )
        self.define_surface_button.setEnabled(False)

    def on_marker_clicked(self, marker_id):
        if self.surface is not None:
            if marker_id in self.surface.markers.keys():
                self.surface.remove_marker(marker_id)
            else:
                marker = next(
                    (m for m in self.detected_markers if m.tag_id == marker_id), None
                )
                img2surface, surface2image = self.localization
                self.surface.add_marker(marker, self.camera, img2surface)

    def on_corner_dragged(self, corner, x, y):
        if self.surface is not None:
            img2surface, _ = self.localization
            self.surface.move_corner(corner, (x, y), img2surface, self.camera)

    def closeEvent(self, event):
        self.timer.stop()
        if self.neon_device:
            self.neon_device.close()

        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = MainWindow(sys.argv[1], int(sys.argv[2]))
    viewer.resize(800, 600)
    viewer.show()
    sys.exit(app.exec())
