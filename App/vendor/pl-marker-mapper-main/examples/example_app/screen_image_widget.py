import numpy as np
from PySide6.QtCore import QPoint, QPointF, Qt, Signal
from PySide6.QtCore import Qt as QtCoreQt
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QLabel


class SceneImageWidget(QLabel):
    marker_clicked = Signal(int)
    corner_dragged = Signal(int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setAlignment(QtCoreQt.AlignmentFlag.AlignCenter)
        self.scene_img = None
        self.surface_markers = []
        self.non_surface_markers = []
        self.surface_boundary_points = None
        self.surface_corner_positions = None
        self.scene_gaze = None

        self.offset = (0, 0)
        self.scale = (1, 1)

        self.mouse_press_pos = None
        self.mouse_dragged = False
        self.pressed_corner = None

    def update_data(
        self,
        scene_img,
        surface_markers,
        non_surface_markers,
        surface_boundary_points,
        surface_corner_positions,
        scene_gaze,
    ):
        self.scene_img = scene_img
        self.surface_markers = surface_markers
        self.non_surface_markers = non_surface_markers
        self.surface_boundary_points = surface_boundary_points
        self.surface_corner_positions = surface_corner_positions
        self.scene_gaze = scene_gaze

        self.update()

    def mousePressEvent(self, event):
        self.mouse_press_pos = event.pos()

        if self.surface_corner_positions is not None:
            for corner_idx, corner in enumerate(self.surface_corner_positions):
                scaled_corner = (
                    int(corner[0] * self.scale[0] + self.offset[0]),
                    int(corner[1] * self.scale[1] + self.offset[1]),
                )
                scaled_corner = QPoint(*scaled_corner)
                if (event.pos() - scaled_corner).manhattanLength() < 10:
                    self.pressed_corner = corner_idx
                    return

    def mouseMoveEvent(self, event):
        if (event.pos() - self.mouse_press_pos).manhattanLength() > 5:
            self.mouse_dragged = True
        if self.pressed_corner is not None:
            new_pos = event.pos()
            new_pos_scaled = (
                (new_pos.x() - self.offset[0]) / self.scale[0],
                (new_pos.y() - self.offset[1]) / self.scale[1],
            )
            self.corner_dragged.emit(
                self.pressed_corner, int(new_pos_scaled[0]), int(new_pos_scaled[1])
            )

            self.update()

    def mouseReleaseEvent(self, event):
        if not self.mouse_dragged:
            self.on_click(event)

        self.mouse_press_pos = None
        self.mouse_dragged = False
        self.pressed_corner = None

    def paintEvent(self, event):
        super().paintEvent(event)
        with QPainter(self) as painter:
            if self.scene_img is not None:
                h, w, c = self.scene_img.shape
                rgb = self.scene_img[..., ::-1].copy()
                qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)
                scaled_pixmap = pixmap.scaled(
                    self.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                painter.drawPixmap(
                    self.rect().center() - scaled_pixmap.rect().center(), scaled_pixmap
                )

                self.scale = scaled_pixmap.width() / w, scaled_pixmap.height() / h
                self.offset = (
                    (self.width() - scaled_pixmap.width()) // 2,
                    (self.height() - scaled_pixmap.height()) // 2,
                )

                if self.surface_markers:
                    self._paint_markers(
                        self.surface_markers,
                        painter,
                        *self.scale,
                        *self.offset,
                        is_surface=True,
                    )
                if self.non_surface_markers:
                    self._paint_markers(
                        self.non_surface_markers,
                        painter,
                        *self.scale,
                        *self.offset,
                        is_surface=False,
                    )

                # Draw surface boundary as polyline if available
                if self.surface_boundary_points is not None:
                    self._paint_surface_boundary(
                        self.surface_boundary_points, painter, *self.scale, *self.offset
                    )

            if self.scene_gaze is not None:
                gaze_x = int(self.scene_gaze[0] * self.scale[0] + self.offset[0])
                gaze_y = int(self.scene_gaze[1] * self.scale[1] + self.offset[1])
                painter.setPen(QPen(QColor(255, 0, 0), 3))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(QPointF(gaze_x, gaze_y), 20, 20)

    @staticmethod
    def _paint_markers(
        markers, painter, scale_x, scale_y, offset_x, offset_y, is_surface=True
    ):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        fill_color = (
            QColor(0, 255, 0, 100) if is_surface else QColor(255, 0, 0, 100)
        )  # Semi-transparent green for surface markers, red for non-surface markers
        for marker in markers:
            if hasattr(marker, "corners"):
                pts = marker.corners
                pts = [
                    (
                        int(pt[0] * scale_x + offset_x),
                        int(pt[1] * scale_y + offset_y),
                    )
                    for pt in pts
                ]
                polygon = [QPointF(x, y) for x, y in pts]
                painter.setBrush(fill_color)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawPolygon(polygon)

    @staticmethod
    def _paint_surface_boundary(
        surface_boundary, painter, scale_x, scale_y, offset_x, offset_y
    ):
        painter.setBrush(Qt.BrushStyle.NoBrush)
        pen = QPen(QColor(0, 0, 255, 200))  # Semi-transparent red
        pen.setWidth(3)
        painter.setPen(pen)
        # Transform boundary points to widget coordinates
        poly_pts = [
            QPointF(pt[0] * scale_x + offset_x, pt[1] * scale_y + offset_y)
            for pt in surface_boundary
        ]

        # Close the polygon by adding the first point at the end
        poly_pts.append(poly_pts[0])
        painter.drawPolyline(poly_pts)

    def on_click(self, event):
        if self.scene_img is None:
            return

        x = (event.x() - self.offset[0]) / self.scale[0]
        y = (event.y() - self.offset[1]) / self.scale[1]

        for marker in self.surface_markers + self.non_surface_markers:
            if hasattr(marker, "corners"):
                if self.point_in_quadrangle(marker.corners, (x, y)):
                    print(f"Clicked inside marker {marker.tag_id} at ({x}, {y})")
                    self.marker_clicked.emit(marker.tag_id)
                    return

    @staticmethod
    def point_in_quadrangle(quad, point):
        """Returns True if the point (x, y) is inside the quadrangle defined by quad (list of 4 (x, y) tuples).
        Uses the winding number algorithm.
        """
        quad_np = np.array(quad)
        px, py = point

        # Close the polygon
        poly = np.vstack([quad_np, quad_np[0]])

        winding_number = 0
        for i in range(4):
            x0, y0 = poly[i]
            x1, y1 = poly[i + 1]
            if y0 <= py:
                if y1 > py and ((x1 - x0) * (py - y0) - (px - x0) * (y1 - y0)) > 0:
                    winding_number += 1
            else:
                if y1 <= py and ((x1 - x0) * (py - y0) - (px - x0) * (y1 - y0)) < 0:
                    winding_number -= 1
        return winding_number != 0
