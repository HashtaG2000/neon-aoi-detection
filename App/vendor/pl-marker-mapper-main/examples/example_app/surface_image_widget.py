from PySide6.QtCore import QPointF, Qt
from PySide6.QtCore import Qt as QtCoreQt
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QLabel


class SurfaceImageWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setAlignment(QtCoreQt.AlignmentFlag.AlignCenter)
        self.surface_img = None
        self.gaze_point = None

        self.offset = (0, 0)
        self.scale = (1, 1)

    def update_data(self, surface_img, gaze_point):
        self.surface_img = surface_img
        self.gaze_point = gaze_point
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        with QPainter(self) as painter:
            if self.surface_img is not None:
                h, w, c = self.surface_img.shape
                rgb = self.surface_img[..., ::-1].copy()
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

            if self.gaze_point is not None:
                gaze_x = int(self.gaze_point[0] * self.scale[0] + self.offset[0])
                gaze_y = int(self.gaze_point[1] * self.scale[1] + self.offset[1])
                painter.setPen(QPen(QColor(255, 0, 0), 3))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(QPointF(gaze_x, gaze_y), 20, 20)
