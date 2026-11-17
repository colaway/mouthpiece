"""Camera preview window (PySide6) with a retro terminal look.

Monochrome UI on pure black (SF Mono / Menlo, three greys, square corners, uppercase
section labels, inverted "> active" selections, faint scanlines). Colour is reserved
for the camera image and a few small accents: amber for keys/alignment points/status,
red for recording.

Shortcuts: R record · S smoothing · M mirror · Q / Esc quit
"""

from __future__ import annotations

import queue
from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt
from PySide6.QtCore import QPointF, QRect, QRectF, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QKeySequence,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPen,
    QPolygonF,
    QRegion,
    QResizeEvent,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from mouthpiece import __version__
from mouthpiece.config import Config
from mouthpiece.preprocess.landmarks import Landmarks
from mouthpiece.preprocess.preview_engine import PreviewEngine, PreviewState

# --- design tokens ------------------------------------------------------------

BG = QColor("#000000")
FG = QColor("#e6e6e6")
DIM = QColor("#7a7a7a")
FAINT = QColor("#4a4a4a")
PANEL = QColor("#0a0a0a")
AMBER = QColor("#ffb000")   # small accents: keys, status, alignment points
REC = QColor("#ff3b30")     # recording only

MONO_FAMILIES = ["SF Mono", "Menlo", "Monaco", "Consolas", "Courier New"]
CSS_MONO = '"SF Mono", Menlo, Monaco, Consolas, monospace'

STYLESHEET = f"""
QMainWindow, QWidget#root {{ background: {BG.name()}; }}
QWidget {{ font-family: {CSS_MONO}; }}
QLabel {{ color: {FG.name()}; font-size: 13px; }}
QLabel#title {{ font-size: 16px; color: {FG.name()}; }}
QLabel#rule {{ color: {FAINT.name()}; font-size: 13px; }}
QLabel#meta {{ color: {DIM.name()}; font-size: 12px; }}
QLabel#section {{ color: {DIM.name()}; font-size: 11px; letter-spacing: 1.6px; }}
QLabel#value {{ color: {DIM.name()}; font-size: 11px; }}
QLabel#stat {{ color: {FG.name()}; font-size: 12px; }}
QLabel#hint {{ color: {DIM.name()}; font-size: 11px; }}
QLabel#toast {{ color: {FG.name()}; background: {BG.name()}; border: 1px solid {FAINT.name()};
    padding: 6px 8px; font-size: 11px; }}
QPushButton {{ color: {FG.name()}; background: transparent; border: none; text-align: left;
    padding: 3px 8px 3px 8px; font-size: 13px; }}
QPushButton:hover {{ color: #ffffff; }}
QPushButton:checked {{ background: {FG.name()}; color: {BG.name()}; }}
"""


def mono_font(point_size: float = 11.0) -> QFont:
    f = QFont()
    f.setFamilies(MONO_FAMILIES)
    f.setStyleHint(QFont.StyleHint.Monospace)
    f.setPointSizeF(point_size)
    return f


def bgr_to_qimage(img: npt.NDArray[np.uint8]) -> QImage:
    """Convert a BGR or grayscale uint8 array to a QImage that owns its data."""
    if img.ndim == 2:
        a = np.ascontiguousarray(img)
        h, w = a.shape
        return QImage(a.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
    rgb = np.ascontiguousarray(img[:, :, ::-1])
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def contain_rect(src_w: int, src_h: int, dst: QRectF) -> tuple[QRectF, float]:
    """Largest rect with the image's aspect ratio centered in `dst` (object-fit: contain).

    The whole camera frame stays visible, so an off-center face is never cropped away."""
    scale = min(dst.width() / src_w, dst.height() / src_h)
    w, h = src_w * scale, src_h * scale
    return QRectF(dst.center().x() - w / 2, dst.center().y() - h / 2, w, h), scale


def leader(label: str, value: str, width: int = 22) -> str:
    """'camera', '25.0 fps' -> 'camera ...... 25.0 fps' (fixed width, monospace)."""
    dots = max(2, width - len(label) - len(value) - 2)
    return f"{label} {'.' * dots} {value}"


def draw_tag(
    p: QPainter, x: float, y: float, text: str, dot: QColor | None = None,
    fg: QColor = FG, bg: QColor = BG, align_right: bool = False,
) -> None:
    """Small square-cornered label on black, terminal style."""
    p.setFont(mono_font(10.5))
    fm = p.fontMetrics()
    pad, h = 8.0, 22.0
    dot_w = 12.0 if dot else 0.0
    w = fm.horizontalAdvance(text) + 2 * pad + dot_w
    if align_right:
        x -= w
    rect = QRectF(x, y, w, h)
    p.setPen(QPen(FAINT, 1))
    p.setBrush(bg)
    p.drawRect(rect)
    if dot:
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(dot)
        p.drawRect(QRectF(x + pad, y + h / 2 - 3, 6, 6))
    p.setPen(fg)
    p.drawText(QRectF(x + pad + dot_w, y, w - 2 * pad - dot_w, h),
               Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, text)


# --- widgets -----------------------------------------------------------------


class VideoView(QWidget):
    """Live video (the only full-colour element) with a monochrome overlay."""

    def __init__(self) -> None:
        super().__init__()
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(480, 360)
        self.state: PreviewState | None = None
        self.mirror = True
        self.error: str | None = None
        self._image: QImage | None = None
        self._blink = False
        timer = QTimer(self)
        timer.timeout.connect(self._tick_blink)
        timer.start(530)

    def _tick_blink(self) -> None:
        self._blink = not self._blink
        self.update()

    def set_state(self, state: PreviewState) -> None:
        self.state = state
        frame = state.frame[:, ::-1] if self.mirror else state.frame
        self._image = bgr_to_qimage(frame)
        self.update()

    def image_rect(self) -> QRectF:
        area = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        if self._image is None:
            return area
        return contain_rect(self._image.width(), self._image.height(), area)[0]

    def _map(self, lm: Landmarks, img_rect: QRectF, scale: float, width: int) -> list[QPointF]:
        pts = []
        for x, y in lm:
            fx = (width - 1 - float(x)) if self.mirror else float(x)
            pts.append(QPointF(img_rect.left() + fx * scale, img_rect.top() + float(y) * scale))
        return pts

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        area = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)

        if self._image is None or self.state is None:
            p.setPen(QPen(FAINT, 1))
            p.drawRect(area)
            p.setFont(mono_font(12))
            p.setPen(DIM)
            cursor = "█" if self._blink else " "
            msg = f"> {self.error}" if self.error else f"> starting camera {cursor}"
            p.drawText(area.adjusted(24, 0, -24, 0),
                       Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, msg)
            p.end()
            return

        s = self.state
        r, scale = contain_rect(self._image.width(), self._image.height(), area)
        p.setClipRect(r)
        p.drawImage(r, self._image)

        top = QLinearGradient(0, r.top(), 0, r.top() + 70)
        top.setColorAt(0, QColor(0, 0, 0, 150))
        top.setColorAt(1, QColor(0, 0, 0, 0))
        p.fillRect(QRectF(r.left(), r.top(), r.width(), 70), QBrush(top))

        frame_w = s.frame.shape[1]
        if s.landmarks is not None:
            self._draw_landmarks(p, s.landmarks, r, scale, frame_w)

        x0, y0 = r.left() + 14, r.top() + 14
        if s.landmarks is not None:
            draw_tag(p, x0, y0, "FACE", dot=AMBER)
        else:
            draw_tag(p, x0, y0, "NO FACE", dot=DIM, fg=DIM)
        draw_tag(p, r.right() - 14, y0, f"{s.camera_fps:4.1f} FPS  {s.process_ms:4.1f} MS",
                 align_right=True)
        if s.recording_s is not None:
            dot = REC if self._blink else QColor(REC.red(), REC.green(), REC.blue(), 70)
            draw_tag(p, r.center().x() - 52, y0, f"REC {s.recording_s:05.1f}s", dot=dot)

        p.setClipping(False)
        self._draw_frame(p, r, REC if s.recording_s is not None else None)
        p.end()

    def _draw_frame(self, p: QPainter, r: QRectF, rec: QColor | None) -> None:
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(rec or FAINT, 2 if rec else 1))
        p.drawRect(r)
        # Viewfinder corner brackets.
        p.setPen(QPen(rec or FG, 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.SquareCap))
        k, m = 18.0, 7.0
        for cx, cy, dx, dy in (
            (r.left() + m, r.top() + m, 1, 1), (r.right() - m, r.top() + m, -1, 1),
            (r.left() + m, r.bottom() - m, 1, -1), (r.right() - m, r.bottom() - m, -1, -1),
        ):
            p.drawLine(QPointF(cx, cy), QPointF(cx + dx * k, cy))
            p.drawLine(QPointF(cx, cy), QPointF(cx, cy + dy * k))

    def _draw_landmarks(
        self, p: QPainter, lm: Landmarks, r: QRectF, scale: float, width: int
    ) -> None:
        pts = self._map(lm, r, scale, width)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(230, 230, 230, 140))
        for pt in pts[:48]:
            p.drawRect(QRectF(pt.x() - 1, pt.y() - 1, 2, 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(230, 230, 230, 230), 1.4))
        p.drawPolygon(QPolygonF(pts[48:60]))
        p.setPen(QPen(QColor(230, 230, 230, 140), 1.0, Qt.PenStyle.DotLine))
        p.drawPolygon(QPolygonF(pts[60:68]))
        # Alignment points as small amber crosses.
        p.setPen(QPen(AMBER, 1.3))
        for i in (28, 33, 36, 39, 42, 45, 48, 54):
            c = pts[i]
            p.drawLine(QPointF(c.x() - 4, c.y()), QPointF(c.x() + 4, c.y()))
            p.drawLine(QPointF(c.x(), c.y() - 4), QPointF(c.x(), c.y() + 4))


class ImagePanel(QWidget):
    """Grayscale image with a 1px border and an optional dashed model-input box."""

    def __init__(self, size: int, inner_ratio: float | None = None) -> None:
        super().__init__()
        self.setFixedSize(size, size)
        self.inner_ratio = inner_ratio
        self._image: QImage | None = None

    def set_image(self, img: npt.NDArray[np.uint8] | None) -> None:
        self._image = bgr_to_qimage(img) if img is not None else None
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.fillRect(r, PANEL)
        if self._image is None:
            p.setFont(mono_font(10.5))
            p.setPen(FAINT)
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, "-- no signal --")
        else:
            p.drawImage(r, self._image)
            if self.inner_ratio:
                m = r.width() * (1 - self.inner_ratio) / 2
                pen = QPen(FG, 1, Qt.PenStyle.CustomDashLine)
                pen.setDashPattern([3, 3])
                p.setPen(pen)
                p.drawRect(r.adjusted(m, m, -m, -m))
        p.setPen(QPen(FAINT, 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(r)
        p.end()


class Sparkline(QWidget):
    """Stepped monochrome history line."""

    def __init__(self, top: float = 0.6, height: int = 44) -> None:
        super().__init__()
        self.setFixedHeight(height)
        self.top = top
        self.values: Sequence[float] = ()

    def set_values(self, values: Sequence[float]) -> None:
        self.values = values
        self.update()

    def _path(self, vals: Sequence[float], r: QRectF, top: float) -> QPainterPath:
        n = len(vals)
        path = QPainterPath()
        for i, v in enumerate(vals):
            x = r.left() + r.width() * i / max(n - 1, 1)
            y = r.bottom() - r.height() * min(max(v, 0.0), top) / top
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, path.currentPosition().y())  # stepped, oscilloscope-like
                path.lineTo(x, y)
        return path

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        p = QPainter(self)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.fillRect(r, PANEL)
        p.setPen(QPen(QColor(255, 255, 255, 16), 1, Qt.PenStyle.DotLine))
        for k in (0.25, 0.5, 0.75):
            y = r.top() + r.height() * k
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
        inner = r.adjusted(2, 4, -2, -3)
        if len(self.values) > 1:
            p.setPen(QPen(FG, 1.2))
            p.drawPath(self._path(self.values, inner, self.top))
        p.setPen(QPen(FAINT, 1))
        p.drawRect(r)
        p.end()


class Scanlines(QWidget):
    """Faint CRT scanlines over the UI chrome (never over the camera image)."""

    def __init__(self, parent: QWidget, exclude: Callable[[], Sequence[QRect]]) -> None:
        super().__init__(parent)
        self.exclude = exclude
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        p = QPainter(self)
        region = QRegion(self.rect())
        for rect in self.exclude():
            region = region.subtracted(QRegion(rect))
        p.setClipRegion(region)
        p.setPen(QPen(QColor(0, 0, 0, 55), 1))
        for y in range(2, self.height(), 3):
            p.drawLine(0, y, self.width(), y)
        p.end()


def section(title: str, value: QLabel | None = None) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setContentsMargins(0, 8, 0, 0)
    t = QLabel(title.upper())
    t.setObjectName("section")
    row.addWidget(t)
    row.addStretch()
    if value is not None:
        row.addWidget(value)
    return row


# --- capture thread ----------------------------------------------------------


class CaptureWorker(QThread):
    state_ready = Signal(object)
    failed = Signal(str)
    clip_ready = Signal(object, object)  # frames, landmarks (push-to-talk release)
    face_changed = Signal(bool)          # a face appeared / disappeared
    started = Signal()                   # camera and landmarker are up

    def __init__(self, cfg: Config, emit_states: bool = True) -> None:
        super().__init__()
        self.cfg = cfg
        self.emit_states = emit_states   # False: no per-frame states (menu-bar app)
        self.commands: queue.Queue[str] = queue.Queue()
        self._running = True

    def stop(self) -> None:
        self._running = False
        self.wait(3000)

    def run(self) -> None:
        from mouthpiece.capture.camera import Camera
        from mouthpiece.preprocess.pipeline import build_landmarker

        c = self.cfg.capture
        try:
            with Camera(c.camera_index, c.fps, c.width, c.height) as cam, \
                    build_landmarker(self.cfg) as lmk:
                engine = PreviewEngine(self.cfg, lmk.detect, native_fps=cam.native_fps)
                self.started.emit()
                face: bool | None = None
                for tf in cam.frames():
                    if not self._running:
                        break
                    while not self.commands.empty():
                        cmd = self.commands.get_nowait()
                        if cmd == "record":
                            engine.toggle_recording()
                        elif cmd == "ptt_start":
                            engine.start_recording()
                        elif cmd == "ptt_stop":
                            self.clip_ready.emit(*engine.take_recording())
                        elif cmd == "ptt_cancel":
                            engine.take_recording()
                        elif cmd == "smoothing":
                            engine.toggle_smoothing()
                    state = engine.process(tf.image, tf.timestamp)
                    if (state.landmarks is not None) != face:
                        face = state.landmarks is not None
                        self.face_changed.emit(face)
                    if self.emit_states:
                        self.state_ready.emit(state)
        except Exception as exc:  # surfaced in the UI instead of a silent dead thread
            self.failed.emit(str(exc))


# --- main window -------------------------------------------------------------


class MenuToggle(QPushButton):
    """Checkable menu item: inverted with a '>' when active."""

    def __init__(self, label: str, checked: bool) -> None:
        super().__init__()
        self.label = label
        self.setCheckable(True)
        self.setChecked(checked)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.toggled.connect(self._refresh)
        self._refresh()

    def _refresh(self) -> None:
        self.setText(f"> {self.label}" if self.isChecked() else f"  {self.label}")


class RecordButton(QPushButton):
    """'[ ● rec ]' with the dot as the only red accent; turns solid red while recording."""

    def __init__(self) -> None:
        super().__init__()
        self.setCheckable(True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setFixedHeight(32)
        self.seconds: float | None = None
        self._hover = False

    def enterEvent(self, event: object) -> None:  # noqa: N802
        self._hover = True
        self.update()

    def leaveEvent(self, event: object) -> None:  # noqa: N802
        self._hover = False
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        p = QPainter(self)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        recording = self.isChecked()
        p.fillRect(r, REC if recording else BG)
        p.setPen(QPen(REC if recording else (DIM if self._hover else FAINT), 1))
        p.drawRect(r)
        p.setFont(mono_font(12))
        fm = p.fontMetrics()
        label = f"stop  {self.seconds or 0.0:05.1f}s" if recording else "rec"
        left, right = "[ ", f" {label} ]"
        dot_w = 8.0
        total = fm.horizontalAdvance(left) + dot_w + fm.horizontalAdvance(right)
        x = r.center().x() - total / 2
        base = r.center().y() + fm.ascent() / 2 - 1
        fg = BG if recording else FG
        p.setPen(fg)
        p.drawText(QPointF(x, base), left)
        x += fm.horizontalAdvance(left)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(BG if recording else REC)
        if recording:
            p.drawRect(QRectF(x, r.center().y() - 4, 8, 8))  # stop square
        else:
            p.drawEllipse(QRectF(x, r.center().y() - 4, 8, 8))
        p.setPen(fg)
        p.drawText(QPointF(x + dot_w, base), right)
        p.end()


class PreviewWindow(QMainWindow):
    SIDE_W = 300

    def __init__(self, cfg: Config, mirror: bool = True) -> None:
        super().__init__()
        self.cfg = cfg
        self.worker: CaptureWorker | None = None
        self.setWindowTitle("mouthpiece — preview")
        self.resize(1240, 760)
        self.setStyleSheet(STYLESHEET)

        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(28, 26, 28, 24)
        outer.setSpacing(30)

        self.video = VideoView()
        self.video.mirror = mirror
        outer.addWidget(self.video, stretch=1)

        side_w = QWidget()
        side_w.setFixedWidth(self.SIDE_W)
        side = QVBoxLayout(side_w)
        side.setContentsMargins(0, 0, 0, 0)
        side.setSpacing(6)
        outer.addWidget(side_w)

        # Header: title with blinking block cursor, ===== rule, meta line.
        self.title = QLabel()
        self.title.setObjectName("title")
        rule = QLabel("=" * 10)
        rule.setObjectName("rule")
        meta = QLabel(f"camera preview · v{__version__}")
        meta.setObjectName("meta")
        side.addWidget(self.title)
        side.addWidget(rule)
        side.addWidget(meta)
        self._cursor_on = True
        self._set_title()

        p = cfg.preprocess
        inner = self.SIDE_W
        size_label = QLabel(f"{p.model_input_size}x{p.model_input_size}")
        size_label.setObjectName("value")
        side.addLayout(section("model input", size_label))
        self.mouth_view = ImagePanel(inner, inner_ratio=p.model_input_size / p.roi_size)
        side.addWidget(self.mouth_view)

        self.open_label = QLabel("--")
        self.open_label.setObjectName("value")
        side.addLayout(section("mouth activity", self.open_label))
        self.spark = Sparkline()
        side.addWidget(self.spark)

        side.addLayout(section("aligned face"))
        self.face_view = ImagePanel(96)
        stats = QVBoxLayout()
        stats.setSpacing(2)
        self.stat_labels: dict[str, QLabel] = {}
        for key in ("camera", "driver", "latency", "smooth"):
            lab = QLabel(leader(key, "--"))
            lab.setObjectName("stat")
            stats.addWidget(lab)
            self.stat_labels[key] = lab
        stats.addStretch()
        row = QHBoxLayout()
        row.setSpacing(14)
        row.addWidget(self.face_view)
        row.addLayout(stats)
        side.addLayout(row)

        side.addLayout(section("controls"))
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(2)
        self.smooth_btn = MenuToggle("smoothing", True)
        self.smooth_btn.clicked.connect(lambda: self._send("smoothing"))
        self.mirror_btn = MenuToggle("mirror", mirror)
        self.mirror_btn.clicked.connect(self._toggle_mirror)
        grid.addWidget(self.smooth_btn, 0, 0)
        grid.addWidget(self.mirror_btn, 0, 1)
        side.addLayout(grid)

        self.record_btn = RecordButton()
        self.record_btn.clicked.connect(lambda: self._send("record"))
        side.addSpacing(4)
        side.addWidget(self.record_btn)
        self._set_record_text(None)

        self.toast = QLabel("")
        self.toast.setObjectName("toast")
        self.toast.setWordWrap(True)
        self.toast.hide()
        side.addWidget(self.toast)
        self._toast_timer = QTimer(self)
        self._toast_timer.setSingleShot(True)
        self._toast_timer.timeout.connect(self.toast.hide)

        self.add_sections(side)
        side.addStretch()
        self.hint = QLabel(self.hint_text())
        self.hint.setObjectName("hint")
        self.hint.setTextFormat(Qt.TextFormat.RichText)
        side.addWidget(self.hint)

        def data_rects() -> list[QRect]:  # camera + image panels stay scanline-free
            rects = [self.video.geometry()]
            for w in (self.mouth_view, self.face_view):
                rects.append(QRect(w.mapTo(root, w.rect().topLeft()), w.size()))
            return rects

        self.scanlines = Scanlines(root, data_rects)
        self.scanlines.raise_()

        cursor_timer = QTimer(self)
        cursor_timer.timeout.connect(self._blink_cursor)
        cursor_timer.start(525)

        for keys, fn in (
            ("R", lambda: self.record_btn.click()),
            ("S", lambda: self.smooth_btn.click()),
            ("M", lambda: self.mirror_btn.click()),
            ("Q", lambda: self.close() and None),
            ("Esc", lambda: self.close() and None),
        ):
            QShortcut(QKeySequence(keys), self, fn)

    # --- extension points (used by the live transcription window) ----------

    def add_sections(self, side: QVBoxLayout) -> None:
        """Hook for subclasses to add panels to the sidebar."""

    def hint_text(self) -> str:
        a = AMBER.name()
        return (
            f"<span style='color:{a}'>[r]</span> rec  "
            f"<span style='color:{a}'>[s]</span> smooth  "
            f"<span style='color:{a}'>[m]</span> mirror  "
            f"<span style='color:{a}'>[q]</span> quit"
        )

    # --- small view helpers ------------------------------------------------

    def _set_title(self) -> None:
        cursor = "█" if self._cursor_on else "&nbsp;"
        self.title.setText(
            f"<span style='color:{FG.name()}'>mouthpiece</span>"
            f"<span style='color:{FG.name()}'>{cursor}</span>"
        )

    def _blink_cursor(self) -> None:
        self._cursor_on = not self._cursor_on
        self._set_title()

    def _set_record_text(self, seconds: float | None) -> None:
        self.record_btn.seconds = seconds
        self.record_btn.update()

    # --- wiring ------------------------------------------------------------

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        central = self.centralWidget()
        if central is not None:
            self.scanlines.setGeometry(central.rect())

    def start(self) -> None:
        self.worker = CaptureWorker(self.cfg)
        self.worker.state_ready.connect(self.update_state)
        self.worker.failed.connect(self.show_error)
        self.worker.clip_ready.connect(self.on_clip)
        self.worker.start()

    def on_clip(self, frames: object, landmarks: object) -> None:
        """Push-to-talk clip finished (only the live window records this way)."""

    def _send(self, cmd: str) -> None:
        if self.worker is not None:
            self.worker.commands.put(cmd)

    def _toggle_mirror(self) -> None:
        self.video.mirror = self.mirror_btn.isChecked()

    def show_error(self, msg: str) -> None:
        self.video.error = msg
        self.video.update()
        self.show_toast(f"error: {msg}", error=True)

    def show_toast(self, msg: str, error: bool = False) -> None:
        self.toast.setStyleSheet(f"border-color: {REC.name()};" if error else "")
        self.toast.setText(f"> {msg}")
        self.toast.show()
        self._toast_timer.start(6000)

    def update_state(self, s: PreviewState) -> None:
        self.video.set_state(s)
        self.mouth_view.set_image(s.aligned.mouth if s.aligned else None)
        self.face_view.set_image(s.aligned.face if s.aligned else None)
        self.spark.set_values(s.openness_history)
        self.open_label.setText(
            f"open {s.openness_history[-1]:.2f}" if s.landmarks is not None else "--"
        )
        self.stat_labels["camera"].setText(leader("camera", f"{s.camera_fps:.1f} fps"))
        self.stat_labels["driver"].setText(
            leader("driver", f"{s.native_fps:.0f} fps" if s.native_fps else "--")
        )
        self.stat_labels["latency"].setText(leader("latency", f"{s.process_ms:.1f} ms"))
        self.stat_labels["smooth"].setText(leader("smooth", "on" if s.smoothing else "off"))
        recording = s.recording_s is not None
        if self.record_btn.isChecked() != recording:
            self.record_btn.setChecked(recording)
        self._set_record_text(s.recording_s)
        if s.message:
            self.show_toast(s.message, error=s.message.startswith("Not saved"))

    def closeEvent(self, event: object) -> None:  # noqa: N802
        if self.worker is not None:
            self.worker.stop()
        super().closeEvent(event)  # type: ignore[arg-type]

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(1240, 760)


def run_preview(cfg: Config, mirror: bool = True) -> None:
    existing = QApplication.instance()
    app = existing if isinstance(existing, QApplication) else QApplication([])
    app.setApplicationName("Mouthpiece")
    app.setFont(mono_font(12))
    win = PreviewWindow(cfg, mirror=mirror)
    win.show()
    win.start()
    app.exec()
