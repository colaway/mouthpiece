"""Small popup near the mouse cursor: a one-line status HUD or the top-3 picker.

It never takes keyboard focus (the app you are typing into keeps it): picker keys arrive
through the global event tap, and clicks on a row choose it. Same retro look as the other
windows: black panel, monospace, amber accents, inverted "> selected" row.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from PySide6.QtCore import QPoint, QRect, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QCursor, QFontMetrics, QMouseEvent, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QApplication, QWidget

from mouthpiece.app.preview_window import AMBER, BG, DIM, FAINT, FG, REC, mono_font

PAD_X, PAD_Y, ROW_H = 12, 8, 22


def popup_position(cursor: QPoint, size: QSize, screen: QRect, offset: int = 18) -> QPoint:
    """Below-right of the cursor, flipped/clamped so the popup stays on screen."""
    x, y = cursor.x() + offset, cursor.y() + offset
    if x + size.width() > screen.right():
        x = cursor.x() - offset - size.width()
    if y + size.height() > screen.bottom():
        y = cursor.y() - offset - size.height()
    x = max(screen.left(), min(x, screen.right() - size.width()))
    y = max(screen.top(), min(y, screen.bottom() - size.height()))
    return QPoint(x, y)


class CursorPopup(QWidget):
    chosen = Signal(int)   # a picker row was clicked

    def __init__(self) -> None:
        # ToolTip windows float above other apps without activating ours (a Qt.Tool window
        # would hide whenever the accessory app is inactive, i.e. always).
        super().__init__(None, Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.WindowStaysOnTopHint
                         | Qt.WindowType.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.font_main = mono_font(13)
        self.font_meta = mono_font(10)
        self.text = ""
        self.color = FG
        self.options: tuple[str, ...] = ()
        self.selected = 0
        self.meta = ""
        self.deadline: float | None = None
        self.timeout_s = 0.0
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self.hide)
        self._tick = QTimer(self)
        self._tick.timeout.connect(self.update)

    # --- modes ---------------------------------------------------------------

    def show_status(self, text: str, color: QColor = FG, hide_after_s: float = 0.0) -> None:
        self.options, self.text, self.color, self.deadline = (), text, color, None
        self._tick.stop()
        self._place()
        if hide_after_s > 0:
            self._hide_timer.start(int(hide_after_s * 1000))
        else:
            self._hide_timer.stop()

    def show_choices(self, options: Sequence[str], meta: str, timeout_s: float) -> None:
        self.options, self.selected, self.meta = tuple(options), 0, meta
        self.timeout_s = timeout_s
        self.deadline = time.monotonic() + timeout_s if timeout_s > 0 else None
        self._hide_timer.stop()
        if self.deadline is not None:
            self._tick.start(40)
        self._place()

    def set_selected(self, i: int, stop_countdown: bool = False) -> None:
        self.selected = i
        if stop_countdown:
            self.deadline = None
            self._tick.stop()
        self.update()

    def hide(self) -> None:
        self._tick.stop()
        self._hide_timer.stop()
        super().hide()

    # --- geometry / painting -------------------------------------------------

    def _content_size(self) -> QSize:
        fm, fm_meta = QFontMetrics(self.font_main), QFontMetrics(self.font_meta)
        if not self.options:
            return QSize(fm.horizontalAdvance(self.text) + 2 * PAD_X + 16, ROW_H + 2 * PAD_Y)
        w = max([fm.horizontalAdvance(f"> 1  {o}") for o in self.options]
                + [fm_meta.horizontalAdvance(self.meta)])
        h = len(self.options) * ROW_H + 18 + 6 + 2 * PAD_Y
        return QSize(max(220, w + 2 * PAD_X + 8), h)

    def _place(self) -> None:
        size = self._content_size()
        cursor = QCursor.pos()
        screen = QApplication.screenAt(cursor) or QApplication.primaryScreen()
        geo = screen.availableGeometry() if screen else QRect(0, 0, 1440, 900)
        self.setGeometry(QRect(popup_position(cursor, size, geo), size))
        self.show()
        self.raise_()
        self.update()

    def _row_rect(self, i: int) -> QRectF:
        return QRectF(PAD_X - 4, PAD_Y + 18 + i * ROW_H, self.width() - 2 * PAD_X + 8, ROW_H)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.fillRect(r, BG)
        p.setPen(QPen(FAINT, 1))
        p.drawRect(r)
        if not self.options:
            p.setFont(self.font_main)
            p.setPen(self.color)
            p.drawText(r.adjusted(PAD_X, 0, -PAD_X, 0), Qt.AlignmentFlag.AlignVCenter, self.text)
            return
        p.setFont(self.font_meta)
        p.setPen(DIM)
        p.drawText(QRectF(PAD_X, PAD_Y, self.width() - 2 * PAD_X, 14),
                   Qt.AlignmentFlag.AlignVCenter, self.meta)
        p.setFont(self.font_main)
        for i, opt in enumerate(self.options):
            row = self._row_rect(i)
            if i == self.selected:
                p.fillRect(row, FG)
                p.setPen(BG)
                label = f"> {i + 1}  {opt}"
            else:
                p.setPen(FG)
                label = f"  {i + 1}  {opt}"
            p.drawText(row.adjusted(4, 0, -4, 0), Qt.AlignmentFlag.AlignVCenter, label)
            if i != self.selected:  # amber digit, like key hints elsewhere
                p.setPen(AMBER)
                fm = QFontMetrics(self.font_main)
                p.drawText(row.adjusted(4 + fm.horizontalAdvance("  "), 0, 0, 0),
                           Qt.AlignmentFlag.AlignVCenter, str(i + 1))
        if self.deadline is not None and self.timeout_s > 0:
            left = max(0.0, (self.deadline - time.monotonic()) / self.timeout_s)
            bar = QRectF(PAD_X, self.height() - PAD_Y - 2, (self.width() - 2 * PAD_X) * left, 2)
            p.fillRect(bar, AMBER)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        for i in range(len(self.options)):
            if self._row_rect(i).contains(event.position()):
                self.chosen.emit(i)
                return


# Re-exported colours for the app's status messages.
STATUS_REC, STATUS_BUSY, STATUS_DIM = REC, AMBER, DIM
