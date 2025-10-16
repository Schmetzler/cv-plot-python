import numpy as np
import cv2
from typing import Callable, Tuple, Any

from cv_plot.core import Axes, Projection

# --- MOUSEEVENT CLASS ---

class MouseEvent:
    """
    Equivalent to CvPlot::MouseEvent.
    A data structure packaging mouse event details along with context 
    needed to convert screen coordinates to plot coordinates.
    """

    def __init__(self, axes: Axes, renderSize: tuple, event: int, x: int, y: int, flags: int):
        # C++: MouseEvent(Axes &axes, cv::Size renderSize, int event, int x, int y, int flags)
        self._axes = axes
        self._renderSize = renderSize
        self._event = event
        self._x = x
        self._y = y
        self._flags = flags

    # --- Public Accessor Methods (Equivalent to C++ const methods) ---

    def axes(self) -> Axes:
        """The Axes object this event occurred on."""
        # C++: Axes &axes() const
        return self._axes

    def renderSize(self) -> tuple:
        """Size of the rendered cv::Mat (window size)."""
        # C++: cv::Size renderSize() const
        return self._renderSize

    def event(self) -> int:
        """Event type (e.g., cv2.EVENT_LBUTTONDOWN)."""
        # C++: int event() const
        return self._event

    def x(self) -> int:
        """X coordinate of the mouse position (in screen/outer coordinates)."""
        # C++: int x() const
        return self._x

    def y(self) -> int:
        """Y coordinate of the mouse position (in screen/outer coordinates)."""
        # C++: int y() const
        return self._y

    def flags(self) -> int:
        """Flags indicating which buttons or keys are pressed (e.g., cv2.EVENT_FLAG_LBUTTON)."""
        # C++: int flags() const
        return self._flags

    def projection(self) -> Projection:
        """Calculates and returns the current Projection object for the Axes."""
        # C++: Projection projection() const
        return self._axes.getProjection(self._renderSize)

    # --- Utility Methods ---

    def outerPoint(self) -> tuple:
        """Position in outer rect (raw screen coordinates)."""
        # C++: cv::Point outerPoint() const
        return (self._x, self._y)

    def innerPoint(self) -> tuple:
        """
        Position relative to the inner plot area (pixels inside the axes bounds).
        This is calculated by subtracting the inner Rect's top-left corner from the outer point.
        """
        # C++: cv::Point innerPoint() const
        outer_x, outer_y = self.outerPoint()
        inner_rect_x, inner_rect_y, _, _ = self.projection().innerRect()
        
        # C++: cv::Point(outer.x - innerRect.x, outer.y - innerRect.y);
        return (outer_x - inner_rect_x, outer_y - inner_rect_y)

    def pos(self) -> tuple:
        """
        Position in logical plot coordinates (unprojected data space).
        This is the most critical calculation.
        """
        # C++: cv::Point2d pos() const
        return self.projection().unproject(self.innerPoint())

# --- MOUSEEVENTHANDLER TYPEDEF ---

# C++: typedef std::function<bool(const MouseEvent&)> MouseEventHandler;

MouseEventHandler = Callable[[MouseEvent], bool]