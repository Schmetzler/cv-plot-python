import numpy as np
import cv2
from typing import Tuple, Any, Optional
import math
from typing import Callable

from cv_plot.core import Axes
from .mouse_event import MouseEvent, MouseEventHandler

class MouseAdapter:
    """
    Equivalent to CvPlot::MouseAdapter.
    Handles default mouse interaction (pan, zoom, autoscale) for an Axes object.
    """

    def __init__(self, axes: Axes):
        """Equivalent to MouseAdapter::MouseAdapter(Axes &axes)."""
        # C++: Impl(Axes &axes) :_axes(axes)
        self._pressPosPix = (0,0)
        self._lastPosPix = (0,0)
        self._mouseEventHandler = None
        self._axes = axes

    # The move constructor (~MouseAdapter(MouseAdapter &&a)) and destructor (~MouseAdapter()) 
    # are handled implicitly by Python's garbage collection and reference semantics.

    def getAxes(self) -> Axes:
        """Equivalent to Axes &MouseAdapter::getAxes()."""
        # C++: return impl->_axes;
        return self._axes

    def setMouseEventHandler(self, mouseEventHandler: MouseEventHandler) -> None:
        """Equivalent to void MouseAdapter::setMouseEventHandler(...)."""
        # C++: impl->_mouseEventHandler = mouseEventHandler;
        self._mouseEventHandler = mouseEventHandler

    # --- Internal Event Handler (From Impl::handleEvent) ---

    def _handle_event(self, mouseEvent: MouseEvent) -> bool:
        """
        Implements the default pan, zoom, and autoscale behavior.
        Returns True if the Axes should re-render.
        """
        update = False
        event = mouseEvent.event()
        flags = mouseEvent.flags()
        outer_point = mouseEvent.outerPoint()
        render_size = mouseEvent.renderSize()

        # 1. Handle BUTTON DOWN (Capture press position for pan/zoom reference)
        # C++: cv::EVENT_MBUTTONDOWN || cv::EVENT_RBUTTONDOWN
        if event == cv2.EVENT_MBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
            self._pressPosPix = outer_point
            self._lastPosPix = outer_point
            update = True # State change, potentially needs re-render

        # 2. Handle MOUSE MOVE (Pan and Zoom operations)
        # C++: cv::EVENT_MOUSEMOVE
        elif event == cv2.EVENT_MOUSEMOVE:
            delta_pix = (outer_point[0] - self._lastPosPix[0], 
                         outer_point[1] - self._lastPosPix[1])
            
            # C++: mouseEvent.flags() & cv::EVENT_FLAG_MBUTTON (Middle button pan)
            if flags & cv2.EVENT_FLAG_MBUTTON:
                self._axes.pan(render_size, delta_pix)
                self._lastPosPix = outer_point
                update = True
            
            # C++: mouseEvent.flags() & cv::EVENT_FLAG_RBUTTON (Right button zoom)
            elif flags & cv2.EVENT_FLAG_RBUTTON:
                # Zoom sensitivity is 2^(delta / 100)
                # C++: scalex = std::pow(2., -deltaPix.x / 100.);
                # C++: scaley = std::pow(2., deltaPix.y / 100.);
                scale_x = math.pow(2., -delta_pix[0] / 100.0)
                scale_y = math.pow(2., delta_pix[1] / 100.0)
                
                self._axes.zoom(render_size, self._pressPosPix, scale_x, scale_y)
                self._lastPosPix = outer_point
                update = True

        # 3. Handle DOUBLE CLICK (Autoscale)
        # C++: cv::EVENT_RBUTTONDBLCLK
        elif event == cv2.EVENT_RBUTTONDBLCLK:
            self._axes.setXLimAuto()
            self._axes.setYLimAuto()
            update = True

        # 4. Handle MOUSE WHEEL (Zoom)
        # C++: cv::EVENT_MOUSEWHEEL
        elif event == cv2.EVENT_MOUSEWHEEL:
            # OpenCV's getMouseWheelDelta is typically flags / 120.0 (or just flags if normalized)
            # Assuming getMouseWheelDelta(flags) is equivalent to flags // 120 in Python context
            delta = flags // 120 
            
            # C++: double scale = std::pow(1.2, - cv::getMouseWheelDelta(mouseEvent.flags()) / 120.);
            # We use 1.2, not 2., for wheel zoom sensitivity
            scale = math.pow(1.2, -delta) 
            
            self._axes.zoom(render_size, outer_point, scale, scale)
            update = True

        return update

    # --- Public Event Dispatcher (From Impl::mouseEvent) ---

    def mouseEvent(self, mouseEvent: MouseEvent) -> bool:
        """
        Equivalent to bool MouseAdapter::mouseEvent(const MouseEvent &mouseEvent).
        Dispatches the event to the built-in handler and the user-defined handler.
        Returns True if a re-render is required.
        """
        update_builtin = False
        update_user = False

        # 1. Handle built-in adapter logic
        # C++: if (handleEvent(mouseEvent)) { update = true; }
        update_builtin = self._handle_event(mouseEvent)
        
        # 2. Handle user-defined event handler
        # C++: if (_mouseEventHandler && _mouseEventHandler(mouseEvent)){ update = true; }
        if self._mouseEventHandler:
            update_user = self._mouseEventHandler(mouseEvent)
        
        # C++: return update; (Logical OR)
        return update_builtin or update_user
