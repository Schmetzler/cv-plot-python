import numpy as np
import cv2
from typing import Tuple, List, Optional, Callable
import time

from .mouse_adapter import MouseAdapter
from .mouse_event import MouseEvent, MouseEventHandler
from cv_plot.core import Axes

# --- GLOBAL CALLBACK WRAPPER ---
# OpenCV callbacks are typically global functions that rely on userdata.

# Use a dictionary to map window names to Window instances
_WINDOW_MAP = {} 

def _cvplot_mouse_callback(event: int, x: int, y: int, flags: int, param: str):
    """
    Python equivalent of the C++ static mouse callback, using the window name 
    (passed as param) to retrieve the Window instance.
    """
    window_name = param
    window = _WINDOW_MAP.get(window_name)

    if window:
        # C++: window.updateSize();
        window._update_size() 
        
        # C++: MouseEvent mouseEvent(window._mouseAdapter.getAxes(), window._mat.size(), event, x, y, flags);
        # _mat.shape[1] = cols, _mat.shape[0] = rows
        mat_size = (window._mat.shape[1], window._mat.shape[0])
        mouse_event = MouseEvent(window._mouseAdapter.getAxes(), mat_size, event, x, y, flags)
        
        # C++: if (window._mouseAdapter.mouseEvent(mouseEvent)) { window.update(); }
        if window._mouseAdapter.mouseEvent(mouse_event):
            window.update()


# --- WINDOW CLASS ---

class Window:
    """
    Equivalent to CvPlot::Window. Manages an OpenCV highgui window and 
    integrates the MouseAdapter for interaction.
    """

    def __init__(self, windowName: str, axes: Axes, cols: int = 640, rows: int = 480):
        """Equivalent to Window::Window(std::string, Axes&, int, int)."""
        self._windowName = windowName

        self._rows = rows
        self._cols = cols
        
        # C++: _mouseAdapter(axes)
        self._mouseAdapter = MouseAdapter(axes)

        # C++: if(valid()){ cv::destroyWindow(windowName); }
        if self.valid():
             cv2.destroyWindow(windowName) # Close any existing window with the same name

        # C++: cv::namedWindow(windowName, cv::WINDOW_NORMAL | cv::WINDOW_FREERATIO);
        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
        
        # C++: cv::resizeWindow(windowName, { cols,rows });
        cv2.resizeWindow(windowName, cols, rows)
        
        # C++: axes.render(_mat, cv::Size(cols, rows));
        # Initialize _mat
        self._mat = self._mouseAdapter.getAxes().render(cols, rows)
        
        # C++: cv::imshow(windowName, _mat);
        cv2.imshow(windowName, self._mat)
        
        # C++: setMouseCallback();
        self._set_mouse_callback()
        
        # Add to global map for callback access
        _WINDOW_MAP[windowName] = self


    def __del__(self):
        """Equivalent to Window::~Window()."""
        # C++: if(valid()){ cv::destroyWindow(_windowName); }
        if self._windowName in _WINDOW_MAP:
             del _WINDOW_MAP[self._windowName]
             
        if self.valid():
            cv2.destroyWindow(self._windowName)

    # --- Public Methods ---

    def getAxes(self) -> Axes:
        """Equivalent to Axes& Window::getAxes()."""
        # C++: return _mouseAdapter.getAxes();
        return self._mouseAdapter.getAxes()

    def update(self) -> None:
        """Equivalent to void Window::update(). Re-renders the plot."""
        # C++: if(!valid()){ return; }
        if not self.valid():
            return
            
        # C++: auto rect = cv::getWindowImageRect(_windowName);
        rect = cv2.getWindowImageRect(self._windowName)
        cols, rows = rect[2], rect[3]
        
        # C++: _mouseAdapter.getAxes().render(_mat, rect.size());
        # Re-initialize mat with correct size (rows, cols, channels)
        self._mat = self._mouseAdapter.getAxes().render(cols, rows)
        
        # C++: if (!_mat.empty()) { cv::imshow(_windowName, _mat); }
        if self._mat.size > 0:
            cv2.imshow(self._windowName, self._mat)

    def setMouseEventHandler(self, mouseEventHandler: MouseEventHandler) -> None:
        """Equivalent to void Window::setMouseEventHandler(...)."""
        # C++: _mouseAdapter.setMouseEventHandler(mouseEventHandler);
        self._mouseAdapter.setMouseEventHandler(mouseEventHandler)

    def valid(self) -> bool:
        """
        Equivalent to bool Window::valid() const. Checks if the OpenCV window is alive.
        """
        # C++: return getWindowProperty(_windowName, cv::WND_PROP_AUTOSIZE) >= 0;
        try:
            return cv2.getWindowProperty(self._windowName, cv2.WND_PROP_AUTOSIZE) >= 0
        except cv2.error:
            # Catch exceptions that can occur if the window is fully destroyed
            return False

    def waitKey(self, delay: int = 0) -> int:
        """Equivalent to int Window::waitKey(int delay)."""
        # C++: return waitKey({ this }, delay);
        return Window.waitKey_static([self], delay)

    # --- Static Methods ---
    
    @staticmethod
    def waitKey_static(windows: List['Window'], delay: int = 0) -> int:
        """Equivalent to static int Window::waitKey(const std::vector<Window*> &, int delay)."""
        
        # C++: const auto endTickCount = cv::getTickCount() + (int64)(delay * 0.001f * cv::getTickFrequency());
        end_time = time.time() + delay / 1000.0 # Python's time is simpler than cv::getTickCount

        while True:
            # C++: int key = cv::waitKey(1);
            key = cv2.waitKey(1)
            
            # C++: if (key != -1) { return key; }
            if key != -1:
                return key

            # C++: if (delay > 0 && cv::getTickCount() - endTickCount >= 0) { return -1; }
            if delay > 0 and time.time() >= end_time:
                return -1

            # C++: if (std::all_of(...) { return -1; }
            if all(not window.valid() for window in windows):
                return -1

            # C++: for (auto window : windows) { window->updateSize(); }
            for window in windows:
                window._update_size()


    # --- Private Utility Methods ---

    def _update_size(self) -> None:
        """Equivalent to void Window::updateSize(). Checks if resize occurred and calls update()."""
        # C++: if (!valid()) { return; }
        if not self.valid():
            return

        # C++: auto rect = cv::getWindowImageRect(_windowName);
        rect = cv2.getWindowImageRect(self._windowName)
        cols, rows = rect[2], rect[3]
        
        # C++: if(rect.size() != _mat.size()) { update(); }
        current_size = (self._mat.shape[1], self._mat.shape[0]) # (cols, rows)
        if cols != current_size[0] or rows != current_size[1]:
            self.update()

    def _set_mouse_callback(self) -> None:
        """Equivalent to void Window::setMouseCallback()."""
        # C++ uses 'this' as userdata pointer, Python uses the window name string as param
        # C++: cv::setMouseCallback(_windowName, [](..., void* userdata) { ... }, this);
        cv2.setMouseCallback(self._windowName, _cvplot_mouse_callback, self._windowName)
    