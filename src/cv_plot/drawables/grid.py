import numpy as np
import cv2
import math
from typing import Tuple, List, Any, Optional, Union

from .axis import YAxis, XAxis
from .line import LineType, LineBase, Orientation
from cv_plot.core import RenderTarget

class GridBase(LineBase):
    """
    Equivalent to CvPlot::HorizontalGrid, inheriting from LineBase.
    Impl fields and logic are merged directly into this class.
    """
    
    # _parent is not needed as methods access self directly

    def __init__(self, orientation : Orientation, axis: Union[XAxis, YAxis, None] = None, lineSpec: str = "-"):
        # C++ Impl constructor calls setColor(cv::Scalar::all(200))
        # We ensure LineBase is initialized first, then override color
        super().__init__(lineSpec)
        
        # C++ Impl initializers
        self._orientation = orientation
        self._axis = axis
        self._enabled = True
        
        # C++ Impl constructor logic: parent.setColor(cv::Scalar::all(200));
        self.setColor((200, 200, 200)) # Set grid color to gray

    # Destructor (~HorizontalGrid) is implicit in Python.
    
    # --- Public Accessors (for chaining) ---

    def setAxis(self, axis: Union[YAxis, XAxis, None]) -> 'GridBase':
        """Equivalent to HorizontalGrid & HorizontalGrid::setYAxis(YAxis * yAxis)."""
        self._axis = axis
        return self

    def setEnabled(self, enabled: bool) -> 'HorizontalGrid':
        """Equivalent to HorizontalGrid & HorizontalGrid::setEnabled(bool enabled)."""
        self._enabled = enabled
        return self

    def getEnabled(self) -> bool:
        """Equivalent to bool HorizontalGrid::getEnabled()."""
        return self._enabled

    # --- Core Methods (Renamed from Impl::render to _render for internal use) ---
    
    def render(self, renderTarget: RenderTarget) -> None:
        """
        Internal render method (from Impl::render).
        Draws the horizontal grid lines.
        """
        # C++: if (!_enabled || _parent.getLineType()==LineType::None) { return; }
        if not self._enabled or self.getLineType() == LineType.NONE:
            return

        # C++: if (!_yAxis) { return; }
        if not self._axis:
            # TODO: calc ticks logic would go here if C++ had it
            return 
        
        ticks = self._axis.getTicks()
        innerMat = renderTarget.innerMat()
        
        # Check if the drawing surface is valid
        if innerMat.size == 0:
            return

        color = self.getColor()
        lineWidth = self.getLineWidth()
        
        # The C++ code implements a simple dash-dot line using multiple line calls
        # Note: This is a simplified dashed line; true dashed lines require a different method.
        # We will replicate the C++ logic exactly: Draw segments (8 pixels total: 4 on, 4 off)
        # plus a fixed 5-pixel line at the start for anti-aliasing on the axis.
        
        for tick in ticks:
            # C++: int tickPix = (int)(renderTarget.project(cv::Point2d(0,tick)).y + .5);
            # Project data y-coordinate to pixel y-coordinate (x is irrelevant for H-line)
            if self._orientation == Orientation.HORIZONTAL:
                tick_pix_y = int(renderTarget.project((0.0, tick))[1] + 0.5)
                
                # 1. Draw dashed/dotted pattern across the inner mat width
                # C++: for (int x = 0; x < innerMat.cols; x += 8) {
                #          cv::line(innerMat, cv::Point(x, tickPix), cv::Point(x + 4, tickPix), color, lineWidth);
                #      }
                cols = innerMat.shape[1]
                for x in range(0, cols, 8):
                    pt1 = (x, tick_pix_y)
                    # Ensure the line doesn't extend past the width boundary
                    pt2 = (min(x + 4, cols - 1), tick_pix_y) 
                    
                    # Draw the 'on' segment (4 pixels long)
                    cv2.line(innerMat, pt1, pt2, color, lineWidth)

                # 2. Draw a fixed 5-pixel line at the start (C++ code's attempt to ensure AA on the axis)
                # C++: cv::line(innerMat, cv::Point(0, tickPix), cv::Point(5, tickPix), color, lineWidth);
                cv2.line(innerMat, (0, tick_pix_y), (5, tick_pix_y), color, lineWidth)
            else:
                tick_pix = int(renderTarget.project((tick, 0.0))[0] + 0.5)
                rows = innerMat.shape[0]
                for y in range(0, rows, 8):
                    pt1 = (tick_pix, y)
                    pt2 = (tick_pix, y + 4) 
                    cv2.line(innerMat, pt1, pt2, color, lineWidth)
                cv2.line(innerMat, (tick_pix, rows-5), (tick_pix, rows), color, lineWidth)

class HorizontalGrid(GridBase):
    def __init__(self, axis: YAxis = None, lineSpec: str = "-"):
        super().__init__(Orientation.HORIZONTAL, axis=axis, lineSpec=lineSpec)

class VerticalGrid(GridBase):
    def __init__(self, axis: XAxis = None, lineSpec: str = "-"):
        super().__init__(Orientation.VERTICAL, axis=axis, lineSpec=lineSpec)