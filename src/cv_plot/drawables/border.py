import numpy as np
import cv2  # Used for the drawing function
from typing import Tuple, List

from cv_plot.core import RenderTarget, Drawable

class Border(Drawable):
    """
    Equivalent to CvPlot::Border.
    The C++ Impl fields are moved directly into this class.
    """
    def __init__(self):
        # Equivalent to Border::Impl::_color = cv::Scalar(0, 0, 0)
        # Assuming cv::Scalar(B, G, R) for black
        super().__init__()

    # Destructor (~Border) is implicit in Python.

    def render(self, renderTarget: RenderTarget) -> None:
        """
        Equivalent to void Border::render(RenderTarget & renderTarget)
        Draws a 1-pixel black border around the inner plotting area.
        """
        # 1. Get innerRect (C++: cv::Rect borderRect = renderTarget.innerRect();)
        x, y, w, h = renderTarget.innerRect()
        
        # Check area (C++: if (!borderRect.area()) { return; })
        if w <= 0 or h <= 0:
            return

        # 2. Expand the rectangle by 1 pixel (C++ logic)
        # C++: borderRect.x--; borderRect.y--; borderRect.height += 2; borderRect.width += 2;
        border_rect_expanded = (
            x - 1, 
            y - 1, 
            w + 2, 
            h + 2
        )
        
        # 3. Draw the rectangle
        # C++: cv::rectangle(renderTarget.outerMat(), borderRect, impl->_color);
        
        # Convert (x, y, w, h) to OpenCV's required corner points:
        pt1 = (border_rect_expanded[0], border_rect_expanded[1])               # Top-Left (x, y)
        pt2 = (border_rect_expanded[0] + border_rect_expanded[2],              # Bottom-Right (x + w, y + h)
               border_rect_expanded[1] + border_rect_expanded[3])
        
        # Draw on the outer Mat using cv2.rectangle (assumes BGR format)
        cv2.rectangle(
            img=renderTarget.outerMat(), 
            pt1=pt1, 
            pt2=pt2, 
            color=self._color,  # (0, 0, 0) is black
            thickness=1         # Default thickness is 1
        )
