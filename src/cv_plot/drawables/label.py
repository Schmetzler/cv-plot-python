import numpy as np
import cv2
from typing import Tuple, List, Any, Optional

from cv_plot.core import RenderTarget, Drawable
from cv_plot.core.util import paint
from .axis import YAxis, XAxis

class YLabel(Drawable):
    """
    Equivalent to CvPlot::YLabel.
    Impl fields and logic are merged directly into this class.
    """

    def __init__(self, label: str = "", yAxis: Optional[YAxis] = None):
        super().__init__()        
        # Initialize mutable state
        self._yAxis = yAxis
        # C++: setLabel(label);
        self.setLabel(label)

    # Destructor (~YLabel) is implicit in Python.
    
    # --- Public Accessors (for chaining) ---

    def setLabel(self, label: str) -> 'YLabel':
        """Equivalent to YLabel & YLabel::setLabel(const std::string & label)."""
        self._label = label
        return self

    def getLabel(self) -> str:
        """Equivalent to std::string YLabel::getLabel()."""
        return self._label

    def setYAxis(self, yAxis: Optional[YAxis]) -> 'YLabel':
        """Equivalent to YLabel& YLabel::setYAxis(YAxis* yAxis)."""
        self._yAxis = yAxis
        return self
    
    # --- Core Methods (Renamed from Impl::render to _render) ---

    def render(self, renderTarget: RenderTarget) -> None:
        """
        Internal render method (from Impl::render).
        Renders the vertical label by drawing on a temporary canvas, rotating it,
        and then pasting it onto the outer Mat.
        """
        outerMat = renderTarget.outerMat()
        inner_x, inner_y, inner_w, inner_h = renderTarget.innerRect()
        
        if inner_w <= 0 or inner_h <= 0 or not self._label:
            return

        # 1. Measure text size
        # C++: cv::Size textSize = cv::getTextSize(_label, _fontFace, _fontScale, _fontThickness, &baseline);
        (text_w, text_h), baseline = cv2.getTextSize(
            self._label, self._font_face, self._font_scale, self._font_thickness
        )
        
        # 2. Setup temporary canvas for horizontal text
        # C++: int margin = _fontThickness + 5; // required for cv::LINE_AA
        margin = self._font_thickness + 5
        
        # C++ temp size: textSize.height + 2 * margin, textSize.width + baseline + 2 * margin
        temp_h = text_h + 2 * margin
        temp_w = text_w + baseline + 2 * margin # Baseline accounts for descenders/slight offset
        
        # Create white background temp image (cv::Vec3b::all(255) in C++)
        temp = np.full((temp_h, temp_w, 3), 255, dtype=np.uint8)
        
        # 3. Draw text horizontally onto temp
        # C++: cv::Point pos((temp.cols - textSize.width) / 2, textSize.height + margin);
        pos_x = (temp_w - text_w) // 2
        pos_y = text_h + margin # Positioned for baseline placement
        
        cv2.putText(
            temp, 
            self._label, 
            (pos_x, pos_y), 
            self._font_face, 
            self._font_scale, 
            self._color, 
            self._font_thickness, 
            cv2.LINE_AA
        )
        
        # 4. Rotate text vertically (90 degrees counter-clockwise)
        # C++: temp = temp.t(); cv::flip(temp, temp, 0);
        # Python: Transpose (temp.T) then flip vertically (code 0)
        temp = temp.T
        temp = cv2.flip(temp, 0)
        
        # 5. Calculate final position on outer Mat
        # Get Y-Axis width (if available)
        yAxisWidth = self._yAxis.getWidth() if self._yAxis else 0
        
        # C++: cv::Point labelTopLeft(left - yAxisWidth - 5 - temp.cols, ycenter - temp.rows / 2);
        left = inner_x
        ycenter = inner_y + inner_h // 2
        
        # Rotated text dimensions
        rotated_w = temp.shape[1]
        rotated_h = temp.shape[0]
        
        # Top-Left X: Left edge of innerRect - Axis width - 5px margin - rotated width
        label_top_left_x = left - yAxisWidth - 5 - rotated_w
        
        # Top-Left Y: Center of innerRect Y - half of rotated height
        label_top_left_y = ycenter - rotated_h // 2
        
        labelTopLeft = (label_top_left_x, label_top_left_y)
        
        # 6. Paint onto the final outer Mat
        # C++: Internal::paint(temp, outerMat, labelTopLeft);
        paint(temp, outerMat, labelTopLeft)

class XLabel(Drawable):
    """
    Equivalent to CvPlot::XLabel.
    Impl fields and logic are merged directly into this class.
    """

    def __init__(self, label: str = ""):
        # C++: XLabel::XLabel(const std::string &label) { setLabel(label); }
        super().__init__()
        self.setLabel(label)

    # Destructor (~XLabel) is implicit in Python.
    
    # --- Utility Method (From Impl) ---

    def _getTextWidth(self, text: str) -> int:
        """Equivalent to int Impl::getTextWidth(const std::string &text)."""
        # C++: return cv::getTextSize(...).width;
        (w, h), baseline = cv2.getTextSize(
            text, 
            self._font_face, 
            self._font_scale, 
            self._font_thickness
        )
        return w

    # --- Public Accessors (for chaining) ---

    def setLabel(self, label: str) -> 'XLabel':
        """Equivalent to XLabel & XLabel::setLabel(const std::string & label)."""
        # C++: impl->_label = label;
        self._label = label
        return self

    def getLabel(self) -> str:
        """Equivalent to std::string XLabel::getLabel()."""
        # C++: return impl->_label;
        return self._label
    
    # --- Core Methods (Renamed from Impl::render to _render) ---

    def render(self, renderTarget: RenderTarget) -> None:
        """
        Internal render method (from Impl::render).
        Draws the centered label beneath the plot area.
        """
        outerMat = renderTarget.outerMat()
        inner_x, inner_y, inner_w, inner_h = renderTarget.innerRect()
        
        if inner_w <= 0 or inner_h <= 0 or not self._label:
            return

        # 1. Calculate positions
        bottom = inner_y + inner_h
        xcenter = inner_x + inner_w // 2
        
        w = self._getTextWidth(self._label)
        
        # C++: cv::Point labelPos(xcenter - w / 2, bottom + 35);
        # Note: In OpenCV, the Y position for putText is the baseline of the text.
        # The offset 35 is used to position the label below the plot area/X-axis.
        label_pos_x = xcenter - w // 2
        label_pos_y = bottom + 35
        labelPos: tuple = (label_pos_x, label_pos_y)

        # 2. Draw text
        # C++: cv::putText(outerMat, _label, labelPos, _fontFace, _fontScale, _color, _fontThickness, cv::LINE_AA);
        cv2.putText(
            outerMat, 
            self._label, 
            labelPos, 
            self._font_face, 
            self._font_scale, 
            self._color, 
            self._font_thickness, 
            cv2.LINE_AA
        )

class Title(Drawable):
    """
    Equivalent to CvPlot::Title.
    Impl fields and logic are merged directly into this class.
    """

    def __init__(self, title: str = ""):
        # C++: Title::Title(const std::string &title) { setTitle(title); }
        super().__init__()

        self.setTitle(title)

    # Destructor (~Title) is implicit in Python.
    
    # --- Public Accessors (for chaining) ---

    def setTitle(self, title: str) -> 'Title':
        """Equivalent to Title & Title::setTitle(const std::string & title)."""
        # C++: impl->_title = title;
        self._title = title
        return self

    def getTitle(self) -> str:
        """Equivalent to std::string Title::getTitle()."""
        # C++: return impl->_title;
        return self._title
    
    # --- Core Methods (Renamed from Impl::render to _render) ---

    def render(self, renderTarget: RenderTarget) -> None:
        """
        Internal render method (from Impl::render).
        Draws the centered title above the plot area.
        """
        outerMat = renderTarget.outerMat()
        inner_x, inner_y, inner_w, inner_h = renderTarget.innerRect()
        
        if not self._title:
            return

        # 1. Measure text size
        # C++: cv::Size size = cv::getTextSize(_title, _fontFace, _fontScale, _fontThickness, &baseline);
        (text_w, text_h), baseline = cv2.getTextSize(
            self._title, 
            self._font_face, 
            self._font_scale, 
            self._font_thickness
        )
        
        # 2. Calculate position
        # X: Center text horizontally above innerRect
        xcenter = inner_x + inner_w // 2
        title_pos_x = xcenter - text_w // 2
        
        # Y: Position the baseline above the plot area.
        # C++: innerRect.y - (size.height * 3) / 2
        # This places the baseline 1.5 times the text height above the inner plot Y start.
        title_pos_y = inner_y - (text_h * 3) // 2 
        
        titlePos: tuple = (title_pos_x, title_pos_y)

        # 3. Draw text
        # C++: cv::putText(outerMat, _title, titlePos, _fontFace, _fontScale, _color, _fontThickness, cv::LINE_AA);
        cv2.putText(
            outerMat, 
            self._title, 
            titlePos, 
            self._font_face, 
            self._font_scale, 
            self._color, 
            self._font_thickness, 
            cv2.LINE_AA
        )
