import numpy as np
import cv2
import math
from typing import Tuple, List, Any, Optional

from cv_plot.core import Drawable, RenderTarget
from cv_plot.core.util import format_value, calc_ticks_linear, calc_ticks_log

class XAxis(Drawable):
    """
    Equivalent to CvPlot::XAxis.
    Impl fields and logic are merged directly into this class.
    """

    def __init__(self):
        super().__init__()

        # Initialize mutable state
        self._ticks = []
        self._is_logarithmic = False

    # Destructor (~XAxis) is implicit in Python.

    # --- Utility Methods (From Impl) ---

    def _getTextWidth(self, text: str) -> int:
        """Calculates the pixel width of the given text."""
        # C++: return cv::getTextSize(text, _fontFace, _fontScale, _fontThickness, &baseline).width;
        (w, h), baseline = cv2.getTextSize(
            text, 
            self._font_face, 
            self._font_scale, 
            self._font_thickness
        )
        return w

    def _estimateLabelWidth(self, x: float, step: float) -> int:
        """Estimates the width of the label for a given tick."""
        label = ""
        if self._is_logarithmic:
            # C++: label = Internal::format(x, true);
            label = format_value(x, True)
        else:
            # C++: int digits = std::max(0, -(int)(std::floor(std::log10(step))));
            digits = max(0, -(int)(math.floor(math.log10(step))))
            
            # C++: double rounded = std::floor(x*std::pow(10, digits))*std::pow(10, -digits);
            rounded = math.floor(x * (10**digits)) * (10**(-digits))
            
            # C++: label = Internal::format(rounded);
            label = format_value(rounded)
            
        return self._getTextWidth(label)

    def _calcTicks(self, renderTarget: RenderTarget) -> None:
        """Calculates the tick positions based on the current viewport and scale."""
        # C++: cv::Rect innerRect = renderTarget.innerRect();
        inner_x, inner_y, inner_w, inner_h = renderTarget.innerRect()
        
        # Unproject points to get data space coordinates
        # C++: double x0 = renderTarget.unproject(cv::Point2d(0, 0)).x;
        x0 = renderTarget.unproject((0.0, 0.0))[0]
        x05 = renderTarget.unproject((inner_w / 2.0, 0.0))[0]
        x1 = renderTarget.unproject((inner_w, 0.0))[0]

        # C++: check for invalid state
        if x1 == x0 or not (math.isfinite(x0) and math.isfinite(x05) and math.isfinite(x1)):
            self._ticks = [x0]
            return

        # Ensure x0 < x1
        if x1 < x0:
            x0, x1 = x1, x0

        # Determine if scale is logarithmic
        # C++: double epsilon = 1e-5; _isLogarithmic = std::abs(2 * (x05 - x0) / (x1 - x0) - 1) > epsilon;
        epsilon = 1e-5
        self._is_logarithmic = abs(2 * (x05 - x0) / (x1 - x0) - 1) > epsilon

        # Estimate tick count for optimal spacing
        step0 = (x1 - x0) / 10
        estimated_label_width = max(self._estimateLabelWidth(x0, step0), self._estimateLabelWidth(x1, step0))
        spacing = 30
        
        # C++: int estimatedTickCount = (int)std::ceil(innerRect.width / (estimatedLabelWidth + spacing));
        estimated_tick_count = math.ceil(inner_w / (estimated_label_width + spacing))

        if self._is_logarithmic:
            # C++: _ticks = Internal::calcTicksLog(x0, x1, estimatedTickCount);
            self._ticks = calc_ticks_log(x0, x1, estimated_tick_count)
        else:
            # C++: _ticks = Internal::calcTicksLinear(x0, x1, estimatedTickCount);
            self._ticks = calc_ticks_linear(x0, x1, estimated_tick_count)

    # --- Rendering Methods (From Impl) ---
    
    def _renderAxisLine(self, renderTarget: RenderTarget) -> None:
        """Draws the main horizontal axis line."""
        outerMat = renderTarget.outerMat()
        inner_x, inner_y, inner_w, inner_h = renderTarget.innerRect()
        
        left = inner_x - 1
        right = inner_x + inner_w
        y = inner_y + inner_h
        
        # C++: cv::line(outerMat, cv::Point(left, y), cv::Point(right, y), _color);
        cv2.line(outerMat, (left, y), (right, y), self._color)

    def render(self, renderTarget: RenderTarget) -> None:
        """
        Equivalent to void XAxis::render(RenderTarget & renderTarget).
        Calculates ticks, draws them, and draws the axis line.
        """
        outerMat = renderTarget.outerMat()
        inner_x, inner_y, inner_w, inner_h = renderTarget.innerRect()

        if inner_w <= 0 or inner_h <= 0:
            return

        self._calcTicks(renderTarget)
        
        bottom = inner_y + inner_h
        labelPos_y = bottom + 20 # Label position Y is constant
        
        for tick in self._ticks:
            # C++: int tickPix = (int)(renderTarget.project(cv::Point2d(tick, 0)).x + .5);
            tick_pix = int(renderTarget.project((tick, 0.0))[0] + 0.5)
            
            # Label generation
            label = format_value(tick, self._is_logarithmic)
            w = self._getTextWidth(label)
            
            # Label position X
            # C++: labelPos.x = (int)(innerRect.x + tickPix - w / 2);
            labelPos_x = int(inner_x + tick_pix - w / 2)
            
            # 1. Draw Text Label
            # C++: cv::putText(outerMat, label, labelPos, _fontFace, _fontScale, _color, _fontThickness, cv::LINE_AA);
            cv2.putText(
                outerMat, 
                label, 
                (labelPos_x, labelPos_y), 
                self._font_face, 
                self._font_scale, 
                self._color, 
                self._font_thickness, 
                cv2.LINE_AA
            )
            
            # 2. Draw Tick Mark
            # C++: cv::line(outerMat, cv::Point(tickPix+innerRect.x, bottom), cv::Point(tickPix + innerRect.x, bottom+4), _color);
            tick_x = inner_x + tick_pix
            cv2.line(outerMat, (tick_x, bottom), (tick_x, bottom + 4), self._color)

        self._renderAxisLine(renderTarget)

    # --- Drawable Interface ---

    def getTicks(self) -> List[float]:
        """Equivalent to const std::vector<double> & XAxis::getTicks()."""
        # C++: return impl->_ticks;
        return self._ticks

class YAxis(Drawable):
    """
    Equivalent to CvPlot::YAxis.
    Impl fields and logic are merged directly into this class.
    """

    def __init__(self):
        super().__init__()
        # Initialize mutable state
        self._ticks = []
        self._is_logarithmic = False

        self._locate_right = False
        self._width = 0

    # Destructor (~YAxis) is implicit in Python.
    
    # --- Utility Methods (From Impl) ---

    def setLocateRight(self, locateRight: bool = True) -> 'YAxis':
        """Equivalent to YAxis& YAxis::setLocateRight(bool locateRight)."""
        self._locate_right = locateRight
        return self

    def _getTextSize(self, text: str) -> Tuple[int, int]:
        """Calculates the pixel width and height of the given text."""
        # C++: cv::Size getTextSize(const std::string &text)
        (w, h), baseline = cv2.getTextSize(
            text, 
            self._font_face, 
            self._font_scale, 
            self._font_thickness
        )
        # Return (width, height)
        return w, h

    def _calcTicks(self, renderTarget: RenderTarget) -> None:
        """Calculates the tick positions based on the current viewport and scale."""
        inner_x, inner_y, inner_w, inner_h = renderTarget.innerRect()
        
        # Unproject points to get data space coordinates
        y0 = renderTarget.unproject((0.0, 0.0))[1]
        y05 = renderTarget.unproject((0.0, inner_h / 2.0))[1]
        y1 = renderTarget.unproject((0.0, inner_h))[1]

        if y1 == y0 or not (math.isfinite(y0) and math.isfinite(y05) and math.isfinite(y1)):
            self._ticks = [y0]
            return

        # Ensure y0 < y1
        if y1 < y0:
            y0, y1 = y1, y0

        # Estimate tick count
        label_size_h = self._getTextSize("1,2")[1] # Use height of a typical label
        spacing = 20
        estimated_tick_count = math.ceil(inner_h / (label_size_h + spacing))

        # Determine if scale is logarithmic
        epsilon = 1e-5
        self._is_logarithmic = abs(2 * (y05 - y0) / (y1 - y0) - 1) > epsilon

        if self._is_logarithmic:
            self._ticks = calc_ticks_log(y0, y1, estimated_tick_count)
        else:
            self._ticks = calc_ticks_linear(y0, y1, estimated_tick_count)

    # --- Rendering Methods (From Impl) ---
    
    def _renderAxisLine(self, renderTarget: RenderTarget) -> None:
        """Draws the main vertical axis line."""
        outerMat = renderTarget.outerMat()
        inner_x, inner_y, inner_w, inner_h = renderTarget.innerRect()
        
        top = inner_y - 1
        bottom = inner_y + inner_h
        
        # Draw on the left side (default location)
        x = inner_x - 1
        
        # C++: cv::line(outerMat, cv::Point(x, top), cv::Point(x, bottom), _color);
        cv2.line(outerMat, (x, top), (x, bottom), self._color)

    def render(self, renderTarget: RenderTarget) -> None:
        """Calculates and draws the ticks, labels, and axis line."""
        outerMat = renderTarget.outerMat()
        inner_x, inner_y, inner_w, inner_h = renderTarget.innerRect()

        if inner_w <= 0 or inner_h <= 0:
            return

        self._calcTicks(renderTarget)
        
        self._width = 0 # Reset width
        margin = 10
        tick_length = 4
        
        for tick in self._ticks:
            # 1. Project position Y
            # C++: int tickPix = (int)(renderTarget.project(cv::Point2d(0, tick)).y + .5);
            tick_pix_y = int(renderTarget.project((0.0, tick))[1] + 0.5)
            
            # Label and size
            label = format_value(tick, self._is_logarithmic)
            size_w, size_h = self._getTextSize(label)
            
            # Initialize positions (LabelPos and TickPos)
            labelPos_x: int
            tickPos_x: int
            
            # Determine X positions based on location
            if not self._locate_right:
                # Left side
                labelPos_x = inner_x - margin - size_w
                tickPos_x = inner_x - tick_length
            else:
                # Right side
                labelPos_x = inner_x + inner_w + margin
                tickPos_x = inner_x + inner_w

            # Y positions
            # C++: labelPos.y = (int)(innerRect.y + tickPix + size.height / 2);
            labelPos_y = int(inner_y + tick_pix_y + size_h / 2)
            tickPos_y = inner_y + tick_pix_y

            # 2. Draw Text Label
            cv2.putText(
                outerMat, 
                label, 
                (labelPos_x, labelPos_y), 
                self._font_face, 
                self._font_scale, 
                self._color, 
                self._font_thickness, 
                cv2.LINE_AA
            )
            
            # 3. Draw Tick Mark
            # C++: cv::line(outerMat, tickPos, tickPos + cv::Point(tickLength-1,0), _color);
            # The tick is a horizontal line segment of length 'tickLength' starting at tickPos_x
            cv2.line(
                outerMat, 
                (tickPos_x, tickPos_y), 
                (tickPos_x + tick_length - 1, tickPos_y), 
                self._color
            )
            
            # 4. Update Width (Max width of labels + margin)
            if margin + size_w > self._width:
                self._width = margin + size_w

        self._renderAxisLine(renderTarget)

    # --- Drawable Interface & Public Accessors ---

    def getTicks(self) -> List[float]:
        """Equivalent to const std::vector<double> & YAxis::getTicks() const."""
        # C++: return impl->_ticks;
        return self._ticks

    def getWidth(self) -> int:
        """Equivalent to int YAxis::getWidth() const."""
        # C++: return impl->_width;
        return self._width
