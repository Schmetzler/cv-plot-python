import numpy as np
from typing import Tuple, List, Any, Optional, Union
from enum import Enum

import cv2
import math

from cv_plot.core import RenderTarget, Drawable
from cv_plot.core.util import draw_cast

class MarkerType(Enum):
    NONE = 0
    CIRCLE = 1
    POINT = 2

class LineType(Enum):
    NONE = 0
    SOLID = 1

class Orientation(Enum):
    HORIZONTAL = 0
    VERTICAL = 1

class LineBase(Drawable):
    """
    Equivalent to CvPlot::LineBase. Inherits from Drawable.
    The C++ Impl fields are moved directly into this class.
    """

    def __init__(self, lineSpec: str = "-"):
        super().__init__()
        
        # Default Impl values
        self._line_type = LineType.NONE
        self._marker_type = MarkerType.NONE
        self._marker_size = 10
        self._color = (255, 0, 0)  # BGR: Blue
        self._line_width = 1
        
        # C++: setLineSpec(lineSpec);
        self.setLineSpec(lineSpec)

    def render(self, renderTarget: RenderTarget) -> None:
        """Base render implementation."""
        # C++: void LineBase::render(RenderTarget & renderTarget) {}
        pass
        
    # --- Accessors (Getters and Setters) ---

    def setLineSpec(self, lineSpec: str) -> 'LineBase':
        """
        Parses a MATLAB-style line specification string to set line type and color.
        Equivalent to LineBase& LineBase::setLineSpec(const std::string &lineSpec)
        """
        # 1. Set Line Type (e.g., '-' means Solid)
        if '-' in lineSpec:
            self.setLineType(LineType.SOLID)
        else:
            self.setLineType(LineType.NONE)

        # 2. Set Color
        colors_map = {
            'b': (255, 0, 0),    # Blue
            'g': (0, 255, 0),    # Green
            'r': (0, 0, 255),    # Red
            'c': (255, 255, 0),  # Cyan
            'y': (0, 255, 255),  # Yellow
            'm': (255, 0, 255),  # Magenta
            'k': (0, 0, 0),      # Black
            'w': (255, 255, 255) # White
        }
        
        # Iterate over colors to find a match
        for char, color_scalar in colors_map.items():
            if char in lineSpec:
                # If multiple colors are specified, the last one found wins (same as C++ logic)
                self.setColor(color_scalar)
    
        if "o" in lineSpec:
            self._marker_type = MarkerType.CIRCLE
        elif "." in lineSpec:
            self._marker_type = MarkerType.POINT
        else:
            self._marker_type = MarkerType.NONE

        return self

    def setLineType(self, lineType: LineType) -> 'LineBase':
        """Equivalent to LineBase& LineBase::setLineType(LineType lineType)."""
        # C++: impl->_lineType = lineType;
        self._line_type = lineType
        return self

    def setColor(self, color) -> 'LineBase':
        """Equivalent to LineBase& LineBase::setColor(cv::Scalar color)."""
        # C++: impl->_color = color;
        self._color = color
        return self

    def setLineWidth(self, lineWidth: int) -> 'LineBase':
        """Equivalent to LineBase& LineBase::setLineWidth(int lineWidth)."""
        # C++: impl->_lineWidth = lineWidth;
        self._line_width = lineWidth
        return self

    def getLineType(self) -> LineType:
        """Equivalent to LineType LineBase::getLineType()."""
        # C++: return impl->_lineType;
        return self._line_type

    def getColor(self):
        """Equivalent to cv::Scalar LineBase::getColor()."""
        # C++: return impl->_color;
        return self._color

    def getLineWidth(self) -> int:
        """Equivalent to int LineBase::getLineWidth()."""
        # C++: return impl->_lineWidth;
        return self._line_width

class HorizontalVerticalLine(LineBase):
    """
    Equivalent to CvPlot::HorizontalLine, inheriting from LineBase.
    Impl fields (_pos, _boundingRectEnabled) are moved into this class.
    """
    # Note: _parent is not needed as self is accessible directly in Python methods

    def __init__(self, orientation : Orientation, pos: float = np.nan, lineSpec: str = "-"):
        # C++: :LineBaseSub(lineSpec), impl(*this)
        super().__init__(lineSpec)
        self.orientation = orientation
        # C++: Impl defaults
        self._pos = np.nan
        self._boundingRect_enabled = False
        
        # C++: setPos(pos); setLineSpec(lineSpec);
        self.setPos(pos)
        # LineSpec is already set by super().__init__(lineSpec)

    # Destructor (~HorizontalLine) is implicit in Python.
    
    # --- Public Accessors (for chaining) ---

    def setPos(self, pos: float) -> 'HorizontalLine':
        """Equivalent to HorizontalLine & HorizontalLine::setPos(double pos)."""
        # C++: impl->_pos = pos; return *this;
        self._pos = pos
        return self

    def setBoundingRectEnabled(self, enabled: bool) -> 'HorizontalLine':
        """Equivalent to HorizontalLine & HorizontalLine::setBoundingRectEnabled(bool enabled)."""
        # C++: impl->_boundingRectEnabled = enabled; return *this;
        self._boundingRect_enabled = enabled
        return self

    # --- Core Methods ---

    def render(self, renderTarget: RenderTarget) -> None:
        """
        Equivalent to void HorizontalLine::render(RenderTarget & renderTarget)
        Draws the horizontal line on the inner mat.
        """
        # C++: if (!std::isfinite(_pos)) { return; }
        if not math.isfinite(self._pos):
            return

        mat = renderTarget.innerMat()
        if mat.size == 0:
            return

        # C++: Constant definitions for sub-pixel anti-aliasing
        shift = 4
        shift_scale = (1 << shift)
        
        # 1. Project position Y
        # C++: auto p = renderTarget.project(cv::Point2d(0,_pos));
        # We only care about the Y-coordinate.

        if self.orientation == Orientation.HORIZONTAL:
            _, projected_y = renderTarget.project((0.0, self._pos))
            scaled = draw_cast(projected_y * shift_scale)
            pt1 = (0, scaled)
            pt2 = (mat.shape[1] * shift_scale, scaled) # mat.shape[1] is width (cols)
        else:
            projected_x, _ = renderTarget.project((self._pos, 0.0))
            scaled = draw_cast(projected_x * shift_scale)
            pt1 = (scaled, 0)
            pt2 = (scaled, mat.shape[0] * shift_scale) # mat.shape[1] is width (cols)

        cv2.line(
            img=mat,
            pt1=pt1,
            pt2=pt2,
            color=self.getColor(),
            thickness=self.getLineWidth(),
            lineType=cv2.LINE_AA,
            shift=shift
        )

    def getBoundingRect(self) -> Optional[tuple]:
        """
        Equivalent to bool HorizontalLine::getBoundingRect(cv::Rect2d &rect).
        Returns the line's position for auto-limits if enabled.
        """
        # C++: return false; (The main method in HorizontalLine.h returned false)
        # However, the Impl logic is what we need to translate if bounding rect is enabled.

        if not self._boundingRect_enabled:
            return None

        if self.orientation == Orientation.HORIZONTAL:
            return (np.nan, self._pos, np.nan, 0.0)
        else:
            return (self._pos, np.nan, 0.0, np.nan)

class HorizontalLine(HorizontalVerticalLine):
    def __init__(self, pos: float = np.nan, lineSpec: str = "-"):
        super().__init__(Orientation.HORIZONTAL, pos, lineSpec)
        
class VerticalLine(HorizontalVerticalLine):
    def __init__(self, pos: float = np.nan, lineSpec: str = "-"):
        super().__init__(Orientation.VERTICAL, pos, lineSpec)

class Series(LineBase):
    """
    Equivalent to CvPlot::Series.
    Impl fields and logic are merged directly into this class.
    """

    # --- Constructor Overloads ---
    
    def __init__(self, data_or_x: Union[np.ndarray, List[Any]] = None, y: Optional[Union[np.ndarray, List[Any]]] = None, lineSpec: str = "-", fill=False):
        super().__init__(lineSpec)

        self._internalX = []
        self._boundingRect = (0.0, 0.0, 0.0, 0.0)
        self.fill = fill
        # Determine which C++ constructor is being called
        self._x = []
        self._y = []
            
        if data_or_x is None:
            pass
        elif y is None:
            data = data_or_x
            # NOTE: Data validation for points/x/y is highly complex and depends
            # on the exact CvPlot internal helper functions, so we simplify it here.
            
            # Assuming data is Y-data (1-channel vector-like)
            # A proper implementation would check if 'data' is 1-channel or 2-channel
            
            # Simple assumption: if a 1D list/array, treat as Y-data
            if self._is_xy_valid(data):
                self.setY(data)
            elif self._is_points_valid(data):
                self.setPoints(data)
            # More complex checks for 2D data (points) are omitted for brevity
            else:
                 raise ValueError("Invalid data format in Series constructor. Expected Y data (1D list/array).")

        else:
            # Series(cv::InputArray x, cv::InputArray y, const std::string &lineSpec)
            super().__init__(lineSpec)          
            self.setX(data_or_x)
            self.setY(y)
            
    # --- Internal Validation and Update Methods (from Impl) ---

    def _is_xy_valid(self, data: Union[np.ndarray, List[Any]]) -> bool:
        """Equivalent to bool Impl::xyValid(cv::InputArray a)."""
        if isinstance(data, np.ndarray):
            # Empty, or 1 channel, and (1xN or Nx1)
            return data.size == 0 or data.ndim <= 1 or (data.ndim == 2 and (data.shape[0] == 1 or data.shape[1] == 1))
        # For lists, we assume it's valid if it can be converted to a list of floats
        return isinstance(data, (list, tuple))

    def _is_points_valid(self, data: Union[np.ndarray, List[Any]]) -> bool:
        """Equivalent to bool Impl::pointsValid(cv::InputArray a)."""
        if isinstance(data, np.ndarray):
            # Empty
            if data.size == 0:
                return True
            # 2-channel (1xN or Nx1)
            if data.ndim == 2 and data.shape[1] == 2:
                return True
            # 1-channel (2xN)
            if data.ndim == 2 and data.shape[0] == 2:
                return True
        # Simplified: allow list of tuples/lists of length 2
        if isinstance(data, (list, tuple)) and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in data):
            return True
        return False

    def _update(self) -> None:
        """Equivalent to void Impl::update(). Calculates internal X and bounding rect."""        
        if len(self._y) == 0:
            self._internalX = []
        elif len(self._x) == 0:
            # C++: _internalX = Internal::makeX(_y);
            self._internalX = list(range(len(self._y)))
        else:
            if len(self._x) != len(self._y):
                self._internalX = []
                # C++ behavior is to clear and return on size mismatch
            else:
                self._internalX = self._x
        if len(self._internalX) == 0:
            self._boundingRect = (0.0, 0.0, 0.0, 0.0)
            return

        minx = min(self._internalX)
        maxx = max(self._internalX)
        miny = min(self._y)
        maxy = max(self._y)
        
        # C++: _boundingRect = cv::Rect2d(minx, miny, maxx - minx, maxy - miny);
        self._boundingRect = (minx, miny, maxx - minx, maxy - miny)

    def _get_shifted_points(self, points: List[tuple], shiftScale: int) -> List[List[tuple]]:
        """Equivalent to std::vector<std::vector<cv::Point>> Impl::getShiftedPoints(...)."""
        groups: List[List[tuple]] = []
        current_group: Optional[List[tuple]] = None
        
        for p_x, p_y in points:
            # C++: if (std::isfinite(p.x) && std::isfinite(p.y))
            if math.isfinite(p_x) and math.isfinite(p_y):
                if current_group is None:
                    groups.append([])
                    current_group = groups[-1]
                
                # C++: group->emplace_back(Internal::drawCast(p.x * shiftScale), Internal::drawCast(p.y * shiftScale));
                x_shifted = draw_cast(p_x * shiftScale)
                y_shifted = draw_cast(p_y * shiftScale)
                current_group.append((x_shifted, y_shifted))
            else:
                # C++: else { group = nullptr; }
                current_group = None
        return groups
    
    # --- Public Data Setter/Getter Methods ---

    def setX(self, x: Union[np.ndarray, List[Any]]) -> 'Series':
        """Equivalent to Series & Series::setX(cv::InputArray x)."""
        if not self._is_xy_valid(x):
            raise ValueError("Invalid x in Series::setX().")
        # C++: impl->_x = Internal::toVector<double>(x);
        self._x = np.array(x).flatten().tolist()
        self._update()
        return self

    def getX(self) -> List[float]:
        """Equivalent to std::vector<double> Series::getX()."""
        return self._x

    def setY(self, y: Union[np.ndarray, List[Any]]) -> 'Series':
        """Equivalent to Series & Series::setY(cv::InputArray y)."""
        if not self._is_xy_valid(y):
            raise ValueError("Invalid y in Series::setY().")
        # C++: impl->_y = Internal::toVector<double>(y);
        self._y = np.array(y).flatten().tolist()
        self._update()
        return self

    def getY(self) -> List[float]:
        """Equivalent to std::vector<double> Series::getY()."""
        return self._y

    def setPoints(self, points: Union[np.ndarray, List[Any]]) -> 'Series':
        """Equivalent to Series & Series::setPoints(cv::InputArray points)."""
        if not self._is_points_valid(points):
            raise ValueError("Invalid points in Series::setPoints().")
        
        p = np.asarray(points)
        if p.size == 0:
            self._x = []
            self._y = []
        # NOTE: Handling of 2-channel vs 1-channel(2xN) array is complex in C++ and simplified here for Python
        elif p.ndim == 2 and p.shape[1] == 2:
            # Standard N x 2 array (like list of Point2d or cv::Mat 2-channel)
            self._x = p[:, 0].flatten().tolist()
            self._y = p[:, 1].flatten().tolist()
        elif p.ndim == 2 and p.shape[0] == 2:
            # 2 x N array (like cv::Mat 1-channel, 2xN)
            self._x = p[0, :].flatten().tolist()
            self._y = p[1, :].flatten().tolist()
        else:
             # Should be caught by validation but as a fallback
             raise ValueError("Internal error: setPoints data structure not recognized.")
             
        self._update()
        return self

    def getPoints(self) -> List[tuple]:
        """Equivalent to std::vector<cv::Point2d> Series::getPoints()."""
        # C++: return impl->getPoints();
        points: List[tuple] = []
        if len(self._internalX) == len(self._y):
            for i in range(len(self._internalX)):
                points.append((self._internalX[i], self._y[i]))
        return points
    
    # --- Marker Specific Methods ---

    def setMarkerType(self, markerType: int) -> 'Series':
        """Equivalent to Series & Series::setMarkerType(MarkerType markerType)."""
        # C++: impl->_markerType = markerType;
        self._marker_type = markerType
        return self

    def getMarkerType(self) -> int:
        """Equivalent to MarkerType Series::getMarkerType()."""
        # C++: return impl->_markerType;
        return self._marker_type

    def setMarkerSize(self, markerSize: int) -> 'Series':
        """Equivalent to Series & Series::setMarkerSize(int markerSize)."""
        # C++: impl->_markerSize = markerSize;
        self._marker_size = markerSize
        return self

    def getMarkerSize(self) -> int:
        """Equivalent to int Series::getMarkerSize()."""
        # C++: return impl->_markerSize;
        return self._marker_size

    # --- Drawable Interface Methods ---

    def getBoundingRect(self) -> Optional[tuple]:
        """Equivalent to bool Series::getBoundingRect(cv::Rect2d &rect)."""
        if not self._internalX:
            return None
        # C++: rect = impl->_boundingRect;
        return self._boundingRect

    def render(self, renderTarget: RenderTarget) -> None:
        """
        Internal render method (from Impl::render).
        Draws the line and/or markers using OpenCV drawing primitives.
        """
        if not self._internalX:
            return
        
        color = self.getColor()
        lineWidth = self.getLineWidth()
        markerType = self.getMarkerType()
        
        if markerType == MarkerType.NONE and lineWidth == 0:
            return
            
        mat = renderTarget.innerMat()
        
        # Fixed point arithmetic for sub-pixel accuracy in OpenCV
        shift = 3
        shiftScale = (1 << shift)
        
        # 1. Project data points to pixel coordinates
        points: List[tuple] = []
        if len(self._internalX) == len(self._y):
            for i in range(len(self._internalX)):
                # C++: points.push_back(renderTarget.project(cv::Point2d(_internalX[i], _y[i])));
                pt = renderTarget.project((self._internalX[i], self._y[i]))
                if np.isfinite(pt).all():
                    points.append(pt)
        
        shiftedPoints = None
        
        # 2. Draw Line (Polyline)
        # C++: if (_parent.getLineType() == LineType::Solid)
        if self.getLineType() == LineType.SOLID or self.fill:
            # C++: if (shiftedPoints.empty()) { shiftedPoints = getShiftedPoints(points, shiftScale); }
            if shiftedPoints is None:
                shiftedPoints = self._get_shifted_points(points, shiftScale)
            
            # Convert List[List[Point]] to the format expected by cv2.polylines (List[np.ndarray])
            polylines_data = [np.array(group, dtype=np.int32) for group in shiftedPoints]
            
            # C++: cv::polylines(mat, shiftedPoints, false, color, lineWidth, cv::LINE_AA, shift);
            if self.getLineType() == LineType.SOLID:
                cv2.polylines(mat, polylines_data, self.fill, color, lineWidth, cv2.LINE_AA, shift)
            if self.fill:
                cv2.fillPoly(mat, polylines_data, color, shift=shift)
                
        
        # 3. Draw Markers (Circle)
        # C++: if (_markerType == MarkerType::Circle)
        if markerType == MarkerType.CIRCLE:
            if shiftedPoints is None:
                shiftedPoints = self._get_shifted_points(points, shiftScale)
            
            # C++: const int radius = (_markerSize * shiftScale) / 2;
            radius = (self._marker_size * shiftScale) // 2
            
            for group in shiftedPoints:
                for point in group:
                    # C++: cv::circle(mat, point, radius, color, lineWidth, cv::LINE_AA, shift);
                    cv2.circle(mat, point, radius, color, lineWidth, cv2.LINE_AA, shift)

        # 4. Draw Markers (Point)
        # C++: if (_markerType == MarkerType::Point)
        if markerType == MarkerType.POINT:
            # C++: cv::Vec3b colorv3((unsigned char)color.val[0], (unsigned char)color.val[1], (unsigned char)color.val[2]);
            colorv3 = np.array(color, dtype=np.uint8) # BGR
            
            for p_x, p_y in points:
                # Cast to integer pixel coordinates
                p = (int(p_x), int(p_y))
                
                # C++: if (p.x >= 0 && p.x < mat.cols && p.y >= 0 && p.y < mat.rows)
                if 0 <= p[0] < mat.shape[1] and 0 <= p[1] < mat.shape[0]:
                    # C++: mat(p.y, p.x) = colorv3; (Single pixel assignment)
                    mat[p[1], p[0]] = colorv3 # Access using (row, col) = (y, x)
