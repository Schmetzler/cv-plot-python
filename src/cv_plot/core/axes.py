import numpy as np
import cv2
import math
from typing import Tuple, Optional, List, Any, Union

from .transformation import *
from .projection import RawProjection, Projection, RenderTarget
from .drawable import Drawable, DrawableContainer
from .util import normalize
from cv_plot.drawables import XAxis,XLabel,YAxis,YLabel,Title

class Axes(DrawableContainer):
    """
    Unified class structure equivalent to CvPlot::Axes and CvPlot::Axes::Impl.
    The PIMPL pattern is replaced by direct class members and methods.
    """
    def __init__(self):
        super().__init__()
        # --- Impl Fields (State) ---
        self._leftMargin: int = 80
        self._rightMargin: int = 30
        self._topMargin: int = 40
        self._bottomMargin: int = 45
        
        self._xLim: Tuple[float, float] = (0.0, 1.0)
        self._yLim: Tuple[float, float] = (0.0, 1.0)
        
        self._xLimAuto: bool = True
        self._yLimAuto: bool = True
        self._xTight: bool = False
        self._yTight: bool = False
        self._tightBox: bool = False
        self._yReverse: bool = False
        self._fixedAspectRatio: bool = False
        
        self._aspectRatio: float = 1.0
        
        self._transformation: Optional[Transformation] = None
        self._xLog: bool = False
        self._yLog: bool = False

    # --- INTERNAL UTILITY METHODS (Former Axes_Impl methods) ---
    def _calcBoundingRect(self) -> tuple:
        rect = self.getBoundingRect()
        if not rect:
            rect = (0,0,1,1)
        rect = normalize(rect)
        if self._transformation:
            rect = self._transformation.transform_bounding_rect(rect)
            rect = normalize(rect)
        return rect

    def _calcXLim(self, boundingRect: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """Calculates X limits based on auto-state and bounding rect."""
        if self._xLimAuto:
            x, y, w, h = boundingRect
            if not (math.isfinite(x) and math.isfinite(w)): return (0.0, 1.0)
            if w == 0.0: return (x - 1.0, x + 1.0)
            return (x, x + w)
        return self._xLim

    def _calcYLim(self, boundingRect: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """Calculates Y limits based on auto-state and bounding rect."""
        if self._yLimAuto:
            x, y, w, h = boundingRect
            if not (math.isfinite(y) and math.isfinite(h)): return (0.0, 1.0)
            if h == 0.0: return (y - 1.0, y + 1.0)
            return (y, y + h)
        return self._yLim


            
    def _getViewport(self, boundingRect: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        """Calculates the final viewport limits, applying auto-limits and padding."""
        xLim = self._calcXLim(boundingRect)
        yLim = self._calcYLim(boundingRect)
        
        viewport_x, viewport_y = xLim[0], yLim[0]
        viewport_w, viewport_h = xLim[1] - xLim[0], yLim[1] - yLim[0]
        
        ex = 0.1 # Expansion factor
        
        if not self._xTight:
            viewport_x -= viewport_w * ex / 2
            viewport_w *= 1 + ex
            
        if not self._yTight:
            viewport_y -= viewport_h * ex / 2
            viewport_h *= 1 + ex
            
        return (viewport_x, viewport_y, viewport_w, viewport_h)

    def _normalizeLims(self) -> None:
        """Ensures xLim and yLim are ordered (min, max)."""
        if self._xLim[0] > self._xLim[1]:
            self._xLim = (self._xLim[1], self._xLim[0])
            
        if self._yLim[0] > self._yLim[1]:
            self._yLim = (self._yLim[1], self._yLim[0])
            
    def _beginZoomOrPan(self) -> None:
        """Freezes auto limits before zooming/panning starts."""
        boundingRect = self._calcBoundingRect()
        if self._xLimAuto:
            self._xLim = self._calcXLim(boundingRect)
            self._xLimAuto = False
        if self._yLimAuto:
            self._yLim = self._calcYLim(boundingRect)
            self._yLimAuto = False

    def _setLimsFromPoints(self, rawProjection: RawProjection, topLeftPix: Tuple[float, float], bottomRightPix: Tuple[float, float]) -> None:
        """Sets limits by unprojecting screen coordinates."""
        topLeft = rawProjection.unproject(topLeftPix, False)
        bottomRight = rawProjection.unproject(bottomRightPix, False)
        
        self._xLim = (topLeft[0], bottomRight[0])
        self._yLim = (topLeft[1], bottomRight[1])
        
        self._normalizeLims()

    def _setLogTransformation(self) -> None:
        """Sets the correct log/lin transformation based on _xLog and _yLog flags."""
        if not self._xLog and not self._yLog:
            self.setTransformation(None)
        elif not self._xLog and self._yLog:
            self.setTransformation(LinLogTransformation())
        elif self._xLog and not self._yLog:
            self.setTransformation(LogLinTransformation())
        elif self._xLog and self._yLog:
            self.setTransformation(LogLogTransformation())

    # --- CORE LOGIC METHODS (Former Axes_Impl methods) ---

    def _getRawProjection(self, destinationSize: Tuple[int, int]) -> RawProjection:
        """Calculates the RawProjection model based on viewport, inner rect, and fixed ratio logic."""
        dst_w, dst_h = destinationSize
        
        if dst_w <= self._leftMargin + self._rightMargin or \
           dst_h <= self._topMargin + self._bottomMargin:
            return RawProjection()
            
        innerRect_x = self._leftMargin
        innerRect_y = self._topMargin
        innerRect_w = dst_w - self._leftMargin - self._rightMargin
        innerRect_h = dst_h - self._topMargin - self._bottomMargin
        innerRect = (innerRect_x, innerRect_y, innerRect_w, innerRect_h)
        
        boundingRect = self._calcBoundingRect()
        viewport = self._getViewport(boundingRect)
        
        if self._fixedAspectRatio:
            if self._tightBox and self._xLimAuto and self._yLimAuto:
                ratio = (viewport[3] / viewport[2]) * self._aspectRatio
                innerRect = fix_ratio(innerRect, ratio, False)
            else:
                ratio = (innerRect[3] / innerRect[2]) / self._aspectRatio
                viewport = fix_ratio(viewport, ratio, True)

        rawProjection = RawProjection()
        
        rawProjection.kx = innerRect[2] / viewport[2]
        rawProjection.ky = (innerRect[3] / viewport[3]) * (1.0 if self._yReverse else -1.0)

        rawProjection.offset = (
            -viewport[0] * rawProjection.kx,
            -(viewport[1] + (0.0 if self._yReverse else viewport[3])) * rawProjection.ky
        )
        
        rawProjection.transformation = self._transformation
        rawProjection.inner_rect = innerRect
        
        return rawProjection

    def zoom(self, size: Tuple[int, int], outerPos: Tuple[int, int], scalex: float, scaley: float) -> None:
        """Zooms the view based on the current limits and a center point."""
        self._beginZoomOrPan()
        
        if self._fixedAspectRatio:
            scalex = scaley = math.sqrt(scalex * scaley)
            
        rawProjection = self._getRawProjection(size)
        
        if rawProjection.area == 0: return
            
        topLeftPix0 = rawProjection.project((self._xLim[0], self._yLim[0]), False)
        bottomRightPix0 = rawProjection.project((self._xLim[1], self._yLim[1]), False)
        
        diag0_x = bottomRightPix0[0] - topLeftPix0[0]
        diag0_y = bottomRightPix0[1] - topLeftPix0[1]
        
        innerRectTopLeft = (rawProjection.inner_rect[0], rawProjection.inner_rect[1])
        
        pos0_x = outerPos[0] - innerRectTopLeft[0] - topLeftPix0[0]
        pos0_y = outerPos[1] - innerRectTopLeft[1] - topLeftPix0[1]
        
        topLeftPix_x = topLeftPix0[0] + pos0_x * (1 - scalex)
        topLeftPix_y = topLeftPix0[1] + pos0_y * (1 - scaley)
        topLeftPix = (topLeftPix_x, topLeftPix_y)
        
        bottomRightPix = (topLeftPix_x + diag0_x * scalex, topLeftPix_y + diag0_y * scaley)
        
        self._setLimsFromPoints(rawProjection, topLeftPix, bottomRightPix)

    def pan(self, size: Tuple[int, int], delta: Tuple[int, int]) -> None:
        """Pans the view based on a pixel delta."""
        self._beginZoomOrPan()
        
        rawProjection = self._getRawProjection(size)
        
        if rawProjection.area == 0: return
            
        topLeftPix0 = rawProjection.project((self._xLim[0], self._yLim[0]), False)
        bottomRightPix0 = rawProjection.project((self._xLim[1], self._yLim[1]), False)
        
        topLeftPix = (topLeftPix0[0] - delta[0], topLeftPix0[1] - delta[1])
        bottomRightPix = (bottomRightPix0[0] - delta[0], bottomRightPix0[1] - delta[1])
        
        self._setLimsFromPoints(rawProjection, topLeftPix, bottomRightPix)

    def setTransformation(self, transformation: Optional[Transformation]) -> 'Axes':
        """Sets a new coordinate transformation, preserving current limits in data space."""
        xlim = self.getXLim()
        ylim = self.getYLim()
        
        self._transformation = transformation
        
        if not self._xLimAuto: self.setXLim(xlim)
        if not self._yLimAuto: self.setYLim(ylim)
        return self

    # --- PUBLIC API METHODS (Original Axes methods) ---

    def getProjection(self, size: Tuple[int, int]) -> Projection:
        """Returns the calculated Projection object."""
        return Projection(self._getRawProjection(size))

    def render(self, cols: Union[tuple,int, np.ndarray], rows: Optional[int] = None) -> Union[np.ndarray, None]:
        """Handles all three C++ render overloads."""
        mat = None
        if isinstance(cols, np.ndarray):
            mat = cols
            rows, cols = mat.shape[:2]
        elif isinstance(cols, (tuple, list)):
            rows = cols[1]
            cols = cols[0]
        elif rows is None:
            rows = cols

        destinationSize = (cols, rows)
        if cols < 0 or rows < 0: destinationSize = (0, 0)
            
        rawProjection = self._getRawProjection(destinationSize)
        
        if mat is None:
            mat = np.full(shape=(rows, cols, 3), fill_value=255, dtype=np.uint8)
        else:
            mat.fill(255)
        
        renderTarget = RenderTarget(rawProjection, mat)
        
        for drawable in self.drawables():
            if drawable.alpha < 1.0:
                rt = RenderTarget(rawProjection, renderTarget.outerMat().copy())
                drawable.render(rt)
                renderTarget.outerMat()[:,:] = cv2.addWeighted(renderTarget.outerMat(), 1-drawable.alpha, rt.outerMat(), drawable.alpha, 0)
            else:
                drawable.render(renderTarget)

        return mat

    def setMargins(self, left: int, right: int, top: int, bottom: int) -> 'Axes':
        self._leftMargin, self._rightMargin, self._topMargin, self._bottomMargin = left, right, top, bottom
        return self
        
    def xLabel(self, label: str) -> 'Axes':
        self.findOrCreate(XLabel).setLabel(label)
        return self

    def yLabel(self, label: str) -> 'Axes':
        self.findOrCreate(YLabel).setLabel(label).setYAxis(self._find(YAxis))
        return self

    def title(self, title: str) -> 'Axes':
        self.findOrCreate(Title).setTitle(title)
        return self

    def enableHorizontalGrid(self, enable: bool = True) -> 'Axes':
        grid = self.findOrCreate(HorizontalGrid).setEnabled(enable)
        if enable: grid.setYAxis(self._find(YAxis))
        return self

    def enableVerticalGrid(self, enable: bool = True) -> 'Axes':
        grid = self.findOrCreate(VerticalGrid).setEnabled(enable)
        if enable: grid.setXAxis(self._find(XAxis))
        return self

    # --- Limit Accessors ---

    def setXLim(self, xlim: Union[float,Tuple[float, float]], upper=None ) -> 'Axes':
        if isinstance(xlim, float):
            if upper is None:
                upper = xlim
                xlim = 0
            xlim = (xlim, upper)
        if self._transformation: xlim = self._transformation.transformXLim(xlim)
        self._xLim = xlim
        self._xLimAuto = False
        return self

    def getXLim(self) -> Tuple[float, float]:
        if self._transformation: return self._transformation.untransformXLim(self._xLim)
        return self._xLim

    def setXLimAuto(self, xLimAuto: bool = True) -> 'Axes':
        self._xLimAuto = xLimAuto
        return self

    def getXLimAuto(self) -> bool: return self._xLimAuto

    def setYLim(self, ylim: Union[float, Tuple[float, float]], upper=None) -> 'Axes':
        if isinstance(ylim, float):
            if upper is None:
                upper = ylim
                ylim = 0
            ylim = (ylim, upper)
        if self._transformation: ylim = self._transformation.transformYLim(ylim)
        self._yLim = ylim
        self._yLimAuto = False
        return self

    def getYLim(self) -> Tuple[float, float]:
        if self._transformation: return self._transformation.untransformYLim(self._yLim)
        return self._yLim

    def setYLimAuto(self, yLimAuto: bool = True) -> 'Axes':
        self._yLimAuto = yLimAuto
        return self

    def getYLimAuto(self) -> bool: return self._yLimAuto

    # --- Other Property Accessors ---
    def setYReverse(self, reverse: bool = True) -> 'Axes': self._yReverse = reverse; return self
    def getYReverse(self) -> bool: return self._yReverse
    def setFixedAspectRatio(self, fixed: bool = True) -> 'Axes': self._fixedAspectRatio = fixed; return self
    def getFixedAspectRatio(self) -> bool: return self._fixedAspectRatio
    def setAspectRatio(self, aspectRatio: float) -> 'Axes': self._aspectRatio = aspectRatio; return self
    def getAspectRatio(self) -> float: return self._aspectRatio
    def setXTight(self, tight: bool = True) -> 'Axes': self._xTight = tight; return self
    def getXTight(self) -> bool: return self._xTight
    def setYTight(self, tight: bool = True) -> 'Axes': self._yTight = tight; return self
    def getYTight(self) -> bool: return self._yTight
    def setTightBox(self, tight: bool = True) -> 'Axes': self._tightBox = tight; return self
    def getTightBox(self) -> bool: return self._tightBox
    def setXLog(self, log: bool = True) -> 'Axes': self._xLog = log; self._setLogTransformation(); return self
    def getXLog(self) -> bool: return self._xLog
    def setYLog(self, log: bool = True) -> 'Axes': self._yLog = log; self._setLogTransformation(); return self
    def getYLog(self) -> bool: return self._yLog

