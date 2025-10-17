import numpy as np
import cv2
from typing import Tuple, List, Union, Any, Optional
import math
import copy

from cv_plot.core import RenderTarget, Drawable
from cv_plot.core.util import paint

class ImageUtils:
    """Helper methods converted from the CvPlot::Imp namespace."""
    
    @staticmethod
    def to_mat1b(mat: np.ndarray) -> np.ndarray:
        """
        Equivalent to cv::Mat1b toMat1b(const cv::Mat& mat).
        Normalizes any single-channel matrix (including floating point) to CV_8UC1 (0-255).
        Handles Inf/NaN edge cases in floating-point data.
        """
        
        # 1. Floating point check and finite mask (mat == mat is true only for finite numbers)
        if mat.dtype in [np.float32, np.float64]:
            floating = True
            mask = np.isfinite(mat)
        else:
            floating = False
            mask = None

        # 2. Find min/max of finite values
        if floating and np.any(mask):
            finite_vals = mat[mask]
            min_val = np.min(finite_vals)
            max_val = np.max(finite_vals)
        elif not floating:
            # Use whole image if not floating (or if all NaN/Inf)
            if mat.size == 0:
                min_val, max_val = 0, 0
            else:
                min_val, max_val = cv2.minMaxLoc(mat)[:2]
        else:
            # All NaN/Inf or empty
            return np.full(mat.shape, 127, dtype=np.uint8)

        # 3. Handle Inf/NaN edge cases (where minVal or maxVal is still Inf)
        if np.isinf(min_val) or np.isinf(max_val) or (max_val - min_val) < 1e-9:
            finite_val = 127
            if np.isfinite(min_val) and (max_val - min_val) < 1e-9: # maxVal = minVal
                pass # remains 127
            elif np.isfinite(min_val):
                finite_val = 0
            elif np.isfinite(max_val):
                finite_val = 255
                
            mat1b = np.full(mat.shape, finite_val, dtype=np.uint8)
            
            # Set specific Inf/Min/Max to 0/255
            mat1b[mat == min_val] = 0
            mat1b[mat == max_val] = 255
            return mat1b
            
        # 4. Standard normalization to [0, 255]
        # C++: const double alpha = 255.0 / (maxVal - minVal);
        # C++: const double beta = -minVal * alpha;
        alpha = 255.0 / (max_val - min_val)
        beta = -min_val * alpha
        
        mat1b = cv2.convertScaleAbs(mat, alpha=alpha, beta=beta)
        return mat1b
        
    @staticmethod
    def to_mat3b(mat: np.ndarray, code: int) -> np.ndarray:
        """Equivalent to cv::Mat3b toMat3b(const cv::Mat& mat, int code). Converts to BGR (CV_8UC3)."""
        if mat.size == 0:
            return np.empty((0, 0, 3), dtype=np.uint8)
        return cv2.cvtColor(mat, code)

    @staticmethod
    def to_bgr(mat: np.ndarray, nan_color: tuple) -> np.ndarray:
        """
        Equivalent to cv::Mat3b toBgr(const cv::Mat& mat, cv::Scalar nanColor).
        Main color conversion function.
        """
        if mat.size == 0:
            return np.empty((0, 0, 3), dtype=np.uint8)

        mat_dtype = mat.dtype
        mat_channels = 1 if mat.ndim < 3 else mat.shape[2]
        
        # 1. CV_8UC3 (already BGR)
        if mat_dtype == np.uint8 and mat_channels == 3:
            return mat
        
        # 2. CV_8UC4 (BGRA)
        elif mat_dtype == np.uint8 and mat_channels == 4:
            return ImageUtils.to_mat3b(mat, cv2.COLOR_BGRA2BGR)
            
        # 3. Floating point (CV_32F, CV_64F)
        elif mat_dtype in [np.float32, np.float64]:
            # Convert to 8UC1 (normalized) then to BGR
            mat1b = ImageUtils.to_mat1b(mat)
            mat3b = ImageUtils.to_mat3b(mat1b, cv2.COLOR_GRAY2BGR)
            
            # Handle NaN values
            # C++: mat3b.setTo(nanColor, mat != mat);
            if nan_color != (0, 0, 0): # Check against default (black, which is Scalar())
                is_nan = np.isnan(mat)
                if np.any(is_nan):
                    mat3b[is_nan] = nan_color
            return mat3b
            
        # 4. Integer (CV_16S, CV_16U, CV_32S) -> Normalize to 8UC1 then BGR
        elif mat_dtype in [np.int16, np.uint16, np.int32]:
            mat1b = ImageUtils.to_mat1b(mat)
            return ImageUtils.to_mat3b(mat1b, cv2.COLOR_GRAY2BGR)
            
        # 5. CV_8UC1 (Grayscale)
        elif mat_dtype == np.uint8 and mat_channels == 1:
            return ImageUtils.to_mat3b(mat, cv2.COLOR_GRAY2BGR)
            
        else:
            raise RuntimeError(f"Image: mat type {mat_dtype} with {mat_channels} channels not supported")


# --- IMAGE CLASS (PIMPL Removed) ---

class Image(Drawable):
    """
    Equivalent to CvPlot::Image.
    Impl fields and logic are merged directly into this class.
    """

    # --- Constructor Overloads ---
    
    def __init__(self, mat: np.ndarray = np.empty(0), 
                 position: Optional[tuple] = None):
        super().__init__()

        self._position = (0,0,0,0)
        self._autoPosition: bool = True
        self._interpolation: int = cv2.INTER_AREA
        self._nanColor = (0, 0, 0) # BGR: Black
    
        self._flippedMat = np.array([])
        self._flippedBgr = np.array([])
        self._flipVert: bool = False
        self._flipHorz: bool = False

        self.setMat(mat)

        if position is not None:
            # C++: Image(const cv::Mat &mat, const cv::Rect2d &position)
            self.setPosition(position)
        else:
            # C++: Image(const cv::Mat &mat = cv::Mat())
            self._autoPosition = True # setMat already calls updateAutoPosition

    # --- Internal Impl Methods ---
    
    def _updateFlipped(self) -> None:
        """
        Equivalent to void Impl::updateFlipped(). 
        Handles flipping the source images (_mat and _matBgr) if necessary.
        """
        if self._mat.size == 0:
            return
            
        # Determine flip code
        if self._flipVert or self._flipHorz:
            if self._flipVert and self._flipHorz:
                code = -1  # flip both axes
            elif self._flipVert:
                code = 0   # flip around X-axis (vertical)
            else: # _flipHorz
                code = 1   # flip around Y-axis (horizontal)
            
            # Clone check (simplified: assume a clone is always needed in Python to separate memory)
            self._flippedMat = cv2.flip(self._mat, code)
            self._flippedBgr = cv2.flip(self._matBgr, code)
            
        else:
            # Reference copy (no flip)
            self._flippedMat = self._mat
            self._flippedBgr = self._matBgr

    def _updateAutoPosition(self) -> None:
        """
        Equivalent to void Impl::updateAutoPosition(). 
        Sets the position based on matrix dimensions if auto-positioning is enabled.
        """
        if self._autoPosition and self._mat.size > 0:
            # C++: _position = cv::Rect2d(0, 0, _mat.cols, _mat.rows);
            h, w = self._mat.shape[:2]
            self._position = (0.0, 0.0, float(w), float(h))

    def render(self, renderTarget: RenderTarget) -> None:
        """Equivalent to void Impl::render(RenderTarget & renderTarget)."""
        if self._mat.size == 0:
            return
            
        innerMat = renderTarget.innerMat()

        # 1. Project the position rectangle to pixel coordinates
        x, y, w, h = self._position
        tl = renderTarget.project((x, y))
        br = renderTarget.project((x + w, y + h))
        matRectDst = (tl[0], tl[1], br[0] - tl[0], br[1] - tl[1])
        
        # 2. Check for required flip and update
        flipVert = tl[1] > br[1] # tl.y > br.y
        flipHorz = tl[0] > br[0] # tl.x > br.x
        
        if flipHorz != self._flipHorz or flipVert != self._flipVert:
            self._flipHorz = flipHorz
            self._flipVert = flipVert
            self._updateFlipped()
            
        # 3. Paint the image onto the innerMat
        # C++: Internal::paint(_flippedBgr, innerMat, matRectDst, _interpolation, _flippedMat);
        paint(self._flippedBgr, innerMat, matRectDst, self._interpolation, self._flippedMat)

    # --- Public Accessors and Mutators ---

    def setMat(self, mat: np.ndarray) -> 'Image':
        """Equivalent to Image& Image::setMat(const cv::Mat & mat)."""
        self._mat = mat
        # C++: impl->_matBgr = Imp::toBgr(impl->_mat, impl->_nanColor);
        self._matBgr = ImageUtils.to_bgr(self._mat, self._nanColor)
        self._updateFlipped()
        self._updateAutoPosition()
        return self

    def getMat(self) -> np.ndarray:
        """Equivalent to cv::Mat Image::getMat() const."""
        return self._mat

    def setPosition(self, position: tuple) -> 'Image':
        """Equivalent to Image& Image::setPosition(const cv::Rect2d & position)."""
        self._position = position
        self._autoPosition = False
        return self

    def getPosition(self) -> tuple:
        """Equivalent to cv::Rect2d Image::getPosition()."""
        return self._position

    def setAutoPosition(self, autoPosition: bool = True) -> 'Image':
        """Equivalent to Image & Image::setAutoPosition(bool autoPosition)."""
        self._autoPosition = autoPosition
        self._updateAutoPosition()
        return self

    def getAutoPosition(self) -> bool:
        """Equivalent to bool Image::getAutoPosition() const."""
        return self._autoPosition

    def setInterpolation(self, interpolation: int) -> 'Image':
        """Equivalent to Image & Image::setInterpolation(int interpolation)."""
        self._interpolation = interpolation
        return self

    def getInterpolation(self) -> int:
        """Equivalent to int Image::getInterpolation() const."""
        return self._interpolation

    def setNanColor(self, nanColor: tuple) -> 'Image':
        """Equivalent to Image& Image::setNanColor(cv::Scalar nanColor)."""
        if nanColor == self._nanColor:
            return self
        self._nanColor = nanColor
        # C++: setMat(impl->_mat); (Reloads BGR mat with new nanColor)
        self.setMat(self._mat) 
        return self

    def getNanColor(self) -> tuple:
        """Equivalent to cv::Scalar Image::getNanColor() const."""
        return self._nanColor
    
    # --- Drawable Interface ---

    def getBoundingRect(self) -> Optional[tuple]:
        """Equivalent to bool Image::getBoundingRect(cv::Rect2d &rect)."""
        if self._mat.size == 0:
            return None
        # C++: rect = impl->_position;
        return self._position
