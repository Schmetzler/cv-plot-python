import collections
import math
import cv2
from typing import Deque, Optional, Tuple, Any, List
from . import RenderTarget

class Drawable:
    """
    Equivalent to CvPlot::Drawable.
    The C++ PIMPL fields (_name) are moved directly into this class.
    """

    def __init__(self, name: str = ""):
        # C++ PIMPL initialization: impl->_name = std::move(name);
        self._font_face: int = cv2.FONT_HERSHEY_SIMPLEX
        self._font_scale: float = 0.4
        self._font_thickness: int = 1
        self._color = (0,0,0)
        self._name = name

        self.alpha = 1.0

    # The C++ move constructor and destructor are implicit in Python.
    # Drawable(Drawable &&a) and ~Drawable() are not needed.

    def render(self, renderTarget: RenderTarget) -> None:
        """
        Equivalent to virtual void render(RenderTarget &renderTarget);
        Base implementation does nothing.
        """
        # C++: virtual void Drawable::render(RenderTarget & renderTarget) {}
        pass

    def getBoundingRect(self) -> Optional[tuple]:
        """
        Equivalent to virtual bool getBoundingRect(cv::Rect2d &rect);
        Base implementation returns false (no bounding box).
        
        Note: In Python, we use a mutable list (rect_out) to pass the result back.
        """
        # C++: bool Drawable::getBoundingRect(cv::Rect2d &rect) { return false; }
        return None

    def setName(self, name: str) -> 'Drawable':
        """
        Equivalent to Drawable& setName(std::string name);
        Sets the name and returns self for method chaining.
        """
        # C++: impl->_name = name; return *this;
        self._name = name
        return self

    def getName(self) -> str:
        """
        Equivalent to const std::string& getName();
        """
        # C++: return impl->_name;
        return self._name

class DrawableContainer:
    """
    Equivalent to CvPlot::DrawableDeque, managing a deque of Drawable objects.
    """
    def __init__(self):
        self._drawables = collections.deque()

    def drawables(self) -> Deque[Drawable]:
        """Returns a reference to the deque of drawables."""
        return self._drawables

    # --- Creation and Finding ---

    def create(self, SomeDrawable: type[Drawable], *args: Any, **kwargs: Any) -> Drawable:
        """Creates an instance of SomeDrawable and appends it to the deque."""
        ptr: Drawable = SomeDrawable(*args, **kwargs)
        self._drawables.append(ptr)
        return ptr

    def find_iterator(self, SomeDrawable: type[Drawable], name: Optional[str] = None) -> Tuple[Optional[Drawable], int]:
        """Internal helper for finding an element by type and optional name."""
        for i, ptr in enumerate(self._drawables):
            is_type_match = isinstance(ptr, SomeDrawable)
            is_name_match = (name is None) or (ptr.getName() == name)
            
            if is_type_match and is_name_match:
                return (ptr, i)
        return (None, -1)

    def find(self, SomeDrawable: type[Drawable], name: Optional[str] = None) -> Optional[Drawable]:
        """Returns the first matching drawable or None."""
        found_drawable, _ = self.find_iterator(SomeDrawable, name)
        return found_drawable

    def find_by_object(self, drawable: Drawable) -> int:
        """Returns the index of the object or -1."""
        try:
            return self._drawables.index(drawable)
        except ValueError:
            return -1

    def findOrCreate(self, SomeDrawable: type[Drawable], name: Optional[str] = None) -> Drawable:
        """Finds a matching drawable or creates a new one."""
        found_drawable = self.find(SomeDrawable, name)
        if found_drawable:
            return found_drawable
        
        new_drawable = self.create(SomeDrawable)
        
        if name is not None:
            new_drawable.setName(name)
            
        return new_drawable

    # --- getBoundingRect Logic ---

    def getBoundingRect(self) -> Optional[tuple]:
        """Calculates the combined bounding box (x, y, w, h) of all drawables. Returns None if empty."""
        hasRect = False
        
        # Combined rect (x, y, w, h)
        combined_rect_x, combined_rect_y, combined_rect_w, combined_rect_h = 0.0, 0.0, 0.0, 0.0
        
        for drawable in self._drawables:
            rect = drawable.getBoundingRect()
            
            if not rect:
                continue
            
            r_x, r_y, r_w, r_h = rect
            
            # Check for non-finite values (NaN/Inf)
            if not (math.isfinite(r_x) and math.isfinite(r_y) and math.isfinite(r_w) and math.isfinite(r_h)):
                continue

            # First valid rectangle found
            if not hasRect:
                combined_rect_x, combined_rect_y = r_x, r_y
                combined_rect_w, combined_rect_h = r_w, r_h
                hasRect = True
                continue

            # Combine subsequent rectangles
            
            # X-axis combination
            min_x = min(combined_rect_x, r_x)
            max_x = max(combined_rect_x + combined_rect_w, r_x + r_w)
            combined_rect_x = min_x
            combined_rect_w = max_x - min_x
            
            # Y-axis combination
            min_y = min(combined_rect_y, r_y)
            max_y = max(combined_rect_y + combined_rect_h, r_y + r_h)
            combined_rect_y = min_y
            combined_rect_h = max_y - min_y

        if hasRect:
            return (combined_rect_x, combined_rect_y, combined_rect_w, combined_rect_h)
        else:
            return None
