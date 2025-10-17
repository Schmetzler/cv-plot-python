import numpy as np
import cv2
import math
from typing import List, Tuple, Any, Union, Optional

# Define the namespace structure, though Python doesn't enforce C++ namespaces
# We'll use a module-level structure or a utility class if needed, but simple
# functions are fine for internal utilities.

# --- Helper Functions and Equivalents for Internal::Imp ---

def _pixel_value(data: np.ndarray, row: int, col: int) -> float:
    """
    Equivalent to Imp::pixelValue. Gets the value of a pixel and converts it to float.
    In Python/NumPy, this is straightforward indexing and conversion.
    """
    if data.ndim > 2 and data.shape[2] > 1:
        # Assuming single channel is expected for pixel value retrieval for text
        # If it's multi-channel, the C++ version converts a 1x1 patch to a 1-channel double Mat, which is complex.
        # Simplification: return the first channel's value.
        value = data[row, col, 0]
    else:
        value = data[row, col]
        
    return float(value)

def _to_string(value: float, digits: int) -> str:
    """
    Equivalent to Imp::toString (non-scientific formatting).
    Uses standard Python string formatting.
    """
    # std::fixed ensures non-scientific notation
    # stream.precision(digits) is equivalent to f-string/format precision
    return f"{value:.{digits}f}"

def _pixel_text(data: np.ndarray, row: int, col: int, 
                font_face: int, font_scale: float, font_thickness: int, 
                max_size: Tuple[int, int], 
                # text, size are output parameters in C++
                ) -> Union[Tuple[str, Tuple[int, int]], bool]:
    """
    Equivalent to Imp::pixelText. Tries to format a pixel value as a string 
    that fits within maxSize.
    Returns (text, (width, height)) on success, False on failure.
    """
    if data.ndim > 2 and data.shape[2] > 1:
        # C++: if (data.channels() > 1) { return false; } //TODO
        # This implementation requires further thought for multi-channel data.
        return False 
    
    max_w, max_h = max_size
    
    def _check_text(candidate: str) -> Union[Tuple[str, Tuple[int, int]], bool]:
        """Helper to check if the candidate text fits the max_size."""
        (w, h), baseline = cv2.getTextSize(
            candidate, font_face, font_scale, font_thickness
        )
        if w <= max_w and h <= max_h:
            # size in Python is (w, h), baseline is separate
            return candidate, (w, h)
        return False

    dtype = data.dtype
    
    # Check integer types
    if dtype in (np.int16, np.uint16, np.int32, np.uint8):
        pix = int(_pixel_value(data, row, col))
        return _check_text(str(pix))

    # Check floating point types
    elif dtype in (np.float32, np.float64):
        pix = _pixel_value(data, row, col)
        
        if np.isnan(pix):
            return _check_text("NaN")
            
        # C++ uses the custom digits function
        digits_ = digits(pix) 
        
        # Iterate backwards to find the shortest string that still represents the value
        for i in range(digits_, -1, -1):
            text_candidate = _to_string(pix, i)
            result = _check_text(text_candidate)
            if result:
                return result
        
        return False # Failed to fit even with 0 digits

    # Default/Unsupported types
    return False

# --- Core Utility Functions (Equivalent to CvPlot::Internal::* in .cpp) ---

def digits(value: float) -> int:
    """
    Equivalent to digits(double value). Calculates the number of significant 
    digits after the decimal point required to represent the number accurately.
    """
    epsilon = 0.0001
    value = float(value)
    if value == 0: return 0
    
    # C++: std::abs(value - rounded) / step < epsilon
    for i in range(11): # Loop up to 10 significant digits
        step = math.pow(10, -i)
        rounded = round(value * math.pow(10, i)) * step
        
        # The C++ error calculation is complex; simplified logic: 
        # check if value is close to the rounded version scaled by the step.
        error = abs(value - rounded) / step
        
        if error < epsilon:
            return i
            
    return 10 # Default fallback

def format_value(value: float, exponentional: bool = False) -> str:
    """
    Equivalent to format(double value, bool exponentional).
    Formats a number as a string, optionally using scientific notation (simplified).
    """
    if exponentional:
        # C++: "1e" + std::to_string(exp) - highly simplified scientific notation
        if value == 0: return "0"
        exp = math.floor(math.log10(abs(value)))
        return f"1e{exp}"
    else:
        # Use custom digits calculation for precision
        return _to_string(value, digits(value))

def normalize(rect: tuple) -> tuple:
    """
    Equivalent to template<typename T> void normalize(cv::Rect_<T> &rect).
    Normalizes a rectangle so width and height are non-negative.
    NOTE: In Python cv2, Rect objects are often just tuples/lists, but we'll 
    use an intermediate list/numpy array for manipulation.
    """
    # Input Rect is (x, y, w, h). We modify in place conceptually.
    x, y, w, h = rect[:4]
    
    if w < 0:
        x += w
        w *= -1
    if h < 0:
        y += h
        h *= -1
        
    return (x, y, w, h) # Return as the same structure type

def fix_ratio(rect: tuple, ratio: float, extend: bool) -> tuple:
    """
    Equivalent to template<typename T> cv::Rect_<T> fixRatio(...)
    Adjusts a rectangle's dimensions to match a given aspect ratio, 
    either by extending or shrinking it.
    """
    # Normalize the input rect first
    x, y, w, h = normalize(rect)[:4]
    
    cur_ratio = h / w if w != 0 else float('inf')
    fixed_x, fixed_y, fixed_w, fixed_h = x, y, w, h
    
    # (curRatio < ratio) == extend
    # If extend is True, we extend when curRatio < ratio (too wide, increase height)
    # If extend is False, we extend when curRatio >= ratio (too tall, increase width)
    
    # Check if we should adjust height (i.e., too wide or need to shrink height)
    if (cur_ratio < ratio and extend) or (cur_ratio >= ratio and not extend):
        new_h = w * ratio
        diff_h = new_h - h
        fixed_h = new_h
        fixed_y -= diff_h / 2
    else: # Adjust width (i.e., too tall or need to shrink width)
        new_w = h / ratio
        diff_w = new_w - w
        fixed_w = new_w
        fixed_x -= diff_w / 2
        
    # Cast back to int if original rect was cv::Rect (int-based)
    if isinstance(rect, tuple) and all(isinstance(i, int) for i in rect[:4]):
        fixed_x, fixed_y, fixed_w, fixed_h = [int(round(val)) for val in [fixed_x, fixed_y, fixed_w, fixed_h]]
    
    return (fixed_x, fixed_y, fixed_w, fixed_h)

def calc_ticks_linear(x0: float, x1: float, estimated_tick_count: int) -> List[float]:
    """
    Equivalent to calcTicksLinear. Calculates nice, evenly spaced linear ticks.
    """
    if x1 < x0:
        x0, x1 = x1, x0
        
    estimated_tick_count = max(1, estimated_tick_count)
    
    # Calculate a raw step size
    step0 = abs(x1 - x0) / estimated_tick_count
    
    # Find the nearest power of 10
    if step0 == 0:
        return [x0]
    
    step0_log = math.ceil(math.log10(step0))
    step = math.pow(10, step0_log)
    
    # Adjust step to 1, 2, or 5 times the power of 10
    if step / 5 > step0:
        step /= 5
    elif step / 2 > step0:
        step /= 2
        
    # Calculate the first tick
    first = math.ceil(x0 / step) * step # Always step up from x0
    
    # Generate ticks
    ticks = []
    
    # Use numpy.arange for cleaner generation, and round to handle float precision issues
    # We add a small epsilon to the end to ensure the last tick is included
    end = x1 + step * 1e-9 
    
    # Generate ticks
    current_tick = first
    while current_tick <= end:
        # Use round to mitigate float accumulation error
        rounded_tick = round(current_tick / step) * step
        if rounded_tick >= x0 and rounded_tick <= x1:
            ticks.append(rounded_tick)
        current_tick += step
        
    return ticks

def calc_ticks_log(x0: float, x1: float, estimated_tick_count: int) -> List[float]:
    """
    Equivalent to calcTicksLog. Calculates major ticks for a logarithmic scale.
    """
    estimated_tick_count = max(1, estimated_tick_count)
    
    if x1 < x0:
        x0, x1 = x1, x0
        
    if x0 <= 0:
        # Log scale requires positive values.
        # Handling for x0 <= 0 is complex; often clip to a small positive value.
        # We'll use the C++ behavior which seems to rely on log10 being defined.
        # Let's assume input has been pre-validated for a log scale (x > 0).
        if x0 <= 0: x0 = 1e-5
        
    log0 = math.ceil(math.log10(x0))
    log1 = math.floor(math.log10(x1))
    
    # Calculate step in powers of 10
    step = max(1, 1 + math.floor((log1 - log0) / estimated_tick_count))
    
    ticks = []
    # Loop through the powers of 10
    current_log = log0
    while current_log <= log1:
        tick = math.pow(10, current_log)
        # Only include ticks within the range (in case log0 or log1 were outside the range due to ceil/floor)
        if tick >= x0 and tick <= x1:
             ticks.append(tick)
        current_log += step
        
    return ticks

def draw_cast(value: float) -> int:
    """
    Equivalent to drawCast. Casts a double to int, clamping to safe limits 
    for OpenCV's drawing functions (which often use 16-bit signed ints).
    """
    max_val = 32000
    min_val = -32000
    return int(np.clip(value, min_val, max_val))

def bounding_rect(src_rect2d: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
    """
    Equivalent to boundingRect. Calculates the smallest integer cv::Rect 
    that fully contains a cv::Rect2d.
    """
    # src_rect2d is (x, y, w, h)
    x, y, w, h = normalize(src_rect2d)
    
    rect_x = math.floor(x)
    rect_y = math.floor(y)
    # The width/height calculation is critical: it goes from the floor of the start
    # to the ceiling of the end, which ensures coverage.
    rect_w = math.ceil(x + w) - rect_x
    rect_h = math.ceil(y + h) - rect_y
    
    return (int(rect_x), int(rect_y), int(rect_w), int(rect_h))



def _paint_simple(src: np.ndarray, dst: np.ndarray, pos: Tuple[int, int]) -> Optional[np.ndarray]:
    """
    Equivalent to void paint(const cv::Mat &src, cv::Mat &dst, const cv::Point &pos).
    Copies a source image into a destination image at a given position, 
    clipping to bounds.
    """
    if src.size == 0:
        return
        
    # Source Rect: (0, 0, src.cols, src.rows)
    src_w, src_h = src.shape[1], src.shape[0]
    dst_w, dst_h = dst.shape[1], dst.shape[0]
    pos_x, pos_y = pos
    
    # Destination Rect relative to Pos: (pos.x, pos.y, src.w, src.h)
    r_dst_x_rel, r_dst_y_rel, r_dst_w_rel, r_dst_h_rel = pos_x, pos_y, src_w, src_h

    # Clip Destination Rect to Dst bounds
    r_dst_x = max(0, r_dst_x_rel)
    r_dst_y = max(0, r_dst_y_rel)
    r_dst_x_end = min(dst_w, r_dst_x_rel + r_dst_w_rel)
    r_dst_y_end = min(dst_h, r_dst_y_rel + r_dst_h_rel)
    
    r_dst_w = r_dst_x_end - r_dst_x
    r_dst_h = r_dst_y_end - r_dst_y

    if r_dst_w <= 0 or r_dst_h <= 0:
        return

    # Source Rect relative to R_Dst
    r_src_x = r_dst_x - r_dst_x_rel
    r_src_y = r_dst_y - r_dst_y_rel
    r_src_w = r_dst_w
    r_src_h = r_dst_h
    
    # Perform copy: dst[y:y+h, x:x+w] = src[y_src:y_src+h, x_src:x_src+w]
    dst[r_dst_y:r_dst_y_end, r_dst_x:r_dst_x_end] = \
        src[r_src_y:r_src_y + r_src_h, r_src_x:r_src_x + r_src_w].copy()
    return dst

def paint(src: np.ndarray, dst: np.ndarray, pos: tuple, 
                 interpolation: int = cv2.INTER_LINEAR, data: np.ndarray = None) -> Optional[np.ndarray]:
    """
    Equivalent to void paint(const cv::Mat3b &src, cv::Mat3b &dst, const cv::Rect2d &pos, ...)
    Scales a source image (src) and draws it onto a destination image (dst) at a 
    floating-point position (pos), with optional grid/text overlay based on 'data'.
    """
    if src.size == 0:
        return
    if len(pos) == 2:
        return _paint_simple(src, dst, pos)

    if data is not None and (data.shape[0] != src.shape[0] or data.shape[1] != src.shape[1]):
        raise ValueError("Bad data size: data and src must have the same height/width.")

    # Rects
    r_src = (0, 0, src.shape[1], src.shape[0])
    r_dst = (0, 0, dst.shape[1], dst.shape[0])
    
    # pos is (x, y, w, h) - the target rect in destination coordinates
    pos_x, pos_y, pos_w, pos_h = pos
    
    # Scaling factors (C++ code uses kx, ky for pixels/src_unit)
    kx = pos_w / r_src[2] if r_src[2] != 0 else 1.0 # pixel width
    ky = pos_h / r_src[3] if r_src[3] != 0 else 1.0 # pixel height

    # Inverse factors (used for mapping dst coords back to src)
    kx_i = 1.0 / kx if kx != 0 else 0.0
    ky_i = 1.0 / ky if ky != 0 else 0.0
    
    # Inverse translation components
    dx_i = -kx_i * pos_x
    dy_i = -ky_i * pos_y
    
    # 1. VISIBILITY RECTS
    
    # Destination visible rect (intersection of pos and dst bounds)
    r_dst_vis = (
        max(r_dst[0], pos_x),
        max(r_dst[1], pos_y),
        min(r_dst[0] + r_dst[2], pos_x + pos_w) - max(r_dst[0], pos_x),
        min(r_dst[1] + r_dst[3], pos_y + pos_h) - max(r_dst[1], pos_y)
    )
    if r_dst_vis[2] <= 0 or r_dst_vis[3] <= 0: return

    # Source visible rect (back-projected from r_dst_vis)
    r_src_vis_x = r_dst_vis[0] * kx_i + dx_i
    r_src_vis_y = r_dst_vis[1] * ky_i + dy_i
    r_src_vis_w = r_dst_vis[2] * kx_i
    r_src_vis_h = r_dst_vis[3] * ky_i
    r_src_vis = (r_src_vis_x, r_src_vis_y, r_src_vis_w, r_src_vis_h)

    # Source visible rect (expanded to nearest integer boundaries and clipped to r_src)
    r_src_vis_ex_xywh = bounding_rect(r_src_vis)
    r_src_vis_ex = (
        max(r_src[0], r_src_vis_ex_xywh[0]),
        max(r_src[1], r_src_vis_ex_xywh[1]),
        min(r_src[0] + r_src[2], r_src_vis_ex_xywh[0] + r_src_vis_ex_xywh[2]) - max(r_src[0], r_src_vis_ex_xywh[0]),
        min(r_src[1] + r_src[3], r_src_vis_ex_xywh[1] + r_src_vis_ex_xywh[3]) - max(r_src[1], r_src_vis_ex_xywh[1])
    )
    if r_src_vis_ex[2] <= 0 or r_src_vis_ex[3] <= 0: return

    # Destination visible rect (re-projected from r_src_vis_ex)
    r_dst_vis_ex_x = r_src_vis_ex[0] * kx + pos_x
    r_dst_vis_ex_y = r_src_vis_ex[1] * ky + pos_y
    r_dst_vis_ex_w = r_src_vis_ex[2] * kx
    r_dst_vis_ex_h = r_src_vis_ex[3] * ky
    r_dst_vis_ex = (int(r_dst_vis_ex_x), int(r_dst_vis_ex_y), int(r_dst_vis_ex_w), int(r_dst_vis_ex_h))

    if r_dst_vis_ex[2] <= 0 or r_dst_vis_ex[3] <= 0: return

    # 2. RESIZE AND PAINT
    
    # Extract the visible part from src
    src_vis_ex = src[r_src_vis_ex[1]:r_src_vis_ex[1] + r_src_vis_ex[3], 
                     r_src_vis_ex[0]:r_src_vis_ex[0] + r_src_vis_ex[2]]
    
    # Resize to the target destination size
    resized = cv2.resize(src_vis_ex, (r_dst_vis_ex[2], r_dst_vis_ex[3]), 
                         interpolation=interpolation)
    
    # Paint the resized image onto the destination
    paint(resized, dst, (r_dst_vis_ex[0], r_dst_vis_ex[1]))
    
    # 3. GRID DRAWING
    
    if kx > 10 and ky > 10:
        r_dst_vis_i = bounding_rect(pos) # C++: const cv::Rect rDstVisI = cv::Rect(pos) & rDst;
        # Since r_dst_vis is already the intersection with r_dst, use it.
        # Ensure we use integer coordinates for drawing.
        r_dst_vis_i_x, r_dst_vis_i_y, r_dst_vis_i_w, r_dst_vis_i_h = map(int, r_dst_vis)
        
        color = (255, 255, 255) # White color for grid
        
        # Vertical lines (columns)
        for c in range(r_src_vis_ex[0], r_src_vis_ex[0] + r_src_vis_ex[2]):
            x = int(pos_x + c * kx)
            cv2.line(dst, (x, r_dst_vis_i_y), (x, r_dst_vis_i_y + r_dst_vis_i_h), color)

        # Horizontal lines (rows)
        for r in range(r_src_vis_ex[1], r_src_vis_ex[1] + r_src_vis_ex[3]):
            y = int(pos_y + r * ky)
            cv2.line(dst, (r_dst_vis_i_x, y), (r_dst_vis_i_x + r_dst_vis_i_w, y), color)
            
    # 4. TEXT OVERLAY
    
    if data is not None and kx > 20 and ky > 20:
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        font_thickness = 1
        max_text_size = (int(kx - 5), int(ky - 5))
        
        for r in range(r_src_vis_ex[1], r_src_vis_ex[1] + r_src_vis_ex[3]):
            for c in range(r_src_vis_ex[0], r_src_vis_ex[0] + r_src_vis_ex[2]):
                
                text_result = _pixel_text(data, r, c, font_face, font_scale, font_thickness, max_text_size)
                
                if text_result:
                    text, text_size = text_result
                else:
                    # Failed to fit
                    text = "..."
                    (w, h), _ = cv2.getTextSize(text, font_face, font_scale, font_thickness)
                    text_size = (w, h)
                    
                    if text_size[0] > max_text_size[0] or text_size[1] > max_text_size[1]:
                        continue

                # Calculate center of the cell in destination coordinates
                center_x = int(pos_x + (c + 0.5) * kx)
                center_y = int(pos_y + (r + 0.5) * ky)
                
                # Text position: align center of text to center of cell
                text_pos_x = center_x - text_size[0] // 2
                text_pos_y = center_y + text_size[1] // 2 # OpenCV origin is bottom-left of text
                text_pos = (text_pos_x, text_pos_y)
                
                # Determine text color based on background luminance
                # C++ assumes src is cv::Mat3b (BGR)
                bg_pix = src[r, c] 
                # Simple average luminance check
                dark = (bg_pix[0] + bg_pix[1] + bg_pix[2]) / 3 < 127
                # White text on dark background, black text on light background
                color = (255, 255, 255) if dark else (0, 0, 0)
                
                cv2.putText(dst, text, text_pos, font_face, font_scale, color, font_thickness, cv2.LINE_AA)
    return dst
