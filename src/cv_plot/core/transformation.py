import numpy as np
import math

# --- Base Transformation Class (Equivalent to C++ abstract class Transformation) ---

class Transformation:
    """
    Base class for coordinate transformations (e.g., Log, Exp, etc.)
    Defines the interface for transforming points, rectangles, and limits.
    """
    
    # In Python, we don't have explicit virtual = 0, we raise a NotImplementedError
    # in the base class methods that must be overridden.
    
    def transform(self, point: tuple) -> tuple:
        """Applies the forward transformation on a point (x, y)."""
        raise NotImplementedError("Subclasses must implement abstract method 'transform'")

    def untransform(self, point: tuple) -> tuple:
        """Applies the inverse transformation on a point (x, y)."""
        raise NotImplementedError("Subclasses must implement abstract method 'untransform'")

    def transform_bounding_rect(self, rect: tuple) -> tuple:
        """
        Transforms a bounding rectangle (x, y, w, h).
        Returns a rectangle defined by two transformed corner points.
        (x_min_out, y_min_out, x_max_out, y_max_out)
        """
        r_x, r_y, r_w, r_h = rect
        # C++: return cv::Rect2d(transform(cv::Point2d(r.x, r.y)), transform(cv::Point2d(r.x + r.width, r.y + r.height)));
        
        # Transform the top-left corner
        p1_x, p1_y = self.transform((r_x, r_y))
        
        # Transform the bottom-right corner
        p2_x, p2_y = self.transform((r_x + r_w, r_y + r_h))
        
        # Return as (x1, y1, x2, y2)
        return (p1_x, p1_y, p2_x, p2_y)

    def transform_xlim(self, xlim: tuple) -> tuple:
        """Transforms the x-axis limits (x_min, x_max)."""
        # C++: return std::make_pair(transform(cv::Point2d(xlim.first, 0)).x, transform(cv::Point2d(xlim.second, 0)).x);
        x_min_in, x_max_in = xlim
        x_min_out, _ = self.transform((x_min_in, 0.0))
        x_max_out, _ = self.transform((x_max_in, 0.0))
        return (x_min_out, x_max_out)

    def transform_ylim(self, ylim: tuple) -> tuple:
        """Transforms the y-axis limits (y_min, y_max)."""
        # C++: return std::make_pair(transform(cv::Point2d(0, ylim.first)).y, transform(cv::Point2d(0, ylim.second)).y);
        y_min_in, y_max_in = ylim
        _, y_min_out = self.transform((0.0, y_min_in))
        _, y_max_out = self.transform((0.0, y_max_in))
        return (y_min_out, y_max_out)

    def untransform_xlim(self, xlim: tuple) -> tuple:
        """Applies inverse transformation on the x-axis limits."""
        # C++: return std::make_pair(untransform(cv::Point2d(xlim.first, 0)).x, untransform(cv::Point2d(xlim.second, 0)).x);
        x_min_in, x_max_in = xlim
        x_min_out, _ = self.untransform((x_min_in, 0.0))
        x_max_out, _ = self.untransform((x_max_in, 0.0))
        return (x_min_out, x_max_out)

    def untransform_ylim(self, ylim: tuple) -> tuple:
        """Applies inverse transformation on the y-axis limits."""
        # C++: return std::make_pair(untransform(cv::Point2d(0, ylim.first)).y, untransform(cv::Point2d(0, ylim.second)).y);
        y_min_in, y_max_in = ylim
        _, y_min_out = self.untransform((0.0, y_min_in))
        _, y_max_out = self.untransform((0.0, y_max_in))
        return (y_min_out, y_max_out)

# --- Static/Helper Method (Used by all Log transformations) ---

def transform_log_lim(lim: tuple) -> tuple:
    """
    Static method equivalent to LogLogTransformation::transformLogLim.
    Applies safety checks for log function domain (x > 0) and applies log.
    """
    lim_min, lim_max = lim
    
    # C++: if (lim.first <= 0) { lim.first = 1e-5; }
    if lim_min <= 0:
        lim_min = 1e-5
        
    # C++: if (lim.second <= lim.first) { lim.second = lim.first * 10; }
    if lim_max <= lim_min:
        lim_max = lim_min * 10
        
    # C++: lim.first = std::log(lim.first); lim.second = std::log(lim.second);
    lim_min = np.log(lim_min)
    lim_max = np.log(lim_max)
    
    return (lim_min, lim_max)

# --- LogLogTransformation (Log on both axes) ---

class LogLogTransformation(Transformation):
    """Applies a logarithmic transformation to both X and Y axes."""
    
    def transform(self, point: tuple) -> tuple:
        # C++: return cv::Point2d(std::log(point.x), std::log(point.y));
        # Using numpy.log for natural logarithm (base e)
        return (np.log(point[0]), np.log(point[1]))

    def untransform(self, point: tuple) -> tuple:
        # C++: return cv::Point2d(std::exp(point.x), std::exp(point.y));
        # Using numpy.exp
        return (np.exp(point[0]), np.exp(point[1]))

    def transform_xlim(self, xlim: tuple) -> tuple:
        # C++: return LogLogTransformation::transformLogLim(xlim);
        return transform_log_lim(xlim)

    def transform_ylim(self, ylim: tuple) -> tuple:
        # C++: return LogLogTransformation::transformLogLim(ylim);
        return transform_log_lim(ylim)

# --- LinLogTransformation (Linear on X, Log on Y) ---

class LinLogTransformation(Transformation):
    """Applies a linear transformation to X and a logarithmic transformation to Y."""
    
    def transform(self, point: tuple) -> tuple:
        # C++: return cv::Point2d(point.x, std::log(point.y));
        return (point[0], np.log(point[1]))

    def untransform(self, point: tuple) -> tuple:
        # C++: return cv::Point2d(point.x, std::exp(point.y));
        return (point[0], np.exp(point[1]))

    def transform_ylim(self, ylim: tuple) -> tuple:
        # C++: return LogLogTransformation::transformLogLim(ylim);
        return transform_log_lim(ylim)

# --- LogLinTransformation (Log on X, Linear on Y) ---

class LogLinTransformation(Transformation):
    """Applies a logarithmic transformation to X and a linear transformation to Y."""
    
    def transform(self, point: tuple) -> tuple:
        # C++: return cv::Point2d(std::log(point.x), point.y);
        return (np.log(point[0]), point[1])

    def untransform(self, point: tuple) -> tuple:
        # C++: return cv::Point2d(std::exp(point.x), point.y);
        return (np.exp(point[0]), point[1])

    def transform_xlim(self, xlim: tuple) -> tuple:
        # C++: return LogLogTransformation::transformLogLim(xlim);
        return transform_log_lim(xlim)