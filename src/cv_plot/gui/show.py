import numpy as np
import cv2
from typing import Union, Any, List

from cv_plot.core import Axes, plot, plotImage
from .window import Window

def show(windowName: str, axes: Axes, cols: int = 640, rows: int = 480) -> None:
    """
    Equivalent to inline void show(const std::string &windowName, Axes &axes, int rows, int cols).
    Creates a window for the given Axes and waits for a key press (or window close).
    """
    # C++: Window window(windowName, axes, rows, cols);
    window = Window(windowName, axes, cols, rows)
    
    # C++: window.waitKey();
    window.waitKey()

def showPlot(x, y=None, lineSpec: str = "-") -> None:
    """
    Equivalent to inline void showPlot(cv::InputArray x, cv::InputArray y, const std::string &lineSpec).
    Creates a plot from (x, y) data and displays it in a window named "CvPlot".
    """
    # C++: auto axes = plot(x, y, lineSpec);
    axes = plot(x, y, lineSpec)
    
    # C++: show("CvPlot", axes);
    show("CvPlot", axes)

def showImage(mat: np.ndarray) -> None:
    """
    Equivalent to inline void showImage(const cv::Mat &mat).
    Creates a plot containing the image and displays it.
    """
    # C++: auto axes = plotImage(mat);
    axes = plotImage(mat)
    
    # C++: show("CvPlot", axes);
    show("CvPlot", axes)