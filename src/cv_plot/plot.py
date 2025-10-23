from cv_plot.core import Axes
from cv_plot.drawables import *
import numpy as np
import cv2

def makePlotAxes() -> Axes:
    axes = Axes()
    axes.create(Border)
    xAxis = axes.create(XAxis)
    yAxis = axes.create(YAxis)
    axes.create(VerticalGrid, xAxis)
    axes.create(HorizontalGrid, yAxis)
    return axes

def makeImageAxes() -> Axes:
    axes = Axes()
    axes.setFixedAspectRatio(True)
    axes.setYReverse(True)
    axes.setXTight(True)
    axes.setYTight(True)
    axes.setTightBox(True)
    axes.create(Border)
    axes.create(XAxis)
    axes.create(YAxis)
    return axes

def plot(x, y=None, lineSpec = "-") -> Axes:
    axes = makePlotAxes()
    axes.create(Series, x, y, lineSpec)
    return axes

def plotImage(img, pos = None) -> Axes:
    axes = makeImageAxes()
    axes.create(Image, img, pos)
    return axes

def show(axes_or_img, width=None, height=None, name="IMG", blocking=True):
    if isinstance(axes_or_img, Axes):
        if width is None:
            width = 400
        if height is None:
            height = 400
        img = axes_or_img.render(width, height)
    elif isinstance(axes_or_img, np.ndarray):
        img = axes_or_img
    else:
        raise ValueError("axes_or_img must be either an Axes object or an np.ndarray")
    cv2.imshow(name, img)
    if blocking:
        try:
            while True:
                if cv2.waitKey(25) >= 0:
                    break
        finally:
            cv2.destroyAllWindows()
    else:
        cv2.waitKey(1)
    return
