from cv_plot.core import Axes
from cv_plot.drawables import *
import numpy as np

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
