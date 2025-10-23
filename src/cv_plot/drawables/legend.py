### Adapted from https://github.com/bjornborg/cv-plot.git

import numpy as np
import cv2
from typing import Tuple, List, Any, Optional

from cv_plot.core import RenderTarget, Drawable, Axes
from . import Series

class LegendLabel(Drawable):
    """
    Equivalent to CvPlot::LegendLabel. Renders a text label in the outer canvas.
    Impl fields are now explicitly defined as instance variables.
    """
    # Impl fields merged into instance variables

    def __init__(self):
        """
        Equivalent to LegendLabel::LegendLabel(). Initializes Impl fields.
        """
        super().__init__()

        self._text = ""
        self._position = (0, 0)

    def setPosition(self, a_x: int, a_y: int) -> None:
        """Equivalent to void LegendLabel::setPosition(...)."""
        self._position = (a_x, a_y)

    def setText(self, a_str: str) -> None:
        """Equivalent to void LegendLabel::setText(...)."""
        self._text = a_str

    def render(self, renderTarget: RenderTarget) -> None:
        """Equivalent to void LegendLabel::render(...)."""
        
        size, baseline = cv2.getTextSize(self._text, self._font_face, self._font_scale, self._font_thickness, )
        size_h = size[1]
        
        proj_x, proj_y = renderTarget.project(self._position)
        outer_x, outer_y = renderTarget.innerToOuter((proj_x, proj_y))
        
        offset_x = size_h * 2
        offset_y = size_h / 2
        
        pos_x = int(outer_x + offset_x)
        pos_y = int(outer_y + offset_y)
        pos: Point = (pos_x, pos_y)
        
        cv2.putText(renderTarget.outerMat(), self._text, pos, self._font_face, self._font_scale, self._color, self._font_thickness, cv2.LINE_AA)
        

# ... (LegendLabel definition follows)

# --- LEGEND CLASS ---

class Legend(Drawable):
    """
    Equivalent to CvPlot::Legend. Renders a legend box.
    Impl fields are now explicitly defined as instance variables.
    """

    def __init__(self, parentAxes=None, omitNoName=True):
        """
        Equivalent to Legend::Legend(). Initializes Impl fields.
        """
        super().__init__()
        
        self._parentAxes = parentAxes
        self._width = 180
        self._height = 60
        self._margin = 10
        self._omit = omitNoName

    def setParentAxes(self, a_parentAxes: Axes) -> None:
        """Equivalent to void Legend::setParentAxes(...)."""
        self._parentAxes = a_parentAxes

    def render(self, renderTarget: RenderTarget) -> None:
        """Equivalent to void Legend::render(...)."""
        if self._parentAxes is None:
            return

        if self._omit:
            series_vec: List[Series] = [drawable for drawable in self._parentAxes.drawables() if isinstance(drawable, Series) and drawable.getName()]
        else:
            series_vec: List[Series] = [drawable for drawable in self._parentAxes.drawables() if isinstance(drawable, Series)]

        if not series_vec:
            return

        legend_axes = Axes()
        num_series = len(series_vec)
        
        legend_axes.setMargins(5, self._width - 2 * self._margin - 60, 5, 5)
        legend_axes.setXLim(-0.2, 1.2)
        legend_axes.setYLim(-0.2, num_series - 1 + 0.2)
        legend_axes.setYReverse()
        
        for i, series in enumerate(series_vec):
            y_pos = float(i)
            
            # Create dummy series
            preview_series: Series = legend_axes.create(Series, 
                np.array([0, 0.25, 0.5, 0.75, 1]), 
                np.full(5, y_pos, dtype=np.uint8)
            )
            preview_series.setLineType(series.getLineType()) \
                          .setLineWidth(series.getLineWidth()) \
                          .setColor(series.getColor()) \
                          .setMarkerType(series.getMarkerType()) \
                          .setMarkerSize(series.getMarkerSize())

            # Create label
            label: LegendLabel = legend_axes.create(LegendLabel)
            label.setPosition(1, int(i))
            label.setText(series.getName())
            
        # Determine destination rectangle
        inner_mat = renderTarget.innerMat()
        inner_cols, inner_rows = inner_mat.shape[1], inner_mat.shape[0]
        
        rect_x = inner_cols - self._width - self._margin
        rect_y = self._margin
        rect_w = self._width
        rect_h = self._height
        
        if rect_x >= 0 and rect_x + rect_w <= inner_cols and rect_y >= 0 and rect_y + rect_h <= inner_rows:
            
            roi = inner_mat[rect_y : rect_y + rect_h, rect_x : rect_x + rect_w]
            
            legend_axes.render(roi)
            
            # Draw a black border
            cv2.rectangle(inner_mat, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (0, 0, 0), 1)
