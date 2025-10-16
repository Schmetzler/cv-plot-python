from typing import Optional
from cv_plot.core import RenderTarget, Drawable, DrawableContainer

class Group(DrawableContainer, Drawable):
    """
    Equivalent to CvPlot::Group, using multiple inheritance from 
    DrawableDeque and Drawable.
    """
        
    def render(self, renderTarget: RenderTarget) -> None:
        """
        Equivalent to void Group::render(RenderTarget &renderTarget).
        Renders all internal drawables in sequence.
        """
        # C++: for (const auto &drawable : drawables()) { drawable->render(renderTarget); }
        for drawable in self.drawables():
            drawable.render(renderTarget)

    def getBoundingRect(self) -> Optional[tuple]:
        """
        Equivalent to bool Group::getBoundingRect(cv::Rect2d &rect).
        Delegates the calculation to the DrawableDeque base class.
        """
        return DrawableContainer.getBoundingRect(self)
        