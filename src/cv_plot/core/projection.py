import numpy as np
from cv_plot.core.transformation import Transformation

class RawProjection():
    def __init__(self, offset=(0.0, 0.0), kx=1.0, ky=1.0, 
                 transformation: Transformation = None, inner_rect=None):
        # Equivalent to cv::Point2d offset (using a tuple/list)
        self.offset = offset  
        
        # Equivalent to double kx, ky
        self.kx = kx
        self.ky = ky
        
        # Equivalent to Transformation *transformation
        self.transformation = transformation 
        
        # Equivalent to cv::Rect innerRect (using a tuple or a custom object)
        # Using a simple tuple (x, y, w, h) for simplicity.
        self.inner_rect = inner_rect 

    def project(self, point: tuple, with_transformation: bool = True) -> tuple:
        """
        Projects a point into the display space.
        Equivalent to the C++ project method.
        """
        px, py = point

        if self.transformation and with_transformation:
            px, py = self.transformation.transform((px, py))

        # return cv::Point2d(offset.x + point.x*kx, offset.y + point.y*ky);
        projected_x = self.offset[0] + px * self.kx
        projected_y = self.offset[1] + py * self.ky
        
        return (projected_x, projected_y)

    def unproject(self, point: tuple, with_transformation: bool = True) -> tuple:
        """
        Unprojects a point from the display space back to data space.
        Equivalent to the C++ unproject method.
        """
        # point = cv::Point2d((point.x - offset.x) / kx, (point.y - offset.y) / ky);
        unproj_x = (point[0] - self.offset[0]) / self.kx
        unproj_y = (point[1] - self.offset[1]) / self.ky
        
        unproj_point = (unproj_x, unproj_y)

        if self.transformation and with_transformation:
            # return transformation->untransform(point);
            return self.transformation.untransform(unproj_point)
            
        return unproj_point

    @property
    def area(self):
        if self.inner_rect is None:
            return 0
        return self.inner_rect[2] * self.inner_rect[3]

class Projection():
    def __init__(self, rawProjection : RawProjection):
        self._rawProjection = rawProjection

    def project(self, point: tuple) -> tuple:
        return self._rawProjection.project(point)
    
    def unproject(self, point: tuple) -> tuple:
        return self._rawProjection.unproject(point)
    
    def innerRect(self):
        return self._rawProjection.inner_rect
    
    def outerToInner(self, outer: tuple) -> tuple:
        return (outer[0] - self._rawProjection.inner_rect[0], outer[1] - self._rawProjection.inner_rect[1])

    def innerToOuter(self, inner: tuple) -> tuple:
        return (inner[0] + self._rawProjection.inner_rect[0], inner[1] + self._rawProjection.inner_rect[1])

class RenderTarget(Projection):
    def __init__(self, rawProjection : RawProjection, outerMat : np.ndarray):
        super().__init__(rawProjection)
        x,y,w,h = rawProjection.inner_rect
        self._outerMat = outerMat
        self._innerMat = outerMat[y:y+h,x:x+w]

    def outerMat(self):
        return self._outerMat

    def innerMat(self):
        return self._innerMat
