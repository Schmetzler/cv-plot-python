# cv-plot-python

A python implementation of the CvPlot library from [PROFACTOR](https://www.profactor.at/).
I needed a fast drawing library as using matplotlib within an application is cumbersome and slow. So I searched for another opportunity using pure OpenCV and found this... So I reimplemented it in python using the Gemini Flash 2.5 Model from Google.

## Usage

In this library you must basically build the whole plot yourself. You have to add the Axis, the grid and of course the actual plot. There are some basic helper methods found in `cv_plot.plot`. You can create a LinePlot and a ScatterPlot (with the `Series` class) and can show `Image`s as well as `HorizontalLine`s and `VerticalLine`s. You can use this as follows:

```python
import cv_plot.plot as cvplt
import cv2

ax = cvplt.makePlotAxes()
s = ax.create(cvplt.Series, x=[0,1,2,3,4], y=[2,3,1,5,6], lineSpec="r-") # this would be a red LinePlot, to create a green ScatterPlot just replace 'r-' with 'g.'
img = ax.render(600,400) # now you have an standard OpenCV image (which is a np.ndarray with type np.uint8)
# show it in a window
plt.show(img)
```

To use alpha blending you have to set the alpha value AFTER creating the Drawable, like so:

```python
# Can also now fill a polygon
s = ax.create(cvplt.Series, x=[0,1,2,3,2,1], y=[2,3,1,5,6,-2], lineSpec="r-", fill=True)
s.alpha=0.3
```

You can do a little bit more in the sense of styling and setting y and x limits but thats basically it.

### Legend

As I had some circular import errors to use a Legend (as it imports `Axes` from `core`) you have to import it after `import cv_plot.plot as cvplt` with something like:

`from cv_plot.drawables.legend import Legend`

## Extending

You can add additionaly drawables by inheriting from `cv_plot.core.Drawable`. There you must especially implement the `render(RawProjection)` function and the `getBoundingRect()` function (to automatical determine the axis limits). With the `RawProjection` you can `project` points from data space into display space and reverse with  `unproject`. The `RawProjection` contains an `inner_rect` which is the inside of the axis boundaries (and mainly the drawing area).

## Notes

I removed the `cv_plot.gui` part as it has its flaws and also didn't work correctly in the cpp version (at least with the Qt backend I was not able to zoom into the axis). But I added the `cv_plot.plot.show` function to allow easy showing of the plot.

---- ORIGINAL README BELOW ----

![CvPlot](doc/img/contours.PNG)

[![Build status](https://github.com/Profactor/cv-plot/workflows/CI/badge.svg)](https://github.com/Profactor/cv-plot/actions)
[![Build status](https://ci.appveyor.com/api/projects/status/2bqhfcoh0q4w2gc8?svg=true)](https://ci.appveyor.com/project/Profactor/cv-plot)
[![Build Status](https://app.travis-ci.com/Profactor/cv-plot.svg?branch=master)](https://app.travis-ci.com/Profactor/cv-plot)

# Motivation
Yes, another C++ plotting library. Because CvPlot is

- Purely OpenCV based
- [Highly adaptable and extendable](doc/tutorial.md#custom-drawables)
- [Fast](doc/img/benchmark.gif)
- [Easy to integrate](doc/integration.md)

CvPlot was developed at [PROFACTOR](https://www.profactor.at/) for realtime image plotting. It comes with some basic "Drawables", including Series, Image, Axis, Grid, Titles, etc. Drawables can easily be modified, replaced and extended, using standard OpenCV drawing functions. CvPlot comes with an interactive [viewer](doc/img/show.gif), based on cv::imshow(). The viewer can easily be integrated into any C++ GUI framework (e.g. Qt/Qml in [CvPlotQt](https://github.com/Profactor/cv-plot-qt)).

# Warning
CvPlot is NOT and will never be a full featured plotting library. Many features are missing, but you can easily add them using [custom drawables](doc/tutorial.md#custom-drawables).

# Documentation
[Screenshots](doc/screenshots.md)

[Integration](doc/integration.md)

[Tutorial](doc/tutorial.md)

[Other C++ OpenCV Libraries](doc/other-libraries.md)




