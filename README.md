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

## Funding

The work was conducted as part of the research project **KI-StudiUm** at the [**Westsächsische Hochschule Zwickau**](https://www.whz.de/english/), which was funded by the [**Federal Ministry of Research, Technology and Space**](https://www.bmftr.bund.de/EN/Home/home_node.html) as part of the federal-state initiative "KI in der Hochschulbildung" under the funding code `16DHBKI063`.

<picture>
     <source srcset="assets/bmftr-en-dark.svg" media="(prefers-color-scheme: dark)">
     <img src="assets/bmftr-en-light.svg" height="75px">
</picture>
<picture>
     <source srcset="assets/whz-en-dark.svg" media="(prefers-color-scheme: dark)">
     <img src="assets/whz-en-light.svg" height="75px">
</picture>
