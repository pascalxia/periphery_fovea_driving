import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#import matplotlib.figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import tensorflow as tf
import io
import numpy as np
import scipy as scipy # Ensure PIL is also installed: pip install pillow

'''
matplotlib_summary code:
Code for generating a tensorflow image summary of a custom matplotlib plot.
Usage: matplotlib_summary(plotting_function, argument1, argument2, ..., name="summary name")
plotting_function is a function which take the matplotlib figure as the first argument and numpy
  versions of argument1, ..., argumentn as the additional arguments and draws the matplotlib plot on the figure
matplotlib_summary creates and returns a tensorflow image summary
'''
class MatplotlibSummaryOpFactory:
    def __init__(self):
        self.counter = 0
        
    def _wrap_pltfn(self, plt_fn):
        def plot(*args):
            f = matplotlib.figure.Figure()
            canvas = FigureCanvas(f)
            ax = f.add_subplot(1, 1, 1)
            args = [ax] + list(args)
            plt_fn(*args)
            buf = io.BytesIO()
            f.savefig(buf, format='png')
            buf.seek(0)
            im = scipy.misc.imread(buf)
            buf.close()
            return im
        return plot
        
    def __call__(self, plt_fn, *args, name=None):
        if name is None:
            self.counter += 1
            name = "matplotlib-summary_%d" % self.counter
        image_tensor = tf.py_func(self._wrap_pltfn(plt_fn), args, tf.uint8)
        image_tensor.set_shape([None, None, 4])
        return tf.summary.image(name, tf.expand_dims(image_tensor, 0))

'''
END matplotlib_summary code
'''
