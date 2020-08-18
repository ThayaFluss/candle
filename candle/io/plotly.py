"""
Plotly: https://plot.ly/python/
"""
import plotly.express as px
import numpy as np
import plotly

def plotter(log_file, fig_file=None,
    xlabel=None, ylabel=None):
    with open(log_file) as f:
        a = f.readlines()
    y = [ float(x) for x in a ]
    y = np.array(y)
    x = np.arange( len(y))
    if xlabel is None or ylabel is None:
        fig = px.line(x=x, y=y )
    else:
        fig = px.line(x=x, y=y, labels={'x':xlabel, 'y':ylabel})

    if fig_file is None:
        print("(plotter){} ---> fig: figure is not closed.".format(log_file))
        fig.show()
    else:
        print("(plotter){} ---> {}".format(log_file, fig_file))
        #fig.write_image(fig_file)
        plotly.io.write_image(fig, fig_file)
        return

