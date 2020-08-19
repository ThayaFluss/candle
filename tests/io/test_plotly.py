import unittest 
from candle.io.plotly import *
from candle.io.util import touch

class TestPlotly(unittest.TestCase):
    def test_plotter(self):
        filename = "log/temp/test_plotly_plotter.log"
        figfile = "plot/temp/test_plotly_plotter.png"
        touch(filename)
        touch(figfile)

        with open(filename, "w") as f:
            f.write("0\n")
            f.write("1\n")
            f.write("2\n")

        plotter(filename, figfile, xlabel="this is x", ylabel="this is y")

