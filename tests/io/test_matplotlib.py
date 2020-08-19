import unittest 

from candle.io.matplotlib import *
from candle.io.util import touch

class TestMatplotlib(unittest.TestCase):
    def test_plotter(self):
        filename = "log/tests/test_matplotlib_plotter.log"
        figfile = "plot/tests/test_matplotlib_plotter.png"
        touch(filename)
        touch(figfile)

        with open(filename, "w") as f:
            f.write("0\n")
            f.write("1\n")
            f.write("2\n")

        plotter(filename, figfile, xlabel="this is x", ylabel="this is y")



