import unittest 


from candle.models.mlp import *

class TestMLP(unittest.TestCase):
    def test_init(self):
        net = MLP(10,100,10)

    def test_init_shrink(self):
        net = MLPShrink(10,100,50, 10)


