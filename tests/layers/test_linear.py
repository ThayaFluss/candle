import unittest 

from candle.layers.linear import *
import numpy as np
import torch

class TestLinear(unittest.TestCase):
    def test_init(self):
        layer = Linear(100, 200, bias=True)
        x = torch.tensor(np.random.randn(1,100), dtype=torch.float32)
        y = layer(x)



