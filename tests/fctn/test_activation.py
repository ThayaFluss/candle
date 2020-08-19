import unittest 
from candle.fctn.activation import *
import torch

class TestModule(unittest.TestCase):
    def test_Nhard_tnah(self):
        nhtanh = Nhard_tanh()
        x = torch.tensor([1.,2.], dtype=torch.float32)
        x.requires_grad = True
        y = nhtanh(x)
        l =  torch.mean(y**2)
        l.backward()


    