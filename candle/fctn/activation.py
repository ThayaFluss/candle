#######  activation functions ######

import torch
import torch.nn.functional as F


def silu(x):
    """
    Smoothed ReLU (Sigmoid-ReLU)
    """
    return x*torch.sigmoid(x)


def shifted_relu(x):
    return F.relu(x+0.5) - 0.5

def hard_tanh(x):
    return F.relu(x+1, inplace=False) - F.relu(x-1, inplace=False) - 1


from torch.nn import Module
import math
class Nact(Module):
    """
    Normalized activation for orthogonal MLP
    """
    def __init__(self, act, i_var, gain):
        super().__init__()
        self.act = act
        self.i_var = i_var
        self.gain = gain
        self.i_std = math.sqrt(i_var)

    def forward(self,x):
        return  self.act(self.i_std*self.gain*x)/self.i_std

    def extra_repr(self):
        print("self.act={}".format(self.act))
        print("self.i_var={}".format(self.i_var))
        print("self.gain={}".format(self.gain))
        
class Nsilu(Nact):
    """
    Normalized SiLU
    """
    def __init__(self,  sp_var=1/4, L=100):
        ### default values
        i_var = 1.60e-4
        gain = 1.9994

        if L == 100:
            if sp_var == 1/4:
                self.sp_var=sp_var
                i_var = 1.60e-4
                gain = 1.9994
            else:
                print("(Nsilu)[ValueWarning]sp_var={}: use default i_var & gain".format(sp_var))
        else:
            print("(Nsilu)[ValueWarning]L={}: use default i_var & gain".format(L))

        super().__init__(silu,i_var, gain)



class Nhard_tanh(Nact):
    def __init__(self, sp_var = 1/4, L = 100):
        ### default values
        #i_var = 1.25e-1
        #gain = 1.0013
        thres = 1e-8
        if L == 100:
            if abs(sp_var - 1/4)<thres :
                i_var = 1.25e-1
                gain = 1.0013
                self.sp_var=1/4
            elif abs(sp_var - 1/16)<thres:
                i_var = 0.087
                gain = 1.0004
                self.sp_var = 1/16
            elif abs(sp_var - 1/64) < sp_var:
                i_var = 0.074
                gain = 1.0002
                self.sp_var = 1/64
            else:
                print("(Nhard_tanh)[ValueWarning]sp_var={}: use default i_var & gain".format(sp_var))
                raise ValueError
        else:
            print("(Nhard_tanh)[ValueWarning]L={}: use default i_var & gain".format(L))
            raise ValueError
        super().__init__(hard_tanh,i_var, gain)

