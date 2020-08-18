import torch
import torch.nn as nn
import torch.nn.functional as F


import candle.fctn.activation as acti
from candle.fctn.activation import Nhard_tanh as Nhtanh
from candle.io.util import touch

import os

class MLP(nn.Module):
    def __init__(self, L, i_dim, o_dim=10,act=Nhtanh(),bias=False):
        super().__init__()

        self.hidden = []
        self.act = act
        self.L = L ### number of hidden layers , which does not contain the last layer.
        self.bias = bias
        self.i_dim = i_dim
        self.o_dim = o_dim
  
        for k in range(self.L):
            layer = nn.Linear(i_dim, i_dim, bias=bias)
            self.hidden.append(layer) 
            self.add_module("fc_{}".format(k), layer)

        self.out  = nn.Linear(i_dim, o_dim,bias=True)

        self.initialize_weight()

    def initialize_weight(self,initializer=nn.init.orthogonal_, gain=1., initializer_b=nn.init.normal_, gain_b=1):
        for layer in self.hidden:
            initializer(layer.weight, gain)
            if layer.bias is not None:
                initializer_b(layer.bias, gain_b)

        initializer(self.out.weight, 1.)
        if self.out.bias is not None:
            nn.init.uniform_(self.out.bias,-1/self.o_dim,1/self.o_dim)
        
    def forward(self,x):
        for idx in range(self.L):
            x = self.hidden[idx](x)
            x = self.act(x)
        x = self.out(x)
        return x




class MLPShrink(nn.Module):
    def __init__(self, L, i_dim, m_dim, o_dim=10,act=Nhtanh(),bias=False):  
        """
        input: i 
        -->
        i x m 
        -->
        m x m  (L-times)
        --> 
        m x o 
        --> output
        ----------------
        Parameters :
        　L: int
            number of hidden layers , which does not contain the last layer.
        """
        super().__init__()
        self.hidden = []
        self.L = L
        self.i_dim = i_dim
        self.m_dim = m_dim
        self.o_dim = o_dim
        self.act = act
        
        for k in range(self.L):
            if k == 0:
                layer = nn.Linear(i_dim, m_dim, bias=bias)
            else:
                layer = nn.Linear(m_dim, m_dim, bias=bias)

            self.hidden.append(layer) 
            self.add_module("fc_{}".format(k), layer)

        self.out  = nn.Linear(m_dim, o_dim,bias=True)    
        self.initialize_weight()

    def initialize_weight(self,initializer=nn.init.orthogonal_, gain=1., initializer_b=nn.init.normal_, gain_b=1):
        for layer in self.hidden:
            initializer(layer.weight, gain)
            if layer.bias is not None:
                initializer_b(layer.bias, gain_b)

        initializer(self.out.weight, 1.)
        if self.out.bias is not None:
            nn.init.uniform_(self.out.bias,-1/self.o_dim,1/self.o_dim)
        
    def forward_hidden_pre(self,x):
        for idx in range(self.L-1):
            #print(torch.var(x))
            x = self.hidden[idx](x)
            x = self.act(x)

        x = self.hidden[self.L-1](x)
        return x

    def forward_hidden(self,x):
        x  =self.forward_hidden_pre(x)
        return self.act(x)


    def forward_hidden_m(self, x, num_hidden):
        for idx in range(num_hidden):
            x = self.hidden[idx](x)
            x = self.act(x)
        return x
        

    def forward(self, x):
        x =self.forward_hidden(x)
        x= self.out(x)
        return x


