import torch
import torch.nn as nn
from candle.fctn.activation import Nhard_tanh as Nhtanh
from candle.fctn.activation import shifted_relu, hard_tanh
import math

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1,  dilation=1):
        """
        Parameters:
            inplanes: int
                number of input channels
            planes: int
                number of output channels
        """
        super(BasicBlock, self).__init__()

        self.inplanes =  inplanes
        self.planes = planes

        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        norm_layer = nn.BatchNorm2d
        self.bn1 = norm_layer(planes)
        #self.act = nn.ReLU(inplace=True)
        #self.act = Nhtanh()
        self.act = shifted_relu

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.stride = stride


        if inplanes == planes:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                norm_layer(planes),
            )


    def forward(self, x):
        identity = x

        #print(x.shape,  torch.sqrt(torch.mean(x**2)),  torch.var(x))

        out = 2*self.conv1(x)

        #print(out.shape,  torch.sqrt(torch.mean(out**2)),  torch.var(out))

        out = self.bn1(out)
        out = self.act(out)
        #print(out.shape,  torch.sqrt(torch.mean(out**2)),  torch.var(out))


        out = self.conv2(out)
        out = self.bn2(out)
        #print(out.shape,  torch.sqrt(torch.mean(out**2)),  torch.var(out))

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        #print(out.shape,  torch.sqrt(torch.mean(out**2)),  torch.var(out))

        out = self.act(out)#/math.sqrt(2)

        return out

class SimpleBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1,  dilation=1):
        """
        Parameters:
            inplanes: int
                number of input channels
            planes: int
                number of output channels
        """
        super(SimpleBlock, self).__init__()

        self.inplanes =  inplanes
        self.planes = planes

        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        norm_layer = nn.BatchNorm2d
        self.bn1 = norm_layer(planes)
        #self.act = nn.ReLU(inplace=True)
        #self.act = Nhtanh()
        self.act = shifted_relu


        self.stride = stride


        if inplanes == planes:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                norm_layer(planes),
            )


    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)


        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.act(out)

        return out




class ResNet(nn.Module):
    def __init__(self, num_classes=10,):
        super().__init__()

        self.hidden_layers = [
            ### 32x 32 --> 16 x 16 
            BasicBlock(inplanes=3, planes=32, stride=2),
            BasicBlock(inplanes=32, planes=32, stride=1),

            ### 16 x 16 --> 8 x 8 
            BasicBlock(inplanes=32, planes=128, stride=2),
            BasicBlock(inplanes=128, planes=128, stride=1),

            ###  8 x 8  --> 4 x 4
            BasicBlock(inplanes=128, planes=256, stride=2),
            BasicBlock(inplanes=256, planes=256, stride=1),

            ### 4 x 4 --> 2 x 2
            BasicBlock(inplanes=256, planes=512, stride=2),
            #BasicBlock(inplanes=512, planes=512, stride=1),

            ### 2 x 2 --> 1 x 1
            #SimpleBlock(inplanes=512, planes=1024, stride=2)
        ]

        for idx, layer in enumerate(self.hidden_layers):
            self.add_module("block_{}".format(idx), layer)


        #self.conv = conv3x3(512,512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.bn = nn.BatchNorm1d(512)
        self.fc = nn.Linear(512, num_classes, bias=True)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.orthogonal_(self.fc.weight,1.)

    def _make_layers(self, inplanes, planes, stride):
        block = BasicBlock(inplanes, planes, stride)
        self.add_module()
        return block
        
    def forward(self,x):
        for layer in self.hidden_layers:
            #print(x.shape,  torch.sqrt(torch.mean(x**2)),  torch.var(x))
            x = layer(x)
        x = self.avgpool(x)
        #import pdb; pdb.set_trace()
        x = torch.flatten(x,1)
        #x = self.bn(x)
        x = self.fc(x)
        return x
