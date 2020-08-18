import unittest 


from candle.models.resnet import ResNet

class TestResNet(unittest.TestCase):
    def test_init(self):
        net = ResNet()


