import unittest 


from candle.tpl.util import *

class TestModule(unittest.TestCase):
    def test_train_test_set(self):
        train_set, test_set = train_test_set()
    