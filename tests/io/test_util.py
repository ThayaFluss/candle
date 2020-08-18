import unittest 
from candle.io.util import *

class  TestUtil(unittest.TestCase):
    def test_touch(self):
        path = "log/tests/test_touch.log"
        touch(path)

