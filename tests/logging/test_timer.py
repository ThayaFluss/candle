import unittest 
from candle.logging.timer import Timer
import time

class TestTimer(unittest.TestCase):
    def test_timer(self):
        clock = Timer()
        clock.tic()
        time.sleep(1)
        clock.toc()
        self.assertTrue(clock.total_time > 0)

