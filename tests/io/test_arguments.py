import unittest 
import numpy as np
import time

from candle.io.arguments import parser_init
from candle.io.arguments import write_config, read_config

class TestModule(unittest.TestCase):
    def test_parser_init(self):
        parser = parser_init()
        args = parser.parse_args(["--device_id", "0"])
        self.assertTrue(args.device_id == 0)    



    def test_write_config(self):
        jobname = "tests/test_write_config"

        def _tester(x, a=0,b=1,c=2):
            write_config(jobname)
            return 0
        x = [8,9., "alice"]
        _tester(x)
    

    def test_write_config_of_job(self):
        """
        If Error does not disapper, then try
            rm -r log/tests
        """
        jobname = "tests/test_read_config_of_job"
        def _tester(x, a=0,b=1,c=2):
            write_config(jobname)
            return 0

        x = [8,9., "alice"]
        _tester(x)
        time.sleep(1)
        x = ["yutaka"]
        _tester(x)
        obj = read_config(jobname)
        self.assertTrue( len(obj.keys()) >= 2)