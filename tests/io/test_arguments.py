import unittest 
import numpy as np
import time

from candle.io.arguments import parser_init, log_dir
from candle.io.arguments import write_config, read_config

class TestModule(unittest.TestCase):
    def test_parser_init(self):
        parser = parser_init()
        args = parser.parse_args(["--device_id", "0"])
        self.assertTrue(args.device_id == 0)    



    def test_write_config(self):
        dirname = "log/temp/test_write_config"

        def _tester(x, a=0,b=1,c=2):
            write_config(dirname)
            return 0
        x = [8,9., "alice"]
        _tester(x)
    

    def test_write_config_of_job(self):
        """
        If Error does not disapper, then try
            rm -r log/temp
        """
        jobname = "temp/test_read_config_of_job"
        dirname = log_dir(jobname)
        def _tester(x, a=0,b=1,c=2):
            write_config(dirname)
            return 0

        x = [8,9., "alice"]
        _tester(x)
        obj = read_config(dirname)
        self.assertTrue( len(obj.keys()) >= 1)