import unittest 


from candle.io.arguments import parser_init

class TestModule(unittest.TestCase):
    def test_parser_init(self):
        parser = parser_init()
        args = parser.parse_args(["--device_id", "0"])
        self.assertTrue(args.device_id == 0)    

