import dedata
import unittest
import subprocess
import logging

class TestInit(unittest.TestCase):
    def setUp(self):
        subprocess.call("../test_data/preptest.sh")
    def test_init(self):
        logging.basicConfig(filename='test.log', level =logging.DEBUG,format='%(asctime)s - %(levelname)s - %(message)s')
        with dedata.dEData('../test_data/temp/test.cfg') as data:
            data.InitSidechain_hdf()
            data.ExamineSidechain_hdf()
            data.InitStats_hdf()
            data.InitPCA_hdf()

if __name__=="__main__":
    unittest.main()
