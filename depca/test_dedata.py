import dedata
import unittest
import subprocess

class TestInit(unittest.TestCase):
    def setUp(self):
        subprocess.call("../test_data/preptest.sh")
    def test_init(self):
        with dedata.dEData('../test_data/temp/test.cfg') as data:
            data.InitSidechain_hdf()
            data.ExamineSidechain_hdf()
            data.InitStats_hdf()
            data.InitPCA_hdf()
