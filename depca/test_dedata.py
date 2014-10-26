"""Test cases for the depca.dedata module."""
import dedata
import unittest
import subprocess
import logging

class TestInit(unittest.TestCase):
    """
    Tests for dedata initialization functions.
    """
    def setUp(self):
        """
        Prepare the tests with preptest.sh, which creates a 
        temp directory and configuration files for your system.
        """
        subprocess.call("../test_data/preptest.sh")
    def test_init(self):
        """
        Tests initialization of all of the HDF5 files using
        csv files provided in the testing folder.
        """
        logging.basicConfig(filename='test.log', level =logging.DEBUG,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        with dedata.dEData('../test_data/temp/test.cfg') as data:
            data.InitSidechain_hdf()
            data.ExamineSidechain_hdf()
            data.InitStats_hdf()
            data.InitPCA_hdf()

if __name__ == "__main__":
    unittest.main()
