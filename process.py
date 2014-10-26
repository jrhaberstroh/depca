import depca.convert as conv
import depca.sidechain_corr as sc
import h5py 
import ConfigParser
import numpy as np
import iopro
import sys


    

if __name__ == '__main__':
    with dedata.dEData('./conf.cfg') as data:
        #data.InitSidechain_hdf()
        #data.ExamineSidechain_hdf()
        #data.InitStats_hdf()
        data.InitPCA_hdf()
