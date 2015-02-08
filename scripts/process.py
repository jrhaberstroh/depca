import depca.dedata as dedata
import logging

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    with dedata.dEData('./conf.cfg') as data:
        data.InitSidechain_hdf()
        data.InitStats_hdf()
        data.InitPCA_hdf()
