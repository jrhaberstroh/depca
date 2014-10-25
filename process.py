import depca.convert as conv
import depca.sidechain_corr as sc
import h5py 
import ConfigParser
import numpy as np
import iopro
import sys
import depca.dedata as dedata

    
#def ComputePCA():
#    with open('./f0postProcess.cfg') as fp:
#        config = ConfigParser.RawConfigParser()
#        config.readfp(fp)
#        sc_h5file = config.get('sidechain','h5file')
#        time_h5tag = config.get('sidechain','time_h5tag')
#        h5stats= config.get('sidechain','h5stats')
#        h5eavtag = config.get('sidechain','h5eavtag')
#        h5corrtag = config.get('sidechain','h5crtag')
#        pca_h5file = config.get('sidechain','pcafile')
#
#    sc_file  = h5py.File(sc_h5file)
#    print sc_file.keys()
#    sc_ds    = sc_file[time_h5tag]
#    stat_file  = h5py.File(h5stats)
#
#    print "Loading covariance and averages '{},{}' from hdf5 file {}...".format(h5corrtag,h5eavtag,h5stats)
#    corr   = stat_file[h5corrtag]
#    Eav_ij = stat_file[h5eavtag]
#
#    print "Computing Modes..."
#    eigval_in, eigvec_inj, impact_in = sc.ComputeModes(corr)
#    print "Eigenvector dimension: {}".format(eigvec_inj.shape)
#
#    conv.ApplyPCA_hdf5(sc_ds, Eav_ij, eigvec_inj, pca_h5file, time_h5tag, site=0, overwrite=True)
#
#    sc_file.close()
#    stat_file.close()

    

if __name__ == '__main__':
    with dedata.dEData('./f0postProcess.cfg') as data:
        #data.InitSidechain_hdf()
        #data.ExamineSidechain_hdf()
        data.InitStats_hdf()
        #data.InitPCA_hdf()
