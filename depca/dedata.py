import numpy as np
import h5py
import ConfigParser
import sys
import logging
import sidechain_corr as sc

HAS_IOPRO = True
HAS_NUMBA = True
try:
    import iopro
    logging.debug("Loaded iopro successfully")
except ImportError:
    HAS_IOPRO=False
    logging.debug("Failed to load iopro")
try:
    from numbapro import vectorize
    logging.debug("Loaded numbapro successfully")
except ImportError:
    HAS_NUMBA=False
    logging.debug("Failed to load numbapro")

def dEhdf5_init(hdf_file, hdf_dsname, open_flag, Ncoarse=0, ntimes=0, dt_ps=None, chunktime=1000):
    with h5py.File(hdf_file,open_flag) as h5_out:
        if not hdf_dsname:
            return
        # Check if the dataset exists
        h5keys = h5_out.items()
        goodkeys = [key[0] == hdf_dsname for key in h5keys]
        if any(goodkeys):
            ds = h5_out[hdf_dsname]
        else:
            ds = h5_out.create_dataset(hdf_dsname, shape=(ntimes,Ncoarse), chunks=(chunktime,Ncoarse), maxshape=(None, Ncoarse), dtype='f32')

        if dt_ps:
            ds.attrs['dt_unit'] = "picoseconds"
            ds.attrs['dt'] = dt_ps

if HAS_NUMBA:
    @vectorize(['float64(float64,float64)'], target='cpu')
    def ParallelSub64(a,b):
        return a - b
    @vectorize(['float32(float32,float32)'], target='cpu')
    def ParallelSub32(a,b):
        return a - b

def PCARotate_hdf5(E_tj, corr, Eav_j, pca_h5ds):
    logging.debug( "Computing Modes...")
    eigval_n, eigvec_nj, impact_n = sc.ComputeModes(corr)
    logging.debug( "Eigenvector dimension: {}".format(eigvec_nj.shape))
    tf = E_tj.shape[0]
    t0 = 0
    Ncoarse = E_tj.shape[1]
    # Perform chunk size computations
    GB = float(1E9)
    RAM_GB = .5
    RAM_nfloat = RAM_GB * 1E9 / 8
    RAM_ntimes = RAM_nfloat / Ncoarse
    RAM_nchunk = int( np.ceil((tf - t0) / float(RAM_ntimes)) )
    RAM_time_per_chunk = (tf - t0) / RAM_nchunk
    logging.debug( "Number of chunks needed: {} of {} GB each".format(RAM_nchunk, RAM_time_per_chunk * Ncoarse * 8 / GB))
    RAM_return_times = (RAM_nchunk*RAM_time_per_chunk)
    RAM_return_tot = (RAM_nchunk*RAM_time_per_chunk) * 8 * (Ncoarse)
    logging.debug( "Disk space needed for output data: {} GB".format(RAM_return_tot / GB))
    for chunk_num in xrange(RAM_nchunk):
        logging.debug( "Chunk {}:".format(chunk_num+1))
        # Each chunk reads [t0_chunk, tf_chunk)
        t0_chunk = (  chunk_num   * RAM_time_per_chunk) + t0
        tf_chunk = ((chunk_num+1) * RAM_time_per_chunk) + t0
        t0_return = t0_chunk - t0
        tf_return = tf_chunk - t0
        logging.debug( "Loading chunk into RAM...")
        RAM_E_t_ij = E_tj[t0_chunk:tf_chunk,:]

        # Build dE for chunk
        logging.debug( "Centering chunk with dataset's mean...")
        if HAS_NUMBA:
            try:
                RAM_dE_t_j = ParallelSub32( RAM_E_t_ij, Eav_j[:])
            except TypeError:
                try:
                    logging.warning( "32 bit parallel float computation failed, trying 64 bit...")
                    RAM_dE_t_j = ParallelSub64( RAM_E_t_ij, Eav_j[:])
                    logging.debug( "32 bit parallel float success.")
                except TypeError as e:
                    logging.error( "64 bit failed, too.")
                    raise e
        else:
            RAM_dE_t_j = RAM_E_t_ij - Eav_j[:]
        logging.debug( "Rotating chunk...")
        RAM_dE_rotated = np.inner(RAM_dE_t_j, eigvec_nj.T)
        logging.debug( "Writing chunk..." )
        pca_h5ds[t0_chunk:tf_chunk, :] = RAM_dE_rotated[:,:]
        logging.debug( 'Chunk {}  done'.format(chunk_num+1))

# Python recipe 577096
def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is one of "yes" or "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = raw_input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

def append_index(dsname, index):
    return dsname + "%d" % index


class dEData():
    def __init__(self, config = './f0postProcess.cfg'):
        with open(config) as fp:
            config = ConfigParser.RawConfigParser()
            config.readfp(fp)
            self.sc_h5file = config.get('sidechain','h5file')
            self.time_h5tag = config.get('sidechain','time_h5tag')
            self.h5stats= config.get('sidechain','h5stats')
            self.h5eavtag = config.get('sidechain','h5eavtag')
            self.h5corrtag = config.get('sidechain','h5crtag')
            self.pca_h5file = config.get('sidechain','pcafile')
            self.Nsites = config.getint('sidechain','Nsites')
            self.csv_files = config.get('sidechain','csvfiles')
            self.sc_file = None
            self.stat_file = None
            self.pca_file= None
    def __enter__(self):
        self.sc_file = None
        self.stat_file = None
        self.pca_file= None
        return self

    def InitSidechain_hdf(self, force=False):
        print "Initializing Sidechain data..."
        if not force and not query_yes_no("Are you sure you want to re-write {}?".format(self.sc_h5file), default="no"):
            print "File rewrite skipped."
            return
        Nsites = self.Nsites

        # Load all of the CSV file paths into an array
        with open(self.csv_files) as f:
            csv_files = f.readlines()
        logging.debug("Loading data from {} csv files...".format(len(csv_files)))
        
        dset_tags = [append_index(self.time_h5tag, i) for i in xrange(1,Nsites+1)]

        # Load each CSV sequentially into an HDF file that is created on the first step
        first_file=True
        for i,file in enumerate(csv_files):
            logging.debug( "File #{}: {}".format(i, file.strip()))
            if HAS_IOPRO:
                x = iopro.loadtxt(file.strip(),delimiter=',')
            else:
                x = numpy.loadtxt(file.strip(),delimiter=',')
            assert(format(x[0::Nsites].shape ==  x[(Nsites-1)::Nsites].shape))
            assert(len(x) % Nsites == 0)
            logging.debug( "\tX total shape: {}".format((x.shape[0]/Nsites, Nsites, len(x[0,:]))))
            Ntimes = len(x) / Nsites

            # Create the HDF file if we are looping through for the first time using the sizes from the csv files
            if first_file:
                Ncoarse = len(x[0])
                logging.debug("\tCreating new datasets, "+
                        "loaded file with Ncoarse = {} and Ntimes = {}".format(Ncoarse, Ntimes))
                dEhdf5_init(self.sc_h5file, None, 'w')
                for dset_tag in dset_tags:
                    dEhdf5_init(self.sc_h5file, dset_tag, 'a', Ncoarse=Ncoarse, dt_ps=.005)
                first_file=False

            h5_out =  h5py.File(self.sc_h5file,'a')
            try:
                dsets  = [h5_out[dset_tag] for dset_tag in dset_tags]
                for i,ds in enumerate(dsets):
                    oldlen = ds.shape[0]
                    newlen = oldlen + Ntimes
                    ds.resize(newlen, axis=0)
                    ds[oldlen:newlen,:] = x[i::Nsites,:]
                h5_out.close()
            except:
                logging.error( "Write failed...")
                h5_out.close()
                logging.error("Error: {}".format(sys.exc_info()[0]))
                raise
    def ExamineSidechain_hdf(self):
        should_close = False
        if not self.sc_file:
            self.sc_file = h5py.File(self.sc_h5file)
            should_close = True
        print self.sc_file.keys()
        if should_close:
            self.sc_file.close()
    def GetSidechain_hdf(self, i):
        if not self.sc_file:
            self.sc_file  = h5py.File(self.sc_h5file)
        return self.sc_file[append_index(self.time_h5tag,i)]
    def CloseSidechain_hdf(self):
        if self.sc_file:
            self.sc_file.close()
            self.sc_file = None

    def InitStats_hdf(self, force = False):
        """
        Computes mean sidechain energy, covariance of sidechain energies, and total energy gap
        """
        print "Initializing stats file..."
        if not force and not query_yes_no(
                "Are you sure you want to re-write {}?".format(self.h5stats), default="no"):
            print "File rewrite skipped."
            return
        self.stat_file = h5py.File(self.h5stats, 'w')
        self.stat_file.close()
        for i in xrange(1,self.Nsites+1):
            logging.debug( "Chromophore {}...".format(i))
            ds_i = self.GetSidechain_hdf(i)
            corr_iab, Eavg_ia = sc.ChunkCovariance(ds_i)
            total_it = sc.ChunkTotal(ds_i)
            self.stat_file = h5py.File(self.h5stats, 'a')
            self.stat_file.create_dataset(append_index(self.h5corrtag,i), data=corr_iab)
            self.stat_file.create_dataset(append_index(self.h5eavtag ,i), data=Eavg_ia)
            self.stat_file.create_dataset(append_index(self.time_h5tag ,i), data=total_it)
            self.stat_file.close()
    def GetStats_hdf(self, i):
        """
        Returns mean sidechain energy, covariance of sidechain energies, and total energy gap
        """
        if not self.stat_file:
            self.stat_file = h5py.File(self.h5stats)
        logging.debug("Keys in stat file: {}".format(self.stat_file.keys()))
        return self.stat_file[append_index(self.h5eavtag, i)], \
                self.stat_file[append_index(self.h5corrtag,i)], \
                self.stat_file[append_index(self.time_h5tag,i)]
    def CloseStats_hdf(self):
        """
        Closes stats HDF file
        """
        if self.stat_file:
            self.stat_file.close()
            self.stat_file = None

    def InitPCA_hdf(self,force=False):
        print "Initializing PCA file..."
        if not force and not query_yes_no(
                "Are you sure you want to re-write {}?".format(self.pca_h5file), default="no"):
            print "File rewrite skipped."
            return
        self.pca_file = h5py.File(self.pca_h5file, 'w')
        self.pca_file.close()
        for i in xrange(1, self.Nsites+1):
            sc_ds          = self.GetSidechain_hdf(i)
            Eav_ds,corr_ds = self.GetStats_hdf(i)
            self.pca_file = h5py.File(self.pca_h5file,'a')
            pca_ds_i = self.pca_file.create_dataset(append_index(self.time_h5tag, i), sc_ds.shape)
            PCARotate_hdf5(sc_ds, corr_ds, Eav_ds, pca_ds_i)
    def GetPCA_hdf(self,i):
        if not self.pca_file:
            self.pca_file = h5py.File(self.pca_h5file)
        return self.pca_file[append_index(self.time_h5tag, i)]
    def ClosePCA_hdf(self):
        if self.pca_file:
            self.pca_file.close()

    def __exit__(self, type, value, traceback):
        self.CloseSidechain_hdf()
        self.CloseStats_hdf()
        self.ClosePCA_hdf()

