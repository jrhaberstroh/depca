import h5py
import ConfigParser
import sys
import iopro
import depca.sidechain_corr as sc

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

    def InitSidechain_hdf(self):
        print "Initializing Sidechain data..."
        if not query_yes_no("Are you sure you want to re-write {}?".format(self.sc_h5file), default="no"):
            print "File rewrite skipped."
            return
        Nsites = self.Nsites

        # Load all of the CSV file paths into an array
        with open(self.csv_files) as f:
            csv_files = f.readlines()
        print("Loading data from {} csv files...".format(len(csv_files)))
        
        dset_tags = [append_index(self.time_h5tag, i) for i in xrange(1,Nsites+1)]

        # Load each CSV sequentially into an HDF file that is created on the first step
        first_file=True
        for i,file in enumerate(csv_files):
            print "File #{}: {}".format(i, file.strip())
            x = iopro.loadtxt(file.strip(),delimiter=',')
            assert(format(x[0::Nsites].shape ==  x[(Nsites-1)::Nsites].shape))
            assert(len(x) % Nsites == 0)
            print "\tX total shape: {}".format((x.shape[0]/Nsites, Nsites, len(x[0,:])))
            Ntimes = len(x) / Nsites

            # Create the HDF file if we are looping through for the first time using the sizes from the csv files
            if first_file:
                Ncoarse = len(x[0])
                print "\tCreating new datasets, loaded file with Ncoarse = {} and Ntimes = {}".format(Ncoarse, Ntimes)
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
                print "Write failed..."
                h5_out.close()
                print("Error: {}".format(sys.exc_info()[0]))
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

    def InitStats_hdf(self):
        #raise NotImplementedError("Initializing stats not implemented. Just use raw sidechain data for now.")
        # TODO: Rewrite this to use the AvgAndCorrelateSidechains that does NOT have the index for chromophore
        # Perform the computation
        self.stat_file = h5py.File(self.h5stats, 'w')
        self.stat_file.close()
        for i in xrange(1,self.Nsites+1):
            print "Chromophore {}...".format(i)
            ds_i = self.GetSidechain_hdf(i)
            corr_iab, Eavg_ia = sc.ChunkCovariance(ds_i)
            self.stat_file = h5py.File(self.h5stats, 'a')
            self.stat_file.create_dataset(append_index(self.h5corrtag,i), data=corr_iab)
            self.stat_file.create_dataset(append_index(self.h5eavtag ,i), data=Eavg_ia)
            self.stat_file.close()
    def GetStats_hdf(self, i):
        if not self.stat_file:
            self.stat_file = h5py.File(self.h5stats)
        return self.stat_file[append_index(self.h5eavtag, i)], self.stat_file[append_index(self.h5corrtag,i)]
    def CloseStats_hdf(self):
        if self.stat_file:
            self.stat_file.close()
            self.stat_file = None
    
    def ComputePCA():
        self.pca_file = h5py.File(self.h5pca, 'w')
        self.pca_file.close()
        for i in xrange(1, self.Nsites+1):
            sc_ds          = self.GetSidechain_hdf(i)
            corr_ds,Eav_ds = self.GetStats_hdf(i)

            self.pca_file = h5py.File(self.pca_h5file,'a')
            slef.pca_file.create_dataset(append_index(self.h5timetag, i), sc_ds.shape)

            print "Computing Modes..."
            eigval_n, eigvec_nj, impact_n = sc.ComputeModes(corr)
            print "Eigenvector dimension: {}".format(eigvec_nj.shape)
    
            #conv.ApplyPCA_hdf5(sc_ds, Eav_ij, eigvec_inj, pca_h5file, time_h5tag, site=0, overwrite=True)
    
            #GB = float(1E9)
            #RAM_GB = .5
            #RAM_nfloat = RAM_GB * 1E9 / 8
            #RAM_ntimes = RAM_nfloat / Ncoarse
            #RAM_nchunk = int( np.ceil((tf - t0) / float(RAM_ntimes)) )
            #RAM_time_per_chunk = (tf - t0) / RAM_nchunk
            #print "Number of chunks needed: {} of {} GB each".format(RAM_nchunk, RAM_time_per_chunk * Ncoarse * 8 / GB)
            #RAM_return_times = (RAM_nchunk*RAM_time_per_chunk)
            #RAM_return_tot = (RAM_nchunk*RAM_time_per_chunk) * 8 * (Ncoarse)
            #print "Disk space needed for output data: {} GB".format(RAM_return_tot / GB)
            #dE_t_j = np.zeros((RAM_time_per_chunk,Ncoarse))
            #return_modes_nt = np.zeros((Ncoarse, RAM_return_times))
            #dDEresidual_t = np.zeros((RAM_return_times))
            #for chunk_num in xrange(RAM_nchunk):
            #    print "Chunk {}:".format(chunk_num+1),;sys.stdout.flush()
            #    # Each chunk reads [t0_chunk, tf_chunk)
            #    t0_chunk = (  chunk_num   * RAM_time_per_chunk) + t0
            #    tf_chunk = ((chunk_num+1) * RAM_time_per_chunk) + t0
            #    t0_return = t0_chunk - t0
            #    tf_return = tf_chunk - t0
            #    print "Loading chunk into RAM...",; sys.stdout.flush()
            #    RAM_E_t_ij = E_t_ij[t0_chunk:tf_chunk,site,:]

            #    @vectorize(['float64(float64,float64)'], target='cpu')
            #    def ParallelSub64(a,b):
            #        return a - b
            #    @vectorize(['float32(float32,float32)'], target='cpu')
            #    def ParallelSub32(a,b):
            #        return a - b
            #    # Build dE for chunk
            #    print "Computing chunk dE...",; sys.stdout.flush()
            #    try:
            #        RAM_dE_t_j = ParallelSub32( RAM_E_t_ij, E_avg_ij[site,:])
            #    except TypeError:
            #        try:
            #            print "32 bit parallel float computation failed, trying 64 bit...",;sys.stdout.flush()
            #            RAM_dE_t_j = ParallelSub64( RAM_E_t_ij, E_avg_ij[site,:])
            #            print "Success."
            #        except TypeError as e:
            #            print "64 bit failed."
            #            raise e
            #    print "Rotating chunk...",; sys.stdout.flush()
            #    RAM_dE_rotated = np.inner(RAM_dE_t_j[:,:], modeweight_inj[site,:,:].T)
            #    print "Writing chunk...",; sys.stdout.flush()
            #    with h5py.File(hdf_file, 'a') as pca_file:
            #        pca_tin = pca_file[hdf_dsname]
            #        pca_tin[t0_chunk:tf_chunk, site, :] = RAM_dE_rotated[:,:]
            #    print 'Chunk {}  done'.format(chunk_num+1)
            sc_file.close()
            stat_file.close()
    def InitPCA_hdf(self):
        raise NotImplementedError("Initializing stats not implemented. Just use raw sidechain data for now.")
    def GetPCA_hdf(self):
        if not self.pca_file:
            self.pca_file = h5py.File(self.pca_h5file)
        return self.pca_file[self.time_h5tag]
    def ClosePCA_hdf(self):
        if self.pca_file:
            self.pca_file.close()
            self.pca_file = None

    def __exit__(self, type, value, traceback):
        self.CloseSidechain_hdf()
        self.CloseStats_hdf()
        self.ClosePCA_hdf()

