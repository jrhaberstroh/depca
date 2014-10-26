import numpy as np
import h5py
import matplotlib.pyplot as plt
import numpy.linalg as LA
import sys
from copy import deepcopy
import logging

def BootstrapPCA(E_ti, Cov_ia, fraction=.1, resample=10):
    return 0


def ChunkCovariance(E_ti, t0=0, tf=None, t_stride=1):
    if tf == None:
        tf = E_ti.shape[0]
    numFrames = min(tf-t0, E_ti.shape[0]-t0) / t_stride
    tf = t0 + (numFrames * t_stride)
    logging.debug( "\tComputing covariance with times {} - {}, stride of {}".format(t0, tf, t_stride))
    logging.debug( "\tArray size: ({},{})".format( E_ti.shape[0], E_ti.shape[1] ))
    max_floats = 4E9 / 8
    max_times = max_floats / float(E_ti.shape[1])
    dset_times = (tf-t0) / t_stride
    chunks = int(np.ceil(dset_times / max_times))
    chunk_size = dset_times/chunks
    logging.debug( "\tNumber of chunks = {},".format(chunks))
    logging.debug( "Chunk size: {} [{:.1f} GB]".format(chunk_size, chunk_size*8*E_ti.shape[1]/1E9))
    logging.debug( "\tTruncated datapoints: {}".format((tf - t0) - (chunk_size * chunks * t_stride)))
    if (chunks > 1):
        raise NotImplementedError("No chunk feature yet implemented")
    logging.debug( "\tLoading data...")
    RAM_Datasubset = E_ti[t0:tf:t_stride,:]
    logging.debug( "Computing covariance...")
    cov = np.cov(RAM_Datasubset, rowvar=0 )
    logging.debug( "Computing mean...")
    mean = RAM_Datasubset.sum(axis=0)
    mean /= numFrames
    logging.debug( "Done.")
    return cov, mean


def AvgAndCorrelateSidechains(E_t_ia, fEnd = None, fStart = 0, fStride=1):
    if not fEnd:
        fEnd = E_t_ia.shape[0]
    assert(fStart < fEnd)
    numFrames = min(fEnd-fStart, E_t_ia.shape[0]-fStart) / fStride
    fEnd = fStart + (numFrames * fStride)
    
    num_chromo = E_t_ia.shape[1]
    num_vars   = E_t_ia.shape[2]
    Corr_i_ab = np.zeros( (num_chromo, num_vars, num_vars))
    AvgEia  = np.zeros( (num_chromo, num_vars) )
    
    print "Assuming 4GB RAM usage for chunks..."  
    
    for i in xrange(num_chromo):
        site_num = i + 1
        print "Chromophore {}".format(site_num)
        print "\tArray size: ({},{})".format(E_t_ia.shape[2], E_t_ia.shape[0])
        max_floats = 4E9 / 8
        max_times = max_floats / E_t_ia.shape[2]
        dset_times = (fEnd-fStart) / fStride
        chunks = int(np.ceil(dset_times / max_times))
        chunk_size = dset_times/chunks
        print "\tNumber of chunks = {},".format(chunks),
        print "Chunk size: {} [{:.1f} GB]".format(chunk_size, chunk_size*8*E_t_ia.shape[2]/1E9)
        print "\tTruncated datapoints: {}".format((fEnd - fStart) - (chunk_size * chunks * fStride))
        if (chunks > 1):
            raise NotImplementedError("No chunk feature yet implemented")
        print "\tLoading data for chromophore {}...".format(site_num), ; sys.stdout.flush()
        RAM_Datasubset = E_t_ia[fStart:fEnd:fStride,i,:]
        print "Computing covariance...", ;sys.stdout.flush()
        Corr_i_ab[i,:,:] = np.cov(RAM_Datasubset, rowvar=0 )
        print "Computing mean...", ;sys.stdout.flush()
        AvgEia[i,:]  = RAM_Datasubset.sum(axis=0)
        AvgEia[i,:]  /= numFrames
        print "Done."
    return Corr_i_ab, AvgEia

def ComputeModes(corr, sort=True):
	"""
	\input   corr	- a lenth Ni array of two-dimensional correlation matrix, dimension M

	\output  vi     - Ni x N eigenvalues
	         wi	- Ni x N eigenvectors of length N 
		impact_i- Ni x N values of the scale factors to conserve energy

	All outputs are sorted by vi*impact_i, with pairings preserved
	All eigenvectors are selected to have positive sum of components, and a factor of -1 is applied to those which do not.
	"""
	#print "Computing Eigenvalues..."
        v,w = LA.eigh(corr)
        w = w.transpose()

        # NOTE: impact will take the same sign as the eigenvector, which is allowed since eigenvectors are still e-vecs under scaling.
	# impact is the mode-specific weighting factor to conserve energy under basis rotation
        impact = []
        for vn,wn in zip(v, w):
            #wn /= LA.norm(wn)
            n_factor = sum(wn)
            if (vn < 0 and n_factor > 0) or (vn > 0 and n_factor < 0):
                wn *= -1
                n_factor *= -1
            #print n_factor
            impact.append(n_factor)

	# --------------------------SORT-------------------------------
        if sort:
	    w_copy = deepcopy(w)
	    tup = zip(v, impact, w_copy)
            # Sort as a group and assign back to the original variables
	    tup.sort(key=lambda x: x[0] * x[1] * x[1])
	    for j in xrange(len(tup)):
	            v[j] = tup[j][0]
	            impact[j] = tup[j][1]
	            w[j] = tup[j][2]

	return v, w, impact



def TimeCorr(f1, f2):
    assert f1.size == f2.size
    if any(f1 != 0.0) or any(f2 != 0.0):
        corrarr = np.correlate(f1, f2, 'full')[f1.size-1:]
        numavg  = corrarr.size - np.arange(corrarr.size)
        corrarr /= numavg
        return corrarr

def hdfSpectralCorr(hdf_file, timetag):
    site = 0
    sc1 = 3
    sc2 = 8
    offset = 0
    nframes = 50000
    start = offset
    stop  = offset + nframes
    with h5py.File(hdf_file, 'r') as f:
        E_tij = f[timetag]
        dt = E_tij.attrs['dt']
        print E_tij.shape
        f1_t = E_tij[start:stop,site,sc1] - np.mean(E_tij[:,site,sc1])
        f2_t = E_tij[start:stop,site,sc2] - np.mean(E_tij[:,site,sc2])
        mycorr = TimeCorr(f1_t, f2_t)
        mycorr = mycorr[0:len(mycorr)/10]
# SPECTRAL DENSITY CODE
#        Jw = np.absolute(np.fft.rfft(mycorr))
#        w_cm  = np.fft.rfftfreq(len(mycorr),dt) * 5.308 * 2. * np.pi
#        print w_cm.shape
#        print Jw.shape
#        #plt.plot(w_cm, Jw * w_cm)
#        N = 16
#        smooth = pd.rolling_mean(Jw * w_cm, N)
#        smooth = np.roll(smooth, -N/2)
#        plt.plot(w_cm, smooth, linewidth=2)
#        plt.show()
    

def HDFStoreCt_v1(hdf_file, timetag, cttag, site_a, site_b, chromo=1, offset = 0, nframes = 200000):
    start = offset
    stop  = offset + nframes
    chromo -= 1
    with h5py.File(hdf_file, 'r') as f:
        E_tij = f[timetag]
        dt = E_tij.attrs['dt']
        print E_tij.shape
        f1_t = E_tij[start:stop,chromo,site_a] - np.mean(E_tij[:,chromo,site_a])
        f2_t = E_tij[start:stop,chromo,site_b] - np.mean(E_tij[:,chromo,site_b])
    mycorr = TimeCorr(f1_t, f2_t)
    mycorr = mycorr[0:len(mycorr)/10]
    return mycorr
    #plt.plot(mycorr)
    #plt.show()
    
    #with h5py.File(hdf_file, 'r') as f:
    #    C_abt = f[cttag]


def HDFStoreCt_v2(hdf_file, timetag, cttag, site_a, site_b, chromo=1, offset = 0, nframes = 200000, corr_len=20000):
    start = offset
    stop  = offset + nframes
    chromo -= 1
    mycorr = np.zeros((corr_len))

    f1_t = np.zeros((stop-start))
    f2_t = np.zeros((stop-start))
    #lock.acquire()
    try:
        with h5py.File(hdf_file, 'r') as f:
            E_tij = f[timetag]
            dt = E_tij.attrs['dt']
            print E_tij.shape
            if (start >= E_tij.shape[0]):
                raise RuntimeError("offset > len(E_tij) [{} > {}]; Use a smaller offset.".format(start,E_tij.shape[0]))
            if (stop > E_tij.shape[0]):
                raise RuntimeError("offset + nframes > len(E_tij) [{} > {}]; Use a smaller offset or fewer frames.".format(stop,E_tij.shape[0]))
            f1_t[:] = E_tij[start:stop,chromo,site_a] - np.mean(E_tij[:,chromo,site_a])
            f2_t[:] = E_tij[start:stop,chromo,site_b] - np.mean(E_tij[:,chromo,site_b])
    except RuntimeError as re:
        #lock.release()
        raise re

    #lock.release()
    #mycorr = TimeCorr(f1_t, f2_t)
    for tc in xrange(corr_len):
        mycorr[tc] = np.inner(f1_t[0:nframes-tc],f2_t[tc:nframes]) / (nframes - tc)
    return mycorr


def HDFStoreCt_v3(hdf_file, timetag, cttag, site_a, site_b, chromo=1, offset = 0, nframes = 200000, corr_len=20000):
    start = offset
    stop  = offset + nframes
    chromo -= 1

    f1_t = np.zeros((stop-start))
    f2_t = np.zeros((stop-start))
    with h5py.File(hdf_file, 'r') as f:
        E_tij = f[timetag]
        dt = E_tij.attrs['dt']
        print E_tij.shape
        if (start >= E_tij.shape[0]):
            raise RuntimeError("offset > len(E_tij) [{} > {}]; Use a smaller offset.".format(start,E_tij.shape[0]))
        if (stop > E_tij.shape[0]):
            raise RuntimeError("offset + nframes > len(E_tij) [{} > {}]; Use a smaller offset or fewer frames.".format(stop,E_tij.shape[0]))
        f1_t[:] = E_tij[start:stop,chromo,site_a] - np.mean(E_tij[:,chromo,site_a])
        f2_t[:] = E_tij[start:stop,chromo,site_b] - np.mean(E_tij[:,chromo,site_b])

    numavg  = nframes - np.arange(corr_len)
    mycorr = np.real(np.fft.ifft( np.fft.fft(f2_t) * np.conj(np.fft.fft(f1_t)) )[0:corr_len])/numavg

    return mycorr

