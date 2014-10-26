
def BootstrapPCA(E_ti, Cov_ia, fraction=.1, resample=10):
    return 0

def GetSubsamples(array1d, fraction, quantity):
    num_in_sample = int(len(array1d) * fraction)
    subsamples = np.zeros((quantity, num_in_sample))
    print len(array1d)
    print num_in_sample
    for i in xrange(quantity):
        sample_ind = random.sample(xrange(len(array1d)), num_in_sample)
        print array1d[sample_ind]
        subsamples[i,:] = array1d[sample_ind]
    return subsamples
