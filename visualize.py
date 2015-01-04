"""
Several core examples of plotting for the depca module
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import depca.visualize.base_viz as viz
import depca.dedata as dedata
import depca.robust as robust
import logging
import numpy as np

def plotgapdist():
    """
    Plotting distributions for the data -- 1d and 2d histograms
    """
    with dedata.dEData('./conf.cfg') as data:
        print "Running with {}...".format(type(data))
        print "Plotting..."

        ## 1D histogram with subsampling
        #legend = None
        #site = 1
        #sidechain = data.GetPCA_hdf(site)
        #for mode in xrange(1, 10):
        #    subsamples = robust.GetSubsamples(sidechain[:, mode], .05, 10)
        #    print subsamples.shape
        #    print subsamples[0, :]
        #    viz.Plot1DHist(subsamples, plottype = 'png', 
        #        fname = '/home/jhaberstroh/Dropbox/Physics' +
        #        '/subgroup/2014-10-28/FMO{}_hist{}'.format(site,mode), 
        #        legend=legend, displace_by=0.0)

        ## 2D histogram
        #sites = np.arange(1, 5)
        #modes = np.arange(1, 5)
        #sites, modes1 = np.meshgrid(sites, modes)
        #sites, modes1 = np.array(sites).flatten(), np.array(modes1).flatten()
        #modes2 = modes1 + 1
        #for site, mode1, mode2 in zip(sites, modes1, modes2):
        #    print site, mode1, mode2
        #    sidechain = data.GetPCA_hdf(site)
        #    viz.Plot2DHist(
        #            sidechain[:, mode1], sidechain[:, mode2], 
        #            plottype = 'png', 
        #            fname = '/home/jhaberstroh/Dropbox/Physics/subgroup' +
        #            '/2014-10-28/FMO{}_dens{}-{}'.format(site, mode1, mode2))

        # Energy total series
        for i in xrange(1,8):
            legend = None
            _,_,gap_t = data.GetStats_hdf(i)
            #subsamples = robust.GetSubsamples(gap_t[:], .05, 10)
            
            #viz.Plot1DHist(subsamples, plottype = 'png', 
            #    fname = '/home/jhaberstroh/Dropbox/Physics' +
            #    '/subgroup/2014-10-28/FMO{}_gap_hist'.format(i), 
            #    legend=legend, displace_by=0.0, parabola=True)
            print len(gap_t)
            window = 4000
            roll_dt = np.zeros(len(gap_t) / window)
            for time in xrange(len(roll_dt)):
                roll_dt[time] = np.mean(gap_t[time*window:(time+1)*window])
            roll_dt -= np.mean(roll_dt)
            times = window*.005 * np.arange(0,len(roll_dt)) * .001 
            plt.plot(times, roll_dt)
            plt.title("Rolling average for chromophore {}, window of {}ps".format(i, window*.005))
            plt.xlabel("Time, ns")
            plt.ylabel("Change in Energy Gap, cm-1")
            plt.savefig('/home/jhaberstroh/Dropbox/Physics/subgroup' +
                    '/2014-10-28/FMO/FMO{}_roll.png'.format(i))
            plt.clf()






if __name__ == "__main__":
    #logging.basicConfig(level=logging.DEBUG)
    plotgapdist()
