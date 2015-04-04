import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.stats as st
import numpy.random as rnd
import scipy
from mpl_toolkits.mplot3d import Axes3D
import time
import h5py

pi = np.pi

def eval_fourier(data, num, end):
    """
    Evaluates the first fourier coefficient of windows in a given set of data 
    and returns a 1-D array of the first fourier coefficients. 
    
    Inputs:
        data (1-D array): a 1-D time series (trajectory) 
        num (integer): corresponds to the number of points per window
        end (integer): the end time of data (can be set to 'length(data)'')

    Outputs:
        A 1-D np array of chi-square values corresponding to each window size. 
            Note that there are as many chi-square values in the return array as there
            are window sizes. And of course the number of windows depends on the length
            of the data and the parameter 'num'.

    """
    pts = len(data) # number of points in the data
    dt = round(end) / pts # amount of time between points 
    windows = pts // num # number of windows
    wintime = num * dt
    seg = divide_trajectory(np.array(data), num) # segmented data
    w = ((2*pi) / (num * dt)) # frequency based on num
    tseg = np.linspace(0, wintime - dt, num) # time steps for each window
    coeffs = []
    for i in range(windows):
        coeff = (1. / num) * np.sum(seg[i,:] * np.cos((2*pi * tseg) / wintime))
        coeffs.append(coeff)
    return np.array(coeffs)


def eval_fft(data,num,end):
    """
    Same as 'eval_fourier' function above except it uses Python's FFT method

    """
    pts = len(data)
    dt = round(end) / pts
    windows = pts // num
    wintime = num * dt
    seg = divide_trajectory(np.array(data),num)
    rcoeffs = []
    
    for i in range(windows):
        coeff = (np.fft.rfft(seg[i,:]).real)[1]
        rcoeffs.append(coeff)
    return np.array(rcoeffs)


def fourier_chi(data, nums, end, eval_method = eval_fourier, bins=100, plot=False):
    """
    Evaluates the chi-square value of Fourier Coefficient distributions for 
    specified window sizes. Each window size will result in one chi-square 
    value. 

    Inputs:
        data (1-D array): a 1-D time series 
        nums (integer array): correspond to the numbers of points per window
        end (integer): the end time of data (can be set to 'length(data)'')

    Outputs:
        A 1-D np array that corresponds to chi-square values of Fourier Coefficient
            distributions for each window size. Note that there are as many chi-square
            values in the return array as there are elements in 'nums'
    """
    if (np.array(nums) <= 1).any():
        raise ValueError("Chi squared can only be computed with samples " +
                "per chunk > 2")

    chisquared = []
    for num in nums:
        a = eval_method(data, num, end)
        chisquared.append(chisq_hist(a, bins, plot))
    return np.array(chisquared)


def divide_trajectory(trajectory, pts_per_chunk):
    """
    Reshapes a trajectory into chunks (windows) of specified size

    Inputs:
        trajectory (1-D array): a 1-D time series 
        pts_per_chunk: number of desired points per chunk 

    Outputs:
        A 2-D array that is the reshaped 'trajectory'. It has as many columns 
            as there are point per chunk and as many rows as there are chunks. 
            Note that the number of chunks is of course the total number of 
            points divided by 'pts_per_chunk'
    """
    numb_of_chunks = len(trajectory) // pts_per_chunk
    trajectory = trajectory[:pts_per_chunk * numb_of_chunks]
    expected_part = trajectory.reshape(numb_of_chunks, pts_per_chunk)
    return expected_part


def chisq_hist(data, number_bins=100, plot = False):
    """
    Evaluates the chi-square value associated with a given distribution by comparing the 
    distribution to its Gaussian envelope.

    Inputs:
        data (1-D array): a 1-D time series

    Outputs: 
        A float number that corresponds to the chi-square value of the distribution

    """

    N = len(data)
    if plot:
        plt.hist(data, number_bins, normed=False, log=True)
        plt.show()
    unnormed, bin_loc = np.histogram(data, number_bins, normed=False)
    bin_loc = adjust_bins(bin_loc)
    expect = calc_expect(data, bin_loc, number_bins)
    error = np.sqrt(unnormed)
    chisq = calc_chisq(unnormed, expect, error)
    return chisq


def calc_expect(data, bin_loc, number_bins):
    """
    This is a helper function.

    Calculates and returns a 1-D array of data corresponding to a Gaussian of the 
    same standard deviation and mean (Gaussian envelope)

    Inputs:
        data (1-D array): a 1-D time series
        bin_loc (1-D array): the x-axis location of the bins
        number_bins (integer): the number of points the Gaussian is to be generated for

    Output:
        1-D array of the expected Gaussian envelope. Note that there are as many points 
        in this Gaussian envelope as there are bins in our data's histogram.
    """
    mu, sigma = np.mean(data), np.std(data)
    expectnorm = mlab.normpdf(bin_loc, mu, sigma)
    difference = diff_factor(data, number_bins)
    expect = difference * expectnorm
    return expect


def calc_chisq(observed, expected, error):
    """
    This is a helper function.

    Calculates and returns the chi-squared value of a set of data compared
    to a Gaussian given: the data, the expected gaussian, and the error. 

    Inputs:
        observed (1-D array): the observed data which is our time series
        expected (1-D array): the expected value of the data (This is given 
                            by a gaussian of the same standard deviation and
                            mean as the data)
        error (1-D array): inherent sampling error in the data (in the case of 
                            histograms as we are using, this is given by the
                            square root of the size of the bin)
    Output:
        An integer corresponding to the chi-square of the data
    """
    N = len(observed) # total number of points in the observed distribution
    total = 0
    for i in range(N):
        if error[i] != 0:
            total += ((expected[i] - observed[i])**2 / error[i]**2)
    return total / N


def diff_factor(data, number_bins):
    """
    Evaluates the difference factor between Python's normed histogram and 
    Python's unnormed histogram (this factor is not necessarily the number 
    of points in the distribution when using plt.hist command's norm parameter). 

    Inputs:
        data (1-D array): a 1-D time series
        number_bins: number of desired bins that constructs a histogram of data

    Outputs:
        a float number that once multiplied by the normalized distribution will
            result in the unnomalized distribution. 
    """
    normed, bl0   = np.histogram(data, number_bins, normed=True)
    unnormed, bl1 = np.histogram(data, number_bins, normed=False)
    for i in range(number_bins):
        if unnormed[i] != 0:
            return unnormed[i] / normed[i]

def adjust_bins(bin_loc):
    """
    This is a helper function.

    Calculates and returns adjusted bin locations.

    When using the numpy histogram function, the bins for the histogram are
    given x-axis values that are not centered (e.g. the bin that holds all 
    points from 0 to 5 is denoted by 0). This function adjusts these locations
    such that they denote the center of the bin (changing the bin location in
    the previous example to 2.5).

    Inputs:
        bin_loc (1-D array): the bin locations to be adjusted

    Output:
        1-D array of the adjusted bin locations
    """
    bin_loc = bin_loc[:len(bin_loc) - 1]
    bin_width = bin_loc[1] - bin_loc[0]
    bin_loc = bin_loc + (bin_width / 2)
    return bin_loc


def GenerateData(field, end, number_points, ksi=1):
    """
    Generates the trajectory of an overdamped brownian particle confined to a
    potential whose derivative is specified. 

    Inputs: 
        field: either 'single well', 'double well' or 'asym'. Note that 
               one can define the derivative of 'whatever' as desired below.
        end: the time that corresponds to the end of the trajectory
        number_points: number of points that make up the trajectory
        ksi (optional): viscosity term in the langevin equation

    Output: 
        A one-dimensional array that is our desired time series (trajectory) 
    """
    types = {'single well': lambda x: -x, 
             'double well': lambda x: 4*3*(x - x**3),
             'asym': lambda x: 10*((x-.1) - (x+.1)**3) 
             }
    D = 1
    potential = types[field]
    times = np.linspace(0, end, number_points)
    dt = times[1] - times[0]
    rand = rnd.normal(0, 1, number_points)
    traj = np.zeros(number_points)
    rand_coeff = np.sqrt(2 * D * dt)
    for i in range(1, number_points):
        traj[i] = traj[i-1] + (1./ksi)*(potential(traj[i-1])*dt + rand_coeff*rand[i])
    return traj

###################
### Sample Code ###
###################

# # Uncomment this section to run #
# data = GenerateData('single well', 100, 10000)
# # Generates a time series to test #
# chisquares = fourier_chi(data, [3,4,5], len(data))
# # Calculates the chi-square values for specified window sizes (3, 4, 5) #
