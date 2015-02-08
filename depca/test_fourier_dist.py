import unittest
import fourier_dist as fourier

class TestFourier(unittest.TestCase):
    def test_chi(self):
        # Uncomment this section to run #
        # Generates a time series to test #
        data = fourier.GenerateData('single well', 100, 10000)
        # Calculates the chi-square values for specified window sizes (3, 4, 5) #
        chisquares = fourier.fourier_chi(data, range(2,100), len(data))
    def test_method_match(self):
        data = fourier.GenerateData('single well', 100, 10000)
        chi0 = fourier.fourier_chi(data, range(2,100), len(data))
        chi1 = fourier.fourier_chi(data, range(2,100), len(data),
                eval_method = fourier.eval_fft)
        for i in xrange(len(chi0)):
            self.assertAlmostEqual(chi0[i], chi1[i])


if __name__ == "__main__":
    unittest.main()
