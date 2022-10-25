"""
A module to hold some maths or statistics oriented functions woring on numpy arrays.
"""

import numpy as np
import scipy.stats


import logging
logger = logging.getLogger(__name__)


def rmsd(x, y):
	"""
	Returns the RMSD between two numpy arrays
	(only beginners tend to inexactly call this the RMS... ;-)
	
	http://en.wikipedia.org/wiki/Root-mean-square_deviation
	
	This function also works as expected on masked arrays.	
	"""
	return np.sqrt(np.mean((x - y)**2.0))

def rmsd_delta(delta):
	"""
	Idem, but directly takes the estimation errors.
	Useful, e.g., to pass as reduce_C_function for hexbin plots.
	"""
	return np.sqrt(np.mean(np.asarray(delta)**2.0))


def linreg(x, y, prob=0.68):
	"""
	A linear regression y = m*x + c, with confidence intervals on m and c.
	
	As a safety measure, this function will refuse to work on masked arrays.
	Indeed scipy.stats.linregress() seems to silently disregard masks...
	... and as a safety measure, we compare against scipy.stats.linregress().
	
	"""
	
	if len(x) != len(y):
		raise RuntimeError("Your arrays x and y do not have the same size")
	
	if np.ma.is_masked(x) or np.ma.is_masked(y):
		raise RuntimeError("Do not give me masked arrays")
	
	n = len(x)
	xy = x * y
	xx = x * x
	
	b1 = (xy.mean() - x.mean() * y.mean()) / (xx.mean() - x.mean()**2)
	b0 = y.mean() - b1 * x.mean()
	
	#s2 = 1./n * sum([(y[i] - b0 - b1 * x[i])**2 for i in xrange(n)])
	s2 = np.sum((y - b0 - b1 * x)**2) / n
	
	alpha = 1.0 - prob
	c1 = scipy.stats.chi2.ppf(alpha/2.,n-2)
	c2 = scipy.stats.chi2.ppf(1-alpha/2.,n-2)
	#print 'the confidence interval of s2 is: ',[n*s2/c2,n*s2/c1]
	
	c = -1 * scipy.stats.t.ppf(alpha/2.,n-2)
	bb1 = c * (s2 / ((n-2) * (xx.mean() - (x.mean())**2)))**.5
	#print 'the confidence interval of b1 is: ',[b1-bb1,b1+bb1]
	
	bb0 = c * ((s2 / (n-2)) * (1 + (x.mean())**2 / (xx.mean() - (x.mean())**2)))**.5
	#print 'the confidence interval of b0 is: ',[b0-bb0,b0+bb0]
	
	ret = {"m":b1-1.0, "c":b0, "merr":bb1, "cerr":bb0}
	
	# A little test (for recent numpy, one would use np.isclose() for this !)
	#slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
	'''
	if not abs(slope - b1) <= 1e-6 * abs(slope):
		raise RuntimeError("Slope error, %f, %f" % (slope, b1))
	if not abs(intercept - b0) <= 1e-6 * abs(intercept):
		raise RuntimeError("Intercept error, %f, %f" % (intercept, b0))
	'''
	return ret


def linregw(x,y,w, sigma_shape=0.0):
        from scipy.optimize import curve_fit
        #sigma = sigma_shape * np.sqrt(1.0/np.clip(w, 1e-18, 1e18))
        sigma = np.sqrt(sigma_shape**2 + 1.0/np.clip(w, 1e-18, 1e18))
        #sigma = 1.0/np.sqrt(np.clip(w, 1e-18, 1e18))
        #sigma = 1.0/(np.clip(np.power(w,10), 1e-18, 1e18))
        #sigma = 1.0/np.clip(w, 1e-18, 1e18)

        absolute_sigma=False

        def f(x, a, b): return a * x + b

        p0 = [1.0, 0.0] # initial parameter estimate
        popt, pcov = curve_fit(f, x, y, p0, sigma, absolute_sigma=absolute_sigma)
        perr = np.sqrt(np.diag(pcov))
        
        m = popt[0] - 1.0
        c = popt[1]
        merr = perr[0]
        cerr = perr[1]
        ret = {"m":m, "c":c, "merr":merr, "cerr":cerr}
        return ret 

