import random
import numpy as np
import scipy.stats
import scipy.optimize
import common

DEFAULT_VAL = -10000.

START_CL = 2.
START_CR = -2.
START_SIGMA = 2.
START_PREFACTOR = -1.
START_RHO = 0.5


# Process generation functions ###################################################


def _c_l(theta, phi):
	return theta + phi


def _c_r(rho, theta, phi):
	return -1.*(rho * theta + phi)


def dar(n, m=0., rho=0.5, start=1):

	"""
	Generator of a DAR(1) process

	:param n: int, length of the series
	:param m: float, average parameter
	:param rho: float, correlation parameter
	:param start: int, starting value of the series
	:return: int, +1 or -1
	"""

	if rho < 0. or rho > 1.:
		raise ValueError('0 < rho < 1')

	if m < -1. or m > 1.:
		raise ValueError('-1 < m < 1')

	if start not in [-1, 1]:
		raise ValueError('start should be -1 or +1')

	epsilon_previous = start
	i = 0

	while i < n:

		vv = 1
		if random.random() > rho:
			vv = 0

		zeta = 1
		if random.random() > (1.+m)/2.:
			zeta = -1

		epsilon_new = vv * epsilon_previous + (1-vv) * zeta

		yield epsilon_new

		epsilon_previous = epsilon_new
		i += 1


def mrr(n, m=0., rho=0.5, theta=1., phi=1., sigma=1.):

	"""
	Generator of the MRR process

	:param n: number of steps
	:param m: dar mean parameter
	:param rho: dar correlation parameter
	:param theta: impact parameter
	:param phi: spread parameter
	:param sigma: volatility parameter
	:return: the trade initiation variable epsilon_i and the price return y_i
	"""

	epsilon_gen = dar(n+1, m=m, rho=rho, start=1)

	epsilon_previous = next(epsilon_gen)
	i = 0

	while i < n:

		epsilon_new = next(epsilon_gen)
		yield epsilon_new, _c_l(theta, phi) * float(epsilon_new) + \
			  _c_r(rho, theta, phi) * float(epsilon_previous) \
			  + random.gauss(0., sigma)
		epsilon_previous = epsilon_new


# Moments estimation functions #########################################################


def _acv_population(x, prefactor, rho):
	""" Functional form of the auto-covariance function """
	return prefactor * rho**x


def _fit_acv_population(xx, acv_sample):
	"""
	Find the best fitting parameters of the auto-covariance function
	:param xx: np.array, i-th lag
	:param acv_sample: np.array, i-th sample auto-covariance
	:return: prefactor and rho
	"""

	if len(xx) != len(acv_sample):
		raise ValueError('xx and yy have different length')

	# casting to np.array
	xx_in = np.array(xx)
	acv_sample_in = np.array(acv_sample)

	# non-linear fit
	out = scipy.optimize.curve_fit(_acv_population, xx_in, acv_sample_in, [START_PREFACTOR, START_RHO])

	# return the prefactor and rho
	return out[0][0], out[0][1]


def rho_qq_hat(sample, n_lags):
	"""
	Estimate rho and the 1-lag sample auto-covariance

	:param sample: list, sample
	:param n_lags: int
	:return: rho_hat, qq_hat
	"""

	# estimate the auto-covariance function
	acv_sample_0 = common.acvar(sample, n_lags)

	# remove the first point
	acv_sample = np.array(acv_sample_0[1:])

	# first n integers
	xx = np.arange(1, n_lags+1)

	# fit the auto-covariance
	prefactor_hat, rho_hat = _fit_acv_population(xx, acv_sample)

	return rho_hat, prefactor_hat*rho_hat


# Parameter Inference Functions #########################################################


def _vv(x, y, s, r):
	return s**2 + (x+y)**2 + 2.*(r-1.)*x*y


def _qq(x, y, r):
	return r*(x+y)**2 + x*y*(1-r)**2


def _kk(x, y, s, r):
	return (x+y)**4 + 4.*(r-1.)*x*y*(x**2+y**2) + 6.*s**2*((x+y)**2 + 2.*(r-1.)*x*y) + 3.*s**4


def _equations(params, sample_moments_rho):

	cl, cr, sigma = params
	vv_s, kk_s, qq_s, rho_s = sample_moments_rho
	return (_vv(cl, cr, sigma, rho_s) - vv_s,
			_kk(cl, cr, sigma, rho_s) - kk_s,
			_qq(cl, cr, rho_s) - qq_s)


def _find_params(params_start, *sample_moments_rho):

	"""
	Find the parameters cl, cr and sigma that better explain the sample moments
	:param params_start: list, starting point of the fsovle algorithm
	:param sample_moments_rho: list, estimated sample moments vv, kk and qq and the estimated rho
	:return: tuple, inferred cl, cr and sigma
	"""
	try:
		out = tuple(scipy.optimize.fsolve(_equations, x0=params_start, args=sample_moments_rho))
	except RuntimeWarning:
		out = (-1000., -1000., -1000.)
	return out


def th_ph_s_hat(vv_sample, kk_sample, qq_sample, rho_hat):

	"""
	Find the parameters theta, phi and sigma that better explain the sample moments
	:param vv_sample: float, sample centered second moment
	:param kk_sample: float, sample centered fourth moment
	:param qq_sample: float, sample 1-lag covariance
	:param rho_hat: float, estimated paramete
	:return: tuple, inferred theta, phi and sigma
	"""

	# tuple of sample moments and rho_hat
	moments_rho_sample = (vv_sample, kk_sample, qq_sample, rho_hat)

	# starting value of the inference parameter
	params_start = [START_CL, START_CR, START_SIGMA]

	# default value of the sample parameters
	theta_hat = DEFAULT_VAL
	phi_hat = DEFAULT_VAL
	sigma_hat = DEFAULT_VAL

	while theta_hat < 0. or phi_hat < 0. or sigma_hat < 0.:

		# find the parameters cl, cr and sigma that better explain the sample moments
		cl_hat, cr_hat, sigma_hat = _find_params(params_start, moments_rho_sample)

		# retrieving theta from cl and cr
		theta_hat = (cl_hat+cr_hat)/(1.-rho_hat)

		# retrieving phi from cl and cr
		phi_hat = cl_hat - theta_hat

		# new starting parameters
		params_start = [i + np.random.normal(0., 0.1) for i in params_start]

	return theta_hat, phi_hat, sigma_hat


# Auxiliary functions #####################################################################


def _c_2(m):
	return 1. - m**2


def _c_11(m, rho):
	return rho*(1.-m**2)


def _c_101(m, rho):
	return rho**2*(1.-m**2)


def _c_1001(m, rho):
	return rho**3*(1.-m**2)


def _c_3(m):
	return -2.*m*(1.-m**2)


def _c_21(m, rho):
	return -2.*m*(1-m**2)*rho


def _c_4(m):
	return (1.+3.*m**2)*(1.-m**2)


def _c_31(m, rho):
	return rho*(1.+3.*m**2)*(1.-m**2)


def _c_22(m, rho):
	return 1. - 2.*m**2 + m**4 + 4.*m**2*rho*(1.-m**2)


def _gm_1_1(m, rho, theta, phi, sigma):
	return sigma**2 + (_c_l(theta, phi)**2 + _c_r(rho, theta, phi)**2)*_c_2(m) \
		   + 2.*_c_l(theta, phi)*_c_r(rho, theta, phi)*_c_11(m, rho)


def _gm_1_2(m, rho, theta, phi):
	return (_c_l(theta, phi)**3 + _c_r(rho, theta, phi)**3)*_c_3(m) \
		   + 3.*(_c_l(theta, phi) + _c_r(rho, theta, phi))*_c_21(m, rho)


def _gm_1_3(m, rho, theta, phi, sigma):
	return (_c_l(theta, phi)**4 + _c_r(rho, theta, phi)**4)*_c_4(m) \
			+ 4.*_c_l(theta, phi)*_c_r(rho, theta, phi)*(_c_l(theta, phi)**2 + _c_r(rho, theta, phi)**2)*_c_31(m,rho) \
			+ 6.*_c_l(theta, phi)**2*_c_r(rho, theta, phi)**2*_c_22(m,rho) \
			+ 6.*sigma**2*((_c_l(theta, phi)**2 + _c_r(rho, theta, phi)**2)*_c_2(m) + 2.*_c_l(theta, phi)*_c_r(rho, theta, phi)*_c_11(m,rho)) \
			+ 3.*sigma**4


def _gm_1_4(m, rho, theta, phi):
	return (_c_l(theta, phi)**2 + _c_r(rho, theta, phi)**2)*_c_11(m, rho) \
			+ _c_l(theta, phi)*_c_r(rho, theta, phi)*(_c_2(m) + _c_101(m, rho))


def _gm_1_5(m, rho, theta, phi):
	return (_c_l(theta, phi)**2 + _c_r(rho, theta, phi)**2)*_c_101(m, rho) \
			+ _c_l(theta, phi)*_c_r(rho, theta, phi)*(_c_11(m, rho) + _c_1001(m, rho))

def c_moment_2_th(m, rho, theta, phi, sigma):
	return _gm_1_1(m, rho, theta, phi, sigma)


def c_moment_4_th(m, rho, theta, phi, sigma):
	return _gm_1_3(m, rho, theta, phi, sigma)


def acvar_1_th(m, rho, theta, phi):
	return _gm_1_4(m, rho, theta, phi)
