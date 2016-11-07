import random
import numpy as np
import scipy.stats
import scipy.optimize

CL_START = 2.
CR_START = -2.
SIGMA_START = 2.
DEFAULT_VAL = -10000.

def _c_l(theta, phi):
	return theta + phi


def _c_r(rho, theta, phi):
	return -1.*(rho * theta + phi)


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
		yield epsilon_new, _c_l(theta, phi) * float(epsilon_new) + _c_r(rho, theta, phi) * float(epsilon_previous) + random.gauss(0., sigma)
		epsilon_previous = epsilon_new


def estimate_moments(sample):

	"""
	Estimate the moments of the transaction price returns of a MRR process

	:param sample: iterable of the trade initiation variable espilon_i
		and transaction price return y_i
	:return: tuple of the moments y^2, y^3, y^4, y*y(-1), y*y(-2)
	"""
	mm_1_1 = 0.
	mm_1_2 = 0.
	mm_1_3 = 0.
	mm_1_4 = 0.
	mm_1_5 = 0.

	y_p = 0.
	y_pp = 0.

	count = 0

	for e, y in sample:
		count += 1
		mm_1_1 += y**2
		mm_1_2 += y**3
		mm_1_3 += y**4
		mm_1_4 += y*y_p
		mm_1_5 += y*y_pp
		y_pp = y_p
		y_p = y

	return mm_1_1/(1.*count), mm_1_2/(1.*count), mm_1_3/(1.*count), mm_1_4/(1.*count-1.), mm_1_5/(1.*count-2.)


def evaluate_moments(m, rho, theta, phi, sigma):
	"""
	Evaluate the moments of the transaction price returns of a MRR process
	:param m: dar mean parameter
	:param rho: dar correlation parameter
	:param theta: impact parameter
	:param phi: spread parameter
	:param sigma: volatility parameter
	:return: tuple of the moments y^2, y^3, y^4, y*y(-1), y*y(-2)
	"""

	output = (
		_gm_1_1(m, rho, theta, phi, sigma), _gm_1_2(m, rho, theta, phi),
		_gm_1_3(m, rho, theta, phi, sigma), _gm_1_4(m, rho, theta, phi),
		_gm_1_5(m, rho, theta, phi)
	)

	return output


######################################################################################################


def _estimate_acv(sample, n_lags):

	"""
	Estimate the first n-lags auto-covariance of a sample
	:param sample: list, a sample of transaction price returns
	:param n_lags: int, number of lags
	:return: list, sample auto-covariance
	"""

	# empty output
	acv = []

	# 0-lag auto-covariance
	acv.append(np.cov(sample, sample)[0, 1])

	# i-lag autocovariance
	for i in range(1, n_lags+1):
		acv.append(np.cov(sample[i:], sample[:-i])[0, 1])
		#acv.append(np.corrcoef(sample[i:], sample[:-i])[0, 1] * np.var(sample))
	return acv


def c_moment_2(sample):
	""" Estimate the sample second centered moment """
	return np.var(sample)


def c_moment_4(sample):
	""" Estimate the sample fourth centered moment """
	return scipy.stats.kurtosis(sample, fisher=False) * c_moment_2(sample)**2


def _estimate_vkq(sample):

	"""
	Estimate the sample second and fourth central moments and 1-lag covariance

	:param sample: list, a sample of transaction price returns
	:return: tuple, sample moments
	"""

	# second central moment
	vv_e = np.var(sample)

	# fourth central moment
	kk_e = scipy.stats.kurtosis(sample, fisher=False) * vv_e**2

	# 1-lag covariance
	q1_e = _estimate_acv(sample, 1)[1]

	return vv_e, kk_e, q1_e


def _fit_mean(x, y):

	"""
	Average the slope of the linear fit of y vs x

	:param x: list, x values
	:param y: list, y values
	:return: float, average slope
	"""

	tmp = []
	for i in range(3, len(x)+1):
		tmp.append(scipy.stats.linregress(x[:i], y[:i]))
	tmp_2 = list(zip(*tmp))
	#return np.mean(np.exp(tmp))
	return np.mean(tmp_2[0]), np.mean(tmp_2[1])


def _estimate_rho(sample, n_lags):

	"""
	Estimate the sample rho

	:param sample: list, transaction price returns
	:param n_lags: int, number of lags for the fitting
	:return: float, rho hat

	Find the best fitting parameter rho_i that describe the decay
	of the first i lags of the sample auto-covariance function (times -1).
	Repeat the procedure for 3<i<n_lags.
	Returns the average of the estimated rho_i.
	"""

	# auto-covariance of the sample
	acv = _estimate_acv(sample, n_lags)[1:]

	# positive auto-covariance
	yy_0 = np.array([-1.*i for i in acv])

	# list of integers from 1 to n
	xx_0 = np.arange(1, len(acv)+1)

	# removing negative p_acv
	sel_pos = yy_0 > 0.
	xx_1 = xx_0[sel_pos]
	yy_1 = yy_0[sel_pos]

	if len(xx_1) < 3:
		# if the number of non-negative values is less than 3, we return nan
		# since it would be impossible to fit rho
		print('Not enough positive points, return nan')
		return np.nan
	else:
		# rho hat
		rho_sample_log = _fit_mean(xx_1, np.log(yy_1))
		#return rho_sample_log
		return np.exp(rho_sample_log[0]), np.exp(rho_sample_log[1])


######################################################################################################

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

	return tuple(scipy.optimize.fsolve(_equations, x0=params_start, args=sample_moments_rho))


def _solve_parameters(vv_sample, kk_sample, qq_sample, rho_hat):

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
	params_start = [CL_START, CR_START, SIGMA_START]

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
		#params_start[0] = np.abs(params_start[0] + np.random.normal(0., 1.))
		#params_start[1] = -1.*np.abs(params_start[1] + np.random.normal(0., 1.))
		#params_start[2] = np.abs(params_start[2] + np.random.normal(0., 1.))

	return theta_hat, phi_hat, sigma_hat


def estimate_parameters(sample, n_lag_rho):

	# moments estimation
	vv_sample, kk_sample, qq_sample = _estimate_vkq(sample)

	# rho estimation
	rho_hat = _estimate_rho(sample, n_lag_rho)

	if np.isnan(rho_hat):
		return np.nan, np.nan, np.nan, np.nan
	else:
		# parameters estimation
		theta_hat, phi_hat, sigma_hat = _solve_parameters(vv_sample, kk_sample, qq_sample, rho_hat)
		return rho_hat, theta_hat, phi_hat, sigma_hat


if __name__ == '__main__':

	n_steps = 2**18

	m = 0.
	rho = 0.7
	theta = 1.
	phi = 0.3
	sigma = 1.

	if True:

		def thacv(x, a, b):
			return a*b**x

		for i_sample in range(1):
			sample_gen = mrr(n=n_steps, m=m, rho=rho, theta=theta, phi=phi, sigma=sigma)
			sample_list = list(zip(*sample_gen))
			sample = sample_list[1]
			n_lags = 7
			yy = np.array(_estimate_acv(sample, n_lags)[1:])
			xx = np.arange(1, n_lags+1)

			out = scipy.optimize.curve_fit(thacv, xx, yy, [-1., 1.])
			aa_hat = out[0][0]
			rho_hat= out[0][1]
			print(aa_hat, rho_hat)
			qq_hat = aa_hat * rho_hat
			qq_population = _gm_1_4(0, rho, theta, phi)
			print(qq_population, qq_hat, (qq_population-qq_hat)/qq_population)

	if False:

		tmp = [[], [], [], []]
		for i in range(50):
			print(i)
			sample_gen = mrr(n=n_steps, m=m, rho=rho, theta=theta, phi=phi, sigma=sigma)
			sample_list = list(zip(*sample_gen))
			sample = sample_list[1]
			vv_sample, kk_sample, qq_sample = _estimate_vkq(sample)
			tmp[0].append(vv_sample)
			tmp[1].append(kk_sample)
			tmp[2].append(qq_sample)
			a, b = _estimate_rho(sample, 3)
			tmp[3].append(-1.*a*b)

		vv_population = _gm_1_1(0, rho, theta, phi, sigma)
		kk_population = _gm_1_3(0, rho, theta, phi, sigma)
		qq_population = _gm_1_4(0, rho, theta, phi)
		moments_population = [vv_population, kk_population, qq_population, qq_population]
		for i in [0, 1, 2, 3]:
			print(moments_population[i], np.mean(tmp[i]), np.std(tmp[i]), np.std(tmp[i])/np.mean(tmp[i]))

	if False:
		import matplotlib.pyplot as plt
		plt.clf()
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.scatter([_gm_1_4(0, rho, theta, phi)]*len(tmp), tmp)
		ax.set_xlim([-0.15,0.02])
		ax.set_ylim([-0.15,0.02])
		ax.set_xlabel(r'$Q(2)$')
		ax.set_ylabel(r'$\hat{Q}(2)$')
	#	plt.savefig('../plot/plot_04_q_2.pdf')
		plt.show()