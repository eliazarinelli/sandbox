import random

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
	return 1.-2.*m**2+m**4+4.*m**2*rho*(1.-m**2)


def _gm_1_1(m, rho, theta, phi, sigma):
	return sigma**2 + (_c_l(theta, phi)**2 + _c_r(rho, theta, phi)**2)*_c_2(m) \
			+ 2.*_c_l(theta, phi)*_c_r(rho, theta, phi)*_c_11(m, rho)


def _gm_1_2(m, rho, theta, phi):
	return (_c_l(theta, phi)**3+_c_r(rho, theta, phi)**3)*_c_3(m) \
		   + 3.*(_c_l(theta, phi)+_c_r(rho, theta, phi))*_c_21(m, rho)


def _gm_1_3(m, rho, theta, phi, sigma):
	return (_c_l(theta, phi)**4+_c_r(rho, theta, phi)**4)*_c_4(m) \
			+ 4.*_c_l(theta, phi)*_c_r(rho, theta, phi)*(_c_l(theta, phi)**2+_c_r(rho, theta, phi)**2)*_c_31(m,rho) \
			+ 6.*_c_l(theta, phi)**2*_c_r(rho, theta, phi)**2*_c_22(m,rho) \
			+ 6.*sigma**2*((_c_l(theta, phi)**2+_c_r(rho, theta, phi)**2)*_c_2(m) + 2.*_c_l(theta, phi)*_c_r(rho, theta, phi)*_c_11(m,rho)) \
			+ 3.*sigma**4


def _gm_1_4(m, rho, theta, phi, sigma):
	return (_c_l(theta, phi)**2+_c_r(rho, theta, phi)**2)*_c_11(m, rho) \
			+ _c_l(theta, phi)*_c_r(rho, theta, phi)*(_c_11(m, rho)+_c_101(m, rho))


def _gm_1_5(m, rho, theta, phi, sigma):
	return (_c_l(theta, phi)**2+_c_r(rho, theta, phi)**2)*_c_101(m, rho) \
			+ _c_l(theta, phi)*_c_r(rho, theta, phi)*(_c_11(m, rho)+_c_1001(m, rho))



def dar(n, m=0., rho=0.5, start=1):
	"""
	Generator of a DAR(1) process

	:param n: int, length of the series
	:param m: float, average parameter
	:param rho: float, correlation parameter
	:param start: int, starting value of the series
	:return: int, +1 or -1
	"""
	if rho < -1. or rho > 1.:
		raise ValueError('-1 < rho < 1')

	if m < 0. or m > 1.:
		raise ValueError('0 < m < 1')

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

	c_l = theta + phi
	c_r = -1.*(rho * theta + phi)

	epsilon_gen = dar(n+1, m=m, rho=rho, start=1)

	epsilon_previous = next(epsilon_gen)
	i = 0

	while i < n:
		epsilon_new = next(epsilon_gen)
		yield epsilon_new, c_l * float(epsilon_new) + c_r * float(epsilon_previous) + random.gauss(0., sigma)
		epsilon_previous = epsilon_new


def estimate_moments(sample):

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


if __name__ == '__main__':

	a = mrr(100, rho=0., theta=0., phi=1., sigma=0.)
	for i, j in a:
		print(i,j)