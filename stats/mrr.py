import random


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

	c_l = theta + phi
	c_r = -1.*(rho * theta + phi)

	epsilon_gen = dar(n+1, m=m, rho=rho, start=1)

	epsilon_previous = next(epsilon_gen)
	i = 0

	while i < n:
		epsilon_new = next(epsilon_gen)
		yield epsilon_new, c_l * float(epsilon_new) + c_r * float(epsilon_previous) + random.gauss(0., sigma)
		epsilon_previous = epsilon_new

if __name__ == '__main__':

	a = mrr(100, rho=0., theta=0., phi=1., sigma=0.)
	for i, j in a:
		print(i,j)