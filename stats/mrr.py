import random


def dar(n, rho=0.5, m=0., start=1):
	"""
	Generator of a DAR(1) process

	:param n: int, length of the series
	:param rho: float, correlation parameter
	:param m: float, average parameter
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
