import numpy as np

SU = 1
SD = -1

def _phi_interaction(x_1, x_2, J, beta):
	"""
	Exponential of the two spins interaction term

	:param x_1: int, first spin = +/- 1
	:param x_2: int, second spin = +/- 1
	:param J: float, coupling
	:param beta: float, inverse temperature
	:return: float
	"""
	return np.exp(beta*J*float(x_1)*float(x_2))


def _phi_field(x_1, h, beta):
	"""
	Exponential of the one spin interaction term

	:param x_1: int, spin = +/- 1
	:param h: float, external magnetic field
	:param beta: float, inverse temperature
	:return: float
	"""
	return np.exp(beta*h*x_1)


def _propagate_message(beta, J, h, mu_up):
	"""
	Next positive message from left

	:param beta: float, inverse temperature
	:param J: float, coupling
	:param h: float, external magnetic field
	:param mu_up: float, incoming positive message
	:return: float, new positive message
	"""

	# non normalised positive message
	tmp_up = _phi_interaction(SU, SU, J, beta) * _phi_field(SU, h, beta) * mu_up \
			 + _phi_interaction(SU, SD, J, beta) * _phi_field(SU, h, beta) * (1.-mu_up)

	# non normalised negative message
	tmp_dw = _phi_interaction(SD, SU, J, beta) * _phi_field(SD, h, beta) * mu_up \
			 + _phi_interaction(SD, SD, J, beta) * _phi_field(SD, h, beta) * (1.-mu_up)

	# normalised positive message
	return tmp_up / (tmp_up+tmp_dw)


