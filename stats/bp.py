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


def _propagate_message(beta, jj, hh, mu_up):
	"""
	Next positive message from left

	:param beta: float, inverse temperature
	:param jj: float, coupling
	:param hh: float, external magnetic field
	:param mu_up: float, incoming positive message
	:return: float, new positive message
	"""

	# non normalised positive message
	tmp_up = _phi_interaction(SU, SU, jj, beta) * _phi_field(SU, hh, beta) * mu_up \
			 + _phi_interaction(SU, SD, jj, beta) * _phi_field(SD, hh, beta) * (1.-mu_up)

	# non normalised negative message
	tmp_dw = _phi_interaction(SD, SU, jj, beta) * _phi_field(SU, hh, beta) * mu_up \
			 + _phi_interaction(SD, SD, jj, beta) * _phi_field(SD, hh, beta) * (1.-mu_up)

	# normalised positive message
	return tmp_up / (tmp_up+tmp_dw)


def _propagate_messages_chain(beta, coupling, field, message_start, direction):

	"""
	Propagation of the messages along the chain, from left or right

	:param beta: float, inverse temperature
	:param coupling: list, 2-body couplings J_{i,i+1}
	:param field: field, external magnetic field h_I
	:param message_start: message on the leaf
	:param direction: str, left or right
	:return: messages_out, list the propagated messages
	"""

	# Checking the consistency of coupling and field
	if len(coupling) != len(field)-1:
		raise ValueError('Length of coupling and of input non consistent')

	# Checking direction input
	if direction not in ['left', 'right']:
		raise ValueError('wrong direction: must be left or rigth')

	# Initialisation of the output
	messages_output = [message_start]

	# Initialisation of previous message to the value on the leaf
	message_previous = message_start

	if direction == 'left':
		j_h_list = zip(coupling, field[:-1])
	elif direction == 'right':
		j_h_list = zip(coupling[::-1], field[-2::-1])

	for jj, hh in j_h_list:

		# propagating messagae
		message_next = _propagate_message(beta, jj, hh, message_previous)

		# appending message to the ouptput
		messages_output.append(message_next)

		# updating previous message
		message_previous = message_next

	if direction == 'right':
		# reverting messages
		messages_output.reverse()

	return messages_output
