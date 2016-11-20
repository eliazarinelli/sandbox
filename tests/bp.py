import unittest
import numpy as np

import stats.bp

class TestIsingChain(unittest.TestCase):

	def test_propagate_message_beta_0(self):

		"""
		Propagation of a message with beta = 0
		Regardless of the inputs, the propagated message should be 0.5
		"""

		# Inputs
		beta_input = 0.
		jj_input = 1.
		hh_input = 1.
		mu_up_list = np.linspace(0., 1., 10)

		# Expected output
		expected_output = 0.5

		for mu_up_input in mu_up_list:
			# Propagating the input message
			actual_output = stats.bp._propagate_message(beta_input, jj_input, hh_input, mu_up_input)

			# Testing that the actual and expected outputs are the same
			self.assertEqual(actual_output, expected_output)

	def test_propagate_message_j_0(self):
		"""
		Propagation of a message with j = 0
		Regardless of the inputs, the propagated message should be 0.5
		Due to the fact that the spin is isolated
		"""

		# Inputs
		beta_input = 1.
		jj_input = 0.
		hh_input = 1.
		mu_up_list = np.linspace(0., 1., 10)

		# Expected output
		expected_output = 0.5

		for mu_up_input in mu_up_list:
			# Propagating the input message
			actual_output = stats.bp._propagate_message(beta_input, jj_input, hh_input, mu_up_input)

			# Testing that the actual and expected outputs are the same
			self.assertEqual(actual_output, expected_output)

	def test_propagate_message_beta_inf(self):

		"""
		Propagation of a message with beta = inf
		If J>0 and h>>0, we expect that the propagated message is identical 1.
		If J>0 and h=0, we expect that the propagated message is identical to the input one
		If J>0 and h<<0, se expect that the propagated message is identical to 0.
		"""

		# Inputs
		beta_input = 50.
		jj_input = 1.
		hh_input = 1.
		mu_up_list = np.linspace(0.1, 1., 10)

		for mu_up_input in mu_up_list:
			# Expected output
			expected_output = 1.

			# Propagating the input message
			actual_output = stats.bp._propagate_message(beta_input, jj_input, hh_input, mu_up_input)

			# Testing that the actual and expected outputs are the same
			self.assertEqual(actual_output, expected_output)

		# Inputs
		beta_input = 50.
		jj_input = 1.
		hh_input = -1.
		mu_up_list = np.linspace(0.1, 0.9, 10)

		for mu_up_input in mu_up_list:
			# Expected output
			expected_output = 0.

			# Propagating the input message
			actual_output = stats.bp._propagate_message(beta_input, jj_input, hh_input, mu_up_input)

			# Testing that the actual and expected outputs are the same
			self.assertAlmostEquals(actual_output, expected_output)

		# Inputs
		beta_input = 50.
		jj_input = 1.
		hh_input = 0.
		mu_up_list = np.linspace(0.1, 1., 9)

		for mu_up_input in mu_up_list:
			# Expected output
			expected_output = mu_up_input

			# Propagating the input message
			actual_output = stats.bp._propagate_message(beta_input, jj_input, hh_input, mu_up_input)

			# Testing that the actual and expected outputs are the same
			self.assertAlmostEquals(actual_output, expected_output)
