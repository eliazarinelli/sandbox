import unittest
import stats.mrr as mrr

import numpy as np

# Process Generation ###############################################################


class TestDar(unittest.TestCase):

	def test_rho_1(self):

		""" if rho=1 we expect all the spins to be equal to start """

		n_steps = 100
		dd = mrr.dar(n=n_steps, m=0., rho=1., start=1)
		n_up_down = sum(dd)
		self.assertEqual(n_up_down, n_steps)

		dd = mrr.dar(n=n_steps, m=0., rho=1., start=-1)
		n_up_down = sum(dd)
		self.assertEqual(n_up_down, -1*n_steps)


class TestMrr(unittest.TestCase):

	def test_no_noise(self):

		""" If sigma=0 we expect all the trade price returns to be in
		[theta*(1-rho), theta*(1+rho)+2phi, -(1+rho)theta-2phi, -theta*(1-rho)] """

		n_steps = 100
		m_in = 0.
		phi_in = 1.
		rho_in = 0.5
		theta_in = 1.
		sigma_in = 0.

		mm = mrr.mrr(n=n_steps, m=m_in, rho=rho_in, theta=theta_in, phi=phi_in, sigma=sigma_in)

		expected_values = [theta_in*(1.-rho_in), theta_in*(1.+rho_in)+2.*phi_in,
							-(1.+rho_in)*theta_in-2.*phi_in, -1.*theta_in*(1.-rho_in)]

		for i, j in mm:
			self.assertIn(j, expected_values)


# Parameter Inference ###########################################################################


class Test_fit_acv_population(unittest.TestCase):

	def test_standard_case(self):

		""" We add noise to the analytic function and retrieve the parameters """

		# Model parameters that we want to recover
		prefactor_input = 1.
		rho_input = 0.5

		# First 10 integers
		xx_input = list(range(1, 10))

		# The population autoc-ovariance function with the input model parameters
		# plus noise
		sigma_noise = 0.0001
		yy_input = [mrr._acv_population(i, prefactor=prefactor_input, rho=rho_input)
					+ np.random.normal(0., sigma_noise) for i in xx_input]

		# The expected output are the input model parameters
		expected_output = (prefactor_input, rho_input)

		# The actual output are the model parameters inferred by the fitting procedure
		actual_output = mrr._fit_acv_population(xx_input, yy_input)

		# We test that the expected and the actual output coincide up to a given tolerance
		tolerance = 0.01
		for i, j in zip(expected_output, actual_output):
			self.assertTrue(np.abs(i-j) < tolerance)


class Test_find_params(unittest.TestCase):

	def test_standard_case(self):

		"""
		We test that giving giving as starting point of the optimisation algoirthm
		close to the expected output, we algorith returns the expected output
		"""

		# Input model parameters
		rho_input = 0.5
		theta_input = 0.7
		phi_input = 0.9
		sigma_input = 1.

		# cl and cr
		c_l = mrr._c_l(theta_input, phi_input)
		c_r = mrr._c_r(rho_input, theta_input, phi_input)

		# Population moments
		vv_population = mrr._vv(c_l, c_r, sigma_input, rho_input)
		kk_population = mrr._kk(c_l, c_r, sigma_input, rho_input)
		qq_population = mrr._qq(c_l, c_r, rho_input)

		# Second input of the function
		moments_population_and_rho = [vv_population, kk_population, qq_population, rho_input]

		# First input of the function
		# The starting point for the fsolve algorithm, we consider the model inputs
		# plus noise
		expected_output = [c_l, c_r, sigma_input]
		starting_point = [i + np.random.normal(0, 0.001) for i in expected_output]

		# Actual output
		actual_output = mrr._find_params(starting_point, moments_population_and_rho)

		# Testing that the expected and actual output are the same up to a tolerance
		tolerance = 0.00001
		for i, j in zip(actual_output, expected_output):
			self.assertTrue(np.abs(i-j) < tolerance)
