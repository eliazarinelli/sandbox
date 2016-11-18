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
