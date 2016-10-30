import unittest
import stats.mrr as mrr


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

	def test_gm_1_1(self):
		m = 0.
		rho = 0.
		theta = 1.
		phi = 1.
		sigma = 1.

		actual_output = mrr._gm_1_1(m, rho, theta, phi, sigma)
		expected_output = 6.

		self.assertEqual(expected_output, actual_output)

class TestEstimateMoments(unittest.TestCase):

	def test_basic_case(self):

		""" Test the moments_estimation function in case of no randomness,
		e.g. all the y_i = 2 """

		n_steps = 1000
		input_epsilon =[1]*n_steps
		input_y = [2]*n_steps

		# creating an iterator from the input list
		zip_e_y = zip(input_epsilon, input_y)

		# estimation of the moments
		actual_output = mrr.estimate_moments(zip_e_y)

		# expected output
		expected_output = (4., 8., 16., 4., 4.)

		for i, j in zip(actual_output, expected_output):
			self.assertEqual(i, j)


