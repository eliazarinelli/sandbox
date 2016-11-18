import unittest
import stats.common as cm

import numpy as np


class TestAcvar(unittest.TestCase):

	def test_trivial_case(self):

		""" Auto-covariance of a constant list should be 0 """

		# generating sample
		sample_length = 1000
		sample = [1.]*sample_length

		# expected output
		n_legs = 5
		expected_output = [0.]*(n_legs+1)

		# actual output
		actual_output = cm.acvar(sample, n_legs)

		self.assertEqual(expected_output, actual_output)


class TestCmoment2(unittest.TestCase):

	def test_trivial_case(self):

		""" Variance of a sample of independent normal random variables """

		# Input, sample of independent normal random variables
		population_variance = 1.
		sample_input = np.random.normal(0., population_variance, 10000)

		# Expected output
		expected_output = population_variance

		# Estimating the sample variance
		actual_output = cm.c_moment_2(sample_input)

		# Testing that the expected and actual output are the same up to a tolerance
		tolerance = 0.05
		self.assertTrue(np.abs(actual_output-expected_output) < tolerance)


class TestCmoment4(unittest.TestCase):

	def test_trivial_case(self):

		""" 4th central momement of a sample of independent normal random variables """

		# Input, sample of independent normal random variables
		sample_input = np.random.normal(0., 0.1, 1000)

		# Expected output
		expected_output = 0.

		# Estimating the sample moment
		actual_output = cm.c_moment_4(sample_input)

		# Testing that the expected and actual output are the same up to a tolerance
		tolerance = 0.01
		self.assertTrue(np.abs(actual_output-expected_output) < tolerance)
