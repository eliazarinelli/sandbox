import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
import bp
import collections

if True:
	NN = 2 ** 17
	parameters_in = {'mm': 0.3,
	                 'rho': 0.7,
	                 'sigma': 1.,
	                 'theta': 3.,
	                 'phi': 0.5}

	epsilon_true = bp.generate_dar(parameters_in, NN)

	yy = bp.generate_observations('HIM', parameters_in, epsilon_true)

	mm_true = bp.calculate_moments('HIM', parameters_in)

	mm_est = bp.estimate_moments(yy)

	sigma_m = np.sqrt(mm_est['gamma_2'])
	parameters_range = {'mm': [-1., 1., 10],
	                    'rho': [0., 0.9, 10],
	                    'sigma': [0., 4. , 20],
	                    'theta': [0., 4. , 20],
	                    'phi': [0., 2. , 20]}

	lambda_0 = .0025
	cc, parameters_out = bp.estimate_parameters('HIM', yy, parameters_range, lambda_0)
	print(cc)

	for i in parameters_out.keys():
		print(i, parameters_in[i], np.median(parameters_out[i]))

	counter=collections.Counter(parameters_out['sigma'])
	plt.plot(counter.keys(), counter.values(), 'bo')
	plt.show()


if False:
	name_stock = 'AAPL'
	path_input = '/Users/eliazarinelli/Desktop/ising_classification/lobster/data/'

	# reading data
	epsilon_true, p_star = bp.read_data(path_input, name_stock)

	NN = len(epsilon_true)

	p_star = p_star/100.

	yy = p_star[1:]-p_star[:-1]

	mm_est = bp.estimate_moments(yy)

	mm_est_in = [mm_est['gamma_1'], mm_est['gamma_2'], mm_est['gamma_3'],
	             mm_est['gamma_4'], mm_est['qq_1'], mm_est['qq_2']]

	print(mm_est_in)
	x0 = [0., 0.8, 2., 2., 1.]

	ret = optimize.basinhopping(bp.cost_function, x0, minimizer_kwargs={'args': (mm_est_in,)}, niter=200)

	print(ret.x)


if False:
	name_stock = 'AAPL'
	path_input = '/Users/eliazarinelli/Desktop/ising_classification/lobster/data/'

	# reading data
	epsilon_true, p_star = bp.read_data(path_input, name_stock)

	NN = len(epsilon_true)

	p_star = p_star/100.

	yy = p_star[1:]-p_star[:-1]

	mm_est = bp.estimate_moments(yy)

	sigma_m = np.sqrt(mm_est['gamma_2'])
	parameters_range = {'mm': [-0.1, 0.1, 10],
	                    'rho': [0.6, 0.9, 10],
	                    'sigma': [2., 2*sigma_m , 10],
	                    'theta': [0., 2. , 10],
	                    'phi': [0., 2. , 10]}

	lambda_0 = .0065
	cc, parameters_out = bp.estimate_parameters('HIM', yy, parameters_range, lambda_0)
	print(cc)

	for i in parameters_out.keys():
		print(i, np.mean(parameters_out[i]))

	counter=collections.Counter(parameters_out['phi'])
	plt.plot(counter.keys(), counter.values(), 'bo')
	plt.show()


