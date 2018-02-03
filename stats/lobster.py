import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import bp as bp


def read_data(path, name):

	path_input = path 
	name_stock = name

	# reading data
	data = pd.read_csv(path_input + name_stock+'.csv', header=0)
	data.columns = ['time','type','id','size','price','sign']

	sel_1 = data.loc[:,'type'] == 4
	sel_2 = data.loc[:,'type'] == 5
	sel_all = sel_1 | sel_2

	mo = data.loc[sel_all,:]

	gp_1 = mo.groupby(['time','price','sign'], as_index=False)

	a = gp_1['size'].sum()

	gp_2 = a.groupby(['time','sign'], as_index=False)

	b = gp_2['price'].first()

	b.loc[:,'sign'] = -1. * b.loc[:,'sign'] 

	p_star = np.array(b.loc[:,'price'])
	epsilon = np.array(b.loc[:,'sign'])

	return epsilon, p_star



l_name_stock = ['AAPL', 'AMZN', 'GOOG', 'INTC', 'MSFT']

#l_name_stock = ['AAPL', 'AMZN', 'GOOG']
l_name_stock = ['AAPL']

for name_stock in l_name_stock:
	path_input = '/Users/eliazarinelli/Desktop/ising_classification/lobster/data/'

	# reading data
	epsilon_true, p_star = read_data(path_input, name_stock)

	p_star = p_star/1000.

	yy = p_star[1:]-p_star[:-1]

	#n, bins, patches = plt.hist(yy, 200, normed=1)
	#plt.show()

	NN = len(p_star)
	mm = 0.
	rho = 0.7
	sigma = 0.5*np.std(yy)
	tt = 10.
	theta = tt*sigma
	pp = 0.5
	phi = pp * sigma

	ee_mmo = bp.hat_epsilon_MMO(yy, rho, theta, phi, mm, sigma)
	ee_dtr = bp.hat_epsilon_DTR(yy)
	
	print(1.*sum(epsilon_true==ee_dtr)/(1.*NN))
	print(1.*sum(epsilon_true==ee_mmo)/(1.*NN))


	if True:	
		# setting the list of trying parameters
		rho_list = np.linspace(0.5, 0.9, 2**3)
		mm_list = np.linspace(-0.01, 0.01, 2**2)

		sigma_list = np.std(yy) * np.linspace(0.05, 1., 2**4)
		theta_list = np.std(yy) * np.linspace(1., 10., 2**4)
		phi_list = np.std(yy) * np.linspace(0.1, 1., 2**4)

		# setting lambda parameters
		lambda_all = 0.2
		lambda_0 =  lambda_all
		lambda_1 =  lambda_all
		lambda_2 =  lambda_all
		lambda_3 =  lambda_all


		# calculating the moments of the observation
		se = bp.estimate_moments(yy)					


		# initialisation of counters
		count_0 = 0
		count_1 = 0 
		count_2 = 0

		#NN = len(yy)
		magn = np.zeros(NN)

		bool_inf = True


		for mm in mm_list:
			for rho in rho_list:
				for theta in theta_list:
					for phi in phi_list:
						for sigma in sigma_list:
							# calculating the moments with the tentative parameters
							cc_l = theta + phi
							#cc_r = -1.*(rho*theta + phi)
							cc_r = -1.*phi
							
							mm_2 = sigma**2 + (1.-mm**2)*(cc_l**2+cc_r**2+2.*rho*cc_l*cc_r)
							
							tmp = (2*mm**3-2.*mm)*(cc_l**3 + cc_r**3) + 3.*cc_l*cc_r*(cc_l+cc_r)*(-2.)*mm*(1-mm**2)*rho
							mm_3 = tmp/np.sqrt(mm_2**3)

							tmp = (1-mm**2)*(cc_l**2+cc_r**2+cc_l*cc_r*(rho+1./rho))*rho
							qq_1 = tmp/mm_2

							tmp = (1-mm**2)*(cc_l**2+cc_r**2+cc_l*cc_r*(rho+1./rho))*rho**2
							qq_2 = tmp/mm_2

							#sel_0 = ((se[1]-mm_2)/(np.sqrt(NN)*lambda_0))**2 < 1.
							#sel_1 = ((se[2]-mm_3)/(np.sqrt(NN)*lambda_1))**2 < 1.
							#sel_2 = ((se[3]-qq_1)/(np.sqrt(NN)*lambda_2))**2 < 1.
							#sel_3 = ((se[4]-qq_2)/(np.sqrt(NN)*lambda_3))**2 < 1.
							
							sel_0 = (se[1]-mm_2)**2/( se[1]**2*np.sqrt(NN)*lambda_0) < 1.
							sel_1 = (se[2]-mm_3)**2/( se[2]**2*np.sqrt(NN)*lambda_1) < 1.
							sel_2 = (se[3]-qq_1)**2/( se[3]**2*np.sqrt(NN)*lambda_2) < 1.
							sel_3 = (se[4]-qq_2)**2/( se[4]**2*np.sqrt(NN)*lambda_3) < 1.

							count_0 += 1
							if(sel_0 and sel_1 and sel_2 and sel_3):
								count_1 += 1
								print(count_1)
								if bool_inf:						
									tmp = bp.local_magn(yy, rho, theta, phi, mm, sigma)
									if ( sum(np.isnan(tmp)) ==0 ):
										count_2 += 1
										magn = magn + tmp 
		print(count_1)
		if count_2>0:
			magn = magn/count_2
			ee_inf_approx = np.sign(magn)
			oo_inf_app = 1.*sum(epsilon_true==ee_inf_approx)/(1.*NN)
			print(oo_inf_app)





