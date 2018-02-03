import numpy as np
import pandas as pd
import scipy.stats as sps

def generate_dar(parameters, NN):

    mm = parameters['mm']
    rho = parameters['rho']

    # generating the trade-sign configuration ep
    ep = []
    ep_previous = 1
    for i in range(NN):
        vv = 0
        if np.random.rand() < rho:
            vv = 1
        zz = -1
        if np.random.rand() < (1.+mm)/2.:
            zz = +1
        ep_i = vv * ep_previous + (1-vv)*zz
        ep.append(ep_i)
        ep_previous = ep_i

    return ep

def generate_observations(model, parameters, epsilon):

    # generating the observation yy
    mm = parameters['mm']
    rho = parameters['rho']
    sigma = parameters['sigma']
    theta = parameters['theta']
    phi = parameters['phi']

    NN = len(epsilon)

    if model == 'PIM':
        cc_l = 1.*(theta + phi)
        cc_r = -1.*(phi)

        yy = []
        for i in range(NN-1):
            yy.append( cc_l*(epsilon[i+1]-mm) + cc_r*(epsilon[i]-mm) + (cc_l+cc_r)*mm + np.random.normal(0,sigma))

    if model == 'HIM':
        cc_l = 1.*(theta + phi)
        cc_r = -1.*(rho*theta + phi)

        yy = []
        for i in range(NN-1):
            yy.append( cc_l*(epsilon[i+1]-mm) + cc_r*(epsilon[i]-mm) + np.random.normal(0,sigma))

    return yy


def cost_function(x, x_measured):

    mm = x[0]
    rho = x[1]
    sigma = x[2]
    theta = x[3]
    phi = x[4]

    C_2_0 = 1.-mm**2
    C_2_1 = rho*(1.-mm**2)
    C_2_2 = rho**2*(1.-mm**2)
    C_2_3 = rho**3*(1.-mm**2)
    C_3_0 = -2.*mm*(1.-mm**2)
    C_3_1 = -2.*mm*(1.-mm**2)*rho
    C_4_0 = 1.+2.*mm**2-3.*mm**4
    C_4_1 = rho*(1.+3.*mm**2)*(1.-mm**2)
    C_4_2 = 1.-2.*mm**2+mm**4+4.*mm**2*rho*(1.-mm**2)

    cc_l = theta + phi
    cc_r = -1.*phi
    mm_1 = (cc_l + cc_r)*mm
    mm_2 = (cc_l**2+cc_r**2)*C_2_0 + 2.*cc_l*cc_r*C_2_1 + sigma**2
    mm_3 = (cc_l**3+cc_r**3)*C_3_0 + 3.*cc_l*cc_r*(cc_l+cc_r)*C_3_1
    mm_4 = (cc_l**4+cc_r**4)*C_4_0 + 4.*cc_l*cc_r*(cc_l**2+cc_r**2)*C_4_1 + \
           6.*(cc_l**2*cc_r**2)*C_4_2 + \
           6.*sigma**2*((cc_l**2+cc_r**2)*C_2_0+2.*cc_l*cc_r*C_2_1) + \
           3.*sigma**4
    dd_1 = (cc_l**2 + cc_r**2)*C_2_1 + cc_l*cc_r*(C_2_0+C_2_2)
    dd_2 = (cc_l**2 + cc_r**2)*C_2_2 + cc_l*cc_r*(C_2_1+C_2_3)

    gamma_1 = mm_1
    gamma_2 = mm_2
    gamma_3 = mm_3/(mm_2**(3./2.))
    gamma_4 = mm_4/mm_2**2 - 3.
    qq_1 = dd_1/mm_2
    qq_2 = dd_2/mm_2

#    cost = (1.-gamma_1/x_measured[0])**2 + (1.-gamma_2/x_measured[1])**2 + \
#           (1.-gamma_3/x_measured[2])**2 + (1.-gamma_4/x_measured[3])**2 + \
#           (1.-qq_1/x_measured[4])**2 + (1.-qq_2/x_measured[5])**2

    cost = (1.-gamma_2/x_measured[1])**2 + (1.-gamma_4/x_measured[3])**2 + \
           (1.-qq_1/x_measured[4])**2

    return cost



def calculate_moments(model, parameters):

    try:
        mm = parameters['mm']
        rho = parameters['rho']
        sigma = parameters['sigma']
        theta = parameters['theta']
        phi = parameters['phi']
    except KeyError:
        print('The model parameter are not provided in '
              'the correct format')
        raise

    C_2_0 = 1.-mm**2
    C_2_1 = rho*(1.-mm**2)
    C_2_2 = rho**2*(1.-mm**2)
    C_2_3 = rho**3*(1.-mm**2)
    C_3_0 = -2.*mm*(1.-mm**2)
    C_3_1 = -2.*mm*(1.-mm**2)*rho
    C_4_0 = 1.+2.*mm**2-3.*mm**4
    C_4_1 = rho*(1.+3.*mm**2)*(1.-mm**2)
    C_4_2 = 1.-2.*mm**2+mm**4+4.*mm**2*rho*(1.-mm**2)

    if model == 'PIM':
        cc_l = theta + phi
        cc_r = -1.*phi
        mm_1 = (cc_l + cc_r)*mm

    if model == 'HIM':
        cc_l = theta + phi
        cc_r = -1.*(rho*theta + phi)
        mm_1 = 0.

    mm_2 = (cc_l**2+cc_r**2)*C_2_0 + 2.*cc_l*cc_r*C_2_1 + sigma**2
    mm_3 = (cc_l**3+cc_r**3)*C_3_0 + 3.*cc_l*cc_r*(cc_l+cc_r)*C_3_1
    mm_4 = (cc_l**4+cc_r**4)*C_4_0 + 4.*cc_l*cc_r*(cc_l**2+cc_r**2)*C_4_1 + \
    6.*(cc_l**2*cc_r**2)*C_4_2 + \
    6.*sigma**2*((cc_l**2+cc_r**2)*C_2_0+2.*cc_l*cc_r*C_2_1) + \
    3.*sigma**4
    dd_1 = (cc_l**2 + cc_r**2)*C_2_1 + cc_l*cc_r*(C_2_0+C_2_2)
    dd_2 = (cc_l**2 + cc_r**2)*C_2_2 + cc_l*cc_r*(C_2_1+C_2_3)

    gamma_1 = mm_1
    gamma_2 = mm_2
    gamma_3 = mm_3/(mm_2**(3./2.))
    gamma_4 = mm_4/mm_2**2 - 3.
    qq_1 = dd_1/mm_2
    qq_2 = dd_2/mm_2

    moments ={'gamma_1' : gamma_1,
              'gamma_2' : gamma_2,
              'gamma_3' : gamma_3,
              'gamma_4' : gamma_4,
              'qq_1' : qq_1,
              'qq_2' : qq_2
              }

    return moments

def estimate_moments(observations):

    yy = observations

    gamma_1 = np.mean(yy)
    gamma_2 = np.var(yy)
    gamma_3 = sps.skew(yy)
    gamma_4 = sps.kurtosis(yy, fisher=True)
    qq_1 = np.corrcoef(yy[1:],yy[:-1])[0,1]
    qq_2 = np.corrcoef(yy[2:],yy[:-2])[0,1]

    moments ={'gamma_1' : gamma_1,
              'gamma_2' : gamma_2,
              'gamma_3' : gamma_3,
              'gamma_4' : gamma_4,
              'qq_1' : qq_1,
              'qq_2' : qq_2
              }
    return moments

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

def estimate_parameters(model, observations, parameter_range, lambda_0):

    try:
        tmp = parameter_range['mm']
        mm_list = np.linspace(tmp[0], tmp[1], tmp[2])
        tmp = parameter_range['rho']
        rho_list = np.linspace(tmp[0], tmp[1], tmp[2])
        tmp = parameter_range['sigma']
        sigma_list = np.linspace(tmp[0], tmp[1], tmp[2])
        tmp = parameter_range['theta']
        theta_list = np.linspace(tmp[0], tmp[1], tmp[2])
        tmp = parameter_range['phi']
        phi_list = np.linspace(tmp[0], tmp[1], tmp[2])
    except KeyError:
        print('Wrong input!')
        return

    if model == 'PIM':
        #sel_moments = ['gamma_1', 'gamma_2', 'gamma_3', 'gamma_4', 'qq_1', 'qq_2']
        sel_moments = ['gamma_2', 'gamma_4', 'qq_1', 'qq_2']
    if model == 'HIM':
        #sel_moments = ['gamma_1', 'gamma_2', 'gamma_3', 'gamma_4', 'qq_1', 'qq_2']
        sel_moments = ['gamma_2', 'gamma_3', 'gamma_4', 'qq_1']

    NN = len(observations)
    moments_est = estimate_moments(observations)

    out = {}
    for i in parameter_range.keys():
        out[i] = []

    count = 0
    for mm in mm_list:
        for rho in rho_list:
            for sigma in sigma_list:
                for theta in theta_list:
                    for phi in phi_list:
                        parameters_in = {'mm': mm,
                         'rho': rho,
                         'sigma': sigma,
                         'theta': theta,
                         'phi': phi}
                        moments_calc = calculate_moments(model, parameters_in)
                        checks = {}
                        for i in moments_est.keys():
                            checks[i] = False
                            if (1.-moments_calc[i]/moments_est[i])**2/(1.*NN*lambda_0**2) < 1.:
                                checks[i] = True
                        check_all = True
                        for i in sel_moments:
                            check_all = check_all and checks[i]
                        if check_all:
                            count += 1
                            for i in parameters_in.keys():
                                out[i].append(parameters_in[i])
    m_out = {}
    #for i in parameters_in.keys():
        #m_out[i] = np.mean(out[i])
    return count, out


def generate_path_MRR(theta, phi, rho, mm, sigma, NN):

    '''
    This function generates a trade-sign configuration epsilon and the observation vector y 
    of the MRR model, equation 1 and 5 of the notes.
    Input: the 4 model parameters theta, phi, rho and sigma and the number of trades NN.
    Output: the trade-sign configuration epsilon and the observation y. 
    '''

    # generating the trade-sign configuration ep
    ep = []
    ep_previous = 1    
    for i in range(NN):
        vv = 0
        if np.random.rand() < rho:
            vv = 1
        zz = -1
        if np.random.rand() < (1.+mm)/2.:
            zz = +1
        ep_i = vv * ep_previous + (1-vv)*zz
        ep.append(ep_i)
        ep_previous = ep_i  
    ep = np.array(ep)

    # generating the observation yy
    aa =  1.*(theta + phi)
    #bb = -1.*(rho*theta + phi) 
    bb = -1.*(phi) 
    cc = -1.*theta*(1.-rho)*mm

    yy = []
    for i in range(NN-1):        
        yy.append(aa*ep[i+1] + bb*ep[i] + cc + np.random.normal(0,sigma))        
    yy = np.array(yy)

    return ep, yy


def hat_epsilon_DTR(yy):

    '''
    This function retrurns the estimated trade-sign configuration hat_epsilon according to the Direct Tick Rule.
    Input: the vector of observations yy.
    Output: the hat_epsilon_DTR
    '''
    tmp = np.append([1],np.sign(yy))

    for i in range(len(tmp)):
        if tmp[i] == 0:
            tmp[i] = tmp[i-1]

    return tmp 


def local_magn(yy, rho, theta, phi, mm, sigma):

    '''
    This function returns the local magnetisations of a trade-sgin configuration of the MRR model given 
    the vector y of the observations and the model parameters Theta.
    Input: the observations yy and the model parameters rho, theta, phi, mm and sigma
    Output: the vector of local magnetisations
    '''

    cc_l = (theta + phi)
    #cc_r = -1.*( rho * theta + phi)
    cc_r = -1.*phi

    # generating coupling
    # coupling of the prior
    jj_p = 1./4. * np.log( ((1.+mm)+rho*(1.-mm)) * ((1.-mm)+rho*(1.+mm)) / ((1.-mm**2)*(1.-rho)**2) ) * np.ones(len(yy))
    # coupling of the likelihood
    #jj_l = (theta+phi)*(rho*theta+phi)/sigma**2 * np.ones(len(yy))
    jj_l = -1.*cc_l*cc_r/sigma**2 * np.ones(len(yy))
    # coupling of the posterior
    jj = jj_p + jj_l

    # generating fields
    # coupling of the prior
    hh_p = 1./2. * np.log( ((1.+mm)+rho*(1.-mm)) / ((1.-mm)+rho*(1.-mm)) ) * np.ones(len(yy)+1)
    # coupling of the likelihood
    hh_l = (np.append([0],yy) * cc_l + np.append(yy,[0]) * cc_r + (cc_l+cc_r)**2*mm)/sigma**2
    # coupling of the posterior
    hh = hh_p + hh_l

    #generating inverse temperature 
    bb = 1
    
    # calculating the marginal probability p_i(epsilon_i=+1)
    pp = calculate_marg(bb, jj, hh)
    
    # retriving magnetisation 
    mm = 2.*pp - 1.

    return mm


def hat_epsilon_MMO(yy, rho, theta, phi, mm, sigma):

    '''
    This function returns the estimation of the trade-sign configuration hat_epsilon 
    according to the rule provided by the Maximum Mean Overlap estimator given 
    the vector y of the observations and the model parameters Theta.
    Input: the observations yy and the model parameters rho, theta, phi and sigma
    Output: the vector of estimations hat_epsilon_MMO
    '''

    # calculating local magnetisations
    mm = local_magn(yy, rho, theta, phi, mm, sigma)

    return np.sign(mm)    




def HH_1(position, sigma_out, sigma_in, beta, JJ, BB):
    #return py.exp(beta*sigma_in*(sigma_out+b_ext))
    return np.exp(beta*sigma_in*(JJ[position]*sigma_out+BB[position]))


def HH_2(position, sigma_out, sigma_in, beta, JJ, BB):
    #return py.exp(beta*sigma_in*(sigma_out+b_ext))
    return np.exp(beta*sigma_in*(JJ[position-1]*sigma_out+BB[position]))


def calculate_marg(beta, JJ, BB):
    
    '''
    Ths function computes the marginal probability of an Ising variable p(epsilon_i=+1) of an Ising chain model.
    Input: the inverse temperature beta, the vector of couplings JJ (of length N-1) 
    and the vector of local external fields BB (of length N).
    Output: the vector of the marginal probailities.
    '''

    if JJ is None:
        JJ = np.ones(len(BB)-1)

    N_OBS = len(BB)
    
    nu_l = np.zeros(N_OBS)
    nu_r = np.zeros(N_OBS)
    marg_p = np.zeros(N_OBS)

    nu_l[0] = 0.5
    nu_r[N_OBS-1] = 0.5

    for i in range(1,N_OBS):
        nu_l_p_nr = nu_l[i-1] * HH_1(i-1,1.,1.,beta,JJ,BB) + (1.- nu_l[i-1]) * HH_1(i-1,1.,-1.,beta,JJ,BB)
        nu_l_n_nr = nu_l[i-1] * HH_1(i-1,-1.,1.,beta,JJ,BB) + (1.- nu_l[i-1]) * HH_1(i-1,-1.,-1.,beta,JJ,BB)
        nu_l[i] = nu_l_p_nr/(nu_l_p_nr + nu_l_n_nr)
        #pippo = nu_l[i]

    for i in reversed(range(N_OBS-1)):
        nu_r_p_nr = nu_r[i+1] * HH_2(i+1,1.,1.,beta,JJ,BB) + (1.- nu_r[i+1]) * HH_2(i+1,1.,-1.,beta,JJ,BB)
        nu_r_n_nr = nu_r[i+1] * HH_2(i+1,-1.,1.,beta,JJ,BB) + (1. - nu_r[i+1]) * HH_2(i+1,-1.,-1.,beta,JJ,BB)
        nu_r[i] = nu_r_p_nr/(nu_r_p_nr + nu_r_n_nr)

    marg_p_nr = nu_l * np.exp(beta*BB) * nu_r
    marg_n_nr = (1-nu_l) * np.exp(-1.*beta*BB) *(1.-nu_r)
    marg_p = marg_p_nr /(marg_p_nr + marg_n_nr)

    
    return marg_p
