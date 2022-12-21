"""

This class simulates the Heston model of

[Cerný, A., & Kallsen, J. (2008)]

in dimension one and computes the
mean-variance hedging solutions
for

[Heath, Platen, Schweizer (2001)]


!!! dim == dim_nohedge == 1 !!!

"""

from equation import Equation
import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal as normal
import HestonPDEsolver_2 as pde


class HestonCall(Equation):

    """

    Constructor of the class

    """

    def __init__(self,eqn_config):

        super(HestonCall, self).__init__(eqn_config)
        self.strike = eqn_config['strike']
        
        ## Initial value of [x, v], COLUMN VECTOR with x underlying and v volatility
        self.xv_init = np.vstack([np.ones((self.dim, 1)) * eqn_config['x_init'], np.ones((self.dim, 1)) * eqn_config['v_init']])  
      
        ## CIR parameters
        self.kappa = eqn_config['kappa'] #speed of mean reversion
        self.theta = eqn_config['theta'] #long term mean
        self.sigma = eqn_config['sigma'] #vol of var
        self.rho = eqn_config['rho'] #correlazione tra w e b
        self.r = eqn_config['r'] #risk-free interest rate  
        self.mu = eqn_config['mu'] #drift coefficient


    def sample(self, num_sample, truncation = None):      
    
        ''' 

        Simulates the trajectories of the BMs and the
        Heston model.

        '''
       
        if truncation == None: truncation = 1e-10
 
        dwb_sample = normal.rvs(size=[num_sample,     
                                     self.dim + self.dim_nohedge,
                                     self.num_time_interval]) * self.sqrt_delta_t

 
        if self.dim + self.dim_nohedge == 1: # if m = 1 and d = 0

            dwb_sample = np.expand_dims(dwb_sample,axis=0)
            dwb_sample = np.swapaxes(dwb_sample,0,1)
        
        ## Tensor in 3 dimensions: (number of paths, dimension, number of time instants)
        
        xv_sample = np.zeros([num_sample, self.dim + self.dim, self.num_time_interval + 1]) 
        xv_sample[:, :, 0] = np.transpose(np.repeat(self.xv_init, num_sample, 1))

        for i in range(self.num_time_interval):

            ## Euler scheme
            xv_sample[:, self.dim:, i+1] =  xv_sample[:, self.dim:, i] + self.kappa * (self.theta - xv_sample[:, self.dim:, i]) * self.delta_t + self.sigma * np.sqrt(np.abs(xv_sample[:, self.dim:, i])) * (self.rho * dwb_sample[:, :self.dim, i] + np.sqrt(1-self.rho*self.rho) * dwb_sample[:, self.dim:, i])

            xv_sample[:, self.dim:, i+1] = tf.math.maximum(xv_sample[:, self.dim:, i+1], truncation)

            xv_sample[:, :self.dim, i+1] =  xv_sample[:, :self.dim, i] * (1 + (self.mu * xv_sample[:, self.dim:, i] - self.r) * self.delta_t + np.sqrt(np.abs(xv_sample[:, self.dim:, i])) * dwb_sample[:, :self.dim, i])           

        ## Truncate the variance to avoid too small values
        #xv_sample[:, self.dim:, :] = tf.math.maximum(xv_sample[:, self.dim:, :], truncation)

        return dwb_sample, xv_sample

  
    @tf.function
    def f_tf(self, t, x, y, z):

        '''
        
        In the notation of the paper:
    
        x : [ S, Y^2 ]       
        y : L        
        z : [\Lambda_1, \Lambda_2 ]
        
        Warning: the drift here is assumed to be of the form
        mu(t) = mu * Y^2 like in the paper by Cerny and Kallsen
    
        '''
        
        ## This syntax is ok only in the one dimensional setting.

        theta = (self.mu * x[:, self.dim:] - self.r)/tf.math.sqrt(x[:, self.dim:])

        return  - theta * theta * y  - 2 * theta * z[:, :self.dim] - z[:, :self.dim] * z[:, :self.dim] / y

   

    def g_tf(self, t, x):

        return tf.constant(1., dtype = 'float64')



    def sample_underQ(self, num_sample, truncation = None):

        '''

        Simulates the trajectories of the BMs and the
        Heston model under the variance optimal Martingale Measure.

        '''

        if truncation == None: truncation = 1e-10

        dwb_sample = normal.rvs(size=[num_sample,
                                     self.dim + self.dim_nohedge,
                                     self.num_time_interval]) * self.sqrt_delta_t

        
        if self.dim + self.dim_nohedge == 1: # if m = 1 and d = 0
            dwb_sample = np.expand_dims(dwb_sample,axis=0)
            dwb_sample = np.swapaxes(dwb_sample,0,1)

        w0 = 0 # \chi_1 (T) = w (0) = 0
        A = - self.mu ** 2
        B = - self.kappa - 2 * self.rho * self.sigma * self.mu
        C = self.sigma ** 2 * (1 - 2 * self.rho ** 2)/2
        D = np.sqrt(B ** 2 - 4 * A * C)
        w0_hat = B/2 + C * w0
        T = self.total_time        
        t_vec = np.linspace(0, T, self.num_time_interval + 1)
       
        Num = lambda tau: (w0_hat + D/2) * np.exp(-D * tau/2) + (w0_hat - D/2) * np.exp(D * tau/2) 
        Den = lambda tau: (w0_hat + D/2) * np.exp(-D * tau/2) - (w0_hat - D/2) * np.exp(D * tau/2)

        w = lambda tau:  -B/2/C + D * Num(tau)/Den(tau)/2/C

        ## Time-dependent coefficients
        zeta1 = lambda t: - self.kappa - self.rho * self.sigma * self.mu + w(T-t) * self.sigma**2 * (1 - self.rho**2)
        zeta0 = self.kappa * self.theta

        ## Tensor in 3 dimensions: (number of paths, dimension, number of time instants)

        xv_sample = np.zeros([num_sample, self.dim + self.dim, self.num_time_interval + 1])
        xv_sample[:, :, 0] = np.transpose(np.repeat(self.xv_init, num_sample, 1))


        for i in range(self.num_time_interval):

            ## Euler scheme
            xv_sample[:, self.dim:, i+1] =  xv_sample[:, self.dim:, i] + (zeta0 + zeta1(t_vec[i]) * xv_sample[:, self.dim:, i]) * self.delta_t + self.sigma * np.sqrt(np.abs(xv_sample[:, self.dim:, i])) * (self.rho * dwb_sample[:, :self.dim, i] + np.sqrt(1-self.rho**2) * dwb_sample[:, self.dim:, i])

            xv_sample[:, self.dim:, i+1] = tf.math.maximum(xv_sample[:, self.dim:, i+1], truncation)

            xv_sample[:, :self.dim, i+1] = xv_sample[:, :self.dim, i] * (1  + np.sqrt(np.abs(xv_sample[:, self.dim:, i])) * dwb_sample[:, :self.dim, i])

        ## Truncate the variance to avoid too small values
        #xv_sample[:, self.dim:, :] = tf.math.maximum(xv_sample[:, self.dim:, :], truncation)

        return dwb_sample, xv_sample


    
    def CKSolutionL(self, num_sample, v_sample):
 
        '''

        In the notation of the paper:

        \chi_0 (t) = y (T - t)
        \chi_1 (t) = w (T - t)

        '''

        w0 = 0 # \chi_0 (T) = w (0) = 0
        y0 = 0 # \chi_1 (T) = y (0) = 0
        A = - self.mu ** 2
        B = - (self.kappa + 2 * self.rho * self.sigma * self.mu)
        C = self.sigma ** 2 * (1 - 2 * self.rho ** 2)/2
        D = np.sqrt(B ** 2 - 4 * A * C)
        F = self.kappa * self.theta
        w0_hat = B/2 + C * w0
        T = self.total_time        

        w = lambda tau: -B/2/C + D * ((w0_hat + D/2) * np.exp(-D * tau/2) + (w0_hat - D/2) * np.exp(D * tau/2))/((w0_hat + D/2) * np.exp(-D * tau/2) - (w0_hat - D/2) * np.exp(D * tau/2))/2/C
        y = lambda tau: y0 + F * (-B/2/C * tau - np.log(((w0_hat + D/2) * np.exp(-D * tau/2) - (w0_hat - D/2) * np.exp(D * tau/2))/D)/C)

        t = np.linspace(0, T, self.num_time_interval + 1)
        
        return np.exp(y(T - t) + np.abs(v_sample) * w(T - t)) 


    def CKSolutionV(self, m1, m2, S_0, S, V_0, V, K):
        
        '''

        We solve the PDE following the approach (and using the --adjusted-- code) of
        
        https://www.ma.imperial.ac.uk/~ajacquie/IC_Num_Methods/IC_Num_Methods_Docs/Literature/ADI_Heston.pdf
        
        m1 : number of points in the grid for [0, S]
        m2 : number of points in the grid for [0, V]
        S_0 : initial value for the price variable
        S : maximal value for the price variable
        V_0 : initial value for the volatility
        V : maximal value for the volatility
        K: strike price
        
        '''
   
        ## The coeffients of the PDE are time-dependent and in particular
        ## they depend on \chi_1 above. In order to not call the function above
        ## I recreate \chi_1 here: 
            
        w0 = 0 # \chi_1 (T) = w (0) = 0
        A = - self.mu ** 2
        B = - (self.kappa + 2 * self.rho * self.sigma * self.mu)
        C = self.sigma ** 2 * (1 - 2 * self.rho ** 2)/2
        D = np.sqrt(B ** 2 - 4 * A * C)
        F = self.kappa * self.theta
        w0_hat = B/2 + C * w0
        T = self.total_time        

        w = lambda tau: -B/2/C + D * ((w0_hat + D/2) * np.exp(-D * tau/2) + (w0_hat - D/2) * np.exp(D * tau/2))/((w0_hat + D/2) * np.exp(-D * tau/2) - (w0_hat - D/2) * np.exp(D * tau/2))/2/C

        t = np.linspace(0, T, self.num_time_interval + 1)
        
        ## Time-dependent coefficients
        chi_1 = w(T-t)
        kappa_vec = self.kappa + self.rho * self.sigma * self.mu - chi_1 * self.sigma**2 * (1 - self.rho**2)
        theta_vec = self.kappa * self.theta / kappa_vec

        
        m = (m1 + 1) * (m2 + 1)  # Matrix A and vector U size
        c = K / 5   # Control parameter for the mesh of S
        d = V / 500   # Control parameter for the mesh of V
        eta = 1/3   # Real parameter for the Finite Difference method
        
        ## Model setup:
        ## Note that in [In't Hout & Foulon (2010)] they have 
        ## r_d : domestic interest rate
        ## r_f : foreign interest rate
        ## hence I set r_d = self.r and r_f = 0
        
        Vec_s, Delta_s, Vec_v, Delta_v, X, Y = pde.make_grid(m1, S, S_0, K, c, m2, V, V_0, d)
        B_0, B_1, B_2, BB = pde.make_boundaries(m1, m2, m, self.r, 0, self.num_time_interval, Vec_s, self.delta_t)
        UU_0 = np.array([[max(Vec_s[i] - K, 0) for i in range(m1 + 1)] for j in range(m2 + 1)])
        U_0 = UU_0.flatten()

        
        ## With HestonPDESolver
        #A_0, A_1, A_2, A = pde.make_matrices(self.num_time_interval, T, m1, m2, m, self.rho, self.sigma, self.r, 0, kappa_vec, theta_vec, Vec_s, Vec_v, Delta_s, Delta_v)
        #price_vector, time = pde.MCS_scheme(m, self.num_time_interval, U_0, self.delta_t, eta, A, A_0, A_1, A_2, B, B_0, B_1, B_2, 0)        
        
    
        ## With HestonPDESolver_2
        price_vector, time = pde.MCS_scheme(m, m1, m2, self.num_time_interval, T,  U_0, self.delta_t, eta,  self.rho, self.sigma, self.r, 0, kappa_vec, theta_vec, BB, B_0, B_1, B_2, Vec_s, Vec_v, Delta_s, Delta_v)

 
        price_grid = np.reshape(price_vector, (m2 + 1, m1 + 1, self.num_time_interval + 1))
        price_grid = np.flip(price_grid, 2)
        
        return price_vector, price_grid, Vec_s, Vec_v



    def CK_derivatives(self, m1, m2, price_vector, Vec_s, Vec_v):
 
        '''

        Computes the derivatives of the solution to the PDE above

        '''

        m = (m1 + 1) * (m2 + 1)  # Matrix A and vector U size
        Delta_s = [Vec_s[i + 1] - Vec_s[i] for i in range(m1)]
        Delta_v = [Vec_v[i + 1] - Vec_v[i] for i in range(m2)]

        A_1, A_2 = pde.make_derivative_matrices(m1, m2, m, Vec_s, Vec_v, Delta_s, Delta_v)
        
        dirac_s = A_1*price_vector
        dirac_s = np.reshape(dirac_s, (m2 + 1, m1 + 1, self.num_time_interval + 1))
        dirac_s = np.flip(dirac_s, 2)

        dirac_v = A_2*price_vector
        dirac_v = np.reshape(dirac_v, (m2 + 1, m1 + 1, self.num_time_interval + 1))
        dirac_v = np.flip(dirac_v, 2)

        return dirac_s, dirac_v


    def CKWealthSDE(self, num_MC, CK_V, x, dirac_s, dirac_v, y = 0):
 
        ## The SDE dependends and on \chi_1 above. In order to not call the function above
        ## I recreate \chi_1 here: 
            
        w0 = 0 # \chi_0 (T) = w (0) = 0
        A = - self.mu ** 2
        B = - (self.kappa + 2 * self.rho * self.sigma * self.mu)
        C = self.sigma ** 2 * (1 - 2 * self.rho ** 2)/2
        D = np.sqrt(B ** 2 - 4 * A * C)
        F = self.kappa * self.theta
        w0_hat = B/2 + C * w0
        T = self.total_time        

        w = lambda tau: -B/2/C + D * ((w0_hat + D/2) * np.exp(-D * tau/2) + (w0_hat - D/2) * np.exp(D * tau/2))/((w0_hat + D/2) * np.exp(-D * tau/2) - (w0_hat - D/2) * np.exp(D * tau/2))/2/C
        t = np.linspace(0, T, self.num_time_interval + 1)
        chi_1 = w(T-t)

        wealth = np.zeros((num_MC, self.num_time_interval + 1))
        wealth[:, 0] = y * np.ones(num_MC)
        phiCK = np.zeros((num_MC, self.num_time_interval))
        for i in range(self.num_time_interval):
            phiCK[:, i] = dirac_s[:, i] + np.divide(self.rho * self.sigma * dirac_v[:, i] + np.multiply(self.mu + self.rho * self.sigma * chi_1[i], CK_V[:, i] - wealth[:, i]), x[:, i])
            wealth[:, i + 1] = wealth[:, i] + np.multiply(phiCK[:, i], x[:, i + 1] - x[:, i])
        

        thetaCK = wealth[:, :self.num_time_interval] - np.multiply(phiCK, x[:, :self.num_time_interval]) 
 
        return thetaCK, phiCK
