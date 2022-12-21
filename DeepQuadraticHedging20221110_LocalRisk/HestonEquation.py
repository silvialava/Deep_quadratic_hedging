"""

This class simulates the Heston model of

[Cerný, A., & Kallsen, J. (2008)]

in dimension one and computes the
local-risk minimization solutions
for

[Heath, Platen, Schweizer (2001)]


!!! dim == dim_nohedge == 1 !!!

"""

from equation import Equation
import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal as normal
import HestonPDEsolver as pde


class HestonCall(Equation):

    '''

    Constructor of the class

    '''

    def __init__(self,eqn_config):

        super(HestonCall, self).__init__(eqn_config)
        self.strike = eqn_config['strike']
        
        ## Initial value of [x, v], COLUMN VECTOR with x underlying and v volatility
        self.xv_init = np.vstack([np.ones((self.dim, 1)) * eqn_config['x_init'], np.ones((self.dim, 1)) * eqn_config['v_init']])  

        ## CIR parameters
        self.kappa = eqn_config['kappa'] #speed of mean reversion
        self.theta = eqn_config['theta'] #long term mean
        self.sigma = eqn_config['sigma'] #vol of var
        self.rho = eqn_config['rho'] #correlation between B and W
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


    def sample_underQ(self, num_sample, truncation = None):      
      
        '''

        Simulates the trajectories of the BMs and the
        Heston model under the minimal martingale measure.

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
            xv_sample[:, self.dim:, i+1] =  xv_sample[:, self.dim:, i] + (self.kappa * (self.theta - xv_sample[:, self.dim:, i]) - self.sigma * self.rho * (self.mu * xv_sample[:, self.dim:, i] - self.r )) * self.delta_t + self.sigma * np.sqrt(np.abs(xv_sample[:, self.dim:, i])) * (self.rho * dwb_sample[:, :self.dim, i] + np.sqrt(1-self.rho*self.rho) * dwb_sample[:, self.dim:, i])

            xv_sample[:, self.dim:, i+1] = tf.math.maximum(xv_sample[:, self.dim:, i+1], truncation)
            
            xv_sample[:, :self.dim, i+1] = xv_sample[:, :self.dim, i] * (1  + np.sqrt(np.abs(xv_sample[:, self.dim:, i])) * dwb_sample[:, :self.dim, i])           
            
        
        ## Truncate the variance to avoid too small values
        #xv_sample[:, self.dim:, :] = tf.math.maximum(xv_sample[:, self.dim:, :], truncation)
        
        return dwb_sample, xv_sample
 


    def get_density(self, dwb_sample, xv_sample):
        
        '''
    
        Returns the density function for the change of measure
        from P to Q   

        '''
        rn = np.zeros((dwb_sample.shape[0], self.num_time_interval + 1))
        rn[:, 0] = np.ones(dwb_sample.shape[0])

        for i in range(self.num_time_interval):
            theta = (self.mu * xv_sample[:, self.dim:, i] - self.r)/np.sqrt(xv_sample[:, self.dim:, i])
            theta = np.squeeze(theta)
            rn[:, i+1] = rn[:, i].dot(np.exp(- theta * dwb_sample[:, self.dim:, i] - 0.5 * theta * theta * self.delta_t))            
        return rn



    @tf.function
    def f_tf(self, t, x, y, z):

        '''
        
        In the notation of the paper:
            
        x : [ S, Y^2 ]
        y : L        
        z : [\Lambda_1, \Lambda_2 ]
    
        '''
    
        ## This syntax is ok only in the one dimensional setting.

        theta = (self.mu * x[:, self.dim:] - self.r)/tf.math.sqrt(x[:, self.dim:])

        return - theta * z[:, :self.dim] 

   

    def g_tf(self, t, x):
        
        '''

        Discounted call payoff

        '''
        
        result = tf.math.maximum(x[:, :self.dim] - self.strike, 0)
        
        return np.exp(- self.r * self.total_time) * result



    def HPS_solution(self, m1, m2, S_0, S, V_0, V, K):
        
        '''
        We solve the PDE arising from the local-risk minimization in
        the paper of Heath, Platen and Schweizer, by 
        adapting the approach of
        
        https://www.ma.imperial.ac.uk/~ajacquie/IC_Num_Methods/IC_Num_Methods_Docs/Literature/ADI_Heston.pdf
        
        m1 : number of points in the grid for [0, S]
        m2 : number of points in the grid for [0, V]
        S_0 : initial value for the price variable
        S : maximal value for the price variable
        V_0 : initial value for the volatility
        V : maximal value for the volatility
        K: strike price
        
        '''   
        
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
        A_0, A_1, A_2, A = pde.make_matrices(m1, m2, m, self.mu,  self.rho, self.sigma, self.r, 0, self.kappa, self.theta, Vec_s, Vec_v, Delta_s, Delta_v)
        B_0, B_1, B_2, B = pde.make_boundaries(m1, m2, m, self.r, 0, self.num_time_interval, Vec_s, self.delta_t)
        
        UU_0 = np.exp(- self.r * self.total_time) * np.array([[max(Vec_s[i] - K, 0) for i in range(m1 + 1)] for j in range(m2 + 1)])
        U_0 = UU_0.flatten()

        price_vector, time = pde.MCS_scheme(m, self.num_time_interval, U_0, self.delta_t, eta, A, A_0, A_1, A_2, B, B_0, B_1, B_2, 0)
        price_grid = np.reshape(price_vector, (m2 + 1, m1 + 1, self.num_time_interval + 1))
        price_grid = np.flip(price_grid, 2)
        
        return price_vector, price_grid, Vec_s, Vec_v


 
    def HPS_derivatives(self, m1, m2, price_vector, Vec_s, Vec_v):
        
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
