"""

This class simulates the multidimensional 
Heston model of

[Gnoatto, Lavagnini, Picarelli (2022)]

"""

from equation import Equation
import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal as normal



class HestonCall(Equation):
   
    '''

    Constructor of the class

    '''

    def __init__(self,eqn_config):

        super(HestonCall, self).__init__(eqn_config)

        self.strike = eqn_config['strike']
        
        ## Initial value of [x, v], COLUMN VECTOR with x underlying and v volatility
        ## x_init e v_init are array of dimension (dim, 1)
        self.xv_init = np.vstack([eqn_config['x_init'], eqn_config['v_init']])
  

        ## Multi-dim CIR parameters

        self.kappa = eqn_config['kappa'] #speed of mean reversion
        self.theta = eqn_config['theta'] #long term mean
        self.sigma = eqn_config['sigma'] #vol of var
        self.rho = eqn_config['rho'] #correlazione tra w e b
        self.r = eqn_config['r'] # risk-free rate, float 
        self.B = eqn_config['B'] # shuffle matrix for drift term, array (dim_nohedge, dim_nohedge)
        self.A = eqn_config['A'] # shuffle matrix for diffusion term, array (dim_nohedge, dim_nohedge)
    

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
        
        theta = self.theta
        kappa = self.kappa
        sigma = self.sigma
        rho = self.rho
        
        ## Tensor in 3 dimensions: (number of paths, dimension, number of time instants)
        
        xv_sample = np.zeros([num_sample, self.dim + self.dim, self.num_time_interval + 1]) 
        xv_sample[:, :, 0] = np.transpose(np.repeat(self.xv_init, num_sample, 1))


        one = np.ones(self.dim_nohedge)       

        for i in range(self.num_time_interval):

            ## Euler scheme

            xv_sample[:, self.dim:, i+1] =  xv_sample[:, self.dim:, i] + kappa * (theta - xv_sample[:, self.dim:, i]) * self.delta_t + sigma * np.sqrt(np.abs(xv_sample[:, self.dim:, i])) * (rho * dwb_sample[:, :self.dim, i] + np.sqrt(one-rho**2) * dwb_sample[:, self.dim:, i])

            xv_sample[:, self.dim:, i+1] = tf.math.maximum(xv_sample[:, self.dim:, i+1], truncation)

            xv_sample[:, :self.dim, i+1] = xv_sample[:, :self.dim, i] + xv_sample[:, :self.dim, i] * (np.transpose(self.B.dot(np.transpose(xv_sample[:, self.dim:, i]))) * self.delta_t + np.transpose(self.A.dot(np.transpose(np.sqrt(np.abs(xv_sample[:, self.dim:, i])) * dwb_sample[:, :self.dim, i]))))

        
        if np.min(xv_sample[:, self.dim:, :]) < truncation: tf.print('!!! Warning: negative values!!! ',  np.min(xv_sample[:, self.dim:, :]))

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
        
        '''
 
        one = tf.ones(self.dim_nohedge, dtype = tf.dtypes.float64)
        diagV =  tf.map_fn(tf.linalg.diag, x[:, self.dim:])

        sigma = tf.linalg.matmul(self.A, tf.math.sqrt(tf.math.abs(diagV)))
        mu = tf.transpose(tf.tensordot(self.B, tf.transpose(x[:, self.dim:]), axes = 1))
        theta = tf.squeeze(tf.linalg.matmul(tf.linalg.inv(sigma), tf.expand_dims(mu - self. r * one, axis = 2)))
        
        return - tf.expand_dims(((tf.linalg.diag_part(tf.tensordot(theta, tf.transpose(theta), axes = 1))) * tf.squeeze(y) + 2 * tf.linalg.diag_part(tf.tensordot(theta, tf.transpose(z[:, :self.dim]), axes = 1)) + tf.linalg.diag_part(tf.tensordot(z[:, :self.dim], tf.transpose(z[:, :self.dim]), axes = 1)) / tf.squeeze(y)), axis = 1)



    def g_tf(self, t, x):

        return tf.ones((x.shape[0], 1), dtype = tf.dtypes.float64)       


    def sample_underQ(self, num_sample, truncation = None):

        '''

        Simulates the trajectories of the BMs and the
        Heston model under the variance optimal Martingale Measure.

        !!!!! WARNING: THIS WORKS ONLY FOR A AND B DIAGONAL MATRICES !!!!!

        '''

        if truncation == None: truncation = 1e-10

        dwb_sample = normal.rvs(size=[num_sample,
                                     self.dim + self.dim_nohedge,
                                     self.num_time_interval]) * self.sqrt_delta_t


        theta = np.expand_dims(self.theta, axis = 1)
        kappa = np.expand_dims(self.kappa, axis = 1)
        sigma = np.expand_dims(self.sigma, axis = 1)
        rho = np.expand_dims(self.rho, axis = 1)
        mu = np.expand_dims(np.diag(self.B), axis = 1)
        one = np.ones((self.dim_nohedge, 1))

        w0 = np.zeros((self.dim, 1)) # \chi_1 (T) = w (0) = 0
        A = - mu ** 2
        B = - kappa - 2 * rho * sigma * mu
        C = sigma ** 2 * (one - 2 * rho ** 2)/2
        D = np.sqrt(B ** 2 - 4 * A * C)
        w0_hat = B/2 + C * w0
        T = self.total_time        
        t_vec = np.linspace(0, T, self.num_time_interval + 1)
        
        Num = lambda tau: (w0_hat + D/2) * np.exp(-D * tau/2) + (w0_hat - D/2) * np.exp(D * tau/2) 
        Den = lambda tau: (w0_hat + D/2) * np.exp(-D * tau/2) - (w0_hat - D/2) * np.exp(D * tau/2)

        w = lambda tau:  -B/2/C + D * Num(tau)/Den(tau)/2/C

        ## Time-dependent coefficients
        zeta1 = lambda t: - kappa - rho * sigma * mu + w(T-t) * sigma**2 * (one - rho**2)
        zeta0 = kappa * theta

        ## Tensor in 3 dimensions: (number of paths, dimension, number of time instants)

        xv_sample = np.zeros([num_sample, self.dim + self.dim, self.num_time_interval + 1])
        xv_sample[:, :, 0] = np.transpose(np.repeat(self.xv_init, num_sample, 1))


        for i in range(self.num_time_interval):

            ## Euler scheme

            xv_sample[:, self.dim:, i+1] =  xv_sample[:, self.dim:, i] + np.transpose((zeta0 + zeta1(t_vec[i])*np.transpose(xv_sample[:, self.dim:, i]))) * self.delta_t + self.sigma * np.sqrt(np.abs(xv_sample[:, self.dim:, i])) * (self.rho * dwb_sample[:, :self.dim, i] + np.sqrt(np.squeeze(one)-self.rho**2) * dwb_sample[:, self.dim:, i])

            xv_sample[:, self.dim:, i+1] = tf.math.maximum(xv_sample[:, self.dim:, i+1], truncation)

            xv_sample[:, :self.dim, i + 1] = xv_sample[:, :self.dim, i] * (1 + self.r * self.delta_t + np.transpose(self.A.dot(np.transpose(np.sqrt(np.abs(xv_sample[:, self.dim:,     i])) * dwb_sample[:, :self.dim, i]))))

        if np.min(xv_sample[:, self.dim:, :]) < truncation: tf.print('!!! Warning: negative values in Q!!! ',  np.min(xv_sample[:, self.dim:, :]))

        ## Truncate the variance to avoid too small values
        #xv_sample[:, self.dim:, :] = tf.math.maximum(xv_sample[:, self.dim:, :], truncation)

        return dwb_sample, xv_sample

    
    def CKSolutionL(self):
 
        '''
        !!!!! WARNING: THIS WORKS ONLY FOR A AND B DIAGONAL MATRICES !!!!!

        In the notation of the paper:

        \chi_0 (t) = y (T - t)
        \chi_1 (t) = w (T - t)I

        !!!!! WARNING: THIS HAS BEEN MODIFIED w.r.t. TO THE ONE-DIMENSONAL VERSION !!!!!
        !!!!! Here theoutput is simply [\chi_0(0), \chi_1(0)] times the dimesion of the assets
        '''

        theta = np.expand_dims(self.theta, axis = 1)
        kappa = np.expand_dims(self.kappa, axis = 1)
        sigma = np.expand_dims(self.sigma, axis = 1)
        rho = np.expand_dims(self.rho, axis = 1)
        mu = np.expand_dims(np.diag(self.B), axis = 1)
        one = np.ones((self.dim_nohedge, 1))

        w0 = np.zeros((self.dim, 1)) # \chi_1 (T) = w (0) = 0
        y0 = np.zeros((self.dim, 1)) # \chi_0 (T) = y (0) = 0
        A = - mu ** 2
        B = - kappa - 2 * rho * sigma * mu
        C = sigma ** 2 * (one - 2 * rho ** 2)/2
        D = np.sqrt(B ** 2 - 4 * A * C)
        F = kappa * theta
        w0_hat = B/2 + C * w0
        T = self.total_time        


        Num = lambda tau: (w0_hat + D/2) * np.exp(-D * tau/2) + (w0_hat - D/2) * np.exp(D * tau/2) 
        Den = lambda tau: (w0_hat + D/2) * np.exp(-D * tau/2) - (w0_hat - D/2) * np.exp(D * tau/2)

        w = lambda tau:  -B/2/C + D * Num(tau)/Den(tau)/2/C
        y = lambda tau: y0 + F * (-B/2/C * tau - np.log(Den(tau)/D)/C)

        return [y(T),w(T)]




