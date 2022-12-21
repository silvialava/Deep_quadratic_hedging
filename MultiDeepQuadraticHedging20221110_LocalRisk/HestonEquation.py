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


        ## Multi-dim. CIR parameters

        self.kappa = eqn_config['kappa'] # speed of mean reversion, array (dim_nohedge, 1)
        self.theta = eqn_config['theta'] # long term mean, array (dim_nohedge, 1)
        self.sigma = eqn_config['sigma'] # vol of var, array (dim_nohedge, 1)
        self.rho = eqn_config['rho'] # correlazione tra w e b, array (dim_nohedge, 1)
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

            xv_sample[:, self.dim:,i + 1] =  xv_sample[:, self.dim:, i] + kappa * (theta - xv_sample[:, self.dim:, i]) * self.delta_t + sigma * np.sqrt(np.abs(xv_sample[:, self.dim:, i])) * (rho * dwb_sample[:, :self.dim, i] + np.sqrt(one-rho**2) * dwb_sample[:, self.dim:, i])
            
            xv_sample[:, self.dim:, i+1] = tf.math.maximum(xv_sample[:, self.dim:, i+1], truncation)

            xv_sample[:, :self.dim, i + 1] = xv_sample[:, :self.dim, i] \
            + xv_sample[:, :self.dim, i] * (np.transpose(self.B.dot(np.transpose(xv_sample[:, self.dim:, i]))) * self.delta_t + np.transpose(self.A.dot(np.transpose(np.sqrt(np.abs(xv_sample[:, self.dim:, i])) * dwb_sample[:, :self.dim, i]))))
        
 
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
 
        return  -tf.expand_dims(tf.linalg.diag_part(tf.tensordot(theta, tf.transpose(z[:, :self.dim]), axes = 1)), axis = 1)


    def g_tf(self, t, x):
        
        '''

        Discounted aggregated call payoff

        '''
        
        result = tf.math.maximum(tf.reduce_sum(x[:, :self.dim], axis = 1, keepdims=True) -  self.dim * self.strike, 0)

        return np.exp(- self.r * self.total_time) * result


    def sample_underQ(self, num_sample, truncation = None):      
      
        '''

        Simulates the trajectories of the BMs and the
        Heston model under the minimal Martingale Measure.

        '''
        
        if truncation == None: truncation = 1e-10

        dwb_sample = normal.rvs(size=[num_sample,     
                                     self.dim + self.dim_nohedge,
                                     self.num_time_interval]) * self.sqrt_delta_t

 
        theta = self.theta
        kappa = self.kappa
        sigma = self.sigma
        rho = self.rho

        ## Tensor in 3 dimensions: (number of paths, dimension, number of time instants)

        xv_sample = np.zeros([num_sample, self.dim + self.dim, self.num_time_interval + 1]) 
        xv_sample[:, :, 0] = np.transpose(np.repeat(self.xv_init, num_sample, 1))
       
        one = np.ones(self.dim_nohedge)

        for i in range(self.num_time_interval):
            
            ## Market price of risk

            diagVi =  np.apply_along_axis(np.diag, 1, xv_sample[:, self.dim:, i])
            sigma_i = np.matmul(self.A, np.sqrt(np.abs(diagVi)))   

            mu_i = np.transpose(self.B.dot(np.transpose(xv_sample[:, self.dim:, i]))) 
            theta_i = np.squeeze(np.matmul(np.linalg.inv(sigma_i), np.expand_dims(mu_i - self.r * one, axis = 2))) 

            ## Euler scheme

            xv_sample[:, self.dim:, i+1] =  xv_sample[:, self.dim:, i] + (kappa * (theta - xv_sample[:, self.dim:, i]) - sigma * np.sqrt(np.abs(xv_sample[:, self.dim:, i])) * rho * theta_i ) * self.delta_t + sigma * np.sqrt(np.abs(xv_sample[:, self.dim:, i])) * (rho * dwb_sample[:, :self.dim, i] + np.sqrt(one-rho**2) * dwb_sample[:, self.dim:, i])
            
            xv_sample[:, self.dim:, i+1] = tf.math.maximum(xv_sample[:, self.dim:, i+1], truncation)

            xv_sample[:, :self.dim, i + 1] = xv_sample[:, :self.dim, i] * (1 + self.r * self.delta_t + np.transpose(self.A.dot(np.transpose(np.sqrt(np.abs(xv_sample[:, self.dim:, i])) * dwb_sample[:, :self.dim, i]))))
        
        if np.min(xv_sample[:, self.dim:, :]) < truncation: tf.print('!!! Warning: negative values in Q!!! ',  np.min(xv_sample[:, self.dim:, :]))

        ## Truncate the variance to avoid too small values
        #xv_sample[:, self.dim:, :] = tf.math.maximum(xv_sample[:, self.dim:, :], truncation)
        
        return dwb_sample, xv_sample
    
    
    def getDensityProcess(self, dwb_sample, xv_sample):
        density = np.zeros([xv_sample.shape[0], self.num_time_interval + 1])
        density[:,0] = 1.0
        
        one = np.ones(self.dim_nohedge)
        
        for i in range(self.num_time_interval):
            
            diagVi =  np.apply_along_axis(np.diag, 1, xv_sample[:, self.dim:, i])
            
            sigma_i = np.matmul(self.A, np.sqrt(np.abs(diagVi)))   
            mu_i = np.transpose(self.B.dot(np.transpose(xv_sample[:, self.dim:, i]))) 
            theta_i = np.squeeze(np.matmul(np.linalg.inv(sigma_i), np.expand_dims(mu_i - self.r * one, axis = 2))) 
             
            stochIntegral = np.sum(theta_i * dwb_sample[:, :self.dim, i],axis = 1)
            #tf.print(stochIntegral)
            compensator = 0.5*np.sum(theta_i * theta_i,axis = 1)*self.delta_t
            #tf.print(compensator)
            density[:,i+1] = density[:,i] * np.exp(-stochIntegral - compensator)
            
        return density


