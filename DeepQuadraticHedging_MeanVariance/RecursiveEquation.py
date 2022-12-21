import numpy as np
import tensorflow as tf
from tqdm import tqdm

class RecursiveEquation(object):
    """Base class for defining PDE related function."""

    def __init__(self, eqn_config):
        self.dim = eqn_config.dim
        self.dim_nohedge = eqn_config.dim_nohedge
        self.total_time = eqn_config.total_time
        self.num_time_interval = eqn_config.num_time_interval
        self.delta_t = self.total_time / self.num_time_interval
        self.sqrt_delta_t = np.sqrt(self.delta_t)
        self.y_init = None
        self.bsde = eqn_config.bsde  # Class Equation, to simulate dw and x
        self.bsde_model = eqn_config.bsde_model

    def sample(self, num_sample):
        """Sample forward SDE."""
        """Sample RN BSDE."""
        raise NotImplementedError

    def f_tf(self, t, x, y, z, rn):
        """Generator function in the PDE."""
        raise NotImplementedError

    def g_tf(self, t, x, rn):
        """Terminal condition of the PDE."""
        raise NotImplementedError

class MVP(RecursiveEquation):

    '''

    MVP = Mean Value Process

    '''

    def __init__(self,eqn_config):
        self.dim = eqn_config.bsde.dim
        self.dim_nohedge = eqn_config.bsde.dim_nohedge
        self.total_time = eqn_config.total_time
        self.num_time_interval = eqn_config.num_time_interval
        self.delta_t = self.total_time / self.num_time_interval
        self.sqrt_delta_t = np.sqrt(self.delta_t)
        self.y_init = None       
        self.strike = eqn_config.bsde.strike
        self.bsde = eqn_config.bsde  # Class Equation, to simulate dw and x
        self.bsde_model = eqn_config.bsde_model

        '''
  
        CIR parameters
           
        '''
   
        self.r = eqn_config.r
        self.mu = eqn_config.mu
 

## Class NonsharedModel to simulate trajectories

    def sample(self, num_sample, truncation):

        sample = self.bsde.sample(num_sample, truncation)
        dw, x = sample
        rn =  self.bsde_model.simulate_path(sample)

        all_one_vec = tf.ones(shape=tf.stack([tf.shape(dw)[0], 1]), dtype=self.bsde_model.net_config['dtype'])
        z = tf.expand_dims(tf.matmul(all_one_vec, self.bsde_model.z_init), axis = 0)
        nets = self.bsde_model.subnet
        lambdas = [nets[t](x[:, :, t+1], False) for t in range(0, self.bsde.num_time_interval-1)]
        lambdas = tf.concat([z, lambdas], axis = 0) 
        lambdas = np.array(lambdas)
        lambdas = tf.convert_to_tensor(lambdas)
        lambdas = tf.transpose(lambdas, [1, 2, 0])
            
        return dw, x, rn, lambdas


    @tf.function
    def f_tf(self, t, x, y, z, rn, lambdas):

        '''
       
        In the notation of the paper:
            
        x : [ S, Y^2 ]
        y : h        
        z : [\eta_1, \eta_2 ]
        rn : p  (Radon-Nikodym derivative, solution of the SRE)
        lambdas = [\Lambda_1, \Lambda_2]
        
        Warning the drift here is assumed to be of the form
        mu(t) = mu * Y^2 like in the paper by Kallsen
                    
        '''
 
        ## This syntax is ok only in the one dimensional setting.

        theta = (self.mu * x[:, self.dim:] - self.r)/tf.math.sqrt( x[:, self.dim:])

        #return  self.r * y + theta * z[:, :self.dim] - lambdas[:, self.dim:] * z[:, self.dim:] / rn
        return  - theta * z[:, :self.dim] + lambdas[:, self.dim:] * z[:, self.dim:] / rn


    @tf.function
    def g_tf(self, t, x, rn, lambdas):
        
        '''

        Call payoff

        '''
        
        return tf.math.maximum(x[:, :self.dim] - self.strike, 0)*np.exp(-self.r*self.total_time)


    def WealthSDE(self, num_MC, v, dw, p, h, lambda_1, eta_1, y= 0): 
        
        ## NOTE that this code assumes that all processes are 1-dim
        ## I need to fix it to allow for multi-dimensional processes!!

        wealth = np.zeros((num_MC, self.num_time_interval + 1)) 
        wealth[:, 0] = y * np.ones(num_MC) 
        pi = np.zeros((num_MC, self.num_time_interval))
        
        for i in range(self.num_time_interval):
            theta_t = np.divide(self.mu * v[:, i] - self.r, np.sqrt(v[:, i]))
            pi[:, i] = np.multiply(np.divide(theta_t + np.divide(lambda_1[:, i], p[:, i]),np.sqrt(v[:, i])), h[:, i] - wealth[:, i]) + np.divide(eta_1[:, i], np.sqrt(v[:, i]))
            wealth[:, i + 1] = wealth[:, i] + (self.r * wealth[:, i] + np.multiply(self.mu * v[:, i]  - self.r, pi[:, i])) * self.delta_t + np.multiply(np.multiply(np.sqrt(v[:, i]), pi[:, i]), dw[:, i])            
        
        pi_0 = wealth[:, :self.num_time_interval] - pi

        return pi_0, pi
