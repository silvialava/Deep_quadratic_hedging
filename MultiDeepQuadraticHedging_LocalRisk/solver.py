#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 17:23:21 2022

@author: silvialavagnini
"""

import logging
import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import tensorflow.keras.layers as layers
DELTA_CLIP = 50.0
import os

class BSDESolver(object):
    """The fully connected neural network model."""
    def __init__(self, config, bsde):
        self.eqn_config = config['eqn_config']
        self.net_config = config['net_config']
        self.bsde = bsde
       
        self.model = NonsharedModel(config, bsde)
        #self.y_init = self.model.y_init

        try:
            lr_schedule = config['net_config']['lr_schedule']
        except AttributeError:
            lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                self.net_config['lr_boundaries'], self.net_config['lr_values'])     
        except KeyError:
            lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                self.net_config['lr_boundaries'], self.net_config['lr_values'])     
    
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-8)

    def train(self):
        start_time = time.time()
        training_history = []
        valid_data = self.bsde.sample(self.net_config['valid_size'], self.eqn_config['v_trunc'])
        
        # begin sgd iteration
        for step in tqdm(range(self.net_config['num_iterations']+1)):
            if step % self.net_config['logging_frequency'] == 0:
                loss = self.loss_fn(valid_data, training=False).numpy()
                relative_loss = self.relative_loss(valid_data, training=False).numpy()
                y_init = self.model.y_init.numpy()[0]
                elapsed_time = time.time() - start_time
                training_history.append([step, loss, relative_loss,  y_init, elapsed_time])
                if self.net_config['verbose']:
                    #logging.info("step: %5u,    loss: %.4e, Y0: %.4e,   elapsed time: %3u" % (
                    #    step, loss, y_init, elapsed_time))
                    print("step: %5u,    loss: %.4e,  relative loss: %.4e,  Y0: %.4e,   elapsed time: %3u" % (
                        step, loss, relative_loss, y_init, elapsed_time))
            self.train_step(self.bsde.sample(self.net_config['batch_size'], self.eqn_config['v_trunc']))            
        return np.array(training_history)

    def loss_fn(self, inputs, training):

        dw, x = inputs
        y_terminal = self.model(inputs, training)
        
        delta = y_terminal - self.bsde.g_tf(self.bsde.total_time, x[:, :, -1])
        # use linear approximation outside the clipped range
        loss = tf.reduce_mean(tf.where(tf.abs(delta) < DELTA_CLIP, tf.square(delta),
                                    2 * DELTA_CLIP * tf.abs(delta) - DELTA_CLIP ** 2))

        return loss

    def relative_loss(self, inputs, training):
    
        '''

        We define the relative loss between x and y as

        relative_loss(x, y) = |x-y|/max(|x|, |y|)

        We then square the components, take the mean and then the squared root. 

        '''

        dw, x = inputs
        y_terminal = self.model(inputs, training)
        #tf.print('y_terminal =', y_terminal)
        #delta = tf.math.abs(y_terminal - self.bsde.g_tf(self.bsde.total_time, x[:, :, -1]))
        delta = tf.math.abs(tf.reduce_mean(y_terminal) - tf.reduce_mean(self.bsde.g_tf(self.bsde.total_time, x[:, :, -1])))
        #denominator = tf.math.maximum(tf.math.abs(y_terminal), tf.math.abs( self.bsde.g_tf(self.bsde.total_time, x[:, :, -1])))
        denominator = tf.math.abs(self.bsde.g_tf(self.bsde.total_time, x[:, :, -1])) # tf.math.abs(y_terminal)
        #relative_loss = tf.sqrt(tf.reduce_mean(tf.math.divide(delta, denominator) ** 2 ))
        #relative_loss = tf.reduce_mean(tf.math.divide(delta, denominator))
        relative_loss = tf.math.divide(delta, tf.reduce_mean(denominator))

        return relative_loss

    def grad(self, inputs, training):
        with tf.GradientTape(persistent=True) as tape:
            loss = self.loss_fn(inputs, training)
        grad = tape.gradient(loss, self.model.trainable_variables)
        del tape
        return grad

    @tf.function
    def train_step(self, train_data):
        grad = self.grad(train_data, training=True)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))     


class NonsharedModel(tf.keras.Model):
    def __init__(self, config, bsde):
        super(NonsharedModel, self).__init__()
        self.config = config
        self.eqn_config = config['eqn_config']
        self.net_config = config['net_config']
        self.bsde = bsde       
        self.dim = bsde.dim
        self.y_init = tf.Variable(np.random.uniform(low=self.net_config['y_init_range'][0],
                                                    high=self.net_config['y_init_range'][1],
                                                    size=[1]),dtype=self.net_config['dtype']
                                  )
        #self.y_init = tf.Variable(self.net_config['y_init_range'][0] ,dtype=self.net_config['dtype'])
        self.z_init = tf.Variable(np.random.uniform(low=-.1, high=.1,
                                                    size=[1, self.eqn_config['dim']+self.eqn_config['dim_nohedge']]),dtype=self.net_config['dtype']
                                  )        
        
        self.subnet = [FeedForwardSubNet(config,bsde.dim + bsde.dim_nohedge) for _ in range(self.bsde.num_time_interval-1)]

   
    def save_model(self, path_dir = 'Model_LR'):
        
        try: 
            os.mkdir(path_dir)
 
            nets = self.subnet
        
            for i in range(self.bsde.num_time_interval-1):
                path = path_dir + '/net_{}'.format(i)
                model = nets[i]
                model.save(path)
 
            path = path_dir + '/z_init.npy'
            np.save(path, self.z_init.numpy())
 
            path = path_dir + '/y_init.npy'
            np.save(path, self.y_init.numpy())

            print('Directory ', '\033[1m' + path_dir + '\033[0m', ' created')
        
        except OSError:
            print('Directory ', '\033[1m' + path_dir + '\033[0m', '  already existing: try another name')
        

    def load_model(self, path_dir = 'Model_LR'):
        
        path = path_dir + '/z_init.npy' 
        self.z_init = np.load(path)
        
        path = path_dir + '/y_init.npy' 
        self.y_init = np.load(path)

        nets = []        
        for i in range(self.bsde.num_time_interval-1):
            path = path_dir + '/net_{}'.format(i)
            model = tf.keras.models.load_model(path)
            nets.append(model)
        
        self.subnet = nets


    @tf.function
    def call(self, inputs, training):
        dw, x = inputs
        time_stamp = np.arange(0, self.eqn_config['num_time_interval']) * self.bsde.delta_t
        all_one_vec = tf.ones(shape=tf.stack([tf.shape(dw)[0], 1]), dtype=self.net_config['dtype'])
        y = all_one_vec * self.y_init
        z = tf.matmul(all_one_vec, self.z_init)
        #tf.print(z.get_shape())
        #tf.print(z)  
        #tf.print(y.get_shape())
        #tf.print(y)
        #tf.print(self.bsde.f_tf(time_stamp[0], x[:, :, 0], y, z))
        
        for t in range(0, self.bsde.num_time_interval-1):
            y = y - self.bsde.delta_t * (
                self.bsde.f_tf(time_stamp[t], x[:, :, t], y, z)
            ) + tf.reduce_sum(z * dw[:, :, t], 1, keepdims=True) 
            #tf.print(y)
            try:          
                z = self.subnet[t](x[:, :, t + 1], training) / self.bsde.dim
                #tf.print(z)
            except TypeError:
                z = self.subnet(tf.concat([time_stamp[t+1]*all_one_vec,x[:, :, t + 1]],axis=1), training=training) / self.bsde.dim
        # terminal time
        y = y - self.bsde.delta_t * self.bsde.f_tf(time_stamp[-1], x[:, :, -2], y, z) + \
            tf.reduce_sum(z * dw[:, :, -1], 1, keepdims=True)
          
        return y         
    
    def predict_step(self, data):
        dw, x = data[0]
        time_stamp = np.arange(0, self.eqn_config['num_time_interval']) * self.bsde.delta_t
        all_one_vec = tf.ones(shape=tf.stack([tf.shape(dw)[0], 1]), dtype=self.net_config['dtype'])
        y = all_one_vec * self.y_init
        z = tf.matmul(all_one_vec, self.z_init)        
        
        history = tf.TensorArray(self.net_config['dtype'], size=self.bsde.num_time_interval+1)     
        history = history.write(0,y)
        
        for t in range(0, self.bsde.num_time_interval-1):
            y = y - self.bsde.delta_t * (
                self.bsde.f_tf(time_stamp[t], x[:, :, t], y, z)
            ) + tf.reduce_sum(z * dw[:, :, t], 1, keepdims=True)
            
            history = history.write(t+1,y)
            try:          
                z = self.subnet[t](x[:, :, t + 1], training=False) / self.bsde.dim
            except TypeError:
                z = self.subnet(tf.concat([time_stamp[t+1]*all_one_vec,x[:, :, t + 1]],axis=1), training=False) / self.bsde.dim
        # terminal time
        y = y - self.bsde.delta_t * self.bsde.f_tf(time_stamp[-1], x[:, :, -2], y, z) + \
            tf.reduce_sum(z * dw[:, :, -1], 1, keepdims=True)
    
        history = history.write(self.bsde.num_time_interval,y)
        history = tf.transpose(history.stack(),perm=[1,2,0])
        return dw,x,history      
    
    def simulate_path(self,num_sample):
        return self.predict(num_sample)[2]           


class FeedForwardSubNet(tf.keras.Model):
    def __init__(self, config,dim):
        super(FeedForwardSubNet, self).__init__()        
        num_hiddens = config['net_config']['num_hiddens']
        self.bn_layers = [
            tf.keras.layers.BatchNormalization(
                momentum=0.99,
                epsilon=1e-6,
                beta_initializer=tf.random_normal_initializer(0.0, stddev=0.1),
                gamma_initializer=tf.random_uniform_initializer(0.1, 0.5)
            )
            for _ in range(len(num_hiddens) + 2)]
        self.dense_layers = [tf.keras.layers.Dense(num_hiddens[i],
                                                   use_bias=False,
                                                   activation=None,)
                             for i in range(len(num_hiddens))]
        # final output should be gradient of size dim
        self.dense_layers.append(tf.keras.layers.Dense(dim, activation=None))

    def call(self, x, training):
        """structure: bn -> (dense -> bn -> relu) * len(num_hiddens) -> dense """
        x = self.bn_layers[0](x, training)
        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
            x = self.bn_layers[i+1](x, training)
            x = tf.nn.relu(x)
        x = self.dense_layers[-1](x)
        return x
