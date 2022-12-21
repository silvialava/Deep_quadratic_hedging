import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import tensorflow as tf
from solver import BSDESolver
import HestonEquation as eqn
import munch
import sys
import os
import random
import time

if __name__ == "__main__":

    ## Specify if we want to train the model (and save it), or to load it   
    train_model = True  # If this is False, the next is not checked 
    save_model = True

    ## Specify if we want to simulate new trajectories (and save them), or to load them   
    simulate_trajectories = True  # If this is False, the next is not checked  
    save_trajectories = False
 
    ## Specify if we want to create the figures (and save them)
    create_figures = True  # If this is False, the next is not checked 
    save_figures = True


    ## Setting
 
    num_time_interval = 100

    dim = 100 # Dimension of hedgeable Brownian motion
    dim_nohedge = dim # Dimension of non-hedgeable Brownian motion
    P = 2048 # Number of outer Monte Carlo loops
    batch_size = 128
    total_time = 1.0
    strike = 100.

    x_init = 100. * np.ones((dim, 1))
    v_init= 0.025 * np.ones((dim_nohedge, 1))
    v_trunc = 1e-8 # Lower truncation level for the variance process.
    # If v_trunc = None then it is set to 1e-10 by default

    sigma = 0.1 * np.ones(dim_nohedge) 
    kappa = 0.5 * np.ones(dim_nohedge)
    theta = 0.05 * np.ones(dim_nohedge)
    rho = -0.45 * np.ones(dim_nohedge)
   
    print('Feller condition satisfied: ', bool(np.prod(2*kappa*theta > sigma**2)))
 
    r = 0.0
    B = 0.1 * np.diag(np.ones(dim_nohedge))     
    A = 1.0 * np.diag(np.ones(dim_nohedge)) 
    
    ## Specify the name of the directory where model and simulations lie
    path_dir = 'LocalRisk{}points_{}dimensional_0'.format(num_time_interval, dim)
    
    ## Algorithm configuration   
    
    config = {
                "eqn_config": {
                    "_comment": "LR of a call option under Heston",
                    "eqn_name": "HestonCall",
                    "total_time": total_time,
                    "dim": dim,
                    "dim_nohedge": dim_nohedge,
                    "num_time_interval": num_time_interval,
                    "strike": strike,
                    "x_init": x_init,
                    "v_init": v_init,
                    "v_trunc": v_trunc,
                    "sigma": sigma,
                    "kappa": kappa,
                    "theta": theta,
                    "rho": rho,
                    "A": A,
                    "B": B,
                    "r": r
                },

                "net_config": {
                    "y_init_range": [0., 100.],
                    "num_hiddens": [dim + dim_nohedge + 20, dim + dim_nohedge + 20, dim + dim_nohedge + 20, dim + dim_nohedge + 20],
                    "lr_values": [5e-2, 5e-3],
                    #"lr_values": [1e-3, 5e-4],
                    "lr_boundaries": [4000],
                    "num_iterations": 8000,
                    "batch_size": batch_size,
                    "valid_size": 256,
                    "logging_frequency": 100,
                    "dtype": "float64",
                    "verbose": True
                }
            }
    
    config = munch.munchify(config) 
    bsde = getattr(eqn, config.eqn_config.eqn_name)(config.eqn_config)
    tf.keras.backend.set_floatx(config.net_config.dtype)
    sys.exit(0) 
    
    ## Simulate the process under Q so to narrow the range for y in the BSDE solver 
    P_sim = 100000
    dwb_testQ, xv_testQ = bsde.sample_underQ(P_sim, v_trunc)
    MC_priceQ = np.mean(np.maximum(np.sum(xv_testQ[:, :dim, -1], axis = 1) - dim * strike, 0))* np.exp(-r * total_time)
    config.net_config['y_init_range'] = [np.maximum(int(MC_priceQ * 0.95), 0), int(MC_priceQ * 1.05)]
    
    print('MC price under Q = ', MC_priceQ)   

    #sys.exit(0)
    ## Apply algorithm
    bsde_solver = BSDESolver(config, bsde)
    
    if train_model:

        ## Train the model
        tic = time.perf_counter()
        training_history = bsde_solver.train()  
        toc = time.perf_counter()
        print(f'Training time: {toc - tic:0.4f}')

        if save_model:

            # Create the directory
            try:    
                os.mkdir(path_dir)
		#print('Directory ', '\033[1m' + path_dir + '\033[0m', ' created')

            except OSError:
                print('Directory ', '\033[1m' + path_dir + '\033[0m', ' already existing: try another name')
            
            ## Save
            path = path_dir + '/Model_LR{}points_{}dim'.format(num_time_interval, dim)
            #bsde_solver.model.save_model(path)
            np.save(path_dir + '/training_history{}_{}dim.npy'.format(num_time_interval, dim), training_history)
            np.save(path_dir + '/training_time{}_{}dim.npy'.format(num_time_interval, dim), toc-tic)
    else:

        ## Load
        path = path_dir +  '/Model_LR{}points_{}dim'.format(num_time_interval, dim)
        bsde_solver.model.load_model(path)
        training_history = np.load(path_dir + '/training_history{}_{}dim.npy'.format(num_time_interval, dim))

    '''
   
    Computing controls and hedging strategies
    
    '''
    
    if simulate_trajectories:
            
        ## Simulate trajectories   
        dwb_test, xv_test = bsde.sample(P_sim, v_trunc)
 
        ## Simulate solution of algorithm
        lr_test = bsde_solver.model.simulate_path([dwb_test, xv_test])   
        
        '''
        ## WARNING: THIS NEEDS TO BE FIXED FOR THE MULTIMENSIONAL CASE
        ## Compute the controls in the BSDE approach       
        all_one_vec = tf.ones(shape=tf.stack([tf.shape(dwb_test)[0], 1]), dtype=bsde_solver.model.net_config['dtype']) 
        z = tf.expand_dims(tf.matmul(all_one_vec, bsde_solver.model.z_init), axis = 0) 
        nets = bsde_solver.model.subnet 
        zetas = [nets[t](xv_test[:, :, t+1], False) for t in range(0, bsde.num_time_interval-1)] 
        zetas = tf.concat([z, zetas], axis = 0)
        zetas = np.array(zetas) 
        zetas = tf.convert_to_tensor(zetas) 
        zetas_test = tf.transpose(zetas, [1, 2, 0])
        '''
        if save_trajectories:
        
            ## Save
            np.save(path_dir + '/xv_test{}_{}dim.npy'.format(num_time_interval, dim), xv_test)
            np.save(path_dir + '/dwb_test{}_{}dim.npy'.format(num_time_interval, dim), dwb_test)
            np.save(path_dir + '/lr_test{}_{}dim.npy'.format(num_time_interval, dim), lr_test)    
            #np.save(path_dir + '/zetas_test{}_{}dim.npy'.format(num_time_interval, dim), zetas_test)
    else: 
        
        ## Load
        xv_test = np.load(path_dir + '/xv_test{}_{}dim.npy'.format(num_time_interval, dim))
        dwb_test = np.load(path_dir + '/dwb_test{}_{}dim.npy'.format(num_time_interval, dim))
        lr_test = np.load(path_dir + '/lr_test{}_{}dim.npy'.format(num_time_interval, dim))
        #lr_test = np.squeeze(lr_test)
        #zetas_test = np.load(path_dir + '/zetas_test{}_{}dim.npy'.format(num_time_interval, dim))

    BSDE_price = np.mean(lr_test[:, 0, 0])
    print('BSDE price = ', BSDE_price, ', ', np.abs(MC_priceQ-BSDE_price)/MC_priceQ*100, '%')
  
    #np.save(path_dir + '/BSDEprice{}_{}dim.npy'.format(num_time_interval, dim), BSDE_price)
    #np.save(path_dir + '/MCprice{}_{}dim.npy'.format(num_time_interval, dim), MC_priceQ)
 
    
    '''
    
    PLOTS
    
    '''
    if create_figures:
 
        ## Plots of the loss
        #import matplotlib
        #matplotlib.rc('font', size=20)
  
        f1 = plt.figure(figsize = (16, 14))
        plt.plot(training_history[1:, 0], np.log(training_history[1:, 1]), 'k', linewidth = 4)   
        plt.xlim((-50, config.net_config.num_iterations+50))
        plt.xticks(fontsize = 25)
        #plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useLocale= True)
        plt.yticks(fontsize = 25)
        #plt.yticks(np.arange(0, 15, 1), fontsize=25)
        plt.xlabel('Iterations', fontsize = 35)
        plt.ylabel('Log loss', fontsize = 35)   

        #plt.show()

        if save_figures:
            f1.savefig(path_dir + '/Log_loss{}_{}dim.pdf'.format(num_time_interval, dim), bbox_inches = 'tight', pad_inches = 0.01)
