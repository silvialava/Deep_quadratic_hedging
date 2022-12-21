import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import tensorflow as tf
from solver import BSDESolver
from RecursiveSolver import RecSolver
import HestonEquation as eqn
import RecursiveEquation as receqn
import HestonPDEsolver_2 as pde
import time
import munch
import sys
import os

if __name__ == "__main__":

    ## Specify if we want to train the model (and save it), or to load it   
    train_model = False # If this is False, the next is not checked 
    save_model = True

    ## Specify if we want to simulate new trajectories (and save them), or to load them   
    simulate_trajectories = False  # If this is False, the next is not checked      
    save_trajectories = True
 
    ## Specify if we want to create the figures (and save them)
    create_figures = False
    save_figures = True

    
    ## Setting
   
    num_time_interval = 10

    dim = 1 # Dimension of hedgeable Brownian motion
    dim_nohedge = 1 # Dimension of non-hedgeable Brownian motion
    P = 2048 # Number of outer Monte Carlo loops
    batch_size = 128
    total_time = 1.0
    strike = 100

    x_init = 100
    v_init= 0.025
    v_trunc = 1e-5 # Lower truncation level for the variance process.
    # If v_trunc = None then it is set to 1e-10 by default

    sigma = 0.1 
    kappa = 0.5
    theta = 0.05 
    rho = -0.45

    r = 0.0
    mu = 0.1

    print('Feller condition satisfied: ', bool(2*kappa*theta > sigma**2))

    ## Specify the name of the directory where model and simulations lie
    path_dir = 'MeanVariance{}points_new'.format(num_time_interval)
  
    ## Algorithm 1 configuration

    config = {
                "eqn_config": {
                    "_comment": "price a call option using Heston",
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
                    "r": r ,
                    "mu": mu,
                },

                "net_config": {
                    "y_init_range": [0.5, 2],
                    "num_hiddens": [dim + dim_nohedge + 2, dim + dim_nohedge + 2, dim + dim_nohedge + 2, dim + dim_nohedge + 2],
                    "lr_values": [5e-2, 5e-3],
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
    Psim = 100000
    dwb_testQ, xv_testQ = bsde.sample_underQ(Psim, v_trunc)
    MC_priceQ = np.mean(np.maximum(xv_testQ[:, 0, -1] - strike, 0)) * np.exp(-r * total_time)
    print('MC price under Q  = ', MC_priceQ)

    ## Apply algorithm 1 
    
    bsde_solver = BSDESolver(config, bsde)
 
    if train_model:

        ## Train the model
        tic1 = time.perf_counter()
        training_history1 = bsde_solver.train()
        toc1 = time.perf_counter()
        print(f'Training time 1: {toc1 - tic1:0.4f}')

        if save_model:

            # Create the directory
            try:
                os.mkdir(path_dir)
                print('Directory ', '\033[1m' + path_dir + '\033[0m', ' created')

            except OSError:
                print('Directory ', '\033[1m' + path_dir + '\033[0m', ' already existing: try another name')

            ## Save
            path = path_dir + '/Model1_MV{}points'.format(num_time_interval)
            bsde_solver.model.save_model(path)
            np.save(path_dir + '/training_history1_{}.npy'.format(num_time_interval), training_history1)
    else:

        ## Load
        path = path_dir +  '/Model1_MV{}points'.format(num_time_interval)
        bsde_solver.model.load_model(path)
        training_history1 = np.load(path_dir + '/training_history1_{}.npy'.format(num_time_interval))


    ## Algorithm 2 configuration
    
    configMVP = {
                "eqn_config": {
                    "_comment": "MVP of a call option",
                    "dim": dim,                
                    "dim_nohedge": dim_nohedge,
                    "eqn_name": "MVP",
                    "total_time": total_time,
                    "num_time_interval": num_time_interval,
                    "v_trunc": v_trunc,
                    "r": r,
                    "mu": mu,
                    "bsde": bsde,
                    "bsde_model": bsde_solver.model
                },
                "net_config": {
                    "y_init_range": [np.maximum(int(MC_priceQ * 0.95), 0), int(MC_priceQ * 1.05)],
                    "num_hiddens": [dim + dim_nohedge + 20, dim + dim_nohedge + 20, dim + dim_nohedge + 20, dim + dim_nohedge + 20],
                    "lr_values": [5e-2, 5e-3],
                    "lr_boundaries": [4000],
                    "num_iterations": 8000,
                    "batch_size": batch_size,
                    "valid_size": 256,
                    "logging_frequency": 100,
                    "dtype": "float64",
                    "verbose": True
                }
            }

    configMVP = munch.munchify(configMVP) 
    mvpbsde = getattr(receqn, configMVP.eqn_config.eqn_name)(configMVP.eqn_config) 
    tf.keras.backend.set_floatx(configMVP.net_config.dtype)
    
    
    ## Apply algorithm 2
    
    mvp_solver = RecSolver(configMVP, mvpbsde)
 
    if train_model:

        ## Train the model
        tic2 = time.perf_counter()
        training_history2 = mvp_solver.train()
        toc2 = time.perf_counter()
        print(f'Training time 2: {toc2 - tic2:0.4f}')
        print(f'Total training time: {toc2 - tic2 + toc1 - tic1:0.4f}')
        
        if save_model:

            ## Save
            path = path_dir + '/Model2_MV{}points'.format(num_time_interval)
            mvp_solver.model.save_model(path)
            np.save(path_dir + '/training_history2_{}.npy'.format(num_time_interval), training_history2)
    else:

        ## Load
        path = path_dir +  '/Model2_MV{}points'.format(num_time_interval)
        mvp_solver.model.load_model(path)
        training_history2 = np.load(path_dir + '/training_history2_{}.npy'.format(num_time_interval))

   
    '''
    
    Computing controls and hedging strategies
 
    '''
    if simulate_trajectories:

        ## Simulate trajectories
        dwb_test, xv_test = bsde.sample(Psim, v_trunc)
 
        ## Simulate solution of algorithm 1
        rn_test = bsde_solver.model.simulate_path([dwb_test, xv_test])
   
        ## Simulate solution of algorithm 2
        all_one_vec = tf.ones(shape=tf.stack([tf.shape(dwb_test)[0], 1]), dtype=bsde_solver.model.net_config['dtype'])
        z = tf.expand_dims(tf.matmul(all_one_vec, bsde_solver.model.z_init), axis = 0)
        nets = bsde_solver.model.subnet
        lambdas = [nets[t](xv_test[:, :, t+1], False) for t in range(0, bsde.num_time_interval-1)]
        lambdas = tf.concat([z, lambdas], axis = 0) 
        lambdas = np.array(lambdas)
        lambdas = tf.convert_to_tensor(lambdas)
        lambdas_test = tf.transpose(lambdas, [1, 2, 0])
        mvp_test = mvp_solver.model.simulate_path([dwb_test, xv_test, rn_test,  lambdas_test])    
        mvp_test = np.squeeze(mvp_test) 
        
        ## Compute the controls in the BSDE approach
        all_one_vec = tf.ones(shape=tf.stack([tf.shape(dwb_test)[0], 1]), dtype=mvp_solver.model.net_config['dtype'])
        z = tf.expand_dims(tf.matmul(all_one_vec, mvp_solver.model.z_init), axis = 0)
        nets = mvp_solver.model.subnet
        etas = [nets[t](xv_test[:, :, t+1], False) for t in range(0, mvpbsde.num_time_interval-1)]
        etas = tf.concat([z, etas], axis = 0) 
        etas = np.array(etas)
        etas = tf.convert_to_tensor(etas)
        etas_test = tf.transpose(etas, [1, 2, 0])

        if save_trajectories:   
        
            ## Save
            np.save(path_dir + '/xv_test{}.npy'.format(num_time_interval), xv_test)
            np.save(path_dir + '/dwb_test{}.npy'.format(num_time_interval), dwb_test)
            np.save(path_dir + '/mvp_test{}.npy'.format(num_time_interval), mvp_test)
            np.save(path_dir + '/rn_test{}.npy'.format(num_time_interval), rn_test)
            np.save(path_dir + '/lambdas_test{}.npy'.format(num_time_interval), lambdas_test)   
            np.save(path_dir + '/etas_test{}.npy'.format(num_time_interval), etas_test) 

    else:

        ## Load 
        xv_test = np.load(path_dir + '/xv_test{}.npy'.format(num_time_interval))
        dwb_test = np.load(path_dir + '/dwb_test{}.npy'.format(num_time_interval))
        mvp_test = np.load(path_dir + '/mvp_test{}.npy'.format(num_time_interval))
        rn_test = np.load(path_dir + '/rn_test{}.npy'.format(num_time_interval))
        lambdas_test = np.load(path_dir + '/lambdas_test{}.npy'.format(num_time_interval))   
        etas_test = np.load(path_dir + '/etas_test{}.npy'.format(num_time_interval))

 
    BSDE_price = mvp_test[0, 0]
    print('BSDE price = ', BSDE_price, ', ', np.abs(MC_priceQ-BSDE_price)/MC_priceQ*100, '%')

    ## Compute the strategies with the BSDE approach
 
    csi0_MV, csi_MV = mvpbsde.WealthSDE(Psim, np.squeeze(xv_test[:, dim:, :]), np.squeeze(dwb_test[:, :dim, :]),  np.squeeze(rn_test), np.squeeze(mvp_test), np.squeeze(lambdas_test[:, :dim, :]), np.squeeze(etas_test[:, :dim, :])) 
    csi_MV = np.divide(csi_MV, np.squeeze(xv_test[:, :dim, :num_time_interval]))
    csi0_MV_hat = csi0_MV/np.squeeze(xv_test[:, :dim, :num_time_interval])

    '''

    Testing the solution in comparison to Cerny and Kallsen's solutions
    
    '''
    
    ## Compute the solution to the PDE  
    ## Grid [0, S] x [0, V]
  
    m1 = 500           # S
    m2 = 100           # V
    
    ## Compute the prices with pde solver: WE ONLY DO THIS ONCE 
    ## We then interpolate once we have the trajectories of the processes S and V
    
    #price_vector, price_grid, Vec_s, Vec_v = bsde.CKSolutionV(m1, m2, x_init, 3*x_init, v_init, 5., strike)
    

    ## Save
    #np.save(path_dir + '/price_grid{}.npy'.format(num_time_interval), price_grid)
    #np.save(path_dir + '/price_vector{}.npy'.format(num_time_interval), price_vector)
    #np.save(path_dir + '/Vec_s{}.npy'.format(num_time_interval), Vec_s)
    #np.save(path_dir + '/Vec_v{}.npy'.format(num_time_interval), Vec_v)
    

    ## Load
    price_grid = np.load(path_dir + '/price_grid{}.npy'.format(num_time_interval))
    price_vector = np.load(path_dir + '/price_vector{}.npy'.format(num_time_interval))
    Vec_s = np.load(path_dir + '/Vec_s{}.npy'.format(num_time_interval))
    Vec_v = np.load(path_dir + '/Vec_v{}.npy'.format(num_time_interval))
    
 
    if simulate_trajectories:

        ## Interpolate the prices   
        price_pde = pde.interpolate_price(Vec_s, Vec_v, price_grid, Psim, num_time_interval, xv_test)   
    
        if save_trajectories:

            ## Save
            np.save(path_dir + '/price_pde{}.npy'.format(num_time_interval), price_pde)
    
    else:
    
        ## Load
        price_pde = np.load(path_dir + '/price_pde{}.npy'.format(num_time_interval))


    L_CK = bsde.CKSolutionL(Psim, xv_test[:, dim:, :])
    L_CK0 = np.mean(L_CK[:, 0])
    print('L CK =', L_CK0)
    print('L_solver =', rn_test[0, 0, 0], ', ', np.abs(rn_test[0, 0, 0]-L_CK0)/L_CK0*100, '%')

    #sys.exit(0)
   
    mean_price_pde = np.mean(price_pde[:, 0])
    print('PDE price = ', mean_price_pde, ', ', np.abs(mean_price_pde-MC_priceQ)/MC_priceQ*100, '%')    

    ## Compute the controls in the Cerny and Kallsen's approach
    
    dirac_s_grid, dirac_v_grid = bsde.CK_derivatives(m1, m2, price_vector, Vec_s, Vec_v)
    dirac_s = pde.interpolate_price(Vec_s, Vec_v, dirac_s_grid, Psim, num_time_interval, xv_test) 
    dirac_v = pde.interpolate_price(Vec_s, Vec_v, dirac_v_grid, Psim, num_time_interval, xv_test) 
    
    eta1_CK = np.multiply(rho * sigma * dirac_v  + np.multiply(dirac_s, np.squeeze(xv_test[:, :dim, :])),  np.sqrt(np.abs(np.squeeze(xv_test[:, dim:, :]))))
    eta2_CK = sigma * np.sqrt(1-rho**2)*np.multiply(dirac_v, np.sqrt(np.abs(np.squeeze(xv_test[:, dim:, :]))))
  
    
    ## Compute the strategies in the Cerny and Kallsen's approach

    csi0_CK, csi_CK = bsde.CKWealthSDE(Psim, price_pde, np.squeeze(xv_test[:, :dim, :]), dirac_s, dirac_v)
    csi0_CK_hat = csi0_CK/np.squeeze(xv_test[:, :dim, :num_time_interval]) 

    ## Compute the MSE

    MSEP_t = (mvp_test-price_pde)**2
    MSEP_t = np.mean(MSEP_t, axis = 0)

    MSE_t = (csi_CK-csi_MV)**2
    MSE_t = np.mean(MSE_t, axis = 0)

    MSE0_t = (csi0_CK_hat-csi0_MV_hat)**2
    MSE0_t = np.mean(MSE0_t, axis = 0)

    if save_trajectories:

        ## Save
        np.save(path_dir + '/mse_price{}.npy'.format(num_time_interval), MSEP_t)
        np.save(path_dir + '/mse_shares{}.npy'.format(num_time_interval), MSE_t)
        np.save(path_dir + '/mse_units{}.npy'.format(num_time_interval), MSE0_t)

    '''
    
    PLOTS
    
    '''
    if create_figures:
 
        ## We plot 10 random trajectories

        rand_vec = np.random.randint(0, P-1, 10)
        time = np.linspace(0, total_time, num_time_interval+1)
        time1 = time[:num_time_interval]

        ## Plots of the strategis

        eps = 0.01
        y_min = np.min([np.min(csi_CK[rand_vec, :]), np.min(csi_MV[rand_vec, :])]) - eps
        y_min0 = np.min([np.min(csi0_CK_hat[rand_vec, :]), np.min(csi0_MV_hat[rand_vec, :])]) * (1 + eps)
        y_max = np.max([np.max(csi_CK[rand_vec, :]), np.max(csi_MV[rand_vec, :])]) + 2 *eps
        y_max0 = np.max([np.max(csi0_CK_hat[rand_vec, :]), np.max(csi0_MV_hat[rand_vec, :])]) + 2*eps
        y_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        y_ticks0 = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0]

        f1 = plt.figure(figsize = (16, 14))
        plt.plot(time1, np.transpose(csi0_CK_hat[rand_vec, :]), linewidth = 4)
        plt.xlim((-eps, total_time))
        plt.ylim((y_min0, y_max0))
        plt.xticks(fontsize=25)
        plt.yticks(y_ticks0, fontsize=25)
        plt.xlabel('Time', fontsize = 35)
        plt.ylabel('Units of cash account', fontsize = 35)
        #plt.title('Units_bank_CK')  

        f2 =  plt.figure(figsize = (16, 14))
        plt.plot(time1, np.transpose(csi_CK[rand_vec, :]), linewidth = 4)
        plt.xlim((-eps, total_time))
        plt.ylim(( y_min , y_max))
        plt.xticks(fontsize = 25)
        plt.yticks(y_ticks, fontsize=25)
        plt.xlabel('Time', fontsize = 35)
        plt.ylabel('Shares of risky asset', fontsize = 35)
        #plt.title('Shares_stock_CK')  

        f3 = plt.figure(figsize = (16, 14))
        plt.plot(time1, np.transpose(csi0_MV_hat[rand_vec, :]), linewidth = 4)
        plt.xlim((-eps, total_time))
        plt.ylim((y_min0, y_max0))
        plt.xticks(fontsize = 25)
        plt.yticks(y_ticks0, fontsize=25)
        plt.xlabel('Time', fontsize = 35)
        plt.ylabel('Units of cash account', fontsize = 35)
        #plt.title('Units_bank_solver') 

        f4 = plt.figure(figsize = (16, 14))
        plt.plot(time1, np.transpose(csi_MV[rand_vec, :]), linewidth = 4)
        plt.xlim((-eps, total_time))
        plt.ylim((y_min, y_max))
        plt.xticks(fontsize = 25)
        plt.yticks(y_ticks, fontsize=25)
        plt.xlabel('Time', fontsize = 35)
        plt.ylabel('Shares of risky asset', fontsize = 35)
        #plt.title('Shares_stock_solver')  

        ## Plots of option prices

        y_minP = np.min([np.min(mvp_test[rand_vec, :]), np.min(price_pde[rand_vec, :])]) - 2*eps
        y_maxP = np.max([np.max(mvp_test[rand_vec, :]), np.max(price_pde[rand_vec, :])])  + 2*eps
        y_ticksP = np.arange(0, y_maxP, 5)

        f5 = plt.figure(figsize = (16, 14))
        plt.plot(time, np.transpose(mvp_test[rand_vec, :]), linewidth = 4)
        plt.xlim((-eps, total_time))
        plt.ylim((y_minP, y_maxP))
        plt.xticks(fontsize = 25)
        plt.yticks(y_ticksP, fontsize=25)
        plt.xlabel('Time', fontsize = 35)
        plt.ylabel('Call option price', fontsize = 35)
        #plt.title('Price_option_solver') 

        f6 = plt.figure(figsize = (16, 14))
        plt.plot(time, np.transpose(price_pde[rand_vec, :]), linewidth = 4)
        plt.xlim((-eps, total_time))
        plt.ylim((y_minP, y_maxP))
        plt.xticks(fontsize = 25)
        plt.yticks(y_ticksP, fontsize=25)
        plt.xlabel('Time', fontsize = 35)
        plt.ylabel('Call option price', fontsize = 35)
        #plt.title('Price_option_CK') 


        ## Plots of MSE 
        '''
        f7 = plt.figure(figsize = (16, 14))
        plt.plot(time, MSEP_t, 'r*', linewidth = 2)
        plt.plot(time1, MSE_t, 'bo', linewidth = 2)
        plt.plot(time1, MSE0_t, 'k+', linewidth = 2)
        plt.xlim((-eps, total_time + 2*eps))
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize=14)
        plt.xlabel('Time', fontsize = 16)
        plt.ylabel('MSE', fontsize = 16)
        plt.legend(['Option price', 'Shares', 'Units'])
        #plt.title('MSE') 
        '''

        #plt.show()

        if save_figures:
            f1.savefig(path_dir + '/Units_bank_CK{}.pdf'.format(num_time_interval), bbox_inches = 'tight', pad_inches = 0.01)
            f2.savefig(path_dir + '/Shares_stock_CK{}.pdf'.format(num_time_interval), bbox_inches = 'tight', pad_inches = 0.01)
            f3.savefig(path_dir + '/Units_bank_solver{}.pdf'.format(num_time_interval), bbox_inches = 'tight', pad_inches = 0.01)
            f4.savefig(path_dir + '/Shares_stock_solver{}.pdf'.format(num_time_interval), bbox_inches = 'tight', pad_inches = 0.01)
            f5.savefig(path_dir + '/Price_option_solver{}.pdf'.format(num_time_interval), bbox_inches = 'tight', pad_inches = 0.01)
            f6.savefig(path_dir + '/Price_option_CK{}.pdf'.format(num_time_interval), bbox_inches = 'tight', pad_inches = 0.01)
            #f7.savefig(path_dir + '/MSE{}.pdf'.format(num_time_interval), bbox_inches = 'tight', pad_inches = 0.01)


'''

        import matplotlib
        matplotlib.rc('font', size=20)

        t_n = np.linspace(0, total_time, 100+1)
        f8 = plt.figure(figsize = (16, 14))
        plt.plot(t_n, MSEP100, '-o', linewidth = 4)
        plt.plot(t_n[::2], MSEP50, '-o', linewidth = 4)
        plt.plot(t_n[::10], MSEP10, '-o', linewidth = 4) 
        plt.xlim((-eps, total_time + 2*eps))
        plt.xticks(fontsize = 25)
        plt.yticks(np.arange(0, 4.5, 0.5), fontsize=25) # for P
        plt.xlabel('Time', fontsize = 35)
        plt.ylabel('MSE option price', fontsize = 35)
        plt.legend(['N = 100', 'N = 50', 'N = 10'], fontsize = 35)
        f8.savefig('MSE_price.pdf', bbox_inches = 'tight', pad_inches = 0.01)


        f9 = plt.figure(figsize = (16, 14))
        plt.plot(t_n[:-1], MSEU100, '-o', linewidth = 4)
        plt.plot(t_n[:-1:2], MSEU50, '-o', linewidth = 4)
        plt.plot(t_n[:-1:10], MSEU10, '-o', linewidth = 4)
        plt.axhline(np.mean(MSEU100), xmin = 0, xmax = total_time, linestyle= 'dashed')
        plt.axhline(np.mean(MSEU50), xmin = 0, xmax = total_time, linestyle= 'dashed',  color = 'orange')
        plt.axhline(np.mean(MSEU10), xmin = 0, xmax = total_time, linestyle= 'dashed', color = 'green')
        plt.xlim((-eps, total_time + 2*eps))
        plt.xticks(fontsize = 25)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useLocale= True)
        plt.yticks(np.arange(0, 0.008, 0.001), fontsize=25) # for psi0
        plt.xlabel('Time', fontsize = 35)
        plt.ylabel('MSE units of cash account', size = 35)
        plt.legend(['N = 100', 'N = 50', 'N = 10', 'N = 100 mean', 'N = 50 mean', 'N = 10 mean'], fontsize = 30, ncol = 2)
        f9.savefig('MSE_units.pdf', bbox_inches = 'tight', pad_inches = 0.01)


        f10 = plt.figure(figsize = (16, 14))
        plt.plot(t_n[:-1], MSES100, '-o', linewidth = 4)
        plt.plot(t_n[:-1:2], MSES50, '-o', linewidth = 4)
        plt.plot(t_n[:-1:10], MSES10, '-o', linewidth = 4)
        plt.axhline(np.mean(MSES100), xmin = 0, xmax = total_time, linestyle= 'dashed')
        plt.axhline(np.mean(MSES50), xmin = 0, xmax = total_time, linestyle= 'dashed',  color = 'orange')
        plt.axhline(np.mean(MSES10), xmin = 0, xmax = total_time, linestyle= 'dashed', color = 'green')
        plt.xlim((-eps, total_time + 2*eps))
        plt.xticks(fontsize = 25)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useLocale= True)
        plt.yticks(np.arange(0, 0.008, 0.001), fontsize=25) # for psi0
        plt.xlabel('Time', fontsize = 35)
        plt.ylabel('MSE shares of risky asset', size = 35)
        plt.legend(['N = 100', 'N = 50', 'N = 10', 'N = 100 mean', 'N = 50 mean', 'N = 10 mean'], fontsize = 30, ncol = 2)
        f10.savefig('MSE_shares.pdf', bbox_inches = 'tight', pad_inches = 0.01)
'''


