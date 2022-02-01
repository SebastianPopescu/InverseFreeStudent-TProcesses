# -*- coding: utf-8 -*-
from __future__ import print_function, division
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np
from kernels import *
from kullback_lieblers import *
DTYPE=tf.float64
    
def condition(X):

    return X + tf.eye(num_rows = X.get_shape().as_list()[0], num_columns =  X.get_shape().as_list()[0], dtype = X.dtype) * 1e-1

def conditional_GP(Xnew, X, sampled_Kmm_inverse_Kmn, sampled_Schur, Xnew_mean_function, l, dim_layer, num_layers, q_mu, q_var_cholesky, 
    log_lengthscales, log_kernel_variance, 
    training_time, num_inducing_points, df_q_inv_wishart, df_p_inv_wishart, df_f_wishart ,cholesky_Kmm, cholesky_Kmm_inverse, Sigma_mm_inverse, 
    g, use_diagnostics,  L_K_A, 
    white=False, full_cov=False):

    #### Training Time ##################################################
    #### sampled_Kmm_inverse_Kmn -- shape (num_inducing, num_batch) #####
    #### sampled_Schur -- shape (num_batch, )

    #### Testing Time ##################################################
    #### sampled_Kmm_inverse_Kmn -- shape (num_samples_testing, num_inducing, num_batch) #####
    #### sampled_Schur -- shape (num_samples_testing, num_batch)

    #### q_mu -- shape (num_inducing, dim_layers)
    #### q_var_cholesky -- shape (dim_layers, num_inducing, num_inducing)

    #### Remainder posterior_cholesky_Kmm_inverse is a lower triangular Cholesky decomposition #####

    type_var = 'full'

    Kmn = RBF(X, Xnew, log_lengthscales, log_kernel_variance)

    ### Compute mean function ###

    if training_time:
        fmean = tf.matmul(sampled_Kmm_inverse_Kmn, q_mu, transpose_a = True)
    else:
        q_mu = tf.tile(tf.expand_dims(q_mu, axis = 0), [tf.shape(sampled_Kmm_inverse_Kmn)[0], 1, 1])
        fmean = tf.linalg.matmul(sampled_Kmm_inverse_Kmn, q_mu, transpose_a = True)        

    if l == num_layers:
        pass 
    else:
        fmean += Xnew_mean_function

    if full_cov:

        if training_time:

            #### Warning -- not sure if we use this, as we would have to sample sampled_Kmm_inverse_Kmn from a matrix variate normal ####
            sampled_Kmm_inverse_Kmn = tf.tile(tf.expand_dims(sampled_Kmm_inverse_Kmn, axis=0),[ dim_layer, 1, 1])
            LTA = tf.matmul(q_var_cholesky, sampled_Kmm_inverse_Kmn, transpose_a = True)
            fvar_epistemic = tf.matmul(LTA, LTA,transpose_a=True)    
        else:
            sampled_Kmm_inverse_Kmn = tf.tile(tf.expand_dims(sampled_Kmm_inverse_Kmn, axis=1),[1, dim_layer, 1, 1])
            q_var_cholesky = tf.tile(tf.expand_dims(q_var_cholesky, axis = 0), [tf.shape(sampled_Kmm_inverse_Kmn)[1], 1, 1])
            LTA = tf.linalg.matmul(q_var_cholesky, sampled_Kmm_inverse_Kmn, transpose_a = True)
            fvar_epistemic = tf.linalg.matmul(LTA, LTA, transpose_a=True)               

    else:
    
        if training_time:
            sampled_Kmm_inverse_Kmn = tf.tile(tf.expand_dims(sampled_Kmm_inverse_Kmn, axis=0),[ dim_layer, 1, 1])
            LTA = tf.matmul(q_var_cholesky, sampled_Kmm_inverse_Kmn, transpose_a = True)
            fvar_epistemic = tf.transpose(tf.reduce_sum(tf.square(LTA), 1, keepdims = False))
        else:
            sampled_Kmm_inverse_Kmn = tf.tile(tf.expand_dims(sampled_Kmm_inverse_Kmn, axis=1),[ 1, dim_layer,  1, 1])
            q_var_cholesky = tf.tile(tf.expand_dims(q_var_cholesky, axis = 0), [tf.shape(sampled_Kmm_inverse_Kmn)[0], 1, 1, 1])
            LTA = tf.linalg.matmul(q_var_cholesky, sampled_Kmm_inverse_Kmn, transpose_a = True)
            fvar_epistemic = tf.transpose(tf.reduce_sum(tf.square(LTA), 2, keepdims = False),[0,2,1])

    if full_cov:
        #### Warning -- we don't actually use this 
        tensor_shape = Xnew.get_shape()   
        Knn = RBF(Xnew, Xnew, log_lengthscales, log_kernel_variance) 
    else:

        Knn = RBF_Kdiag(Xnew, log_kernel_variance) 
    
    if full_cov:
        #### Warning -- to use this, we need to sample sampled_Schur from an inverse wishart distribution ####
        fvar_distributional = sampled_Schur
    else:

        fvar_distributional = sampled_Schur

    if training_time and use_diagnostics:

        Kuu = RBF(X, X, log_lengthscales, log_kernel_variance) 

        #################################### 
        #### compute KL-DIV[q(U)||p(U)] ####
        ####################################        

        kl_term_qu = KL_inverse_free(q_mu = q_mu, q_var_choleksy = q_var_cholesky, 
            Sigma_mm_inverse = Sigma_mm_inverse, type_var='full', white = white,  L_K_A =  L_K_A)    

        #######################################################
        #### compute KL-DIV[q(K_{uu}^{-1}),p(K_{uu}^{-1})] ####
        ####################################################### 

        kl_term_wishart, slack_conj_grad_solution, slack_log_det_Kuu_lower_bound, slack_log_det_Kuu_explicit, kl_term_wishart_actual, num_steps = KL_wishart(df_q = df_f_wishart, 
            df_p = df_f_wishart,  
            Kuu = Kuu, inducing_points_number = num_inducing_points, cholesky_Kmm = cholesky_Kmm,
            cholesky_Kmm_inverse = cholesky_Kmm_inverse, g = g, use_diagnostics = use_diagnostics)

        hopefully_id_matrix_sample = tf.matmul(Sigma_mm_inverse, tf.stop_gradient(Kuu)) 
        Kmm_inverse = tf.matmul(cholesky_Kmm_inverse, cholesky_Kmm_inverse, transpose_b = True)
        hopefully_id_matrix_mean_covariance = tf.matmul(Kmm_inverse, tf.stop_gradient(Kuu)) 
        T_inv = tf.matmul(cholesky_Kmm, cholesky_Kmm, transpose_a = True)


    elif training_time:
        
        Kuu = RBF(X, X, log_lengthscales, log_kernel_variance) 

 
        #### compute KL-DIV[q(U)||p(U)] ####        
        kl_term_qu = KL_inverse_free(q_mu = q_mu, q_var_choleksy = q_var_cholesky, 
            Sigma_mm_inverse = Sigma_mm_inverse, type_var='full', white = white,  L_K_A =  L_K_A)    

        #######################################################
        #### compute KL-DIV[q(K_{uu}^{-1}),p(K_{uu}^{-1})] ####
        ####################################################### 

        kl_term_wishart = KL_wishart(df_q = df_f_wishart, 
            df_p = df_f_wishart,  
            Kuu = Kuu, inducing_points_number = num_inducing_points, cholesky_Kmm = cholesky_Kmm,
            cholesky_Kmm_inverse = cholesky_Kmm_inverse, g = g, use_diagnostics = use_diagnostics) 

    print('*************** ------- at the end of conditional_GP function  ------------  ********************')
    print(fmean)
    print(fvar_epistemic)
    print(fvar_distributional)

    if training_time and use_diagnostics:
        return fmean, fvar_epistemic, fvar_distributional, kl_term_qu, kl_term_wishart, hopefully_id_matrix_sample, hopefully_id_matrix_mean_covariance, slack_conj_grad_solution, slack_log_det_Kuu_lower_bound, slack_log_det_Kuu_explicit, T_inv, Kuu, kl_term_wishart_actual, num_steps
    elif training_time:
        return fmean, fvar_epistemic, fvar_distributional, kl_term_qu, kl_term_wishart
    else:
        return fmean, fvar_epistemic, fvar_distributional


