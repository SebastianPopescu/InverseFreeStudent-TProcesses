from matplotlib import use
import tensorflow as tf
import numpy as np
DTYPE=tf.float64
#from tensorflow_probability.python.internal import prefer_static as ps
import math as m
   
def preconditioned_conj_grad_routine(A, b, x, M, n_iterations):

    M = tf.stop_gradient(M)
    r_previous =  b - tf.linalg.matmul(A,x) 
    z_previous = tf.linalg.matmul(M,r_previous)
    p = z_previous
    #list_error = []

    for i in range(n_iterations+1):
        Ap = tf.linalg.matmul(A,p)
        alpha = tf.linalg.matmul(r_previous, z_previous, transpose_a = True) / (tf.linalg.matmul(p, Ap, transpose_a = True))
        x = x + alpha * p
        r = r_previous - alpha * Ap
        z = tf.linalg.matmul(M, r)
        beta = tf.linalg.matmul(r,z, transpose_a = True) / (tf.linalg.matmul(r_previous, z_previous, transpose_a = True))
        p = z + beta * p
        r_previous = r
        z_previous = z
        #list_error.append(tf.sqrt(tf.reduce_sum(tf.square(tf.matmul(A,x) - b))))
    #return x, list_error
    return x

def condition(X):

    print('********** inside condition function ********')
    print(X)
    X = tf.cast(X, DTYPE)

    return X + tf.eye(tf.shape(X)[0], dtype = DTYPE) * 1e-1

#############################################################
###### KL-div between Multivariate Normal distributions #####
#############################################################

def KL_inverse_free(q_mu, q_var_choleksy,  Sigma_mm_inverse, type_var, white,  L_K_A):

    ### Kl-Div between posterior and prior over inducing point values
    ### q_mu -- shape (num_inducing, dim_output)
    ### q_var_cholesky -- shape (dim_output, num_inducing, num_inducing)
    ### posterior_Kmm_inverse -- shape (num_inducing, num_inducing)

    ### TODO -- the L_K_A from propagate_layers has to be changed #####

    if not white:

        S = tf.matmul(q_var_choleksy, q_var_choleksy, transpose_b = True) ### shape -- (dim_output, num_inducing, num_inducing)
        Sigma_mm_inverse = tf.tile(tf.expand_dims(Sigma_mm_inverse, axis = 0), [tf.shape(q_mu)[1],1,1]) ### shape -- (dim_output, num_inducing, num_inducing)
        L_K_A = tf.tile(tf.expand_dims(L_K_A, axis = 0), [tf.shape(q_mu)[1],1,1]) ### shape -- (dim_output, num_inducing, num_inducing)

        kl_term = - 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(q_var_choleksy))) 
        kl_term -=  2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L_K_A))) 
        kl_term -= tf.cast(tf.shape(q_mu)[0],DTYPE ) * tf.cast(tf.shape(q_mu)[1],DTYPE )
        ### Explicit calculation of trace term 
        ### TODO -- implement Hutchinson trace estimator ### 	
        kl_term += tf.reduce_sum(tf.linalg.trace(tf.matmul(Sigma_mm_inverse, S)))
        q_mu = tf.expand_dims(tf.transpose(q_mu),axis=-1) ### shape (dim_output, num_inducing, 1)
        kl_term += tf.reduce_sum(tf.matmul(tf.matmul(q_mu, Sigma_mm_inverse, transpose_a = True),q_mu))
    
    elif white:

        S = tf.matmul(q_var_choleksy, q_var_choleksy, transpose_b = True) ### shape -- (dim_output, num_inducing, num_inducing)

        kl_term = - 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(q_var_choleksy))) 
        
        kl_term -= tf.cast(tf.shape(q_mu)[0],DTYPE ) * tf.cast(tf.shape(q_mu)[1],DTYPE ) 	
        ### Explicit calculation of trace term 
        ### TODO -- implement Hutchinson trace estimator ### 	
        kl_term += tf.reduce_sum(tf.linalg.trace(S))
        q_mu = tf.expand_dims(tf.transpose(q_mu),axis=-1) ### shape (dim_output, num_inducing, 1)
        kl_term += tf.reduce_sum(tf.matmul(q_mu, q_mu, transpose_a = True))

    return 0.5 * kl_term


######################################################
###### KL-div between Wishart Distributions ##########
######################################################



def KL_wishart(df_q, df_p, Kuu, inducing_points_number, 
    cholesky_Kmm, cholesky_Kmm_inverse, g, use_diagnostics):

    #############################################################################
    ### Explicit calculation of posterior Kuu_inverse  (T in paper notation) ####
    #############################################################################
        
    posterior_Kuu_inverse = tf.matmul(cholesky_Kmm_inverse, cholesky_Kmm_inverse, transpose_b = True)

    #############################################################################
    ### Explicit calculation of posterior Kfu_fu (T^{-1} in paper notation)  ####
    #############################################################################
    posterior_Kuu = tf.linalg.matmul(cholesky_Kmm, cholesky_Kmm, transpose_a = True)

    if use_diagnostics:

        #####################################################################################################################
        ### Remainder -- this is only used for getting an estimate of the slack in the lower bound on log-det|K_{fu,fu}| ####
        ### Warning -- this is using exact log det calculation, will incur massive computational cost #######################
        
        Kuu_conditioned = condition(Kuu)
        L_Kuu = tf.linalg.cholesky(Kuu_conditioned)

        ### Explicit calculation of log-det terms ###
        log_det_Kuu_explicit =  0.5 * df_p * 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L_Kuu)))  
    
    Kuu_tiled = condition(Kuu) 
    Kuu_tiled = tf.expand_dims(Kuu_tiled, axis = 0)
    Kuu_tiled = tf.tile(Kuu_tiled, [g.get_shape().as_list()[0], 1, 1])
    
    Kuu_inv_posterior_tiled = tf.expand_dims(posterior_Kuu_inverse, axis = 0)
    Kuu_inv_posterior_tiled = tf.tile(Kuu_inv_posterior_tiled, [tf.shape(g)[0], 1, 1])

    Kuu_operator = tf.linalg.LinearOperatorFullMatrix(
        matrix = condition(Kuu), is_non_singular=True, is_self_adjoint=True, is_positive_definite=True,
        is_square=True, name='LinearOperatorFullMatrixKuu')
    use_this = tf.stop_gradient(posterior_Kuu_inverse)
    posterior_Kuu_inv_operator = tf.linalg.LinearOperatorFullMatrix(
        matrix = use_this, is_non_singular=True, is_self_adjoint=True, is_positive_definite=True,
        is_square=True, name='LinearOperatorFullMatrixKuuinv')

    output_cg = tf.linalg.experimental.conjugate_gradient(
        operator = Kuu_operator, rhs = g, preconditioner= posterior_Kuu_inv_operator, 
        tol=1e-05, max_iter=tf.shape(Kuu)[0],
        name='conjugate_gradient')

    conj_grad_solution = tf.expand_dims(output_cg[1], axis =-1)
    print('_________________________________________')
    print('Conjugate Gradient solution')
    print(conj_grad_solution)

    if use_diagnostics:
        num_steps = output_cg[0]
        residual_vector = output_cg[2]
        slack_conj_grad_solution = tf.sqrt(tf.reduce_sum(tf.square(tf.reduce_mean(residual_vector, axis=0) )))

    Kuu_posterior_tiled = tf.expand_dims(posterior_Kuu, axis = 0)
    Kuu_posterior_tiled = tf.tile(Kuu_posterior_tiled, [g.get_shape().as_list()[0], 1, 1])
    conj_grad_solution = tf.linalg.matmul(tf.linalg.matmul(tf.expand_dims(g, axis=1), Kuu_posterior_tiled),conj_grad_solution)
    
    log_det_Kuu_lower_bound = - tf.reduce_mean(conj_grad_solution)
    log_det_Kuu_lower_bound +=  inducing_points_number 
    log_det_posterior_Kuu = - 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(cholesky_Kmm_inverse)))
    log_det_Kuu_lower_bound += log_det_posterior_Kuu
    log_det_Kuu_lower_bound = 0.5 * df_p * log_det_Kuu_lower_bound

    if use_diagnostics:
        slack_log_det_Kuu_lower_bound = (log_det_Kuu_explicit - log_det_Kuu_lower_bound) * 2. / df_p
        slack_log_det_Kuu_explicit = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L_Kuu)))  - log_det_posterior_Kuu
        #slack_log_det_Kfu_fu_lower_bound = tf.matmul(posterior_Kfu_fu, Kfu_fu_inv)

    kl_term = - log_det_Kuu_lower_bound   
    #kl_term = - log_det_Kfu_fu_explicit
    kl_term+=  0.5 * df_q * log_det_posterior_Kuu
 
    ### Explicit calculation of trace term ###
    trace_term_hutch_first_part = tf.matmul(g, condition(Kuu)) ### shape (num_hutch_samples, M)
    trace_term_hutch_second_part = tf.matmul(posterior_Kuu_inverse, g, transpose_b = True) ### shape (M, num_hutch_samples)
    trace_term_hutch  = tf.multiply(trace_term_hutch_first_part, tf.transpose(trace_term_hutch_second_part)) ### shape (num_hutch_samples, M)
    trace_term_hutch  = tf.reduce_mean(tf.reduce_sum(trace_term_hutch, axis = 1))

    #kl_term-= 0.5 * df_q * ( num_data+inducing_points_number -  tf.linalg.trace(tf.matmul(posterior_Kfu_fu_inverse, condition(Kfu_fu)))   )
    kl_term+= 0.5 * df_q * ( trace_term_hutch - inducing_points_number )

    if use_diagnostics:
        kl_term_actual = kl_term + log_det_Kuu_lower_bound - log_det_Kuu_explicit

    if use_diagnostics:
        return kl_term, slack_conj_grad_solution, slack_log_det_Kuu_lower_bound, slack_log_det_Kuu_explicit, kl_term_actual, num_steps
    else:
        return kl_term
