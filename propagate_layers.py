# -*- coding: utf-8 -*-
from __future__ import print_function, division
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np
DTYPE=tf.float64
from conditional_GP import *
from kullback_lieblers import *
from GP_details import GP_definition
from kernels import *

### helper functions ###

def condition(X):

    return X + tf.eye(tf.shape(X)[0], dtype = DTYPE) * 1e-1

def sample_chi_squared(df,m):

    normal_matrix = tf.random.normal(shape=(tf.cast(df,tf.int32),tf.cast(df,tf.int32)), dtype = DTYPE)
    sq_normal_matrix = tf.square(normal_matrix)

    upper_sq_normal_matrix = tf.linalg.band_part(sq_normal_matrix, 0, -1) 
    ### select top m rows
    sliced_upper_sq_normal_matrix = tf.slice(upper_sq_normal_matrix,[0,0],[m,-1])

    return tf.reduce_sum(sliced_upper_sq_normal_matrix, axis=-1)

def sample_Wishart(posterior_cholesky, df_q, num_inducing):

    A = tf.random.normal(shape = (num_inducing, num_inducing), dtype = DTYPE)

    sampled_chi_squared_diagonal_terms = sample_chi_squared(df = df_q, m = num_inducing)
    sampled_chi_squared_diagonal_terms = tf.sqrt(sampled_chi_squared_diagonal_terms)
    A = tf.linalg.set_diag(A, tf.reshape(sampled_chi_squared_diagonal_terms,[-1, ]))

    L_K_A = tf.matmul(posterior_cholesky, A)
    sampled_Kmm_inverse = tf.matmul(L_K_A, L_K_A, transpose_b = True) 

    return sampled_Kmm_inverse, L_K_A


class propagate_layers(GP_definition):

    ##################################################################
    ####### Dual form of  Gaussian Processes -- in RKHS space ########
    ##################################################################

    def __init__(self, **kwargs):
        
        GP_definition.__init__(self, **kwargs)

    def propagate_layer(self, X, X_mean_function, l, training_time, g):

        type_var = 'full'
        full_cov = False

        with tf.compat.v1.variable_scope('list_1', reuse = tf.compat.v1.AUTO_REUSE):
            with tf.compat.v1.variable_scope('num_layer_'+str(l), reuse=tf.compat.v1.AUTO_REUSE):

                log_kernel_variance = tf.compat.v1.get_variable(initializer = tf.constant(0.301, dtype = DTYPE),dtype=tf.float64,
                    name='log_kernel_variance')       
                log_lengthscales = tf.compat.v1.get_variable(initializer = tf.constant([-0.301 for _ in range(self.dim_layers[l-1])], dtype = DTYPE),
                    dtype=tf.float64,name='log_lengthscales')

                if training_time and l==1:

                    Z = tf.compat.v1.get_variable(initializer =  tf.constant(self.Z_init, dtype=DTYPE),
                        dtype=DTYPE, name='Z')

                else:

                    Z = tf.compat.v1.get_variable(initializer =  tf.random_uniform_initializer(minval=-2.0,
                        maxval=2.0), shape = (self.num_inducing[l-1], self.dim_layers[l-1]),
                        dtype=DTYPE, name='Z')
                
        with tf.compat.v1.variable_scope('list_2', reuse = tf.compat.v1.AUTO_REUSE):
            with tf.compat.v1.variable_scope('num_layer_'+str(l), reuse=tf.compat.v1.AUTO_REUSE):

                q_mu = tf.compat.v1.get_variable(initializer = tf.random_uniform_initializer(minval=-0.05,
                    maxval=0.05),shape=(self.num_inducing[l-1],self.dim_layers[l]),
                    dtype=DTYPE,name='q_mu')

                if l!=self.num_layers:
                    q_identity_matrix = np.tile(np.expand_dims(1e-5*np.eye(self.num_inducing[l-1], dtype=np.float64),axis=0), (self.dim_layers[l], 1, 1))
                else:
                    q_identity_matrix = np.tile(np.expand_dims(self.posterior_cholesky_Kmm[1],axis=0),(self.dim_layers[l], 1, 1))
                q_cholesky_unmasked = tf.compat.v1.get_variable(initializer = tf.constant(q_identity_matrix, dtype=tf.float64),
                    dtype=DTYPE, name='q_cholesky_unmasked')

                q_var_cholesky = tf.linalg.band_part(q_cholesky_unmasked,-1,0)
            
                #### get degrees of freedom for both posterior and prior ###
                ### Remainder for Wishart Distributions, we need v > dimensions - 1
                ### Remainder for Inverse-Wishart Distributyions, we need v > dimensions + 1 (so that it has a valid mean estimate)

                number_of_inducing_points = self.num_inducing[l-1]
                number_of_inducing_points = float(number_of_inducing_points)

                unrestricted_df_q_inverse_wishart = tf.compat.v1.get_variable(initializer = tf.constant(10.*number_of_inducing_points, dtype = DTYPE), dtype=tf.float64,
                    name='df_q_inverse_wishart', trainable = True) 

                unrestricted_df_q_wishart = tf.compat.v1.get_variable(initializer = tf.constant(10. * number_of_inducing_points, dtype = DTYPE), dtype=tf.float64,
                    name='df_q_wishart', trainable = False) 

        ########## Remainder ############################################################################################# 
        # we take df_q == df_p so that we simplify the computation of the KL-div between Inverse-Wishart distributions ###
        df_q_inv_wishart = tf.math.softplus(unrestricted_df_q_inverse_wishart) 
        df_f_inv_wishart = df_q_inv_wishart + self.num_inducing[l-1] + 1.

        df_q_wishart = tf.math.softplus(unrestricted_df_q_wishart) 
        df_f_wishart = df_q_wishart + self.num_inducing[l-1] + 1.

        #########################################################################################
        ######### Parametrization of Posterior over Big Covariance Matrix #######################
        ######### v is taken to be a Lower Diagonal Matrix ######################################

        with tf.compat.v1.variable_scope('list_2', reuse = tf.compat.v1.AUTO_REUSE):
            with tf.compat.v1.variable_scope('num_layer_'+str(l), reuse=tf.compat.v1.AUTO_REUSE):

                posterior_cholesky_Kmm_inv = tf.compat.v1.get_variable(initializer =  tf.constant(self.posterior_cholesky_Kmm_inv[l], dtype=DTYPE),
                        dtype=DTYPE, name='posterior_cholesky_Kmm_inv')

        print('********* ------------------ *************')

        posterior_cholesky_Kmm_inv = tf.linalg.band_part(posterior_cholesky_Kmm_inv, -1, 0)
        posterior_cholesky_Kmm = tf.linalg.triangular_solve(posterior_cholesky_Kmm_inv, tf.eye(self.num_inducing[l-1], dtype = DTYPE), lower = True)  
        
        ##############################################################
        ### sample from Inverse-Wishart/Inverse-Gamma Distribution ###
        ############################################################## 

        #Kmm_inverse =  tf.matmul(posterior_cholesky_Kmm_inv, posterior_cholesky_Kmm_inv, transpose_b = True) 
        Kmm_inverse, L_K_A = sample_Wishart(posterior_cholesky = posterior_cholesky_Kmm_inv / tf.sqrt(df_f_wishart), df_q = df_f_wishart, num_inducing = self.num_inducing[l-1])
        ### Remainder -- Kmm_inverse is actually a sample ###
    

        Kfu = RBF(X, Z, log_lengthscales, log_kernel_variance)                   
        Kff = RBF(X, X, log_lengthscales, log_kernel_variance)
        Kuu = RBF(Z, Z, log_lengthscales, log_kernel_variance)  
        T_Kuu_T = tf.matmul(tf.matmul(Kmm_inverse, Kuu), Kmm_inverse)    
        posterior_Schur = Kff + tf.matmul(tf.matmul(Kfu, T_Kuu_T), Kfu, transpose_b = True)  
        posterior_Schur -= 2 * tf.matmul(tf.matmul(Kfu, Kmm_inverse), Kfu, transpose_b = True)

        df_inv_gamma = 0.5 * (df_f_inv_wishart +  1.)
        diagonal_posterior_Schur = 0.5 * df_f_inv_wishart * tf.linalg.diag_part(posterior_Schur) 
        df_inv_gamma = tf.ones_like(diagonal_posterior_Schur) * df_inv_gamma 
        #inverse_gamma_object = tf.contrib.distributions.InverseGamma(

        inverse_gamma_object = tfp.distributions.InverseGamma(
            concentration = df_inv_gamma, scale = diagonal_posterior_Schur, 
            name='InverseGamma')                    
        
        sampled_Schur = inverse_gamma_object.sample() #### (num_batch,)
        sampled_Schur = tf.reshape(sampled_Schur, [-1,1]) ### -- shape (num_batch, 1)

        print('***********************************************************************************************************************')
        print('**** size sampled_Schur at training time *******')
        print(sampled_Schur)
        print('***********************************************************************************************************************')


        batched_Kuu_inverse_Kuf = tf.matmul(Kmm_inverse, Kfu, transpose_b = True) ### -- shape (num_inducing, num_batch )
        
        batched_posterior_cholesky_Kmm_inverse = tf.tile(tf.expand_dims(posterior_cholesky_Kmm_inv, axis = 0), [ tf.shape(X)[0], 1, 1])
        batched_sqrt_Schur = tf.tile(tf.expand_dims(tf.sqrt(sampled_Schur), axis=-1), [1, self.num_inducing[l-1], self.num_inducing[l-1]])
        batched_sqrt_Schur = batched_sqrt_Schur / (tf.sqrt(df_f_inv_wishart))
        batched_diagonal_hadamard_product = tf.multiply(batched_sqrt_Schur, batched_posterior_cholesky_Kmm_inverse) ### shape -- (num_batch, num_inducing, num_inducing)

        '''
        mvn_object = tfp.distributions.MultivariateNormalTriL(
            loc=batched_Kuu_inverse_Kuf, scale=batched_diagonal_hadamard_product, validate_args=False, allow_nan_stats=True,
            name='MultivariateNormalLinearOperator')

        sampled_Kmm_inverse_Kmn = mvn_object.sample() #### (num_batch, num_inducing)
        '''
        sampled_Kmm_inverse_Kmn = batched_Kuu_inverse_Kuf + tf.transpose(tf.squeeze(tf.linalg.matmul(batched_diagonal_hadamard_product, 
            tf.random.normal(shape = (tf.shape(batched_diagonal_hadamard_product)[0],tf.shape(batched_diagonal_hadamard_product)[1],1), dtype=DTYPE)), axis=-1)) 

        print('***********************************************************************************************************************')
        print('**** size Kmm_inverse_Kmn at training time *******')
        print(sampled_Kmm_inverse_Kmn)
        print('***********************************************************************************************************************')

        if training_time:

            ######################################################################################################
            #### Remainder -- this is necesary in the computation of the KL-div between inducing point values ####
            ######################################################################################################

            Sigma_mm_inverse, L_K_A = sample_Wishart(posterior_cholesky =  L_K_A / tf.sqrt(df_f_inv_wishart), df_q = df_f_inv_wishart, num_inducing = self.num_inducing[l-1])
            #Sigma_mm_inverse, L_K_A = sample_Wishart(posterior_cholesky =  posterior_cholesky_Kmm_inv / tf.sqrt(df_f_inv_wishart), df_q = df_f_inv_wishart, num_inducing = self.num_inducing[l-1])


        if training_time:

            output_now = conditional_GP(Xnew = X, X = Z, sampled_Kmm_inverse_Kmn = sampled_Kmm_inverse_Kmn, sampled_Schur = sampled_Schur, 
                Xnew_mean_function = X_mean_function, l = l, dim_layer = self.dim_layers[l], num_layers =  self.num_layers, 
                q_mu = q_mu, q_var_cholesky = q_var_cholesky, 
                log_lengthscales = log_lengthscales, log_kernel_variance = log_kernel_variance,                
                training_time = training_time, num_inducing_points = tf.cast(self.num_inducing[l-1], DTYPE), 
                df_q_inv_wishart = df_q_inv_wishart, df_p_inv_wishart = df_q_inv_wishart, 
                df_f_wishart = df_f_wishart, 
                cholesky_Kmm = posterior_cholesky_Kmm, cholesky_Kmm_inverse = posterior_cholesky_Kmm_inv, 
                Sigma_mm_inverse = Sigma_mm_inverse,  g=g, use_diagnostics=self.use_diagnostics, L_K_A =  L_K_A,
                white = False, full_cov = full_cov)
        else:
            output_now = conditional_GP(Xnew = X, X = Z, sampled_Kmm_inverse_Kmn = sampled_Kmm_inverse_Kmn, sampled_Schur = sampled_Schur, 
                Xnew_mean_function = X_mean_function, l = l, dim_layer = self.dim_layers[l], num_layers =  self.num_layers, 
                q_mu = q_mu, q_var_cholesky = q_var_cholesky, 
                log_lengthscales = log_lengthscales, log_kernel_variance = log_kernel_variance,         
                training_time = training_time, num_inducing_points = tf.cast(self.num_inducing[l-1], DTYPE),  
                df_q_inv_wishart = df_q_inv_wishart, df_p_inv_wishart = df_q_inv_wishart, 
                df_f_wishart = df_f_wishart, 
                cholesky_Kmm = None, cholesky_Kmm_inverse = posterior_cholesky_Kmm_inv, Sigma_mm_inverse = None, 
                g = None, use_diagnostics=self.use_diagnostics,  L_K_A = None, 
                white = False, full_cov = full_cov)

        if training_time and self.use_diagnostics:

            output_mean = output_now[0]
            output_var_epistemic = output_now[1]
            output_var_distributional = output_now[2]
            output_var_distributional = tf.reshape(output_var_distributional, [-1,1])  

            kl_qu = output_now[-11]
            kl_wishart = output_now[-10]
            hopefully_id_matrix_sample = output_now[-9]
            hopefully_id_matrix_mean_covariance = output_now[-8]
            slack_conj_grad_solution = output_now[-7]
            slack_log_det_Kuu_lower_bound = output_now[-6]
            slack_log_det_Kuu_explicit = output_now[-5]
            T_inv = output_now[-4]
            Kuu = output_now[-3]
            kl_wishart_actual = output_now[-2]
            num_steps = output_now[-1]
        
        elif training_time:
            
            output_mean = output_now[0]
            output_var_epistemic = output_now[1]
            output_var_distributional = output_now[2]
            output_var_distributional = tf.reshape(output_var_distributional, [-1,1])  

            kl_qu = output_now[-2]
            kl_wishart = output_now[-1]
        
        else:
        
            output_mean = output_now[0] #, [-1, num_samples_testing])
            output_var_epistemic =  output_now[1] #, [-1, num_samples_testing])
            output_var_distributional =  output_now[2] #, [-1, num_samples_testing])

        if training_time and self.use_diagnostics:
            return output_mean, output_var_epistemic, output_var_distributional, kl_qu, kl_wishart, hopefully_id_matrix_sample, hopefully_id_matrix_mean_covariance, slack_conj_grad_solution, slack_log_det_Kuu_lower_bound, slack_log_det_Kuu_explicit, T_inv, Kuu, kl_wishart_actual, num_steps
        elif training_time:
            return output_mean, output_var_epistemic, output_var_distributional, kl_qu, kl_wishart
        else:
            return output_mean, output_var_epistemic, output_var_distributional

    def propagate_layer_expectations(self, X, X_mean_function, l, training_time,  num_samples_testing):

        type_var = 'full'
        full_cov = False

        with tf.compat.v1.variable_scope('list_1', reuse = True):
            with tf.compat.v1.variable_scope('num_layer_'+str(l), reuse=True):

                log_kernel_variance = tf.compat.v1.get_variable(dtype=DTYPE,
                    name='log_kernel_variance')       
                log_lengthscales = tf.compat.v1.get_variable(
                    dtype=DTYPE,name='log_lengthscales')

                Z = tf.compat.v1.get_variable(dtype=DTYPE, name='Z')
                
        with tf.compat.v1.variable_scope('list_2', reuse = True):
            with tf.compat.v1.variable_scope('num_layer_'+str(l), reuse=True):

                q_mu = tf.compat.v1.get_variable(
                    dtype=DTYPE,name='q_mu')

                q_cholesky_unmasked = tf.compat.v1.get_variable(
                    dtype=DTYPE, name='q_cholesky_unmasked')
                q_var_cholesky = tf.linalg.band_part(q_cholesky_unmasked,-1,0)
            
                #### get degrees of freedom for both posterior and prior ###
                ### Remainder for Wishart Distributions, we need v > dimensions - 1
                ### Remainder for Inverse-Wishart Distributyions, we need v > dimensions + 1 (so that it has a valid mean estimate)

                number_of_inducing_points = self.num_inducing[l-1]
                number_of_inducing_points = float(number_of_inducing_points)

                unrestricted_df_q_inverse_wishart = tf.compat.v1.get_variable(dtype=tf.float64,
                    name='df_q_inverse_wishart') 

                unrestricted_df_q_wishart = tf.compat.v1.get_variable(dtype=tf.float64,
                    name='df_q_wishart') 

        ########## Remainder ############################################################################################# 
        # we take df_q == df_p so that we simplify the computation of the KL-div between Inverse-Wishart distributions ###
        
        df_q_inv_wishart = tf.math.softplus(unrestricted_df_q_inverse_wishart) 
        df_f_inv_wishart = df_q_inv_wishart + self.num_inducing[l-1] + 1.

        df_q_wishart = tf.math.softplus(unrestricted_df_q_wishart) 
        df_f_wishart = df_q_wishart + self.num_inducing[l-1] + 1.

        with tf.compat.v1.variable_scope('list_2', reuse = True):
            with tf.compat.v1.variable_scope('num_layer_'+str(l), reuse=True):

                posterior_cholesky_Kmm_inv = tf.compat.v1.get_variable(
                        dtype=DTYPE, name='posterior_cholesky_Kmm_inv')

        posterior_cholesky_Kmm_inv = tf.linalg.band_part(posterior_cholesky_Kmm_inv, -1, 0)

        Kmm_inverse =  tf.matmul(posterior_cholesky_Kmm_inv, posterior_cholesky_Kmm_inv, transpose_b = True) 
        Kfu = RBF(X, Z, log_lengthscales, log_kernel_variance)                   
        Kff = RBF(X, X, log_lengthscales, log_kernel_variance)
        Kuu = RBF(Z, Z, log_lengthscales, log_kernel_variance)  
        T_Kuu_T = tf.matmul(tf.matmul(Kmm_inverse, Kuu), Kmm_inverse)    
        posterior_Schur = Kff + tf.matmul(tf.matmul(Kfu, T_Kuu_T), Kfu, transpose_b = True)  
        posterior_Schur -= 2 * tf.matmul(tf.matmul(Kfu, Kmm_inverse), Kfu, transpose_b = True)

        expectation_Schur = (df_f_inv_wishart / (df_f_inv_wishart -1.))  * tf.linalg.diag_part(posterior_Schur) 
        expectation_Schur = tf.tile(tf.expand_dims( tf.reshape(expectation_Schur, [-1,1]), axis = 0), [num_samples_testing, 1, 1])

        expectation_Kuu_inverse_Kuf = tf.matmul(Kmm_inverse, Kfu, transpose_b = True) ### shape -- (num_inducing, num_batch)
        expectation_Kuu_inverse_Kuf = tf.tile(tf.expand_dims(expectation_Kuu_inverse_Kuf, axis=0), [num_samples_testing, 1, 1]) ### shape -- (num_samples_testing, num_inducing, num_batch)

        output_now = conditional_GP(Xnew = X, X = Z, sampled_Kmm_inverse_Kmn = expectation_Kuu_inverse_Kuf, sampled_Schur = expectation_Schur, 
            Xnew_mean_function = X_mean_function, l = l, dim_layer = self.dim_layers[l], num_layers =  self.num_layers, 
            q_mu = q_mu, q_var_cholesky = q_var_cholesky, 
            log_lengthscales = log_lengthscales, log_kernel_variance = log_kernel_variance,         
            training_time = training_time, num_inducing_points = tf.cast(self.num_inducing[l-1], DTYPE),  
            df_q_inv_wishart = df_q_inv_wishart, df_p_inv_wishart = df_q_inv_wishart, 
            df_f_wishart = df_f_wishart,  
            cholesky_Kmm = None, cholesky_Kmm_inverse = posterior_cholesky_Kmm_inv, Sigma_mm_inverse = None,  
            g = None, use_diagnostics = self.use_diagnostics,  L_K_A = None, 
            white = False, full_cov = full_cov)

        output_mean = output_now[0] #, [-1, num_samples_testing])
        output_var_epistemic =  output_now[1] #, [-1, num_samples_testing])
        output_var_distributional =  output_now[2] #, [-1, num_samples_testing])

        return output_mean, output_var_epistemic, output_var_distributional


