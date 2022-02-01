# -*- coding: utf-8 -*-
from __future__ import print_function, division
import tensorflow as tf
import numpy as np
DTYPE=tf.float64
from propagate_layers import *
from kullback_lieblers import *
from GP_details import GP_definition

################################
### helper functions ###########
################################

def bernoulli(x,p):

    ##### make sure probability is between 0 and 1 #########
    p = inv_probit(p)

    return tf.math.log(tf.where(tf.equal(x,1.0), p, 1-p))

def multiclass_helper(inputul, outputul):

    ###############################################################################
    ##### Zoubin and Kim, 2006 paper on multi-class classification with GPs #######
    ###############################################################################

    softmaxed_input = tf.nn.softmax(inputul)+1e-6

    return tf.math.log(tf.reduce_sum(tf.multiply(softmaxed_input,outputul),axis=1,keepdims=True))

def inv_probit(x):

    jitter = 1e-3  # ensures output is strictly between 0 and 1
    return 0.5 * (1.0 + tf.math.erf(x / np.sqrt(2.0))) * (1 - 2 * jitter) + jitter

def variational_expectations(Y, Fmu, Fvar,log_variance_output):

    pi = np.pi
    pi = tf.cast(pi,tf.float64)
    return -0.5 * tf.math.log(2.0 * pi)  - 0.5 * log_variance_output - 0.5 * (tf.square(Y - Fmu) + Fvar) / tf.exp(log_variance_output)


class cost_functions(GP_definition):

    def __init__(self, **kwargs):
        
        GP_definition.__init__(self, **kwargs)

    ###################################
    #### Regression cost function #####
    ###################################

    def regression(self, f_mean, f_var, Y):

        scale = tf.cast(self.num_data, tf.float64)
        scale /= tf.cast(tf.shape(f_mean)[0], tf.float64) # minibatch size

        with tf.compat.v1.variable_scope('gaussian_likelihood', reuse=tf.compat.v1.AUTO_REUSE):
         
            unrestricted_variance_output = tf.compat.v1.get_variable(initializer = tf.constant(0.36, DTYPE),dtype=DTYPE,
                name='unrestricted_variance_output')

        variance_output = tf.math.softplus(unrestricted_variance_output)
        log_variance_output = tf.math.log(variance_output)

        data_fit_cost = 10.*scale * tf.reduce_sum(variational_expectations(Y, f_mean, f_var, log_variance_output)) 

        print('shape of re_error_final')
        print(data_fit_cost)

        return data_fit_cost

    #########################################
    ### CLassification Cost Function ########
    #########################################

    def classification(self, f_mean, f_var, Y):

        scale = tf.cast(self.num_data, tf.float64)
        scale /= tf.cast(tf.shape(f_mean)[0], tf.float64)  # minibatch size

        ### We sample 5 times and take the average ###
        f_sampled = tf.tile(tf.expand_dims(f_mean, axis=-1), [1, 1, 5]) + tf.multiply(tf.tile(tf.expand_dims(tf.sqrt(f_var), axis=-1), [1, 1, 5]),
            tf.random.normal(shape=(tf.shape(f_mean)[0], tf.shape(f_mean)[1], 5), dtype=DTYPE))	
        f_sampled = tf.reduce_mean(f_sampled, axis=-1, keepdims=False)

        if self.dim_layers[-1]==1:
            ##### Binary classification #####
            data_fit_cost = 10*scale * tf.reduce_sum(bernoulli(p = f_sampled, x = Y))	

        else:
            ###### Multi-class Classification ##### 
            data_fit_cost = scale * tf.reduce_sum(multiclass_helper(inputul = f_sampled, outputul = Y))

        return data_fit_cost

