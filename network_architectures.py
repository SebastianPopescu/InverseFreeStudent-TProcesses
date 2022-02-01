# -*- coding: utf-8 -*-
from __future__ import print_function, division
import tensorflow as tf
import numpy as np
DTYPE=tf.float64
from propagate_layers import *
from kullback_lieblers import *
from GP_details import GP_definition

class network_architectures(GP_definition):

    def __init__(self, **kwargs):
        
        GP_definition.__init__(self, **kwargs)

    def standard_DeepGP(self, X, X_mean_function, training_time, propagate_layers_object, g):

        ###########################################################
        #### This is just for Representation Learning #############
        #### Needs another layer on top for actual predictions ####
        ###########################################################

        if training_time and self.use_diagnostics:

            list_hopefully_id_matrix_sample = []
            list_hopefully_id_matrix_mean_covariance = []
            list_slack_conj_grad_solution = []
            list_slack_log_det_Kuu_lower_bound = []
            list_slack_log_det_Kuu_explicit = []
            list_T_inv = []
            list_Kuu = []     
            list_KL_wishart_actual = []
            list_num_steps = []

        if training_time:
            
            list_KL_qu = []
            list_KL_wishart = [] 

        ### Euclidean Space ###
        for l in range(1, self.num_layers+1):

            output_now = propagate_layers_object.propagate_layer(X = X, X_mean_function = X_mean_function, l = l, training_time = training_time,
                g = g)
            f_mean = output_now[0]
            f_var = output_now[1] + output_now[2]

            if training_time and self.use_diagnostics:
                list_KL_qu.append(output_now[-11])
                list_KL_wishart.append(output_now[-10])
                list_hopefully_id_matrix_sample.append(output_now[-9])
                list_hopefully_id_matrix_mean_covariance.append(output_now[-8])
                list_slack_conj_grad_solution.append(output_now[-7])
                list_slack_log_det_Kuu_lower_bound.append(output_now[-6])
                list_slack_log_det_Kuu_explicit.append(output_now[-5])
                list_T_inv.append(output_now[-4])
                list_Kuu.append(output_now[-3])       
                list_KL_wishart_actual.append(output_now[-2])
                list_num_steps.append(output_now[-1])

            elif training_time:
                list_KL_qu.append(output_now[-2])
                list_KL_wishart.append(output_now[-1])                

            X = f_mean + tf.multiply(tf.sqrt(f_var),
                tf.random.normal(shape=(tf.shape(f_mean)[0], tf.shape(f_mean)[1]), dtype=DTYPE))	

        if training_time and self.use_diagnostics:
            return f_mean, f_var, tf.reduce_sum(list_KL_qu), tf.reduce_sum(list_KL_wishart), list_hopefully_id_matrix_sample, list_hopefully_id_matrix_mean_covariance, list_slack_conj_grad_solution, list_slack_log_det_Kuu_lower_bound, list_slack_log_det_Kuu_explicit, list_T_inv, list_Kuu, tf.reduce_sum(list_KL_wishart_actual), list_num_steps   
        elif training_time:
            return f_mean, f_var, tf.reduce_sum(list_KL_qu), tf.reduce_sum(list_KL_wishart)
        else:
            return f_mean, f_var

    def uncertainty_decomposed_DeepGP(self, X, X_mean_function, propagate_layers_object, num_samples_testing):

        list_mean_epistemic = []
        list_mean_distributional = []
        list_var_epistemic = []
        list_var_distributional = []

        for l in range(1, self.num_layers+1):

            output_now = propagate_layers_object.propagate_layer_expectations(X = X, X_mean_function = X_mean_function, l = l, training_time = False,
                num_samples_testing = num_samples_testing)

            f_mean = output_now[0]
            f_var_epistemic = output_now[1]
            f_var_distributional = output_now[2]

            #f_var = f_var_epistemic + f_var_distributional
            
            list_mean_epistemic.append(f_mean)
            list_mean_distributional.append(tf.zeros_like(f_mean))
            list_var_epistemic.append(f_var_epistemic)
            list_var_distributional.append(f_var_distributional)

            #X = f_mean + tf.multiply(tf.sqrt(f_var),
            #    tf.random.normal(shape=(tf.shape(f_mean)[0], tf.shape(f_mean)[1]), dtype=DTYPE))	

        return list_mean_epistemic, list_mean_distributional, list_var_epistemic, list_var_distributional

