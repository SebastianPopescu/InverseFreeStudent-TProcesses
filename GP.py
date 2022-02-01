# -*- coding: utf-8 -*-
from __future__ import print_function, division
import matplotlib as mpl
mpl.use('Agg')
import tensorflow as tf
import numpy as np
from collections import defaultdict
import random
import argparse
import matplotlib.pyplot as plt
import sys
import os
DTYPE=tf.float64
import seaborn as sns
from sklearn.cluster import  KMeans
from matplotlib import rcParams
import itertools
from scipy.stats import norm
import pandas as pd
import scipy
from GP_details import GP_definition
from propagate_layers import *
from losses import *
from network_architectures import *
sys.setrecursionlimit(10000)


########################
### helper functions ###
########################

def safe_softplus(x, limit=30):
  if x>limit:
    return x
  else:
    return np.log1p(np.exp(x))


# Student's T random variable
def multivariate_t_rvs(m, S, df=np.inf, n=1):
    '''generate random variables of multivariate t distribution
    Parameters
    ----------
    m : array_like
        mean of random variable, length determines dimension of random variable
    S : array_like
        square array of covariance  matrix
    df : int or float
        degrees of freedom
    n : int
        number of observations, return random array will be (n, len(m))
    Returns
    -------
    rvs : ndarray, (n, len(m))
        each row is an independent draw of a multivariate t distributed
        random variable
    '''
    m = np.asarray(m)
    d = len(m)
    if df == np.inf:
        x = 1.
    else:
        x = np.random.chisquare(df, n)/df
    z = np.random.multivariate_normal(np.zeros(d),S,(n,))
    return m + z/np.sqrt(x)[:,None]   # same output format as random.multivariate_normal


def plot_gp_dist(
    ax,
    samples: np.ndarray,
    x: np.ndarray,
    plot_samples=True,
    palette="Reds",
    fill_alpha=0.8,
    samples_alpha=0.1,
    fill_kwargs=None,
    samples_kwargs=None,):
    """A helper function for plotting 1D GP posteriors from trace
        Parameters
    ----------
    ax: axes
        Matplotlib axes.
    samples: numpy.ndarray
        Array of S posterior predictive sample from a GP.
        Expected shape: (S, X)
    x: numpy.ndarray
        Grid of X values corresponding to the samples.
        Expected shape: (X,) or (X, 1), or (1, X)
    plot_samples: bool
        Plot the GP samples along with posterior (defaults True).
    palette: str
        Palette for coloring output (defaults to "Reds").
    fill_alpha: float
        Alpha value for the posterior interval fill (defaults to 0.8).
    samples_alpha: float
        Alpha value for the sample lines (defaults to 0.1).
    fill_kwargs: dict
        Additional arguments for posterior interval fill (fill_between).
    samples_kwargs: dict
        Additional keyword arguments for samples plot.
    Returns
    -------
    ax: Matplotlib axes
    """
    import matplotlib.pyplot as plt

    if fill_kwargs is None:
        fill_kwargs = {}
    if samples_kwargs is None:
        samples_kwargs = {}
    if np.any(np.isnan(samples)):
        warnings.warn(
            "There are `nan` entries in the [samples] arguments. "
            "The plot will not contain a band!",
            UserWarning,
        )

    cmap = plt.get_cmap(palette)
    percs = np.linspace(51, 99, 40)
    colors = (percs - np.min(percs)) / (np.max(percs) - np.min(percs))
    samples = samples.T
    x = x.flatten()
    for i, p in enumerate(percs[::-1]):
        upper = np.percentile(samples, p, axis=1)
        lower = np.percentile(samples, 100 - p, axis=1)
        color_val = colors[i]
        ax.fill_between(x, upper, lower, color=cmap(color_val), alpha=fill_alpha, **fill_kwargs)
    if plot_samples:
        # plot a few samples
        idx = np.random.randint(0, samples.shape[1], 30)
        ax.plot(x, samples[:, idx], color=cmap(0.9), lw=1, alpha=samples_alpha, **samples_kwargs)

    return ax

def draw_gaussian_at(support, sd=1.0, height=1.0, xpos=0.0, ypos=0.0, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    gaussian = np.exp((-support ** 2.0) / (2 * sd ** 2.0))
    gaussian /= gaussian.max()
    gaussian *= height
    return ax.plot(gaussian + xpos, support + ypos, **kwargs)

def timer(start,end):
       hours, rem = divmod(end-start, 3600)
       minutes, seconds = divmod(rem, 60)
       print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def inv_probit_np(x):
    
    jitter = 1e-1  # ensures output is strictly between 0 and 1
    return 0.5 * (1.0 + scipy.special.erf(x / np.sqrt(2.0))) * (1 - 2 * jitter) + jitter

def find_weights(input_dim, output_dim, X):
    
    """
    Find the initial weights of the Linear mean function based on
    input and output dimensions of the layer
    """

    if input_dim == output_dim:
        W = np.eye(input_dim)

    elif input_dim > output_dim:

        _, _, V = np.linalg.svd(X, full_matrices=False)
        W = V[:output_dim, :].T

    elif input_dim < output_dim:
        I = np.eye(input_dim)
        zeros = np.zeros((input_dim, output_dim - input_dim))
        W = np.concatenate([I, zeros], 1)

    W = W.astype(np.float64)

    return W

def create_objects(num_data, dim_input, dim_output,
        num_iterations, num_inducing,
        type_var, num_layers, dim_layers,
        num_batch, Z_init, num_test, learning_rate, base_seed, mean_Y_training, dataset_name,
        use_diagnostics, task_type, posterior_cholesky_Kmm, posterior_cholesky_Kmm_inv, posterior_cholesky_Schur, student_or_gaussian):

    ### Create objects ###
    propagate_layers_object = propagate_layers(num_data = num_data, 
        dim_input = dim_input, 
        dim_output = dim_output, 
        num_iterations = num_iterations, 
        num_inducing = num_inducing,  
        type_var = type_var, 
        num_layers = num_layers, 
        dim_layers = dim_layers,
        num_batch = num_batch, 
        Z_init = Z_init,  
        num_test = num_test, 
        learning_rate = learning_rate, 
        base_seed = base_seed, 
        mean_Y_training = mean_Y_training, 
        dataset_name = dataset_name,
        use_diagnostics = use_diagnostics, 
        task_type = task_type,
        posterior_cholesky_Kmm = posterior_cholesky_Kmm,
        posterior_cholesky_Kmm_inv = posterior_cholesky_Kmm_inv,
        posterior_cholesky_Schur = posterior_cholesky_Schur,
        student_or_gaussian = student_or_gaussian)

    network_architectures_object = network_architectures(num_data = num_data, 
        dim_input = dim_input, 
        dim_output = dim_output, 
        num_iterations = num_iterations, 
        num_inducing = num_inducing,  
        type_var = type_var, 
        num_layers = num_layers, 
        dim_layers = dim_layers,
        num_batch = num_batch, 
        Z_init = Z_init,  
        num_test = num_test, 
        learning_rate = learning_rate, 
        base_seed = base_seed, 
        mean_Y_training = mean_Y_training, 
        dataset_name = dataset_name,
        use_diagnostics = use_diagnostics, 
        task_type = task_type,
        posterior_cholesky_Kmm = posterior_cholesky_Kmm,
        posterior_cholesky_Kmm_inv = posterior_cholesky_Kmm_inv,
        posterior_cholesky_Schur = posterior_cholesky_Schur,
        student_or_gaussian = student_or_gaussian)
    
    cost_functions_object = cost_functions(num_data = num_data, 
        dim_input = dim_input, 
        dim_output = dim_output, 
        num_iterations = num_iterations, 
        num_inducing = num_inducing,  
        type_var = type_var, 
        num_layers = num_layers, 
        dim_layers = dim_layers,
        num_batch = num_batch, 
        Z_init = Z_init,  
        num_test = num_test, 
        learning_rate = learning_rate, 
        base_seed = base_seed, 
        mean_Y_training = mean_Y_training, 
        dataset_name = dataset_name,
        use_diagnostics = use_diagnostics, 
        task_type = task_type, 
        posterior_cholesky_Kmm = posterior_cholesky_Kmm,
        posterior_cholesky_Kmm_inv = posterior_cholesky_Kmm_inv,
        posterior_cholesky_Schur = posterior_cholesky_Schur,
        student_or_gaussian = student_or_gaussian)
    
    return network_architectures_object, cost_functions_object, propagate_layers_object

def RBF_np(X1, X2, log_lengthscales, log_kernel_variance):
       	
	X1 = X1 / np.exp(log_lengthscales)
	X2 = X2 / np.exp(log_lengthscales)
	X1s = np.sum(np.square(X1),1)
	X2s = np.sum(np.square(X2),1)       

	return np.exp(log_kernel_variance) * np.exp(-(-2.0 * np.matmul(X1,np.transpose(X2)) + np.reshape(X1s,(-1,1)) + np.reshape(X2s,(1,-1))) /2)     

@tf.function
def train_step(X_train_batch, Y_train_batch, network_architectures_object, propagate_layers_object, cost_functions_object, X_train_full, indices_minibatch, g, optimizer, step = None, writer = None):

    with tf.GradientTape() as tape:

        output_training = network_architectures_object.standard_DeepGP(X = X_train_batch, 
            X_mean_function = None, training_time = True,
            propagate_layers_object = propagate_layers_object, X_full = X_train_full, indices_minibatch = indices_minibatch, g=g)
        
        data_fit_cost = cost_functions_object.regression(f_mean = output_training[0], f_var = output_training[1], Y = Y_train_batch)

        kl_cost = output_training[2]+output_training[3]
        cost = - data_fit_cost + kl_cost

    var = tape.watched_variables()
    print('_________________________we are watching the following variables____________________')
    print(var)
    #with writer.as_default():
    #    # other model code would go here
    #    tf.summary.scalar("my_metric", 0.5, step=step)

    gradients = tape.gradient(cost, var)
    optimizer.apply_gradients(zip(gradients, var))
      
    l=1
    with tf.compat.v1.variable_scope('list_1', reuse = True):
        with tf.compat.v1.variable_scope('num_layer_'+str(l), reuse=True):

            Z = tf.compat.v1.get_variable(
                dtype=DTYPE, name='Z')

    with tf.compat.v1.variable_scope('list_2', reuse = True):
        with tf.compat.v1.variable_scope('num_layer_'+str(l), reuse=True):

            unrestricted_df_q = tf.compat.v1.get_variable(dtype=tf.float64,
                name='df_q') 
    df_q = tf.math.softplus(unrestricted_df_q)

    with tf.compat.v1.variable_scope('gaussian_likelihood', reuse=True):

        unrestricted_variance_output = tf.compat.v1.get_variable(dtype=tf.float64,
            name='unrestricted_variance_output', trainable = True)
    variance_output = tf.math.softplus(unrestricted_variance_output)
    log_variance_output = tf.math.log(variance_output)

    return data_fit_cost, output_training, df_q, log_variance_output, Z

@tf.function
def test_step(X_test_batch, Y_test_batch, network_architectures_object, propagate_layers_object, cost_functions_object, task_type, dim_output, mean_Y_training, X_train_full, indices_minibatch, num_samples_testing, age_mean):

    list_mean_epistemic, list_mean_distributional, list_var_epistemic, list_var_distributional = network_architectures_object.uncertainty_decomposed_DeepGP(X = X_test_batch, 
        X_mean_function = None, propagate_layers_object = propagate_layers_object, X_full = X_train_full, indices_minibatch = indices_minibatch, num_samples_testing = num_samples_testing)

    f_mean_testing = list_mean_epistemic[0]
    f_mean_testing += age_mean
    f_var_epistemic_testing = list_var_epistemic[0]
    f_var_distributional_testing = list_var_distributional[0]
    f_var_testing = f_var_epistemic_testing + f_var_distributional_testing

    if task_type=='regression':

        f_mean_testing += mean_Y_training

        ########### MAE on Testing data #################
        mae_testing = tf.reduce_mean(tf.abs(Y_test_batch - f_mean_testing))
        ########### Log-likelihood on Testing Data ######
        with tf.compat.v1.variable_scope('gaussian_likelihood', reuse=True):

            unrestricted_variance_output = tf.compat.v1.get_variable(dtype=tf.float64,
                name='unrestricted_variance_output', trainable = True)
        variance_output = tf.math.softplus(unrestricted_variance_output)
        log_variance_output = tf.math.log(variance_output)

        nll_test = tf.reduce_sum(variational_expectations(Y_test_batch, f_mean_testing, f_var_testing, log_variance_output)) 

    elif task_type=='classification':

        if dim_output==1:
            f_mean_testing_squashed = inv_probit(f_mean_testing)
    
        else:
            pass

        #########################################
        ##### Metrics for Accuracy ##############
        #########################################

        if dim_output>1:                

            correct_pred_testing = tf.equal(tf.argmax(f_mean_testing,1), tf.argmax(Y_test_batch,1))
            accuracy_testing = tf.reduce_mean(tf.cast(correct_pred_testing, DTYPE))
        else:

            correct_pred_testing = tf.equal(tf.round(f_mean_testing_squashed), Y_test_batch)
            accuracy_testing = tf.reduce_mean(tf.cast(correct_pred_testing, DTYPE))

        #################################################
        ########### Log-likelihood on Testing Data ######
        #################################################
    
        #### we sample 5 times and average ###

        sampled_testing = tf.tile(tf.expand_dims(f_mean_testing, axis=-1), [1,1,5]) + tf.multiply(tf.tile(tf.expand_dims(tf.sqrt(f_var_testing),axis=-1),[1,1,5]),
            tf.random.normal(shape=(tf.shape(f_mean_testing)[0],tf.shape(f_mean_testing)[1],5), dtype=DTYPE))	
        sampled_testing = tf.reduce_mean(sampled_testing, axis=-1, keepdims=False)

        if dim_output == 1:

            ##### Binary classification #####
            nll_test = tf.reduce_sum(bernoulli(p = sampled_testing, x = Y_test_batch))	

        else:

            ###### Multi-class Classification ######
            nll_test = tf.reduce_sum(multiclass_helper(inputul = sampled_testing, outputul = Y_test_batch))

    if task_type=='regression':
         
        return nll_test, mae_testing

    elif task_type=='classification':

        return nll_test, accuracy_testing

@tf.function
def get_predictions(X_test_batch, network_architectures_object, propagate_layers_object, X_full, indices_minibatch, num_samples_testing, age_mean):

    list_mean_epistemic, list_mean_distributional, list_var_epistemic, list_var_distributional  = network_architectures_object.uncertainty_decomposed_DeepGP(X = X_test_batch, 
        X_mean_function = None, propagate_layers_object = propagate_layers_object, X_full = X_full, indices_minibatch = indices_minibatch, num_samples_testing = num_samples_testing)

    f_mean_testing = list_mean_epistemic[0]
    f_mean_testing+= age_mean
    f_var_epistemic_testing = list_var_epistemic[0]
    f_var_distributional_testing = list_var_distributional[0]

    return f_mean_testing, f_var_epistemic_testing, f_var_distributional_testing



## main function ###
def main_DeepGP( num_data, dim_input, dim_output,
        num_iterations, num_inducing,
        type_var, num_layers, dim_layers,
        num_batch, Z_init,  num_test, learning_rate, base_seed, mean_Y_training, dataset_name,
        use_diagnostics, task_type, X_training, Y_training, X_testing, Y_testing, posterior_cholesky_Kmm, posterior_cholesky_Kmm_inv, posterior_cholesky_Schur,
        num_samples_testing, num_samples_hut_trace_est, student_or_gaussian):

        #tf.random.set_seed(base_seed)

        #train_ds = tf.data.Dataset.from_tensor_slices(
        #    (X_training, Y_training)).shuffle(base_seed).batch(num_batch)

        #test_ds = tf.data.Dataset.from_tensor_slices((X_testing, Y_testing)).batch(num_batch)
        #g = tf.placeholder(tf.float64, shape = (num_samples_hut_trace_est, num_inducing[0]+num_data, 1), name= 'g_h')

        ### Create objects ###
        network_architectures_object, cost_functions_object,  propagate_layers_object = create_objects(num_data, dim_input, dim_output,
            num_iterations, num_inducing, type_var, num_layers, dim_layers,
            num_batch, Z_init, num_test, learning_rate, base_seed, mean_Y_training, dataset_name,
            use_diagnostics, task_type, posterior_cholesky_Kmm, posterior_cholesky_Kmm_inv, posterior_cholesky_Schur, student_or_gaussian)
        
        list_slack_log_det_Kfu_fu_np = []
        list_slack_log_det_Kfu_fu_explicit_np = []
        list_slack_conj_grad_solution_np = []
        list_df_q_np = []
        list_elbo_lower_bound = []
        list_elbo_actual_bound = []
        list_log_variance_output = []
        list_num_steps = []
        list_nll_test = []
        list_time = []

        where_to_save = str(dataset_name)+'/'+str(student_or_gaussian)+'/num_inducing_'+str(num_inducing[0])+'/lr_'+str(learning_rate)+'/seed_'+str(base_seed)
        opt = tf.compat.v1.train.AdamOptimizer(learning_rate)

        writer = tf.summary.create_file_writer('./tensorboard/'+where_to_save)
        cmd='mkdir -p ./tensorboard'
        os.system(cmd)


        for i in range(num_iterations):

            #g_np = np.random.normal(loc=0.0, scale=1.0, size=(num_samples_hut_trace_est, num_inducing[0] + num_data, 1))
            g_np = np.random.normal(loc=0.0, scale=1.0, size=(num_samples_hut_trace_est, num_inducing[0] + num_data))


            data_fit_cost_np, output_training, df_q, log_variance_output, Z = train_step(X_train_batch = X_training, Y_train_batch = Y_training, 
                network_architectures_object = network_architectures_object, propagate_layers_object = propagate_layers_object, 
                cost_functions_object = cost_functions_object, X_train_full = X_training, indices_minibatch = np.arange(X_training.shape[0]), g = g_np, optimizer=opt)

            #_________________________we are watching the following variables____________________
            #(<tf.Variable 'list_1/num_layer_1/log_kernel_variance:0' shape=() dtype=float64>, 0
            #<tf.Variable 'list_1/num_layer_1/log_lengthscales:0' shape=(1,) dtype=float64>, 1
            #<tf.Variable 'list_1/num_layer_1/Z:0' shape=(10, 1) dtype=float64>, 2
            #<tf.Variable 'list_2/num_layer_1/q_mu:0' shape=(10, 1) dtype=float64>, 3
            #<tf.Variable 'list_2/num_layer_1/q_cholesky_unmasked:0' shape=(1, 10, 10) dtype=float64>, 4
            #<tf.Variable 'list_2/num_layer_1/df_q:0' shape=() dtype=float64>, 5
            #<tf.Variable 'posterior_cholesky_Kmm:0' shape=(10, 10) dtype=float64>, 6
            #<tf.Variable 'posterior_cholesky_Schur:0' shape=(100, 100) dtype=float64>, 7
            #<tf.Variable 'gaussian_likelihood/log_variance_output:0' shape=() dtype=float64>), 8
            df_q_np = df_q.numpy()
            log_variance_output_np = log_variance_output.numpy() 
            Z_np = Z.numpy()

            f_mean_training_np = output_training[0]
            f_var_training_np = output_training[1]
            kl_cost_qu_np = output_training[2]
            kl_cost_inverse_wishart_np = output_training[3]
            list_hopefully_id_matrix_sample_np = output_training[4]
            list_hopefully_id_matrix_mean_covariance_np = output_training[5]
            slack_conj_grad_solution_np = output_training[6]
            slack_log_det_Kfu_fu_lower_bound_np = output_training[7]
            slack_log_det_Kfu_fu_explicit_np = output_training[8]
            list_C_np = output_training[9]
            list_Kfu_fu_np = output_training[10] 
            list_hopefully_id_matrix_mean_big_covariance_np = output_training[11]
            kl_cost_inverse_wishart_actual_np = output_training[12]
            num_steps = output_training[13][0]
            list_num_steps.append(num_steps.numpy())

            slack_conj_grad_solution_np = slack_conj_grad_solution_np[0].numpy()
            slack_log_det_Kfu_fu_lower_bound_np = slack_log_det_Kfu_fu_lower_bound_np[0].numpy()
            slack_log_det_Kfu_fu_explicit_np = slack_log_det_Kfu_fu_explicit_np[0].numpy()           
            kl_cost_np = kl_cost_qu_np.numpy() + kl_cost_inverse_wishart_np.numpy()
            kl_cost_actual_np = kl_cost_qu_np.numpy() + kl_cost_inverse_wishart_actual_np.numpy()
            list_slack_log_det_Kfu_fu_np.append(slack_log_det_Kfu_fu_lower_bound_np)
            list_slack_log_det_Kfu_fu_explicit_np.append(slack_log_det_Kfu_fu_explicit_np)
            list_slack_conj_grad_solution_np.append(slack_conj_grad_solution_np)
            list_df_q_np.append(df_q_np)
            list_log_variance_output.append(log_variance_output_np)
            print('****************************')
            print(slack_conj_grad_solution_np)
            print(slack_log_det_Kfu_fu_lower_bound_np)
            print(slack_log_det_Kfu_fu_explicit_np)
            print(df_q_np)
            print(num_steps.numpy())

            elbo_lower_bound = data_fit_cost_np - kl_cost_np
            elbo_actual = data_fit_cost_np - kl_cost_actual_np

            list_elbo_lower_bound.append(elbo_lower_bound.numpy())
            list_elbo_actual_bound.append(elbo_actual.numpy())
            total_nll_np = 0.0
            #for X_test_batch, Y_test_batch, indices_minibatch_testing in test_ds:

                #print('**********************************')
                #print(' Testing batches')
                #print(X_test_batch)
                #print(Y_test_batch)

            nll_test_now, precision_now = test_step(X_testing, Y_testing, network_architectures_object, propagate_layers_object, cost_functions_object,
                task_type, dim_output, mean_Y_training, X_training, np.arange(X_testing.shape[0]), num_samples_testing, age_mean = mean_Y_training)
            print(nll_test_now)
            print(precision_now)
            print('----------')
            total_nll_np+=nll_test_now
            mae_testing_overall_np = precision_now
            total_nll_np = total_nll_np / num_test
            list_nll_test.append(total_nll_np.numpy())

            if task_type=='regression':
                print('at iteration '+str(i) + 're cost :'+str(data_fit_cost_np)+' kl cost qu :'+str(kl_cost_qu_np)+' kl cost inverse wishart :'+str(kl_cost_inverse_wishart_np))	

            elif task_type=='classification':
                print('at iteration '+str(i) + 're cost :'+str(data_fit_cost_np)+' kl cost qu :'+str(kl_cost_qu_np)+' kl cost inverse wishart :'+str(kl_cost_inverse_wishart_np))	
                
            #if i % 250==0 and i!=0:
            if i % 100==0:
                ##############################################
                #### produce the identity  matrix figure #####
                ##############################################

                cmd = 'mkdir -p ./figures/'+where_to_save
                os.system(cmd)

                im = plt.imshow(list_hopefully_id_matrix_sample_np[0], cmap='coolwarm', interpolation='nearest')
                plt.title(r'$\Sigma_{uu}^{-1}K_{uu}$')
                plt.colorbar(im)
                plt.savefig('./figures/'+where_to_save+'/plot_iteration_'+str(i)+'_id_matrix_sample.png')
                plt.close()

                im = plt.imshow(list_hopefully_id_matrix_mean_covariance_np[0], cmap='coolwarm', interpolation='nearest')
                plt.title(r'$\tilde{K_{uu}^{-1}}K_{uu}$')
                plt.colorbar(im)
                plt.savefig('./figures/'+where_to_save+'/plot_iteration_'+str(i)+'_id_matrix_mean_covariance.png')
                plt.close()

                list_hopefully_id_matrix_np = list_Kfu_fu_np[0] - list_C_np[0] + np.eye(list_Kfu_fu_np[0].shape[0]) * 1e-1

                im = plt.imshow(list_hopefully_id_matrix_np, cmap='coolwarm', interpolation='nearest')
                plt.title(r'$K_{fu,fu}  -\tilde{K_{fu,fu}}$')
                plt.colorbar(im)
                plt.savefig('./figures/'+where_to_save+'/plot_iteration_'+str(i)+'diff_Kfu_fu_C.png')
                plt.close()

                list_hopefully_id_matrix_np = list_hopefully_id_matrix_mean_big_covariance_np[0]

                im = plt.imshow(list_hopefully_id_matrix_np, cmap='coolwarm', interpolation='nearest')
                plt.title(r'$K_{fu,fu}\tilde{K_{fu,fu}}$')
                plt.colorbar(im)
                plt.savefig('./figures/'+where_to_save+'/plot_iteration_'+str(i)+'_id_matrix_mean_big_covariance.png')
                plt.close()

        ###################################################################
        ######### Upper and Lower Bounds Diagnostic Plots #################
        ###################################################################

        print(len(range(num_iterations)))
        print(len(list_slack_log_det_Kfu_fu_np))
        print(len(list_slack_log_det_Kfu_fu_explicit_np))
        print(len(list_slack_conj_grad_solution_np))
        print(len(list_df_q_np))
        print(len(list_elbo_lower_bound))
        print(len(list_elbo_actual_bound))
        print(len(list_log_variance_output))
        print(len(list_num_steps))
        print(len(list_nll_test))

        dict = {'Training Iterations' : range(num_iterations),
            'Slack log-determinant bound':list_slack_log_det_Kfu_fu_np,
            'Slack log-determinant C' : list_slack_log_det_Kfu_fu_explicit_np,
            'Conjugate Gradient Solution Error' : list_slack_conj_grad_solution_np,
            'Degrees of Freedom' : list_df_q_np,
            'ELBO Lower' : list_elbo_lower_bound,
            'ELBO Actual' : list_elbo_actual_bound,
            'Log Variance Output' : list_log_variance_output,
            'CG Steps' : list_num_steps,
            'LL Test' : list_nll_test}
        df = pd.DataFrame(dict)

        sns.lineplot(x = df['Training Iterations'], y = df['Slack log-determinant bound'])
        plt.savefig('./figures/'+where_to_save+'/plot_slack_log_det_lower_bound.png')
        plt.close()

        sns.lineplot(x = df['Training Iterations'], y = df['Slack log-determinant C'])
        plt.savefig('./figures/'+where_to_save+'/plot_slack_log_det.png')
        plt.close()

        sns.lineplot(x = df['Training Iterations'], y = df['Conjugate Gradient Solution Error'])
        plt.savefig('./figures/'+where_to_save+'/plot_slack_conj_grad_sol.png')
        plt.close()

        sns.lineplot(x = df['Training Iterations'], y = df['Degrees of Freedom'])
        plt.savefig('./figures/'+where_to_save+'/plot_dof.png')
        plt.close()

        sns.lineplot(x = df['Training Iterations'], y = df['ELBO Lower'])
        plt.savefig('./figures/'+where_to_save+'/plot_elbo_lower.png')
        plt.close()

        sns.lineplot(x = df['Training Iterations'], y = df['ELBO Actual'])
        plt.savefig('./figures/'+where_to_save+'/plot_elbo_actual.png')
        plt.close()

        sns.lineplot(x = df['Training Iterations'], y = df['Log Variance Output'])
        plt.savefig('./figures/'+where_to_save+'/plot_log_variance_output.png')
        plt.close()

        sns.lineplot(x = df['Training Iterations'], y = df['CG Steps'])
        plt.savefig('./figures/'+where_to_save+'/plot_cg_steps.png')
        plt.close()

        print('**** stats on slack *****')
        print(np.median(list_slack_log_det_Kfu_fu_np))
        print(np.median(list_slack_log_det_Kfu_fu_explicit_np))
        print(np.median(list_slack_conj_grad_solution_np))
        
        df.to_csv('./figures/'+where_to_save+'/summary.csv', index = False)









