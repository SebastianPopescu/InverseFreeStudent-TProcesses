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
DTYPE=tf.float32
import seaborn as sns
from sklearn.cluster import  KMeans
from matplotlib import rcParams
import itertools
from scipy.stats import norm
import pandas as pd
import scipy
from DeepGP import  main_DeepGP
sys.setrecursionlimit(10000)
from uci_datasets_fetcher import *

def RBF_np(X1, X2, log_lengthscales, log_kernel_variance):
       	
	X1 = X1 / np.exp(log_lengthscales)
	X2 = X2 / np.exp(log_lengthscales)
	X1s = np.sum(np.square(X1),1)
	X2s = np.sum(np.square(X2),1)       

	return np.exp(log_kernel_variance) * np.exp(-(-2.0 * np.matmul(X1,np.transpose(X2)) + np.reshape(X1s,(-1,1)) + np.reshape(X2s,(1,-1))) /2)     


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_layers', type=int, help = 'the number of total GP layers')
    parser.add_argument('--num_inducing', type=int, help = 'the number of inducing points in the first layer')
    parser.add_argument('--dim_layers', type=int, help='the size of the hidden GP layers, this is a scalar which generalizes across all hidden layers of the DGP')
    parser.add_argument('--learning_rate', type = float, help = 'the constant learning rate to be used for optimization')
    parser.add_argument('--num_iterations', type = int, help = 'the number of iterations for the training process')
    parser.add_argument('--num_batch', type = int , help = 'minibatch size during training process')
    parser.add_argument('--base_seed', type = int, help = 'seed for selecting the minibatch')
    parser.add_argument('--dataset_name', type = str, help = 'name of the UCI dataset to use')
    parser.add_argument('--dim_output', type=int)
    args = parser.parse_args()	

    #####################################
    ######### Fetch UCI dataset #########
    #####################################

    X_training, Y_training, X_testing, Y_testing  = fetch_dataset(name=args.dataset_name)

    print('shape of training set')
    print(X_training.shape)
    print(Y_training.shape)
    print('shape of testing set')
    print(X_testing.shape)
    print(Y_testing.shape)

    num_inducing = [args.num_inducing]
    num_inducing.extend([args.num_inducing for _ in range(args.num_layers-1)])

    dim_layers = [X_training.shape[1]]
    dim_layers.extend([args.dim_layers for _ in range(args.num_layers-1)])
    dim_layers.append(args.dim_output)

    ####################################################################################
    ######## get initalization for first layer inducing points locations ###############
    ####################################################################################
    decimal = int(X_training.shape[0]/10)
    if decimal>args.num_inducing:
        k_mean_output = get_kmeans(X = X_training[:decimal,...], num_inducing = args.num_inducing)
    elif X_training.shape[0] >10000:
        k_mean_output = get_kmeans(X = X_training[:10000,...], num_inducing = args.num_inducing)
    else:
        k_mean_output = get_kmeans(X = X_training, num_inducing = args.num_inducing)		

    X_training = X_training.astype('float32')
    X_testing  = X_testing.astype('float32')
    Y_training = Y_training.astype('float32')
    Y_testing = Y_testing.astype('float32')
    k_mean_output = k_mean_output.astype('float32')


    dict_chol_Kzz = defaultdict()
    Kzz = RBF_np(X1 = k_mean_output, X2 = k_mean_output, log_lengthscales = [-0.301], log_kernel_variance = 0.301)
    Kzz_inv = np.linalg.inv(Kzz)
    chol_Kzz_inv = np.linalg.cholesky(Kzz_inv)
    dict_chol_Kzz[1] = chol_Kzz_inv

    main_DeepGP( num_data = X_training.shape[0], dim_input = X_training.shape[1], dim_output = args.dim_output,
            num_iterations = args.num_iterations, num_inducing = num_inducing,
            type_var = 'full', num_layers = args.num_layers, dim_layers = dim_layers,
            num_batch = args.num_batch, Z_init = k_mean_output,  num_test = X_testing.shape[0], learning_rate = args.learning_rate, base_seed = args.base_seed, mean_Y_training = 0.0, dataset_name = args.dataset_name,
            use_diagnostics = True, task_type = 'classification', X_training = X_training, Y_training = Y_training, X_testing = X_testing, Y_testing = Y_testing, posterior_cholesky_Kmm_inverse = dict_chol_Kzz)












