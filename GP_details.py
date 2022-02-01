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


class GP_definition(object):

    ##########################################################
    #### Parent class that holds all the model parameters ####
    ##########################################################

    def __init__(self, num_data, dim_input, dim_output,
        num_iterations, num_inducing, 
        type_var, num_layers, dim_layers,
        num_batch, Z_init, num_test, learning_rate, base_seed, mean_Y_training, dataset_name,
        use_diagnostics, task_type, posterior_cholesky_Kmm, posterior_cholesky_Kmm_inv):

        self.task_type = task_type
        self.use_diagnostics = use_diagnostics
        self.dataset_name = dataset_name
        self.mean_Y_training = mean_Y_training
        self.base_seed = base_seed
        self.dim_layers = dim_layers
        self.num_layers = num_layers
        self.learning_rate = learning_rate	
        self.num_test = num_test
        self.num_batch = num_batch
        self.type_var = type_var
        self.num_data = num_data
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.num_iterations = num_iterations
        self.num_inducing = num_inducing
        self.Z_init = Z_init
        self.posterior_cholesky_Kmm = posterior_cholesky_Kmm
        self.posterior_cholesky_Kmm_inv = posterior_cholesky_Kmm_inv


