# -*- coding: utf-8 -*-
from __future__ import print_function, division
import tensorflow as tf
import numpy as np
DTYPE=tf.float64

def RBF(X1, X2, log_lengthscales, log_kernel_variance):
       	
	X1 = X1 / tf.exp(log_lengthscales)
	X2 = X2 / tf.exp(log_lengthscales)
	X1s = tf.reduce_sum(tf.square(X1),1)
	X2s = tf.reduce_sum(tf.square(X2),1)       

	return tf.exp(log_kernel_variance) * tf.exp(-(-2.0 * tf.matmul(X1,tf.transpose(X2)) + tf.reshape(X1s,(-1,1)) + tf.reshape(X2s,(1,-1))) /2)      

def RBF_Kdiag(X, log_kernel_variance):
	### returns a list
	return tf.ones((tf.shape(X)[0],1),dtype=tf.float64) * tf.exp(log_kernel_variance)	

def RBF_without_kernel_variance(X1, X2, log_lengthscales, log_kernel_variance):
       	
	X1 = X1 / tf.exp(log_lengthscales)
	X2 = X2 / tf.exp(log_lengthscales)
	X1s = tf.reduce_sum(tf.square(X1),1)
	X2s = tf.reduce_sum(tf.square(X2),1)       

	return tf.exp(-(-2.0 * tf.matmul(X1,tf.transpose(X2)) + tf.reshape(X1s,(-1,1)) + tf.reshape(X2s,(1,-1))) /2)      