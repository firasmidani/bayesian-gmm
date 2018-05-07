#!/usr/bin/env python

# Firas Said Midani
# Start date: 2018-05-27
# Final date: 2018-05-07

# DESCRIPTION Library of functions for analysis of flow cytometry data

# TABLE OF CONTENTS
#
#

# IMPORT NECESSARY LIBRARIES

import numpy as np

from scipy.stats import dirichlet,invwishart,multinomial,multivariate_normal

def updateInvWishartDOF(data,m):
	'''
	updateInvWishartDOF updates the degrees of freedom hyperparameter for Inverse Wishart distribution

	Keyword arguments:
	data -- n-dimensional data as numpy.ndarray
	m -- prior degrees of freedom hyperparameter as int or float

	Returns float
	'''

	n = len(data)

	return n+float(m)

def updateInvWishartScaleMatrix(PriorScaleMatrix,SumSquares,DeviationMeans):
    '''
    updateInvWishartScaleMatrix updates the scale matrix hyperparameter for the Inverse Wishart distribution

    Keyword arguments:
    PriorScaleMatrix -- prior scale matrix hyperparameter as numpy.ndarray
    SumSquares -- sum of squares for deviation of data from mean as numpy.ndarray
    DeviationMeans -- deviation between the prior and estimate mean values as numpy.ndarray

    Returns nump.ndarray
    '''

    return PriorScaleMatrix + SumSquares + DeviationMeans

def computeSumSquares(data):
	'''
	computeSumSquares computes sum of squares for deviation of data from mean

	Keyword arguments:
	data -- n-dimensional data as numpy.ndarray	

	Returns numpy.ndarray
	'''

	# estimate mean values
    y_bar = np.mean(data,axis=0);

    # estimate deviation from mean
    deviations = np.matrix(data - y_bar);
    
    # dot product of deviations from mean
    Squares = [deviation.transpose()*deviation for deviation in deviations]

    # sum deviationd ot products
    SumSquares = np.sum(Squares,axis=0)
    
    return SumSquares
    