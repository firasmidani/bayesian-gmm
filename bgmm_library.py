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

	# number of samples
	n = len(data)

	return n+float(m)

def updateInvWishartScaleMatrix(PriorScaleMatrix,SumSquares,DeviationMeans):
    '''
    updateInvWishartScaleMatrix updates the scale matrix hyperparameter for the Inverse Wishart distribution

    Keyword arguments:
    PriorScaleMatrix -- prior scale matrix hyperparameter as numpy.ndarray (p x p)
    SumSquares -- sum of squares for deviation of data from mean as numpy.ndarray (p x p)
    DeviationMeans -- deviation between the prior and estimate mean values as numpy.ndarray (p x p)

    Returns nump.ndarray (p x p)
    '''

    return PriorScaleMatrix + SumSquares + DeviationMeans

def computeSumSquares(data):
	'''
	computeSumSquares computes sum of squares for deviation of data from mean

	Keyword arguments:
	data -- n-dimensional data as numpy.ndarray (n x p)

	Returns numpy.ndarray (p x p)
	'''

	# estimate mean values
    y_bar = np.mean(data,axis=0);

    # estimate deviation from mean
    deviations = np.matrix(data - y_bar);
    
    # compute squares of deviations from mean
    Squares = [deviation.transpose()*deviation for deviation in deviations]

    # sum deviations of dot products
    SumSquares = np.sum(Squares,axis=0)
    
    return SumSquares

def computeDeviationMeans(data,PriorMean):
    '''
    computeDeviationMeans compute squares for deviation between prior and esitmated mean values

    Keyword arguments:
    data -- n-dimensional data as numpy.ndarray (n x p)
    PriorMean -- prior mean hyperparameter as numpy.ndarray (p x 1)

    Returns numpy.ndarray (p x p)
    '''

	# estimate mean values
    y_bar = np.mean(data,axis=0);

    # estimate deviation from prior mean value
    deviations = np.matrix(y_bar - PriorMean.transpose());

    # compute squares of deviations from prior mean
    Squares = deviation.transpose()*deviation
    
    return Squares
    
def updateMVNormalMu(data,precision,PriorMean):
     '''
    updateMVNormalMu updates the mean hyperparamter for MultiVariate Normal distribution

    Keyword arguments:
    data -- n-dimensional data as numpy.ndarray (n x p)
    precision -- prior precision hyperparameter as float (1 x 1) 
    PriorMean -- prior mean values as numpy.ndarray (p x 1)

    Returns numpy.ndarray (p x p)
    '''
   
   	# number of samples
    n = len(data);

	# estimate mean values
    y_bar = np.mean(data,axis=0); # (1 x p)
    
    return (np.dot(precision,PriorMean).transpose()+np.dot(n,y_bar))/(n+float(precision))
 
def updateMVNormalSigma(data,Sigma,precision):
    '''
    updateMVNormalSigma updates the covariance hyperparameter for MultiVariate Normal distribution

    Keyword arguments:
    data -- n-dimensional data as numpy.ndarray (n x p)
    Sigma -- prior covariance as numpy.ndarray (p x p)
    precision -- priro precision hyperparameter as float (1 x 1)

    Returns numpy.ndarray (p x p)
    '''

    # number of samples
    n = len(data);
    
    return np.dot(float(1)/(float(precision)+n),Sigma)

 def sampleSigma(data_j,m_j,xi_j,psi_j):
 	'''
 	sampleSigma generates posterior sample of covariance using an Inverse Wishart distribution

 	Keyword arguments:
 	data_j -- n-dimensional data as numpy.ndarray (n x p)
 	m_j -- updated degrees of freedom hyperparameter as float (1 x 1)
 	xi_j -- prior mean hyperparameter
 	psi_j -- prior scale matrix hyperparameter as numpy.ndarray (p x p)

 	Returns numpy.ndarray (p x p)
 	'''

 	# number of samples
 	n = float(len(data_j)); 

    # update DOF
    iw_dof = updateInvWishartDOF(data,m_j);

    # update Scale Matirx

  	## 1 -- compute sum of squares
    iw_SumSquares = computeSumSquares(data);
    
    ## 2 -- compute deviation of mean values from prior
    iw_DeviationMeans = computeDeviationMeans(data,xi_j);
    iw_DeviationMeans = np.dot(((n*tau_j)/(n+tau_j)),iw_DeviationMeans); 
    
    ## 3 -- put it together
    iw_scale = updateInvWishartScaleMatrix(psi_j,iw_SumSquares,iw_DeviationMeans);
    
    return invwishart(df=iw_dof,scale=iw_scale).rvs(1)

