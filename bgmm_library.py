#!/usr/bin/env python

# Firas Said Midani
# Start date: 2018-05-07
# Final date: 2018-05-07

# DESCRIPTION Library of functions for analysis of flow cytometry data

# TABLE OF CONTENTS
#
#|-- Moment calcuations
#     |-- computeDeviationMenas
#     |-- computeSumSquares
#
#|-- Algebraic manipulations
#     |-- computePosteriorLabels
#     |-- computeWeightedMVNormalPDF
#
#|-- Distribution sampling
#     |-- computeMVNormalPDF
#     |-- sampleMu
#     |-- sampleSigma
#
#|-- Hyper-parameter updates
#     |-- updateInvWishartDOF
#     |-- updateInvWishartScaleMatrix
#     |-- updateMVNormalMu
#     |-- updateMVNormalSigma
#     |-- updateOmega

# IMPORT NECESSARY LIBRARIES

import numpy as np

from scipy.stats import dirichlet,invwishart,multinomial,multivariate_normal

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

def computeMVNormalPDF(data_i,mu_j,Sigma_j):
    '''
    f computes probability of value sampled from a MultiVariate Normal distribution

    Keyword arguments:
    data_i -- n-dimensional data as numpy.ndarray (1 x p)
    mu_j -- mean as numpy.ndarray (1 x p)
    Sigma_j -- covariance as numpy.ndarray (p x p)

    Returns float
    '''

    return multivariate_normal(mean=mu_j,cov=Sigma_j).pdf(data_i)

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

def computePosteriorLabels(data,labels,Mu_s,Sigma_s,Omega_s):
    '''
    computePosteriorLabels calcualte posterior for each data point belonging to each label

    Keyword argumetns:
    data -- n-dimensional data as numpy.ndarray (n x p)
    labels -- cluster labels (1 x j)
    Mu_s -- mean as numpy.ndarray (1 x p)
    Sigma_s -- covariance as numpy.ndarray (p x p)
    Omega_s -- label weights as numpy.ndarray (1 x j)

    Returns numpy.ndarray (1 x j)
    '''

    proba_list,proba_max_list = [],[];

    for data_i in data:

        # compute posterior label probabilities 
        proba = computeWeightedMVNormalPDF(data_i,labels,Mu_s,Sigma_s,Omega_s);

        # record probabilities and the label with maximum probability
        proba_list.append(proba);
        proba_max_list.append(np.where(proba==np.max(proba))[0][0])

    return proba_list,proba_max_list


def computeWeightedMVNormalPDF(data_i,labels,Mu_s,Sigma_s,Omega_s):
    '''
    t computes probability of a sample belonging to each cluster

    Keyword argumetns:
    data_i -- n-dimensional data as numpy.ndarray (1 x p)
    labels -- cluster labels (1 x j)
    Mu_s -- mean as numpy.ndarray (1 x p)
    Sigma_s -- covariance as numpy.ndarray (p x p)
    Omega_s -- label weights as numpy.ndarray (1 x j)

    Returns numpy.ndarray (1 x j)
    '''

    mixture, proba = [],[]

    # compute label probability for each cluster
    for jj,mj,sj,wj in zip(set(labels),Mu_s,Sigma_s,Omega_s):

        mixture.append(wj*computeMVNormalPDF(data_i,mj,sj))

    # normalize label probability by marginal probability over all labels
    for mix in mixture:

        proba.append(float(mix)/np.sum(mixture));

    return proba

def sampleMu(data_j,tau_j,xi_j,Sigma_s):
    '''
    sampleMu generates posterior sample of mean using a MultiVariate Normal distribution

    Keyword arguments:
    data_j -- n-dimensional data as numpy.ndarray (n x p)
    xi_j -- prior mean hyperparameter (1 x p)
    tau_j -- prior precision hyperparameter (1 x 1)
    Sigma_S -- prior estimate of covariance (p x x)

    Returns 
    '''

    # updates the mean hyperparameter
    updatedMu = np.ravel(updateMVNormalMu(data,tau_j,xi_j));

    # updates the covariance hyperparameter
    updatedSigma = updateMVNormalSigma(data,Sigma_s,tau_j);

    return multivariate_normal(mean=updatedMu,cov=updatedSigma).rvs(1)

def sampleSigma(data_j,m_j,psi_j,tau_j,xi_j):
    '''
    sampleSigma generates posterior sample of covariance using an Inverse Wishart distribution

    Keyword arguments:
    data_j -- n-dimensional data as numpy.ndarray (n x p)
    m_j -- updated degrees of freedom hyperparameter as float (1 x 1)
    psi_j -- prior scale matrix hyperparameter as numpy.ndarray (p x p)
    tau_j -- prior precision hyperparameter (1 x 1)
    xi_j -- prior mean hyperparameter (1 x p)

    Returns numpy.ndarray (p x p)
    '''

    # number of samples
    n = float(len(data_j)); 

    # update DOF
    iw_dof = updateInvWishartDOF(data,m_j);

    # update Scale Matirx

    # 1 -- compute sum of squares
    iw_SumSquares = computeSumSquares(data);

    # 2 -- compute deviation of mean values from prior
    iw_DeviationMeans = computeDeviationMeans(data,xi_j);
    iw_DeviationMeans = np.dot(((n*tau_j)/(n+tau_j)),iw_DeviationMeans); 

    # 3 -- put it together
    iw_scale = updateInvWishartScaleMatrix(psi_j,iw_SumSquares,iw_DeviationMeans);

    return invwishart(df=iw_dof,scale=iw_scale).rvs(1)

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

def updateOmega(data,labels,a):
    '''
    updateOmega updates label weights (i.e. probabilities) using a Dirichlet distribution

    Keyword argumetns:
    data -- n-dimensional data as numpy.ndarray (n x p)
    labels -- sample labels as numpy.ndarr  ay (1 x n)
    a -- prior label weights as numpy.ndarray (1 x j)

    Returns numpy.ndarray (1 x j)
    '''

    # number of unique labels
    n_labels = len(set(labels))

    # coutn of samples with each label
    V = [list(labels).count(ii) for ii in range(n_labels)];

    return dirichlet([xx+yy for xx,yy in zip(a,V)]).rvs(1)[0]
