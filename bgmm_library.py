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
	m -- prior degrees of freedom as int or float

	Returns float
	'''

	n = len(data)

	return n+float(m)

