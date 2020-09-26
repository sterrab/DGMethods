#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 15:14:44 2020

@author: sorayaterrab
"""

import numpy as np

# Legendre Polynomial (degree m ) evaluation of an input array x
def LegendrePoly(m,x):
    x = np.array(x)
    LegPolyEval = np.zeros((x.size, m+1))
    for i in range(m+1):
        if i == 0: 
            LegPolyEval[:,i] = np.ones(x.size)
        elif i == 1: 
            LegPolyEval[:,i] = x
        else: 
            LegPolyEval[:,i] = ((2*i-1)/i )*x*LegPolyEval[:,i-1] - ((i-1)/i )*LegPolyEval[:,i-2]
    return LegPolyEval

# Calculating Cell Centers for a given 1D domain with N elements, and 
    # Calculating Corresponding Legendre Basis Coefficients for given degree
def CellCenters_LegBasisCoeff(xleft, xright, N, degree, InitialCondition):
    dx = (xright-xleft)/N # assumes uniform mesh
    # Guass-Legendre Quadrature: Evaluation
    num_of_eval_pts= degree+1
    xi_eval, weights_eval = np.polynomial.legendre.leggauss(num_of_eval_pts)
    
    #Computing Basis Coefficients and Cell Centers
    CellCenters = np.zeros(N)
    BasisCoeff = np.zeros((N,degree+1))
    
    for m in range(degree+1):
        for i in range(N): 
            CellCenters[i]=xleft + (i+0.5)*dx 
            # In-Cell [-1,1] coordinates:
            nodes_eval = (dx/2)*xi_eval + CellCenters[i]*np.ones(num_of_eval_pts)
            # Projections calculation 
            integrand =  InitialCondition(nodes_eval)*LegendrePoly(degree,xi_eval)[:,m]
            BasisCoeff[i,m]= (2*m+1)/2*sum(weights_eval * integrand)
    
    return dx, CellCenters, BasisCoeff

# Orthonormal Legendre Polynomial (degree m ) evaluation of an input array x
def orthonormalLegendrePoly(m,x):
    x = np.array(x)
    LegPolyEval = np.zeros((x.size, m+1))
    scaling = np.zeros((m+1,m+1))
    for i in range(m+1):
        scaling[i,i] = np.sqrt(i+0.5)
        if i == 0: 
            LegPolyEval[:,i] = np.ones(x.size)
        elif i == 1: 
            LegPolyEval[:,i] = x
        else: 
            LegPolyEval[:,i] = ((2*i-1)/i )*x*LegPolyEval[:,i-1] - ((i-1)/i )*LegPolyEval[:,i-2]
    orthonormalLegPolyEval = LegPolyEval @ scaling
    return orthonormalLegPolyEval

# Calculating Cell Centers for a given 1D domain with N elements, and 
    # Calculating Corresponding Orthonormal Legendre Basis Coefficients for given degree
def CellCenters_OrthoNormalLegBasisCoeff(xleft, xright, N, degree, InitialCondition):
    dx = (xright-xleft)/N # assumes uniform mesh
    # Guass-Legendre Quadrature: Evaluation
    num_of_eval_pts= degree+1
    xi_eval, weights_eval = np.polynomial.legendre.leggauss(num_of_eval_pts)
    
    #Computing Basis Coefficients and Cell Centers
    CellCenters = np.zeros(N)
    BasisCoeff = np.zeros((N,degree+1))
    
    for m in range(degree+1):
        for i in range(N): 
            CellCenters[i]=xleft + (i+0.5)*dx 
            # In-Cell [-1,1] coordinates:
            nodes_eval = (dx/2)*xi_eval + CellCenters[i]*np.ones(num_of_eval_pts)
            # Projections calculation 
            integrand = InitialCondition(nodes_eval)*orthonormalLegendrePoly(degree,xi_eval)[:,m]
            BasisCoeff[i,m]= sum(weights_eval * integrand)
    
    return dx, CellCenters, BasisCoeff

# Approximation and Plot Points in a domain [xleft, xright] given a provided degree
def OrthoNormalLegendreBasisApproximation(xleft, xright, N, degree, InitialCondition, num_of_plot_pts): 
    # Guass-Legendre Points for Plotting
    zeta_plot, _ = np.polynomial.legendre.leggauss(num_of_plot_pts)
    
    # Basis Matrix with columns of m'th degree orthonormal Legendre Polynomial evaluated at xi_plot
    BasisMatrix = orthonormalLegendrePoly(degree,zeta_plot)
    dx, CellCenters, BasisCoefficients = CellCenters_OrthoNormalLegBasisCoeff(xleft, xright, N, degree, InitialCondition)
    
    plot_points = np.zeros((num_of_plot_pts, N))
    Approximation = np.zeros((num_of_plot_pts, N))
    for j in range (len(CellCenters)):
        plot_points[:,j] = CellCenters[j]*np.ones(num_of_plot_pts) + (dx/2)*zeta_plot
        Approximation[:,j] = BasisMatrix @ BasisCoefficients[j,:].T
    
    plot_points_array = plot_points.T.reshape(num_of_plot_pts*N)
    Approximation_array = Approximation.T.reshape(num_of_plot_pts*N)
    
    return plot_points_array, Approximation_array
    
    