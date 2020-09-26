#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 17:53:09 2020

@author: sorayaterrab
"""

import numpy as np

from InitialConditions import *
from MultiWavelet_eval import *
from OrthoLegendreBasis_Projections import *


# Multiwavelet Coefficients and Approximations 
def MultiWavelet_Coefficients(xleft, xright, N_coarse, degree, InitialCondition):
    N_fine = 2*N_coarse  
    dx, CellCenters_fine, BasisCoeff_fine = CellCenters_OrthoNormalLegBasisCoeff(xleft, xright, N_fine, degree, InitialCondition)
    
    # Guass-Legendre Quadrature: Evaluation
    num_of_eval_pts= degree+1
    xi_eval, weights_eval = np.polynomial.legendre.leggauss(num_of_eval_pts)
    
    #Initializing
    integrand_left = np.zeros((degree+1,degree+1))
    integrand_right = np.zeros((degree+1,degree+1))
    PhidotPsi_left = np.zeros(degree+1)
    PhidotPsi_right = np.zeros(degree+1)
    
    MultiWavelet_coeff = np.zeros((N_coarse,degree+1))     
    for i in range(N_coarse):
        # Local Scaling Coordinate in coarse grid:
        xi_eval_MW_left = 0.5*(xi_eval - np.ones(degree+1))
        xi_eval_MW_right = 0.5*(xi_eval + np.ones(degree+1))
 
        for h in range (degree+1): 
            for m in range(degree+1):
                integrand_left[:,m] = orthonormalLegendrePoly(m,xi_eval)[:,-1]*np.transpose(MultiWavelet_eval(degree, h, xi_eval_MW_left))
                PhidotPsi_left[m] = BasisCoeff_fine[2*i,m]*sum(np.multiply(integrand_left[:,m],weights_eval))
                
                integrand_right[:,m] = orthonormalLegendrePoly(m,xi_eval)[:,-1]*np.transpose(MultiWavelet_eval(degree,h, xi_eval_MW_right))
                PhidotPsi_right[m] = BasisCoeff_fine[2*i+1,m]*sum(np.multiply(integrand_right[:,m],weights_eval))
            
            # Evaluation of MultiWavelet coefficient
            MultiWavelet_coeff[i,h]= 0.5*sum(PhidotPsi_left) + 0.5*sum(PhidotPsi_right) 

    return MultiWavelet_coeff


def MultiWavelet_Approximation(xleft, xright, N_coarse, degree, InitialCondition, num_of_plot_pts):
    dx, CellCenters_coarse, _ = CellCenters_OrthoNormalLegBasisCoeff(xleft, xright, N_coarse, degree, InitialCondition)
    MultiWaveletCoeffs = MultiWavelet_Coefficients(xleft, xright, N_coarse, degree, InitialCondition)
    
    # Plot Points using Gauss-Legendre Nodes
    xi_plot, _ =  np.polynomial.legendre.leggauss(num_of_plot_pts)
        
    nodes_plot_coarse = np.zeros((N_coarse,num_of_plot_pts))   
    MultiWavelet_approx = np.zeros((N_coarse,num_of_plot_pts))
    MultiWavelet_basis = np.zeros((degree + 1, num_of_plot_pts ))
    
    for h in range (degree+1):
            MultiWavelet_basis[h,:] = MultiWavelet_eval(degree,h,xi_plot).T
            
    for i in range(N_coarse):
        # Plot points in coarse grid: 
        nodes_plot_coarse[i,:] = (dx/2)*xi_plot + CellCenters_coarse[i]*np.ones(num_of_plot_pts)
        MultiWavelet_approx[i,:] = MultiWaveletCoeffs[i,:] @ MultiWavelet_basis 
    
    nodes_plot_coarse = nodes_plot_coarse.reshape(N_coarse*num_of_plot_pts)
    MultiWavelet_approx = MultiWavelet_approx.reshape(N_coarse*num_of_plot_pts)
    
    return nodes_plot_coarse, MultiWavelet_approx

