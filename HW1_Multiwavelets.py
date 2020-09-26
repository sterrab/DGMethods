#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 18:34:09 2020

@author: sorayaterrab
"""
import numpy as np
import matplotlib.pyplot as plt

from InitialConditions import *
from MultiWavelet_eval import *
from OrthoLegendreBasis_Projections import *
from MultiWaveletBasis_Projections import *

# Initial Conditions - Exact Solution 
InitCondition_ID = input('Select Initial Condition (1,2,3):  ');
InitCondition_ID = int(InitCondition_ID )
if InitCondition_ID == 1: 
    IC = lambda x: InitCondition_1(x)
elif InitCondition_ID == 2: 
    IC = lambda x: InitCondition_2(x)
elif InitCondition_ID == 3: 
    IC = lambda x: InitCondition_3(x)
else: 
    print('Pick Initial Condition 1, 2, or 3. ')    
Exact = lambda x: IC(x)

# Parameters
xl = -1
xr = 1
degree = np.array([0, 1, 2, 3])
N = 16
plot_pts = 6

### Plotting Multiwavelets by Projection Method and Actual Approximation as Reference
plt.figure(1)
for p in range(len(degree)): 
    # Plotting Approximation as Reference on left panels
    x, u_h = OrthoNormalLegendreBasisApproximation(xl, xr, N, p, IC, plot_pts)
    exact = Exact(x)
    plt.subplot(4,2,2*p+1) 
    plt.plot(x, u_h, '-o')
    plt.plot(x, exact, 'k--', label='Exact')
    plt.title('Approximation, $N = $' + str(N) + ', $p = $' +str(p))
    plt.xlabel('$x$')
    plt.ylabel('$u_h(x)$')
    
    # Plotting Multiwavelet projection on right panels
    x_coarse, MWprojection = MultiWavelet_Approximation(xl, xr, int(N/2), p, IC, plot_pts)
    plt.subplot(4,2,2*p+2) 
    plt.plot(x_coarse, MWprojection, '-v')
    plt.title('MultiWavelets by Projection, $N = $' + str(N) + ', $p = $' +str(p))
    plt.xlabel('$x$')
    plt.ylabel('$MW(x)$')

fig = plt.gcf()
fig.set_size_inches(11, 8.5)
fig.tight_layout()
#fig.savefig("MWProjection_IC" + str(InitCondition_ID)+ '.pdf')


### Plotting Multiwavelets by Difference Method with Actual Approximations as Reference
zeta_plot , _ =  np.polynomial.legendre.leggauss(plot_pts)
zeta_plot_c2fneg = 2*zeta_plot[0:int(plot_pts/2)] + np.ones(int(plot_pts/2))
zeta_plot_c2fpos = 2*zeta_plot[int(plot_pts/2):plot_pts] - np.ones(int(plot_pts/2))
zeta_plot_fine = np.zeros(plot_pts)
plt.figure()
for p in range(len(degree)): 
    # Approximations in Coarse Mesh
    x_coarse, u_h_coarse = OrthoNormalLegendreBasisApproximation(xl, xr, int(N/2), p, IC, plot_pts)
    exact = Exact(x_coarse)
    
    # Finding Fine Approximation in order to maintain same Node Evaluation Points across Both Resolutions
    _, _, BasisCoeffFine = CellCenters_OrthoNormalLegBasisCoeff(xl, xr, N, p, IC)
    BasisMatrixNeg = orthonormalLegendrePoly(p,zeta_plot_c2fneg)
    BasisMatrixPos = orthonormalLegendrePoly(p,zeta_plot_c2fpos)
    u_h_fine = np.zeros((int(N/2),plot_pts))
    
    # Finding Fine Approximation
    for i in range(int(N/2)):
        u_h_fine[i,0:int(plot_pts/2)] = BasisMatrixNeg @ BasisCoeffFine[2*i,:].T
        u_h_fine[i,int(plot_pts/2):plot_pts] = BasisMatrixPos @ BasisCoeffFine[2*i+1,:].T 
    
    u_h_fine = u_h_fine.reshape(int(N/2)*plot_pts)
    
    plt.figure(2)
    plt.subplot(4,2,2*p+1) 
    plt.plot(x_coarse, u_h_coarse, '-s', label='N=8')
    plt.plot(x_coarse, u_h_fine, '-o', label='N=16')
    plt.plot(x_coarse, exact, 'k--', label='Exact')
    plt.title('Approximation, $p = $'+ str(p))
    plt.xlabel('$x$')
    plt.ylabel('$u_h(x)$')
    plt.legend()
    
    # Plotting Multiwavelet projection on right panels
    MW_diff = u_h_fine - u_h_coarse
    plt.figure(2)
    plt.subplot(4,2,2*p+2) 
    plt.plot(x_coarse, MW_diff, '-v')
    plt.title('MultiWavelets by Difference, $p = $'+ str(p))
    plt.xlabel('$x$')
    plt.ylabel('$u_{h16}(x)-u_{h8}(x)$')
    
fig = plt.gcf()
fig.set_size_inches(11, 8.5)
fig.tight_layout()
#fig.savefig("MWDifference_IC" + str(InitCondition_ID)+ '.pdf')


# Plotting Both Methods
plt.figure()
for p in range(len(degree)): 
    # Difference Method
    x_coarse, u_h_coarse = OrthoNormalLegendreBasisApproximation(xl, xr, int(N/2), p, IC, plot_pts)
    exact = Exact(x_coarse)
    
    # Finding Fine Approximation in order to maintain same Node Evaluation Points across Both Resolutions
    _, _, BasisCoeffFine = CellCenters_OrthoNormalLegBasisCoeff(xl, xr, N, p, IC)
    BasisMatrixNeg = orthonormalLegendrePoly(p,zeta_plot_c2fneg)
    BasisMatrixPos = orthonormalLegendrePoly(p,zeta_plot_c2fpos)
    u_h_fine = np.zeros((int(N/2),plot_pts))
    
    for i in range(int(N/2)):
        u_h_fine[i,0:int(plot_pts/2)] = BasisMatrixNeg @ BasisCoeffFine[2*i,:].T
        u_h_fine[i,int(plot_pts/2):plot_pts] = BasisMatrixPos @ BasisCoeffFine[2*i+1,:].T      
    u_h_fine = u_h_fine.reshape(int(N/2)*plot_pts)

    MW_diff = u_h_fine - u_h_coarse
    
    #Projections Method
    x_coarse, MWprojection = MultiWavelet_Approximation(xl, xr, int(N/2), p, IC, plot_pts)
    
    #Plotting
    plt.subplot(4,2,2*p+1)
    plt.plot(x_coarse, MWprojection, '-v', label='Projection')
    plt.plot(x_coarse, MW_diff, '-v', label='Difference')
    plt.title('MultiWavelet Approximations, $p = $' +str(p))
    plt.xlabel('$x$')
    plt.ylabel('$MW(x)$')
    plt.legend()
    
    plt.subplot(4,2,2*p+2) 
    plt.plot(x_coarse, np.absolute(MWprojection-MW_diff), '-')
    plt.title('Error between Methods, $p = $' +str(p))
    plt.xlabel('$x$')
    plt.ylabel('$|MW_{Proj.}(x) - MW_{Diff}(x)|$')
    
    
fig = plt.gcf()
fig.set_size_inches(11, 8.5)
fig.tight_layout()
#fig.savefig("MWMethodComparison_IC" + str(InitCondition_ID) +'.pdf')
