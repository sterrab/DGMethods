#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 11:52:45 2020

@author: sorayaterrab
"""
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

from InitialConditions import *
from MultiWavelet_eval import *
from OrthoLegendreBasis_Projections import *


#### Initial Condition and Exact Solution
IC = lambda x: np.sin(np.pi*x)
Exact = lambda x: IC(x)

#### Problem Parameters
a = input('Select wavespeed "a" for linear transport model u_t+au_x=0:');
N = input('Select "N", the number of elements in grid:');
p = input('Select polynomial degree "p" (not order):');
a = int(a)
N = int(N)
p = int(p)
T_final = 10

#### Grid Parameters
xl = -1
xr = 1
DeltaX = (xr-xl)/N
num_of_plot_pts = 6

#### CFL and TimeStep
# CFL condition for p=[0 1 2 3 4 5]
Lambda = [1.256, 0.409, 0.209, 0.13, 0.089, 0.066]

DeltaT = Lambda[p]/np.absolute(a)*DeltaX

##### Mass Matrix M and Matrices D, S1, S2 for the given degree p
def DiscreteVariationalFormMatrices(degree): 
    xi = sym.symbols('xi')
    M = np.zeros((degree+1, degree+1))
    D = np.zeros((degree+1,degree+1))
    S1 = np.zeros((degree+1,degree+1))
    S2 = np.zeros((degree+1,degree+1))
    for i in range(degree+1): 
        for j in range(degree+1): 
            M[i,j] = sym.integrate(sym.legendre(j, xi)*sym.legendre(i, xi), (xi, -1, 1))
            D[i,j] = sym.integrate(sym.legendre(j, xi)*sym.diff(sym.legendre(i, xi), xi), (xi, -1, 1))
            S1[i,j] = sym.legendre(j, 1)*sym.legendre(i, 1)
            S2[i,j] = sym.legendre(j, 1)*sym.legendre(i, -1)
    M_inverse = np.linalg.inv(M)
    return M_inverse, D-S1, S2

##### Strong Stability Preserving Runge-Kutta Scheme of Order 3 (SSPRK3)
def TimeEvolutionStep(degree, u_modes_at_j, u_modes_at_jminus1):
    Minv, DminusS1, S2 = DiscreteVariationalFormMatrices(p)
    L_operator = Minv @ (DminusS1 @ u_modes_at_j + S2 @ u_modes_at_jminus1)
    return L_operator

def SSPRK3(degree, CFLcondition, u_modes_at_j, u_modes_at_jminus1):
    N = u_modes_at_j.shape[1] -1 # given the modes array has N + 1 columns
    u1 = u_modes_at_j + 2* CFLcondition* TimeEvolutionStep(degree, u_modes_at_j, u_modes_at_jminus1)
    u1[:,0] = u1[:,N] 
    u1_minus = np.block([u1[:,N-1].reshape((p+1,1)), u1[:,0:N]]) 
    u2 = 1/4*(3*u_modes_at_j + u1 + 2*CFLcondition * TimeEvolutionStep(degree, u1, u1_minus))
    u2[:,0] = u2[:,N]
    u2_minus = np.block([u2[:,N-1].reshape((p+1,1)), u2[:,0:N]]) 
    u3 = 1/3*(u_modes_at_j + 2*u2 + 4*CFLcondition * TimeEvolutionStep(degree, u2, u2_minus))
    return u3


#### Evolving Modes    
_, CellCenters, Umodes = CellCenters_LegBasisCoeff(xl, xr, N, p, IC)
Umodes = Umodes.T
# Creating Modes matrix with each column representing an element j with ghost nodes added
Umodes_at_j = np.block([Umodes, Umodes[:,0].reshape((p+1,1))])
Umodes_at_jminus1 = np.block([Umodes[:,N-1].reshape((p+1,1)), Umodes])

time = 0
while time+DeltaT < T_final: 
    New_modes = SSPRK3(p, Lambda[p], Umodes_at_j, Umodes_at_jminus1)
    Umodes_at_j = New_modes
    Umodes_at_j[:,0] = New_modes[:,N]
    Umodes_at_jminus1 = np.block([Umodes_at_j[:,N-1].reshape((p+1,1)), Umodes_at_j[:,0:N]])
    time = time + DeltaT

Final_modes = Umodes_at_j[:,0:N]
   
##### Calculating/Plotting Approximation and Error
# Guass-Legendre Points for Plotting
zeta_plot, _ = np.polynomial.legendre.leggauss(num_of_plot_pts)    
# Basis Matrix with columns of m'th degree orthonormal Legendre Polynomial evaluated at zeta_plot
BasisMatrix = LegendrePoly(p,zeta_plot)

plot_points = np.zeros((num_of_plot_pts, N))
Approximation = np.zeros((num_of_plot_pts, N))
for j in range (len(CellCenters)):
    plot_points[:,j] = CellCenters[j]*np.ones(num_of_plot_pts) + (DeltaX/2)*zeta_plot

Approximation = BasisMatrix @ Final_modes

plot_points_array = plot_points.T.reshape(num_of_plot_pts*N)
Approximation_array = Approximation.T.reshape(num_of_plot_pts*N)

# Approximation Plot
Exact_solution = np.sin(np.pi*(plot_points_array-a*time*np.ones(num_of_plot_pts*N)))
plt.figure()
plt.plot(plot_points_array, Approximation_array, 'ob', label = 'N=' + str(N)+', p='+str(p))
plt.plot(plot_points_array, Exact_solution, '--.', color='gray', label='Exact')
plt.legend()
plt.xlabel('$x$')
plt.ylabel('$u_h(x,10)$')
plt.title('Approximation at $T_f=$' + str(T_final))
fig = plt.gcf()
#fig.savefig('Approximation_N'+str(N)+'_p'+str(p)+'.pdf')

# Error Plot
Error = np.absolute(Exact_solution - Approximation_array)
plt.figure()
plt.semilogy(plot_points_array, Error)
plt.xlabel('$x$')
plt.ylabel('$|u(x,10) - u_h(x,10)|$')
plt.title('Error for $N=$' + str(N) + ', $p=$'+ str(p) + ' at $T_f=$' + str(T_final))
fig = plt.gcf()
#fig.savefig('Error_N'+str(N)+'_p'+str(p)+'.pdf')

#### EXAMINING L2-Linf ERRORS and ORDERS
N_vals = [8, 16, 32, 64]

Error_L2 = np.zeros((len(N_vals), 3))
Error_Linf = np.zeros((len(N_vals), 3))
Order_L2 = np.zeros((len(N_vals)-1, 3))
Order_Linf = np.zeros((len(N_vals)-1, 3))
for p in range(1,4):
    print(p)
    for k in range(len(N_vals)):
        N = N_vals[k]
        DeltaX = (xr-xl)/N
        # Lambda/CFL values
        if p == 1 or k == 0: 
            CFL = Lambda[p]
        if p >= 2 and k > 0: 
            CFL = CFL* ((1/2)**(3/(p+1)))
        
        DeltaT = CFL/np.absolute(a)*DeltaX
                
        # Creating Modes matrix with each column representing an element j with ghost nodes added
        _, CellCenters, Umodes = CellCenters_LegBasisCoeff(xl, xr, N, p, IC)
        Umodes = Umodes.T
        Umodes_at_j = np.block([Umodes, Umodes[:,0].reshape((p+1,1))])
        Umodes_at_jminus1 = np.block([Umodes[:,N-1].reshape((p+1,1)), Umodes])
        
        time = 0
        while time+DeltaT < T_final: 
            New_modes = SSPRK3(p, CFL, Umodes_at_j, Umodes_at_jminus1)
            Umodes_at_j = New_modes
            Umodes_at_j[:,0] = New_modes[:,N]
            Umodes_at_jminus1 = np.block([Umodes_at_j[:,N-1].reshape((p+1,1)), Umodes_at_j[:,0:N]])
            time = time+DeltaT
        Final_modes = Umodes_at_j[:,0:N]
           
        ##### Calculating/Plotting Approximation and Error
        # Guass-Legendre Points for Plotting
        zeta_plot, _ = np.polynomial.legendre.leggauss(num_of_plot_pts)    
        # Basis Matrix with columns of m'th degree orthonormal Legendre Polynomial evaluated at zeta_plot
        BasisMatrix = LegendrePoly(p,zeta_plot)
        
        plot_points = np.zeros((num_of_plot_pts, N))
        Approximation = np.zeros((num_of_plot_pts, N))
        for j in range (len(CellCenters)):
            plot_points[:,j] = CellCenters[j]*np.ones(num_of_plot_pts) + (DeltaX/2)*zeta_plot
        
        Approximation = BasisMatrix @ Final_modes
        
        plot_points_array = plot_points.T.reshape(num_of_plot_pts*N)
        Approximation_array = Approximation.T.reshape(num_of_plot_pts*N)
        Exact_solution = np.sin(np.pi*(plot_points_array-a*time*np.ones(num_of_plot_pts*N)))
        
        # Error Calculation
        Error_abs = np.absolute(Exact_solution - Approximation_array)
        Error_L2[k,p-1] = np.linalg.norm(Error_abs, ord=2)
        Error_Linf[k,p-1] = np.linalg.norm(Error_abs, ord=np.inf)
        
        # Order Calculation
        if k > 0: 
            Order_L2[k-1,p-1]= np.log(Error_L2[k-1,p-1]/Error_L2[k,p-1])/np.log(2)
            Order_Linf[k-1,p-1]= np.log(Error_Linf[k-1,p-1]/Error_Linf[k,p-1])/np.log(2)

print('\n L2 Error:')
print(Error_L2)
print('\n Linf Error:')
print(Error_Linf)
print('\n Order based on L2 Error:')
print(Order_L2)
print('\n Order based on Linf Error:')
print(Order_Linf)















