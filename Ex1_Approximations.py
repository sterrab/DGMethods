#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 13:54:26 2020

@author: sorayaterrab
"""
import numpy as np
import matplotlib.pyplot as plt

from InitialConditions import *
from MultiWavelet_eval import *
from OrthoLegendreBasis_Projections import *

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
Nvals = np.array([8, 16])
plot_pts = 6

##### PLOTTING APPROXIMATION and Error plot for N = 16 
for i in range (len(Nvals)): 
    N = Nvals[i]
    plt.figure(i+1)         
    for p in range(len(degree)): 
        x, u_h = OrthoLegendreBasisApproximation(xl, xr, N, p, IC, plot_pts)
        exact = Exact(x)
        plt.subplot(2,2,p+1) 
        plt.plot(x, u_h, '-ob')
        plt.plot(x, exact, 'k--', label='Exact')
        plt.title('N = '  +str(N) + ', p = ' +str(p))
        plt.xlabel('x')
        plt.ylabel('u_h(x)')
        plt.tight_layout()
        if N == 16 and p > 0: 
            error = np.absolute(u_h - exact)
            plt.figure(i+2)
            plt.subplot(3,1,p)
            plt.semilogy(x, error, linewidth=2)
            plt.title('N = '  +str(N) + ', p = ' +str(p))
            plt.xlabel('x')
            plt.ylabel('Abs. Error ')
            plt.tight_layout()
            fig=plt.gcf()
            fig.set_size_inches(8, 8)
            #fig.savefig("Error_IC" + str(InitCondition_ID) + "_N" +str(N)+ "p1-3.pdf")
  
    fig = plt.figure(i+1)
    fig.set_size_inches(8, 8)
    #fig.savefig("Approximation_IC" + str(InitCondition_ID) + "_N" +str(N)+ "p0-3.pdf")

##### PLOTTING Log-Log ERROR and CALCULATING ORDER
p = 3
Nvals = np.array([8, 16, 32, 64]) 
error_l2 = np.zeros(len(Nvals))
error_linf = np.zeros(len(Nvals))
order_l2 = np.zeros(len(Nvals)-1)
order_linf = np.zeros(len(Nvals)-1)
for i in range(len(Nvals)):
    N = Nvals[i]
    x, u_h = OrthoLegendreBasisApproximation(xl, xr, N, p, IC, plot_pts)
    exact = Exact(x)
    error = np.absolute(u_h - exact)
    error_l2[i] = np.linalg.norm(error, ord=2) 
    error_linf[i] = np.linalg.norm(error, ord=np.inf)
    if i>0:
        order_l2[i-1] = np.log(error_l2[i-1]/error_l2[i])/np.log(Nvals[i]/Nvals[i-1])
        order_linf[i-1] = np.log(error_linf[i-1]/error_linf[i])/np.log(Nvals[i]/Nvals[i-1])
        
plt.figure()
plt.loglog(Nvals, error_l2, '-v', label="$L^2$ Error")
plt.loglog(Nvals, error_linf, '-s', label="$L^\inf$ Error")
plt.loglog(Nvals, 1/(Nvals**3), 'k--', label="Order 3 Ref")
plt.loglog(Nvals, 1/(Nvals**4), 'k-.', label="Order 4 Ref")
plt.legend()
plt.xlabel('N')
plt.ylabel('Norm of Absolute Error')
fig = plt.gcf()
fig.set_size_inches(6, 6)
#fig.savefig("LogLogError_IC" + str(InitCondition_ID) +".pdf")


print("\n The L2-Order values are: ")
print(order_l2)
print("\n The Linf-Order values are: ")
print(order_linf)
    
    