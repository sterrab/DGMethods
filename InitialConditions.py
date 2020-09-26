#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 09:43:40 2020

@author: sorayaterrab
"""

import numpy as np
import scipy as sci
import matplotlib.pyplot as plt

import math

def InitCondition_1(input): 
    output = np.sin(2*np.pi * input)
    return output

def InitCondition_2(input): 
    output = np.zeros(len(input))
    for i in range(len(input)):
        if input[i] < -1 or input[i] > 1:
            print("Error: Input outside Domain.")
            break
        elif input[i] >=-0.5 and input[i] <= 0.5: 
            output[i] = 1
    return output


# Functions needed for Initial Condition 3

def F_function(input, alpha, a):
    output = 1-alpha**2*(input-a)**2
    if output > 0: 
        output = np.sqrt(output)
    else:
        output = 0
    return output

def G_function(input, beta, center):
    output = np.exp(-beta*(input-center)**2)
    return output

# Parameters for Initial Condition 3
c = 0.5 
z = -0.7 
delta = 0.005 
alpha = 10
beta = np.log(2)/(36*delta**2)

def InitCondition_3(input):
    output = np.zeros(len(input))
    for i in range(len(input)): 
        if input[i] >= -0.8 and input[i] <= -0.6: 
            output[i]=(1/6)*(G_function(input[i], beta, z-delta) + G_function(input[i], beta, z+delta)) + (2/3)* G_function(input[i], beta, z)
        elif input[i] >= -0.4 and input[i] <= -0.2:
            output[i] = 1
        elif input[i] >= 0 and input[i] <= 0.2:
            output[i] = 1 - abs(10*(input[i]-0.1))
        elif input[i] >= 0.4 and input[i] <= 0.6:
            output[i] = (1/6)*(F_function(input[i], alpha, c-delta)+F_function(input[i], alpha, c+delta)+4*F_function(input[i], alpha, c))
    return output
    
    
