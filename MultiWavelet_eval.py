#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 14:59:10 2020

@author: sorayaterrab
"""

import numpy as np

## Defining Alpert's Multiwavelet Functions \psi_m(x) on x in [-1,1] for up to p = 3. 
#

def MultiWavelet_eval(p,m,input):
    n = len(input)
    psi_m = np.zeros((n,1))
    if max(abs(input)) > 1: 
        print("Error: Input value outside [-1,1].")
        return
    else: 
        if p == 0: 
            for j in range(n):
                if input[j]>0:
                    psi_m[j] = np.sqrt(1/2)
                else:
                    psi_m[j] = - np.sqrt(1/2)
                    
        elif p == 1: 
            for j in range(n):
                if input[j]>0:
                    if m == 0:
                        psi_m[j] = np.sqrt(3/2)* (2*input[j] -1)
                    elif m == 1: 
                        psi_m[j] = np.sqrt(1/2)* (3*input[j] -2)
                else:
                    if m == 0:
                        psi_m[j] = -np.sqrt(3/2)* (2*input[j] +1)
                    elif m == 1: 
                        psi_m[j] = np.sqrt(1/2)* (3*input[j] +2)
        
        elif p == 2: 
            for j in range(n):
                if input[j]>0:
                    if m == 0:
                        psi_m[j] = (1/3) * np.sqrt(1/2) * (30*input[j]**2 - 24*input[j] + 1) 
                    elif m == 1: 
                        psi_m[j] = (1/2) * np.sqrt(3/2) * (15*input[j]**2 - 16*input[j] + 3) 
                    elif m == 2: 
                        psi_m[j] = (1/3) * np.sqrt(5/2) * (12*input[j]**2 - 15*input[j] + 4)      
                else:
                    if m == 0:
                        psi_m[j] = -(1/3) * np.sqrt(1/2) * (30*input[j]**2 + 24*input[j] + 1) 
                    elif m == 1: 
                        psi_m[j] = (1/2) * np.sqrt(3/2) * (15*input[j]**2 + 16*input[j] + 3) 
                    elif m == 2: 
                        psi_m[j] = -(1/3) * np.sqrt(5/2) * (12*input[j]**2 + 15*input[j] + 4)
                        
        elif p == 3: 
            for j in range(n):
                if input[j]>0:
                    if m == 0:
                        psi_m[j] = np.sqrt(15/34) * (28*input[j]**3-30*input[j]**2 + 4*input[j] + 1)
                    elif m == 1: 
                        psi_m[j] = np.sqrt( 1/42) * (210*input[j]**3-300*input[j]**2 + 105*input[j] - 4)
                    elif m == 2: 
                        psi_m[j] = (1/2)*np.sqrt(35/34) * (64*input[j]**3-105*input[j]**2 + 48*input[j] - 5)
                    elif m ==3:
                        psi_m[j] = (1/2)*np.sqrt( 5/42) * (105*input[j]**3-192*input[j]**2 + 105*input[j] - 16)
                else:
                    if m == 0:
                        psi_m[j] = np.sqrt(15/34) * (-28*input[j]**3-30*input[j]**2 - 4*input[j] + 1)
                    elif m == 1: 
                        psi_m[j] = np.sqrt( 1/42) * (210*input[j]**3+300*input[j]**2 + 105*input[j] + 4)
                    elif m == 2: 
                        psi_m[j] = -(1/2)*np.sqrt(35/34) * (64*input[j]**3+105*input[j]**2 + 48*input[j] + 5)
                    elif m ==3:
                        psi_m[j] = (1/2)*np.sqrt( 5/42) * (105*input[j]**3+192*input[j]**2 + 105*input[j] + 16)
        
        else:
            print("Error: Function currently allows for p <= 3 only.")
            return
        
    return psi_m
        
                      
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        