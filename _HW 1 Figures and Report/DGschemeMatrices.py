import numpy as np
import sympy as sym

## Calculating Matrices for Weak Variational Form for u_t+au_x=0
p=5
xi = sym.symbols('xi')
M = np.zeros((p+1, p+1))
D = np.zeros((p+1,p+1))
S1 = np.zeros((p+1,p+1))
S2 = np.zeros((p+1,p+1))
for i in range(p+1): 
    for j in range(p+1): 
        M[i,j] = sym.integrate(sym.legendre(j, xi)*sym.legendre(i, xi), (xi, -1, 1))
        D[i,j] = sym.integrate(sym.legendre(j, xi)*sym.diff(sym.legendre(i, xi), xi), (xi, -1, 1))
        S1[i,j] = sym.legendre(j, 1)*sym.legendre(i, 1)
        S2[i,j] = sym.legendre(j, 1)*sym.legendre(i, -1)

M_inverse = np.linalg.inv(M)
print("M, M^{-1}, D-S1, S2 : ")
print(M, M_inverse)
print(D-S1, S2)



