import numpy as np
import sympy as sp




lam = sp.symbols('lam')

m = [[3,0], [4, 5]]
print(f"Matrix M: \n{m}")
#numpy.transpose(m) this is to transpose the matrix

#np.matmul(A,B)

m_t = np.transpose(m)

print(f" Transpose of the matrix M: \n{m_t}")

m_tm = np.matmul(m_t, m)
size = len(m_tm)
print(f"Matrix M^TM:\n {m_tm}")

#for determinant
#np.linalg.det(A)

#create an identity matrix
#np.identity(size)

#I = np.identity(size)
#lam = sp.symbols('lam')
#char_matrix = sp.Matrix(m_tm - lam * I)
#char_poly = char_matrix.det()
#print(char_poly)

evals, evecs = np.linalg.eig(m_tm)

#Eigenvectors are already normalized
V_t = np.transpose(evecs)

print(f"Transposed Matrix of orthonormal eigenvectors V^T:\n {V_t}")
print (f"Eigenvalues of the matrix M^TM:\n{evals}")

sqrt_evals = []
for i in evals:
    if i > 0 :
        sqrt_evals.append(float(np.sqrt(i)))

print(f"Root square of eigenvalues:\n{sqrt_evals}")

sigma = np.diag(sqrt_evals)

print(f"Diagonal matrix Sigma: \n{sigma}")

U= []
counter = 0
for i in V_t[:len(sqrt_evals)] :

    k = np.matmul(m,i)
    U.append(k/float(sqrt_evals[counter]))
    counter += 1
print(f"U Matrix:\n{U}")
# Convert the list of arrays into a single 2D matrix
U_matrix = np.column_stack(U)

print(f"U Matrix:\n{U_matrix}")

print(f"Back to the original matrix M \n {np.matmul(np.matmul(U_matrix,sigma),V_t)}")



