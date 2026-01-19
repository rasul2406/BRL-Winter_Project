import numpy as np
import sympy as sp
from PIL import Image



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




def eckart(img_m, r):
    r = int(r)
    img_m = img_m.astype(float)
    M, N = img_m.shape


    m_t = np.transpose(img_m)
    m_tm = np.matmul(m_t, img_m)
    size = len(m_tm)


    #sorting
    evals, evecs = np.linalg.eigh(m_tm)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    V = evecs[:, idx]
    V_t = np.transpose(V)
    #Getting the U matrix
    evals_u, evecs_u = np.linalg.eigh(np.matmul(img_m, m_t))
    idx_u = np.argsort(evals_u)[::-1]
    U = evecs_u[:, idx_u]

    # Compression happens here
    s_sqrt = np.sqrt(np.maximum(evals, 0))

    sigma_full = np.zeros((M, N))
    min_dim = min(M, N)

    s_compressed = np.copy(s_sqrt)
    s_compressed[r:] = 0

    np.fill_diagonal(sigma_full[:min_dim, :min_dim], s_compressed[:min_dim])

    for i in range(min(M, N)):
        if s_sqrt[i] > 1e-10:
            # Predict what u_i should look like based on v_i
            prediction = np.dot(img_m, V[:, i])
            # If the prediction points opposite to our U column, flip the U column
            if np.dot(prediction, U[:, i]) < 0:
                U[:, i] = -U[:, i]

    reconstructed = np.matmul(np.matmul(U, sigma_full), V_t)


    reconstructed_clipped = np.clip(reconstructed, 0, 255).astype(np.uint8)
    img_out = Image.fromarray(reconstructed_clipped, mode='L')

    img_out.show()
    img_out.save('output_image.jpg')
    return img_out

img = Image.open('test_img.jpg').convert('L')
img_m = np.array(img)
r= input("Choose till which rank you want to approximate:")
eckart(img_m,int(r))