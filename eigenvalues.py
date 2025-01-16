import numpy as np
np.set_printoptions(suppress=True, precision=2)

def Householder_To_Hessenberg(A):
    m, n = A.shape
    H = np.copy(A)
    
    for k in range(m - 2):  # Fase 1: A -> Hessenberg superior ou tridiagonal
        x = H[k + 1:m, k]  # Vetor abaixo da diagonal
        v = np.copy(x)
        v[0] += np.sign(x[0]) * np.linalg.norm(x)
        v = v / np.linalg.norm(v)
        
        Hk = np.eye(m)  # Matriz identidade de dimensão (m)
        Hk[k + 1:m, k + 1:m] -= 2 * np.outer(v, v)  # Subtrai o refletor

        F = np.eye(m)
        F[k + 1:m, k + 1:m] = Hk[k + 1:m, k + 1:m]   # Refletor de Householder
        
        print("Aplicando refletor a esquerda da matriz A: ")
        H = F @ H
        print(H)
        print("\n")

        print("Aplicando refletor a direita da matriz A: ")
        H = H @ F.T
        print(H)
        print("\n")
        

        print(f"Refletor vk em {k+1}:")
        print(v)
        print("\n")

        print(f"Matriz refletora Q{k+1}:")
        print(F)
        print("\n")
        
        print(f"Matriz A{k+1}:")
        print(H)
        print("\n")
    return H

def QR_Para_Autovalores(A, tol=1e-6, max_iter=1000):
    m, n = A.shape
    H = Householder_To_Hessenberg(A)  # Fase 1
    k = 0
    Qa = np.eye(m)  # Matriz acumuladora Q
    
    while k < max_iter:
        Q, R = np.linalg.qr(H)  # Fase 2: Método QR
        Qa = Qa @ Q
        H = R @ Q
        
        # Critério de convergência: soma dos elementos fora da diagonal
        off_diagonal_sum = np.sum(np.abs(H - np.diag(np.diag(H))))  
        print(f"\tIter: {k+1} off-diagonal sum: {off_diagonal_sum:.6e}\t")
        
        print(f"\nMatriz A{k+1}:")
        print(H)
        print("\n")

        if off_diagonal_sum < tol:
            break
        
        k += 1
        
    lambda_ = np.diag(H)  # Autovalores estão na diagonal de H
    return Qa, lambda_, H

# Exemplo de uso
A = np.array([
    [85, 102, 70, 129, 137],
    [102, 167, 85, 157, 189],
    [70, 85, 110, 91, 151],
    [129, 157, 91, 272, 218],
    [137, 189, 151, 218, 267]
], dtype=float)

print("Matriz inicial A:")
print(A)
print("\n")

Qa, Lambda, H = QR_Para_Autovalores(A, tol=1e-6, max_iter=1000)
print("Autovalores:", Lambda)
