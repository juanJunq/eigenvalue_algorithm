import numpy as np
np.set_printoptions(suppress=True, precision=2)

def Householder_To_Hessenberg(A):
    m, n = A.shape
    H = np.copy(A)
    
    for k in range(m - 2):  # Fase 1: transformando A em Hessenberg
        x = H[k + 1:m, k]  # Vetor abaixo da diagonal
        v = np.copy(x)
        v[0] += np.sign(x[0]) * np.linalg.norm(x)
        v = v / np.linalg.norm(v)
        
        I_k = np.eye(m - k - 1)  # Matriz identidade de dimensão (m-k-1)
        Hk = np.eye(m)  # Inicializa uma matriz identidade de tamanho m
        Hk[k + 1:m, k + 1:m] -= 2 * np.outer(v, v)  # Subtrai o refletor

        # Refletor de Householder
        H[k + 1:m, k:m] -= 2 * np.outer(v, np.dot(v.T, H[k + 1:m, k:m]))
        print("Aplicando refletor a esquerda da matriz A: ")
        print(H)
        print("\n")
        H[0:m, k + 1:m] -= 2 * np.outer(np.dot(H[0:m, k + 1:m], v), v.T)
        
        

        print(f"Refletor vk em {k+1}:")
        print(v)
        print(f"Matriz refletora Hk em {k+1}:")
        print(Hk)
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
