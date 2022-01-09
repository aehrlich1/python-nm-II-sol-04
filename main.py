import numpy as np


# Question 02
def norm(v):
    return v / np.sqrt(np.sum(v**2))


def diff(v1, v2):
    return np.sqrt(np.sum((v1-v2)**2))


def pow_iteration(A, z):
    e = 10e-5
    while True:
        z_k = A @ z
        # z_k = norm(z_k)
        z_k = z_k / np.sum(z_k)

        if(diff(z_k, z) < e):
            return z_k
        else:
            z = z_k


A = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
z = 1/np.sqrt(3)*np.array([1, 1, 1], dtype=np.float64)

# pow_iteration(A, z)


# Question 03
def page_rank(A, p):
    n = len(A)
    v = np.ones(n, dtype=np.float64) / n  # rank vector
    M = (1 - p) * A + p * np.ones((n, n)) / n
    return pow_iteration(M, v)


A = np.array([[0, 1, 0], [0.5, 0, 1], [0.5, 0, 0]], dtype=np.float64)
p = 0.15
#print(page_rank(A, p))


# Question 04
def qr_algorithm(A):
    """
    Compute thr qr algorithm of the given matrix

    Parameters
    ----------
    A : matrix
        A matrix array

    Returns
    -------
    q : 


    """
    e = 1e-5
    i = 0

    while True:
        q, r = np.linalg.qr(A)
        A_k = r @ q

        delta_norm = np.abs(np.linalg.norm(A) - np.linalg.norm(A_k))

        if(delta_norm < e):
            return A_k, q
        else:
            A = A_k
            i += 1

        if(i > 1e4):
            print("No convergence")
            break


def sym(n):
    """
    Generate a symmetric matrix of dimension n x n

    Parameters
    ----------
    n : int
        A matrix array

    """
    A = np.random.rand(n, n)
    A = (A + A.T) / 2

    return A


A, Q = qr_algorithm(sym(5))
#print(A)


# Question 05
A = np.random.rand(15, 30)
u, s, vh = np.linalg.svd(A)

sigma = np.zeros((15, 30))
euclidean = np.sqrt(np.sum(s[10:-1]))
s[10:-1] = 0                                # set the last 5 singular values to 0
np.fill_diagonal(sigma, s)                  # diagonal matrix with `s` on the main diagonal

A_10 = u @ sigma @ vh
frobenius = np.linalg.norm(A - A_10)


print("Question 5\n-------------------------------------------")
print(f'{"Frobenius":20} {"==>":15} {frobenius:.3f}')
print(f'{"Euclidean":20} {"==>":15} {euclidean:.3f}')