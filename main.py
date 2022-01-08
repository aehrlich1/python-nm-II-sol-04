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
    M = (1 - p) * A + p * np.ones((n,n)) / n
    return pow_iteration(M, v)


A = np.array([[0, 1, 0], [0.5, 0, 1], [0.5, 0, 0]], dtype=np.float64)
p = 0.15
print(page_rank(A, p))
