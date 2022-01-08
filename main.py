import numpy as np


# Question 02
def norm(v):
    return v / np.sqrt(np.sum(v**2))


def diff(v1, v2):
    return np.sqrt(np.sum((v1-v2)**2))


def pow_iteration(A, z):
    e = 10e-3
    while True:
        z_k = A @ z
        z_k = norm(z_k)

        if(diff(z_k, z) < e):
            break
        else:
            print(diff(z_k, z))
            z = z_k


A = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
z = 1/np.sqrt(3)*np.array([1, 1, 1], dtype=np.float64)

pow_iteration(A, z)