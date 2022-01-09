import numpy as np

# Question 01
def symmetric_matrix(n):
    """
    Generate a symmetric square matrix of dimension n x n.

    Parameters
    ----------
    n : int
        Size of the square matrix.
    
    Returns
    -------
    A : 2darray
        Matrix of dimension n x n with random values in (0, 1).

    """
    A = np.random.rand(n, n)
    A = (A + A.T) / 2

    return A


def norm(v):
    """
    Return the normalized vector of the provided input vector

    Parameters
    ----------
    v : array
        1D vector to be normalized
    
    Returns
    -------
    n : array
        Return the normalized vector (euclidean length = 1)

    """
    n = v / np.sqrt(np.sum(v**2))
    return n


def power_iteration(A, z):
    e = 1e-9
    i = 0

    while True:
        z_k = A @ z
        z_k = norm(z_k)
        # z_k = z_k / np.sum(z_k)
        delta = np.linalg.norm(z - z_k)

        if(delta < e):
            l = (z_k.T @ A @ z_k) / (z_k.T @ z_k)
            return z_k, l
        else:
            z = z_k
            i += 1

        if(i > 1e4):
            print("No convergence")
            break


def question_01():
    A = symmetric_matrix(10)
    z = norm(np.ones(10))
    v, l = power_iteration(A, z)
    rho = np.linalg.norm(A, ord=2)

    print("\nQuestion 1\n-------------------------------------------")
    print(f'{"Power iteration":20} {"==>":15} {l}')
    print(f'{"Solution":20} {"==>":15} {rho}')
    print(f'{"Difference":20} {"==>":15} {np.abs(rho - l)}')


# Question 02
def question_02():
    A = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
    z = 1/np.sqrt(3)*np.array([1, 1, 1], dtype=np.float64)

    print("\nQuestion 2\n-------------------------------------------")
    # pow_iteration(A, z)


# Question 03
def page_rank(A, p):
    n = len(A)
    v = np.ones(n, dtype=np.float64) / n  # rank vector
    M = (1 - p) * A + p * np.ones((n, n)) / n
    return power_iteration(M, v)


def question_03():
    A = np.array([[0, 1, 0], [0.5, 0, 1], [0.5, 0, 0]], dtype=np.float64)
    p = 0.15

    print("\nQuestion 3\n-------------------------------------------")
    #print(page_rank(A, p))


# Question 04
def qr_algorithm(A):
    """
    Compute thr qr algorithm of the given matrix. Within each
    iteration it performs a qr decomposition using np.linalg.qr
    and recombines the next iterate in reverse order (r @ q).
    The convergence criterion is calculated as the absolute sum
    of the diagonal elements of the matrix A. If the difference
    of this value between each iterate is below a threshold value
    we have achieved convergence.

    Parameters
    ----------
    A : 2darray
        A matrix array

    Returns
    -------
    A : 2darray
        2-Dimensional matrix with eigenvalues in the diagonal
    
    q : 2darray
        2-Dimensional matrix with eigenvectors in each column

    """
    e = 1e-9
    i = 0

    while True:
        q, r = np.linalg.qr(A)
        A_k = r @ q

        delta = np.sum(np.abs(np.diag(A) - np.diag(A_k)))

        if(delta < e):
            return A_k, q
        else:
            A = A_k
            i += 1

        if(i > 1e4):
            print("No convergence")
            break


def question_04():
    R = symmetric_matrix(5)
    A, Q = qr_algorithm(R)
    Eig = np.sort(np.diag(A))
    eig, eigv = np.linalg.eig(R)
    eig = np.sort(eig)
    diff = np.linalg.norm(Eig - eig)

    print("\nQuestion 4\n-------------------------------------------")
    print(f'{"qr algorithm":20} {"==>":15} {Eig}')
    print(f'{"numpy.eig":20} {"==>":15} {eig}')
    print(f'{"Difference":20} {"==>":15} {diff}')


# Question 05
def question_05():
    A = np.random.rand(15, 30)
    u, s, vh = np.linalg.svd(A)

    sigma = np.zeros((15, 30))
    euclidean = np.sqrt(np.sum(s[10:-1]))
    s[10:-1] = 0                                # Set the last 5 singular values to 0
    np.fill_diagonal(sigma, s)                  # Diagonal matrix with `s` on the main diagonal

    A_10 = u @ sigma @ vh
    frobenius = np.linalg.norm(A - A_10)


    print("\nQuestion 5\n-------------------------------------------")
    print(f'{"Frobenius":20} {"==>":15} {frobenius:.3f}')
    print(f'{"Euclidean":20} {"==>":15} {euclidean:.3f}')


# Main
question_01()
question_02()
question_03()
question_04()
question_05()