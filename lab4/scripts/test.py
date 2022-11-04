import numpy as np
from sys import argv

def classical_gram_schmidt(A, x, n: int):
    """
    Computes a basis of the (n + 1)-Krylov subspace of A: the space
    spanned by {x, Ab, ..., A^n x}.

    Arguments
      A: m x m array
      x: initial vector (length m)
      n: dimension of Krylov subspace, must be >= 1
    
    Returns
      Q: m x (n + 1) array, the columns are an orthonormal basis of the
        Krylov subspace.
      H: (n + 1) x n array, A on basis Q. It is upper Hessenberg.  
    """

    eps = 1e-12
    H = np.zeros((n + 1, n))
    Q = np.zeros((A.shape[0], n + 1))
    # Normalize the input vector and use it as the first Krylov vector
    Q[:, 0] = x / np.linalg.norm(x, 2)

    for k in range(1, n + 1):
        # Generate a new candidate vector
        y = np.dot(A, Q[:, k - 1])

        # Subtract the projections on previous vectors
        for j in range(k):
            H[j, k - 1] = np.dot(Q[:, j].T, y)
        # Subtract the projections on previous vectors
        for j in range(k):
            y = y - H[j, k - 1] * Q[:, j]

        H[k, k - 1] = np.linalg.norm(y, 2)
         # Add the produced vector to the list, unless
        if H[k, k - 1] > eps:
            Q[:, k] = y / H[k, k - 1]
         # If that happens, stop iterating
        else:
            return Q, H

    return Q, H


def modified_gram_schmidt(A, x, n: int):
    """
    Computes a basis of the (n + 1)-Krylov subspace of A: the space
    spanned by {x, Ab, ..., A^n x}.

    Arguments
      A: m x m array
      x: initial vector (length m)
      n: dimension of Krylov subspace, must be >= 1

    Returns
      Q: m x (n + 1) array, the columns are an orthonormal basis of the
        Krylov subspace.
      H: (n + 1) x n array, A on basis Q. It is upper Hessenberg.  
    """

    eps = 1e-12
    H = np.zeros((n + 1, n))
    Q = np.zeros((A.shape[0], n + 1))
    # Normalize the input vector and use it as the first Krylov vector
    Q[:, 0] = x / np.linalg.norm(x, 2)

    for k in range(1, n + 1):
        # Generate a new candidate vector
        y = np.dot(A, Q[:, k - 1])

        # Subtract the projections on previous vectors
        for j in range(k):
            H[j, k - 1] = np.dot(Q[:, j].T, y)
            y = y - H[j, k - 1] * Q[:, j]

        H[k, k - 1] = np.linalg.norm(y, 2)

        # Add the produced vector to the list, unless
        if H[k, k - 1] > eps:
            Q[:, k] = y / H[k, k - 1]
        # If that happens, stop iterating
        else: 
            return Q, H

    return Q, H


def main():
    m = 10
    n = 10
    if len(argv) == 3:
        m = int(argv[1])
        n = int(argv[2])

    A = np.random.rand(m, m)
    x = np.zeros(m)
    x[0] = 1

    # print(f"m = {m} | n = {n}")
    # print("A :\n", A)
    # print("x :\n", x)

    CGS_Q, CGS_H = classical_gram_schmidt(A, x, n)
    print("CGS_Q: ")
    for row in CGS_Q:
        print(["%.3f" % x for x in row])
    print("CGS_H: ")
    for row in CGS_H:
        print(["%.3f" % x for x in row])

    # MGS_Q, MGS_H = modified_gram_schmidt(A, x, n)
    # print("MGS_Q = ")
    # for row in MGS_Q:
    #     print(["%.3f" % x for x in row])
    # print("MGS_H = ")
    # for row in MGS_H:
    #     print(["%.3f" % x for x in row])

    # print(f"\ndiff Q: {np.sum(CGS_Q - MGS_Q)}")
    # print(f"diff H: {np.sum(CGS_H - MGS_H)}")

if __name__ == '__main__':
    main()
