from functools import reduce
import numpy as np


#################################################################
# Based on functions defined in Yunus Saatci's Thesis (Ch. 5):
# http://mlg.eng.cam.ac.uk/pub/pdf/Saa11.pdf
#################################################################


def kronecker(K):
    """Return the Kronecker product of list of arrays K:
            K_1 \otimes K_2 \otimes ... \otimes K_D

    Parameters
    ----------
    K: List of array-like
       [K_1, K_2, ..., K_D]
    """
    return reduce(np.kron, K)


def flat_mtprod(tens, mat):
    """A matrix-tensor product
        Z_{i_1, ..., i_D} = \sum_k M_{i_1,k} T_{k, i_2, ..., i_D}
    where tens is the vectorized version of T.

    Parameters
    -----------
    mat : 2D array-like
    tens: (N,1)- or (N,)-shaped array-like

    Returns
    -------
    Z: column vector
       A (column) vectorized version of the matrix-tensor product
    """

    Nm = mat.shape[1]
    Tmat = tens.reshape((Nm, -1))
    Z = np.dot(mat, Tmat)
    return Z.T.reshape((-1, 1))


def kron_mvprod(kron_list, b):
    """Compute the matrix-vector product of kronecker(kron_list).b

    Parameters
    -----------
    kron_list: list of 2D array-like objects
               D matrices [A_1, A_2, ..., A_D] to be Kronecker'ed:
                    A = A_1 \otimes A_2 \otimes ... \otimes A_D
               Product of column dimensions must be N
    b        : array-like
               Nx1 column vector
    """
    # N = np.size(b)
    # print(b)

    return reduce(flat_mtprod, kron_list, b)


def kron_mmprod(kron_list, m):
    """Compute the matrix product of kronecker(kron_list).m

    Parameters
    -----------
    kron_list: list of 2D array-like objects
               D matrices [A_1, A_2, ..., A_D] to be Kronecker'ed:
                    A = A_1 \otimes A_2 \otimes ... \otimes A_D
               Product of column dimensions must be N
    m        : array-like
               NxM matrix
    """

    return np.concatenate([kron_mvprod(kron_list, b) for b in m.T], axis=1)
