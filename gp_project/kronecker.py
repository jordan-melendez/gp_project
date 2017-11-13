from functools import reduce
import numpy as np
from scipy.stats import multivariate_normal


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
    if len(m.shape) == 1:
        m = m[:, None]  # Treat 1D array as Nx1 matrix
    return np.concatenate([kron_mvprod(kron_list, b) for b in m.T], axis=1)


def flattened_outer(a, b):
    return np.outer(a, b).ravel()


def kron_diag(diags):
    """Returns diagonal of kronecker product from list of diagonals.
    """
    return reduce(flattened_outer, diags)


#################################################################
# Statistical classes for use in GP regression. Based on PyMC3's
# GP implementation and Yunus Saatci's Thesis mentioned above
#################################################################


def gaussian_kernel(x, xp, ell):
    return np.exp(-np.subtract.outer(x, xp)**2/ell**2)


class KroneckerNormal:
    """A multivariate normal that makes use of Kronecker structure of covariance.

    Parameters
    ----------
    mu   : array-like
    covs : list of arrays
    noise: float
    """

    def __init__(self, mu, covs, noise):
        # K + noise = Q.(L + noise*I).Q^T
        Lambdas, self.Qs = zip(*map(np.linalg.eigh, covs))  # Unzip tuples
        self.QTs = tuple(map(np.transpose, self.Qs))
        self.eig_noise = kron_diag(Lambdas) + noise
        self.N = len(self.eig_noise)
        self.mu = mu

    def random(self, size=None):
        """Drawn using x = mu + A.z for z~N(0,I) and A=Q.sqrt(Lambda)

        Warning: This does not (yet) match with random draws from numpy
        since A is only defined up to some unknown orthogonal transformation.
        Numpy used svd while we must use eigendecomposition, which aren't
        easily related due to sign ambiguities and permutations of eigenvalues.
        """
        if size is None:
            size = [self.N]
        elif isinstance(size, int):
            size = [size, self.N]
        else:
            raise NotImplementedError

        z = np.random.standard_normal(size)
        sqrtLz = np.sqrt(self.eig_noise) * z
        Az = kron_mmprod(self.Qs, sqrtLz.T).T
        return self.mu + Az

    def quaddist(self, value):
        """The quadratic (x-mu)^T @ K^-1 @ (x-mu)"""
        alpha = kron_mmprod(self.QTs, (value-self.mu).T)
        alpha = alpha/self.eig_noise[:, None]
        alpha = kron_mmprod(self.Qs, alpha)
        quad = np.dot(value-self.mu, alpha)
        quad = np.diag(quad)  # Remove correlations between samples
        return quad

    def logp(self, value):
        quad = self.quaddist(value)
        logdet = np.sum(np.log(self.eig_noise))
        return -1/2 * (quad + logdet + self.N*np.log(2*np.pi))

    def update(self):
        # How will updates to hyperparameters be performed?
        raise NotImplementedError


class MarginalKron:
    """
    """

    def __init__(self, mean_func, cov_func):
        raise NotImplementedError

    def _build_marginal_likelihood(self, X, noise):
        raise NotImplementedError

    def marginal_likelihood(self, X, y, noise, is_observed=True, **kwargs):
        """
        Returns the marginal likelihood distribution, given the input
        locations `X` and the data `y`.
        """
        raise NotImplementedError

    def _build_conditional(self, Xnew, pred_noise, diag, X, y, noise,
                           cov_total, mean_total):
        raise NotImplementedError

    def conditional(self, name, Xnew, pred_noise=False, given=None, **kwargs):
        """
        Returns the conditional distribution evaluated over new input
        locations `Xnew`.
        """
        raise NotImplementedError
