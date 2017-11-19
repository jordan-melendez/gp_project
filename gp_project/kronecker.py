from functools import reduce
from itertools import cycle, zip_longest
import numpy as np
import scipy as sp
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


def cartesian(*arrays):
    """Makes the Cartesian product of arrays.

    Parameters
    ----------
    arrays: list of 1D array-like
            1D arrays where earlier arrays loop more slowly than later ones
    """
    N = len(arrays)
    return np.stack(np.meshgrid(*arrays, indexing='ij'), -1).reshape(-1, N)


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


def flat_chol_solve(b, chol):
    """Solve A.x = b given cholesky decomposition of A
    """
    N = chol.shape[1]
    B = b.reshape((N, -1))
    X = sp.linalg.cho_solve((chol, True), B)
    return X.T.reshape((-1, 1))


def kron_chol_vsolve(chol_list, b):
    """Solve kronecker(kron_list).x = b where chol_list is the
    cholesky decomposition of matrices to be kronecker'ed: kron_list

    Parameters
    -----------
    chol_list: list of 2D array-like objects
               Cholesky decompositions of D matrices [A_1, A_2, ..., A_D]
               to be Kronecker'ed:
                    A = A_1 \otimes A_2 \otimes ... \otimes A_D
               Product of column dimensions must be N
    b        : array-like
               Nx1 column vector
    """
    return reduce(flat_chol_solve, chol_list, b)


def kron_chol_msolve(chol_list, m):
    """Solve kronecker(kron_list).x = m where chol_list is the
    cholesky decomposition of matrices to be kronecker'ed: kron_list

    Parameters
    -----------
    chol_list: list of 2D array-like objects
               Cholesky decompositions of D matrices [A_1, A_2, ..., A_D]
               to be Kronecker'ed:
                    A = A_1 \otimes A_2 \otimes ... \otimes A_D
               Product of column dimensions must be N
    m        : array-like
               NxM matrix
    """
    if len(m.shape) == 1:
        m = m[:, None]  # Treat 1D array as Nx1 matrix
    return np.concatenate([kron_chol_vsolve(chol_list, b) for b in m.T], axis=1)


def flat_lower_solve(b, L):
    """Solve L.x = b given lower triangular matrix L
    """
    N = L.shape[1]
    B = b.reshape((N, -1))
    X = sp.linalg.solve_triangular(L, B, lower=True)
    return X.T.reshape((-1, 1))


def kron_lower_vsolve(lowers, b):
    """Solve kronecker(lowers).x = b where lowers is a list of lower
    triangular matrices.

    Parameters
    -----------
    lowers   : list of 2D array-like objects
               Lower triangular matrices
                    L = L_1 \otimes L_2 \otimes ... \otimes L_D
               Product of column dimensions must be N
    b        : array-like
               Nx1 column vector
    """
    return reduce(flat_lower_solve, lowers, b)


def kron_lower_msolve(lowers, m):
    """Solve kronecker(lowers).x = m where lowers is a list of lower
    triangular matrices.

    Parameters
    -----------
    lowers   : list of 2D array-like objects
               Lower triangular matrices
                    L = L_1 \otimes L_2 \otimes ... \otimes L_D
               Product of column dimensions must be N
    m        : array-like
               NxM matrix
    """
    if len(m.shape) == 1:
        m = m[:, None]  # Treat 1D array as Nx1 matrix
    return np.concatenate([kron_lower_vsolve(lowers, b) for b in m.T], axis=1)


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

    def __init__(self, mu, covs=None, chols=None, EVDs=None, noise=None):
        # K + noise = Q.(L + noise*I).Q^T
        # Lambdas, self.Qs = zip(*map(np.linalg.eigh, covs))  # Unzip tuples
        # self.QTs = tuple(map(np.transpose, self.Qs))
        # self.eig_noise = kron_diag(Lambdas) + noise
        # self.N = len(self.eig_noise)
        self._setup(covs, chols, EVDs, noise)
        self.mu = mu

    def _setup(self, covs, chols, EVDs, noise):
        if len([i for i in [covs, chols, EVDs] if i is not None]) != 1:
            raise ValueError('Incompatible parameterization. '
                             'Specify exactly one of covs, chols, '
                             'or EVDs.')
        self.isEVD = False
        if covs is not None:
            self.covs = covs
            if noise is not None and noise != 0:
                # Noise requires eigendecomposition
                self.isEVD = True
                eigs_sep, self.Qs = zip(*map(np.linalg.eigh, covs))  # Unzip
                self.QTs = list(map(np.transpose, self.Qs))
                self.eigs = kron_diag(eigs_sep)  # Combine separate eigs
                self.eigs += noise
                self.N = len(self.eigs)
            else:
                # Otherwise use cholesky
                self.chols = list(map(np.linalg.cholesky, self.covs))
                self.chol_diags = np.array(list(map(np.diag, self.chols)))
                self.sizes = np.array([len(chol) for chol in self.chols])
                self.N = np.prod(self.sizes)
        elif chols is not None:
            self.chols = chols
            self.chol_diags = np.array(list(map(np.diag, self.chols)))
            self.sizes = np.array([len(chol) for chol in self.chols])
            self.N = np.prod(self.sizes)
        else:
            self.isEVD = True
            eigs_sep, self.Qs = zip(*EVDs)  # Unzip tuples
            self.QTs = list(map(np.transpose, self.Qs))
            self.eigs = kron_diag(eigs_sep)  # Combine separate eigs
            if noise is not None:
                self.eigs += noise
            self.N = len(self.eigs)

    def random(self, size=None):
        """Drawn using x = mu + A.z for z~N(0,I) and
            A = Q.sqrt(Lambda), if isEVD
            A = chol,           otherwise

        Warning: EVD does not (yet) match with random draws from numpy
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
        if self.isEVD:
            sqrtLz = np.sqrt(self.eigs) * z
            Az = kron_mmprod(self.Qs, sqrtLz.T).T
        else:
            Az = kron_mmprod(self.chols, z.T).T
        return self.mu + Az

    def _quaddist(self, value):
        """Computes the quadratic (x-mu)^T @ K^-1 @ (x-mu) and log(det(K))"""
        delta = value - self.mu
        if self.isEVD:
            sqrt_quad = kron_mmprod(self.QTs, delta.T)
            sqrt_quad = sqrt_quad/np.sqrt(self.eigs[:, None])
            logdet = np.sum(np.log(self.eigs))
        else:
            sqrt_quad = kron_lower_msolve(self.chols, delta.T)
            logchols = np.log(self.chol_diags) * self.N/self.sizes[:, None]
            logdet = np.sum(2*logchols)
        # Square each sample
        quad = np.einsum('ij,ij->j', sqrt_quad, sqrt_quad)
        # For theano: quad = tt.batched_dot(sqrt_quad.T, sqrt_quad.T)
        return quad, logdet

    def logp(self, value):
        quad, logdet = self._quaddist(value)
        return -1/2 * (quad + logdet + self.N*np.log(2*np.pi))

    def update(self):
        # How will updates to hyperparameters be performed?
        raise NotImplementedError


class MarginalKron:
    """
    """

    def __init__(self, mean_func, cov_funcs):
        self.mean_func = mean_func
        try:
            self.cov_funcs = list(cov_funcs)
        except TypeError:
            self.cov_funcs = [cov_funcs]

    def _build_marginal_likelihood(self, Xs):
        self.X = cartesian(*Xs)
        mu = self.mean_func(self.X)
        covs = [f(X) for f, X in zip_longest(cycle(self.cov_funcs), Xs)]
        return mu, covs

    def marginal_likelihood(self, Xs, y, noise, is_observed=True, **kwargs):
        """
        Returns the marginal likelihood distribution, given the input
        locations `X` and the data `y`.
        """
        mu, covs = self._build_marginal_likelihood(Xs)
        self.Xs = Xs
        self.y = y
        self.noise = noise
        return KroneckerNormal(mu=mu, covs=covs, noise=noise)

    def total_cov(self, X, Xs=None, diag=False):
        covs = [f(x, xs, diag) for f, x, xs in
                zip_longest(cycle(self.cov_funcs), X.T, Xs.T)]
        return reduce(mul, covs)

    def _build_conditional(self, Xnew, pred_noise, diag, Xs, y, noise,
                           cov_total, mean_total):
        # Old points
        delta = y - self.mean_func(cartesian(Xs))
        Kns = [f(X) for f, X in zip_longest(cycle(self.cov_funcs), Xs)]
        eigs_sep, Qs = zip(*map(np.linalg.eigh, Kns))  # Unzip
        QTs = list(map(np.transpose, Qs))
        eigs = kron_diag(eigs_sep)  # Combine separate eigs
        if noise is not None:
            eigs += noise

        # New points
        Km = self.total_cov(Xnew, diag)
        Knm = self.total_cov(cartesian(Xs), Xnew)
        Kmn = Knm.T

        # Build conditional mu
        alpha = kron_mvprod(QTs, delta)
        alpha = alpha/self.eigs[:, None]
        alpha = kron_mvprod(Qs, alpha)
        mu = np.dot(Kmns, alpha) + self.mean_func(Xnew)

        # Build conditional cov
        A = kron_mmprod(QTs, Knm)
        A = A/np.sqrt(self.eigs[:, None])
        if diag:
            Asq = np.sum(np.square(A), 0)
            cov = Km - Asq
            if pred_noise:
                cov += noise
        else:
            Asq = np.dot(A.T, A)
            cov = Km - Asq
            if pred_noise:
                cov += noise*np.eye(cov.shape)
        return mu, cov

    def conditional(self, name, Xnew, pred_noise=False, given=None, **kwargs):
        """
        Returns the conditional distribution evaluated over new input
        locations `Xnew`.
        """
        raise NotImplementedError
        mu, cov = self._build_conditional(Xnew, pred_noise, False, *givens)
        return MvNormal(mu=mu, cov=cov)
