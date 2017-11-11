import numpy as np
from scipy.stats import multivariate_normal
import unittest
from gp_project import *


class TestKronecker(unittest.TestCase):
    """Test the functions defined in gp_project.kronecker.
    """

    def test_kronecker(self):
        np.random.seed(1)
        # Create random matrices
        [a, b, c] = [np.random.rand(3, 3+i) for i in range(3)]

        np.testing.assert_array_almost_equal(
            kronecker([a, b, c]),       # Custom version
            np.kron(a, np.kron(b, c))   # Standard nested version
            )

    def test_kron_mvprod(self):
        np.random.seed(1)
        # Create random matrices
        Ks = [np.random.rand(3, 3+i) for i in range(3)]
        # Create random vector with correct shape
        tot_size = np.prod([k.shape[1] for k in Ks])
        x = np.random.rand(tot_size).reshape((tot_size, 1))
        # Construct entire kronecker product then multiply
        big = kronecker(Ks)
        slow_ans = np.matmul(big, x)
        # Use tricks to avoid construction of entire kronecker product
        fast_ans = kron_mvprod(Ks, x)

        np.testing.assert_array_almost_equal(slow_ans, fast_ans)

    def test_kron_mmprod(self):
        np.random.seed(1)
        # Create random matrices
        Ks = [np.random.rand(3, 3+i) for i in range(3)]
        # Create random matrix with correct shape
        tot_size = np.prod([k.shape[1] for k in Ks])
        m = np.random.rand(tot_size, 4)  # column size could be anything
        # Construct entire kronecker product then multiply
        big = kronecker(Ks)
        slow_ans = np.matmul(big, m)
        # Use tricks to avoid construction of entire kronecker product
        fast_ans = kron_mmprod(Ks, m)

        np.testing.assert_array_almost_equal(slow_ans, fast_ans)

    def test_kron_diag(self):
        np.random.seed(1)
        # Create random matrices
        Ks = [np.random.rand(5, 5) for i in range(4)]
        slow_ans = np.diag(kronecker(Ks))
        Kdiags = map(np.diag, Ks)
        fast_ans = kron_diag(Kdiags)
        np.testing.assert_array_almost_equal(slow_ans, fast_ans)

    def test_KroneckerNormal_logp(self):
        np.random.seed(1)
        # Make mean, covariance, and noise
        nvars = 3
        lenvars = 5
        xs = np.random.rand(nvars, lenvars)
        xs = np.sort(xs)
        Ks = [gaussian_kernel(x, x, .1) for x in xs]
        tot_size = np.prod([k.shape[1] for k in Ks])
        mu = np.ones(tot_size)
        noise = 1
        # Construct entire kronecker product and feed to multivariate normal
        big = kronecker(Ks)
        K = big + noise * np.eye(tot_size)
        x = np.random.rand(tot_size)
        sp_logp = multivariate_normal.logpdf(x, mean=mu, cov=K)
        # Use smarter method
        kron_logp = KroneckerNormal(mu=mu, covs=Ks, noise=noise).logp(x)
        # Test
        np.testing.assert_array_almost_equal(sp_logp, kron_logp)

    def test_KroneckerNormal_logp_vec(self):
        np.random.seed(1)
        # Make mean, covariance, and noise
        nvars = 3
        lenvars = 5
        xs = np.random.rand(nvars, lenvars)
        xs = np.sort(xs)
        Ks = [gaussian_kernel(x, x, .1) for x in xs]
        tot_size = np.prod([k.shape[1] for k in Ks])
        mu = np.ones(tot_size)
        noise = 1
        # Construct entire kronecker product and feed to multivariate normal
        big = kronecker(Ks)
        K = big + noise * np.eye(tot_size)
        x = np.random.rand(10, tot_size)
        sp_logp = multivariate_normal.logpdf(x, mean=mu, cov=K)
        # Use smarter method
        kron_logp = KroneckerNormal(mu=mu, covs=Ks, noise=noise).logp(x)
        # Test
        np.testing.assert_array_almost_equal(sp_logp, kron_logp)

    # def test_KroneckerNormal_random(self):
    #     np.random.seed(1)
    #     nvars = 2
    #     lenvars = 3
    #     xs = np.random.rand(nvars, lenvars)
    #     xs = np.sort(xs)
    #     Ks = [gaussian_kernel(x, x, .1) for x in xs]
    #     tot_size = np.prod([k.shape[1] for k in Ks])
    #     mu = np.ones(tot_size)
    #     noise = 1e-3
    #     size = 2
    #     # Construct entire kronecker product and feed to multivariate normal
    #     big = kronecker(Ks)
    #     K = big + noise * np.eye(tot_size)
    #     np.random.seed(1)
    #     # sp_kron_random = multivariate_normal.rvs(mean=mu, cov=K)
    #     sp_kron_random = multivariate_normal.rvs(mean=mu, cov=K, size=size)
    #     kron_norm = KroneckerNormal(mu=mu, covs=Ks, noise=noise)
    #     np.random.seed(1)
    #     kron_random = kron_norm.random(size=size)
    #     np.testing.assert_array_almost_equal(sp_kron_random, kron_random)
