import numpy as np
import unittest
from gp_project import *


class TestKronecker(unittest.TestCase):
    """Test the functions defined in gp_project.kronecker.
    """

    def test_kronecker(self):
        # Create random matrices
        [a, b, c] = [np.random.rand(3, 3+i) for i in range(3)]

        np.testing.assert_array_almost_equal(
            kronecker([a, b, c]),       # Custom version
            np.kron(a, np.kron(b, c))   # Standard nested version
            )

    def test_kron_mvprod(self):
        # Create random matrices
        [a, b, c] = [np.random.rand(3, 3+i) for i in range(3)]
        # Create random vector with correct shape
        tot_size = a.shape[1]*b.shape[1]*c.shape[1]
        x = np.random.rand(tot_size).reshape((tot_size, 1))
        # Construct entire kronecker product then multiply
        big = kronecker([a, b, c])
        slow_ans = np.matmul(big, x)
        # Use tricks to avoid construction of entire kronecker product
        fast_ans = kron_mvprod([a, b, c], x)

        np.testing.assert_array_almost_equal(slow_ans, fast_ans)

    def test_kron_mmprod(self):
        # Create random matrices
        [a, b, c] = [np.random.rand(3, 3+i) for i in range(3)]
        # Create random matrix with correct shape
        tot_size = a.shape[1]*b.shape[1]*c.shape[1]
        m = np.random.rand(tot_size, 4)  # column size could be anything
        # Construct entire kronecker product then multiply
        big = kronecker([a, b, c])
        slow_ans = np.matmul(big, m)
        # Use tricks to avoid construction of entire kronecker product
        fast_ans = kron_mmprod([a, b, c], m)

        np.testing.assert_array_almost_equal(slow_ans, fast_ans)
