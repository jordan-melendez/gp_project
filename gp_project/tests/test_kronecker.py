import numpy as np
import unittest
from gp_project import *


class TestKronecker(unittest.TestCase):
    """Test the functions defined in gp_project.kronecker.
    """

    def test_kronecker(self):
        [a, b, c] = [np.random.rand(3, 3+i) for i in range(3)]
        np.testing.assert_array_almost_equal(
            kronecker([a, b, c]),
            np.kron(a, np.kron(b, c))
            )

    def test_kron_mvprod(self):
        [a, b, c] = [np.random.rand(3, 3+i) for i in range(3)]
        tot_size = a.shape[1]*b.shape[1]*c.shape[1]
        x = np.random.rand(tot_size).reshape((tot_size, 1))
        big = kronecker([a, b, c])
        slow_ans = np.matmul(big, x)
        fast_ans = kron_mvprod([a, b, c], x)
        np.testing.assert_array_almost_equal(slow_ans, fast_ans)

    def test_kron_mmprod(self):
        [a, b, c] = [np.random.rand(3, 3+i) for i in range(3)]
        tot_size = a.shape[1]*b.shape[1]*c.shape[1]
        x = np.random.rand(tot_size).reshape((tot_size, 1))
        y = np.random.rand(tot_size).reshape((tot_size, 1))
        big = kronecker([a, b, c])
        m = np.concatenate((x, y), axis=1)
        slow_ans = np.matmul(big, m)
        fast_ans = kron_mmprod([a, b, c], m)
        np.testing.assert_array_almost_equal(slow_ans, fast_ans)
