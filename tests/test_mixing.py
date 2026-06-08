import unittest

import numpy as np

from src import kernels, mixing


class MixingDiagnosticsTest(unittest.TestCase):
    def test_kernel_metric_diagonal_zero(self):
        x = np.array([[0.0], [1.0], [2.0]])
        k = kernels.rbf_kernel(x, bandwidth=1.0)
        d2 = kernels.kernel_metric_squared(k)
        np.testing.assert_allclose(np.diag(d2), 0.0)
        self.assertTrue(np.all(d2 >= 0.0))

    def test_opposite_collision_pairs_detected(self):
        x = np.array([[0.0], [0.01], [10.0], [10.01]])
        y = np.array([1.0, -1.0, 1.0, -1.0])
        d2 = kernels.pairwise_sq_dists(x)
        pairs = mixing.greedy_disjoint_opposite_pairs(d2, y, rho=0.02)
        self.assertEqual(len(pairs), 2)
        summary = mixing.summarize(d2, y, k=1, rho_quantile=1.0)
        self.assertEqual(summary.collision_count, 2)


if __name__ == "__main__":
    unittest.main()
