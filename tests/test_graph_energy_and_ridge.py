import unittest

import numpy as np

from src import kernel_ridge, kernels, mixing


class GraphEnergyAndRidgeTests(unittest.TestCase):
    def test_graph_energy_increases_with_label_mixing(self):
        x = np.array([[0.0], [0.1], [2.0], [2.1]])
        pure_labels = np.array([1.0, 1.0, -1.0, -1.0])
        mixed_labels = np.array([1.0, -1.0, 1.0, -1.0])
        d2 = kernels.pairwise_sq_dists(x)
        adj = mixing.knn_adjacency(d2, k=1, weighted=False)

        pure = mixing.graph_dirichlet_energy(adj, pure_labels)
        mixed = mixing.graph_dirichlet_energy(adj, mixed_labels)

        self.assertLess(pure, mixed)
        self.assertEqual(mixing.graph_disagreement_ratio(adj, pure_labels), 0.0)
        self.assertGreater(mixing.graph_disagreement_ratio(adj, mixed_labels), 0.0)

    def test_binary_kernel_ridge_separates_simple_clusters(self):
        train_x = np.array([[-1.0], [-0.8], [0.8], [1.0]])
        test_x = np.array([[-0.9], [0.9]])
        y_train = np.array([-1.0, -1.0, 1.0, 1.0])
        y_test = np.array([-1.0, 1.0])
        k_train = kernels.linear_kernel(train_x)
        k_test_train = test_x @ train_x.T / train_x.shape[1]

        result = kernel_ridge.fit_binary_kernel_ridge(k_train, y_train, k_test_train, y_test, ridge=1e-4)

        self.assertEqual(result.test_accuracy, 1.0)
        self.assertGreater(result.test_margin_median, 0.0)
        self.assertGreater(result.source_norm, 0.0)

    def test_multiclass_kernel_ridge_handles_noncontiguous_labels(self):
        train_x = np.array([[1.0, 0.0], [0.8, 0.0], [0.0, 1.0], [0.0, 0.8]])
        test_x = np.array([[0.9, 0.0], [0.0, 0.9]])
        y_train = np.array([2, 2, 5, 5])
        y_test = np.array([2, 5])
        k_train = kernels.linear_kernel(train_x)
        k_test_train = test_x @ train_x.T / train_x.shape[1]

        result = kernel_ridge.fit_multiclass_kernel_ridge(k_train, y_train, k_test_train, y_test, ridge=1e-4)

        self.assertEqual(result.test_accuracy, 1.0)
        self.assertGreater(result.test_margin_median, 0.0)


if __name__ == "__main__":
    unittest.main()
