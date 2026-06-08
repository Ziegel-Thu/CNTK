import unittest

import numpy as np

from src import kernels, mixing, spectral


class MulticlassDiagnosticTests(unittest.TestCase):
    def test_label_subspace_tail_is_low_for_label_block_kernel(self):
        labels = np.array([0, 0, 1, 1, 2, 2])
        y_mat = spectral.one_hot_labels(labels, center=True)
        gram = y_mat @ y_mat.T + 1e-6 * np.eye(len(labels))

        summary = spectral.summarize_multiclass(gram, labels)

        self.assertLess(summary.tail[1], 1e-5)
        self.assertGreater(summary.alignment, 0.99)
        self.assertEqual(summary.n_classes, 3)

    def test_multiclass_entropy_is_zero_for_pure_neighborhoods(self):
        x = np.array([[0.0], [0.1], [3.0], [3.1], [6.0], [6.1]])
        labels = np.array([0, 0, 1, 1, 2, 2])
        d2 = kernels.pairwise_sq_dists(x)

        summary = mixing.summarize_multiclass(d2, labels, k=1)

        self.assertEqual(summary.knn_disagreement_ratio, 0.0)
        self.assertEqual(summary.local_label_entropy, 0.0)
        self.assertEqual(summary.local_label_entropy_normalized, 0.0)


if __name__ == "__main__":
    unittest.main()
