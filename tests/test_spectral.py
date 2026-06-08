import unittest

import numpy as np

from src import spectral


class SpectralDiagnosticsTest(unittest.TestCase):
    def test_label_energy_identity_kernel(self):
        y = np.array([1.0, -1.0, 1.0, -1.0])
        k = np.eye(4)
        summary = spectral.summarize(k, y)
        self.assertAlmostEqual(summary.energy[-1], 1.0)
        self.assertAlmostEqual(summary.tail[-1], 0.0)

    def test_label_in_top_eigendirection_has_zero_first_tail(self):
        y = np.array([1.0, 1.0, -1.0, -1.0])
        y_unit = y / np.linalg.norm(y)
        k = 10.0 * np.outer(y_unit, y_unit) + 0.1 * np.eye(4)
        summary = spectral.summarize(k, y)
        self.assertLess(summary.tail[0], 1e-10)
        self.assertGreater(summary.alignment, 0.9)


if __name__ == "__main__":
    unittest.main()
