import unittest

import numpy as np

from refine_perforation import refine_valleys, fit_arc


class RefinementTests(unittest.TestCase):
    def test_recovers_apex_from_biased_threshold_and_off_center_seed(self):
        x = np.arange(200)
        boundary = 70 + np.sqrt(np.maximum(0, 20**2-(x-100)**2))
        yy = np.arange(160)[:, None]
        image = (30 + 180/(1+np.exp(np.clip(-(yy-boundary)/1.2, -80, 80)))).astype(np.float32)
        positions = np.arange(60, 141)
        prior = boundary[positions]-4
        initial = np.array([[104., 87.]])
        result = refine_valleys(image, positions, prior, initial, 64)[0]
        self.assertIsNotNone(result)
        self.assertEqual(result['model'], 'circle')
        self.assertLess(np.linalg.norm(result['point']-[100, 90]), 0.6)
        self.assertLess(np.linalg.norm(result['point']-[100, 90]),
                        np.linalg.norm(initial[0]-[100, 90]))

    def test_one_sided_arc_is_not_extrapolated(self):
        xx = np.linspace(102, 119, 25)
        pts = np.column_stack([xx, 70+np.sqrt(20**2-(xx-100)**2)])
        self.assertIsNone(fit_arc(pts, np.array([104, 89]), 64))

    def test_flat_edge_is_not_fitted_as_hole(self):
        xx = np.arange(80, 121)
        pts = np.column_stack([xx, np.full(len(xx), 90.)])
        self.assertIsNone(fit_arc(pts, np.array([100, 90]), 64))


if __name__ == '__main__':
    unittest.main()
