import unittest

import numpy as np

from diagnose_kmeans_quantization import smooth_dark_background_profile
from kmeans_brightness.candidate_search_experiment import sequential
from kmeans_brightness.ridges import depth_of, line_majority_depth


class KMeansProfileSmoothingTests(unittest.TestCase):
    def context(self, width=20):
        return {
            'positions': np.arange(width, dtype=float),
            'n0': 0,
        }

    def test_moves_premature_background_island_to_paper_majority(self):
        width = 20
        mask = np.zeros((24, width), dtype=np.uint8)
        mask[15:] = 1
        # Four light pixels satisfy boundary_profile's sustained-run rule but
        # are separated from the real paper by a wide dark gap.
        mask[1:5, 2] = 1
        depth = np.full(width, 15., dtype=float)
        depth[2] = 1.
        depth[10] = 18.  # A genuine inward valley must remain untouched.

        corrected, metadata = smooth_dark_background_profile(
            self.context(width), mask, depth,
        )

        self.assertEqual(corrected[2], 15.)
        self.assertEqual(corrected[10], 18.)
        self.assertEqual(metadata['corrected_columns'], 1)
        self.assertEqual(metadata['max_shift_px'], 14.)

    def test_uses_first_light_pixel_below_curve_as_fallback(self):
        width = 20
        mask = np.zeros((24, width), dtype=np.uint8)
        depth = np.full(width, 15., dtype=float)
        depth[2] = 1.
        mask[2, 2] = 1

        corrected, metadata = smooth_dark_background_profile(
            self.context(width), mask, depth,
        )

        self.assertEqual(corrected[2], 2.)
        self.assertEqual(metadata['corrected_columns'], 1)

    def test_experiment_variant_uses_the_shared_implementation(self):
        width = 20
        mask = np.zeros((24, width), dtype=np.uint8)
        mask[15:] = 1
        mask[1:5, 2] = 1

        direct = line_majority_depth(mask, 0)
        through_dispatch = depth_of(mask, 0, 'majority')

        np.testing.assert_array_equal(direct, through_dispatch)
        self.assertEqual(direct[2], 15.)


class RankedKMeansSearchTests(unittest.TestCase):
    @staticmethod
    def item(arcs, ridge='stack', geometry='circle_arcs', status='ok'):
        return {
            'summary': {
                'arcs': arcs, 'ridge': ridge, 'geometry_source': geometry,
                'status': status, 'gauge': 14.5, 'reliable': False,
            },
        }

    def test_seven_arc_stop_skips_weaker_first_profile(self):
        first = self.item(6)
        second = self.item(7)
        chosen, attempts = sequential({
            'stack': [first, second], 'close': [self.item(12, 'close')],
        }, 7)

        self.assertIs(chosen, second)
        self.assertEqual(attempts, 2)

    def test_close_is_only_tried_after_stack_is_exhausted(self):
        close = self.item(7, 'close')
        chosen, attempts = sequential({
            'stack': [self.item(4), self.item(3)], 'close': [close],
        }, 7)

        self.assertIs(chosen, close)
        self.assertEqual(attempts, 3)

    def test_circle_geometry_is_required_for_early_stop(self):
        false_signal = self.item(12, geometry=None)
        valid = self.item(7)
        chosen, attempts = sequential({
            'stack': [false_signal, valid], 'close': [],
        }, 7)

        self.assertIs(chosen, valid)
        self.assertEqual(attempts, 2)


if __name__ == '__main__':
    unittest.main()
