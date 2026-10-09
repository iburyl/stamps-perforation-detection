import unittest

import cv2
import numpy as np

from design_size import measure_design


def synthetic_design(width, height, frame_px=14, margin=40):
    """A stamp-like design: an outer frame band plus interior structure."""
    image = np.full((height, width), 235, np.uint8)
    cv2.rectangle(image, (margin, margin),
                  (width - margin - 1, height - margin - 1), 60,
                  thickness=frame_px)
    cv2.ellipse(image, (width // 2, height // 2),
                (width // 5, height // 4), 0, 0, 360, 90, thickness=9)
    for step in range(6):
        y = margin + frame_px + 30 + step * 24
        cv2.line(image, (margin + frame_px + 20, y),
                 (width - margin - frame_px - 20, y), 150, 3)
    return image


def place(design, canvas, scale_x, scale_y, offset, rng, cancel=True):
    """One noisy, de-centred, independently scaled instance of the design."""
    height, width = canvas
    resized = cv2.resize(design, None, fx=scale_x, fy=scale_y,
                         interpolation=cv2.INTER_CUBIC)
    patch = np.full(canvas, 235, np.uint8)
    y0 = (height - resized.shape[0]) // 2 + offset[1]
    x0 = (width - resized.shape[1]) // 2 + offset[0]
    patch[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
    if cancel:
        start = (rng.integers(0, width), rng.integers(0, height // 2))
        end = (rng.integers(0, width), rng.integers(height // 2, height))
        cv2.line(patch, tuple(int(v) for v in start),
                 tuple(int(v) for v in end), 30, 11)
    noise = rng.normal(0, 3, canvas)
    return np.clip(patch.astype(np.float32) + noise, 0, 255).astype(np.uint8)


class DecomposeTest(unittest.TestCase):
    def test_recovers_independent_scales_and_rotation(self):
        angle = np.radians(2.0)
        rotation = np.array([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]])
        stretch = np.array([[1.03, 0.0], [0.0, 0.97]])
        warp = np.hstack([rotation @ stretch, [[5.0], [-3.0]]]).astype(np.float32)
        parts = measure_design.decompose(warp)
        self.assertAlmostEqual(parts['scale_x'], 1.03, places=6)
        self.assertAlmostEqual(parts['scale_y'], 0.97, places=6)
        self.assertAlmostEqual(parts['rotation_deg'], 2.0, places=4)
        self.assertAlmostEqual(parts['shear'], 0.0, places=6)
        self.assertAlmostEqual(parts['shift_x'], 5.0, places=6)


class BandTest(unittest.TestCase):
    def test_reports_rules_outermost_first(self):
        profile = np.zeros(400)
        profile[20:34] = 1.0      # thin outer rule, centre 26.5
        profile[50:70] = 1.0      # thick main frame band, centre 59.5
        profile[330:350] = 1.0
        profile[366:380] = 1.0
        low, high = measure_design.edge_bands(profile, min_run=5, level=0.5,
                                           count=2)
        self.assertAlmostEqual(low[0]['centre_px'], 26.5, places=3)
        self.assertEqual(low[0]['width_px'], 14)
        self.assertAlmostEqual(low[1]['centre_px'], 59.5, places=3)
        self.assertAlmostEqual(high[0]['centre_px'], 372.5, places=3)
        self.assertAlmostEqual(high[1]['centre_px'], 339.5, places=3)

    def test_ignores_runs_shorter_than_min_run(self):
        profile = np.zeros(200)
        profile[10:12] = 1.0      # speck
        profile[40:54] = 1.0
        profile[150:164] = 1.0
        low, _ = measure_design.edge_bands(profile, min_run=5, level=0.5, count=1)
        self.assertAlmostEqual(low[0]['centre_px'], 46.5, places=3)

    def test_thinly_covered_lines_cannot_look_like_a_rule(self):
        """The 70K failure: a dark margin artefact carried by one or two
        stamps is indistinguishable from a rule until a quorum is required."""
        darkness = np.zeros((40, 40), np.float32)
        darkness[:, :6] = 9.0           # dark, but only two stamps reach it
        darkness[:, 10:16] = 1.0        # the real rule, reached by all ten
        coverage = np.full((40, 40), 10.0, np.float32)
        coverage[:, :6] = 2.0
        loose = measure_design.masked_profile(darkness, coverage, 0, span=1.0,
                                           min_stamps=1)
        strict = measure_design.masked_profile(darkness, coverage, 0, span=1.0,
                                            min_stamps=6)
        low, _ = measure_design.edge_bands(loose, min_run=5, level=0.5, count=1)
        self.assertAlmostEqual(low[0]['centre_px'], 2.5, places=3)
        low, _ = measure_design.edge_bands(strict, min_run=5, level=0.5, count=1)
        self.assertAlmostEqual(low[0]['centre_px'], 12.5, places=3)


class BackProjectionTest(unittest.TestCase):
    """The annotated scan is only honest if the drawn quadrilateral's side
    lengths are the reported width and height, in the right place."""

    def test_rectangle_lands_on_the_scan_at_the_measured_size(self):
        rectangle = {'left_px': 100.0, 'right_px': 400.0,
                     'top_px': 50.0, 'bottom_px': 550.0}
        angle = np.radians(1.7)
        rotation = np.array([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]])
        warp = np.hstack([rotation @ np.diag([1.02, 0.98]),
                          [[7.0], [-4.0]]]).astype(np.float32)
        # A rigid placement, as stamp_patch builds: rotation plus translation.
        place_angle = np.radians(-3.0)
        place_rotation = np.array(
            [[np.cos(place_angle), -np.sin(place_angle)],
             [np.sin(place_angle), np.cos(place_angle)]])
        placement = np.hstack([place_rotation,
                               [[620.0], [310.0]]]).astype(np.float32)

        corners = measure_design.rectangle_on_scan(rectangle, warp, placement)
        self.assertEqual(corners.shape, (4, 2))

        parts = measure_design.decompose(warp)
        width = (rectangle['right_px'] - rectangle['left_px']) * parts['scale_x']
        height = (rectangle['bottom_px'] - rectangle['top_px']) * parts['scale_y']
        self.assertAlmostEqual(np.linalg.norm(corners[1] - corners[0]),
                               width, places=3)
        self.assertAlmostEqual(np.linalg.norm(corners[2] - corners[1]),
                               height, places=3)
        self.assertAlmostEqual(np.linalg.norm(corners[3] - corners[2]),
                               width, places=3)
        # Still a rectangle: the diagonals match and the corner is square.
        self.assertAlmostEqual(np.linalg.norm(corners[2] - corners[0]),
                               np.linalg.norm(corners[3] - corners[1]),
                               places=3)
        self.assertAlmostEqual(
            float(np.dot(corners[1] - corners[0], corners[3] - corners[0])),
            0.0, places=2)

    def test_placement_round_trips_through_stamp_patch(self):
        """The placement returned with a patch must be the one that put it
        there, or the annotation lands somewhere else on the scan."""
        rng = np.random.default_rng(3)
        scan = rng.integers(60, 200, (700, 600, 3), dtype=np.uint8)
        quad = np.array([[180.0, 150.0], [460.0, 170.0],
                         [450.0, 540.0], [170.0, 520.0]])
        _, _, placement = measure_design.stamp_patch(
            scan, quad, (520, 420), erode_px=3)
        identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], np.float32)
        on_canvas = measure_design.apply_affine(placement, quad)
        back = measure_design.apply_affine(
            cv2.invertAffineTransform(placement), on_canvas)
        np.testing.assert_allclose(back, quad, atol=1e-3)
        # An identity warp must map the canvas centre back to the quad centre.
        centre = measure_design.rectangle_on_scan(
            {'left_px': 210.0, 'right_px': 210.0,
             'top_px': 260.0, 'bottom_px': 260.0}, identity, placement)
        np.testing.assert_allclose(centre[0], quad.mean(axis=0), atol=1e-3)


class GoldenTest(unittest.TestCase):
    def test_coverage_counts_stamps(self):
        """Coverage drives the quorum, so its unit has to be stamps, not the
        0/255 of the masks it is accumulated from."""
        size = (20, 20)
        patches = [np.full(size, float(value), np.float32) for value in (1, 3)]
        masks = [np.full(size, 255, np.uint8) for _ in patches]
        masks[1][:, :10] = 0
        warps = [np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], np.float32)
                 for _ in patches]
        weights = [np.ones(size, np.float32) for _ in patches]
        golden, coverage = measure_design.build_golden(
            patches, masks, warps, weights, size)
        self.assertAlmostEqual(float(coverage[0, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(coverage[0, 15]), 2.0, places=5)
        self.assertAlmostEqual(float(golden[0, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(golden[0, 15]), 2.0, places=5)


class CongealTest(unittest.TestCase):
    """The end-to-end claim: known independent scales are recovered from a set
    that is de-centred by more than the frame line is wide, and defaced."""

    def test_recovers_known_scales(self):
        rng = np.random.default_rng(7)
        canvas = (560, 440)
        design = synthetic_design(360, 480)
        truth = [(0.98, 1.01), (1.00, 1.00), (1.02, 0.99),
                 (0.99, 1.02), (1.01, 0.98), (1.00, 1.01)]
        patches, masks = [], []
        for scale_x, scale_y in truth:
            offset = (int(rng.integers(-18, 19)), int(rng.integers(-22, 23)))
            grey = place(design, canvas, scale_x, scale_y, offset, rng)
            mask = np.full(canvas, 255, np.uint8)
            mask[:6, :] = mask[-6:, :] = mask[:, :6] = mask[:, -6:] = 0
            value = grey.astype(np.float32)
            value = (value - value.mean()) / value.std()
            patches.append(value)
            masks.append(mask)

        golden, coverage, warps, _, failures, free_shear = measure_design.congeal(
            patches, masks, canvas, ((4, 3), (2, 2), (1, 2)),
            iterations=200, eps=1e-7, cutoff=3.0, verbose=False)
        self.assertEqual(sum(failures), 0)
        self.assertLess(max(abs(value) for value in free_shear), 0.01)

        rules = measure_design.frame_rectangle(
            golden, coverage, min_run=5, level=0.5, span=0.5,
            min_stamps=0.6 * len(truth))
        self.assertGreaterEqual(len(rules), 1)
        rectangle = rules[0]

        # Gauge normalisation pins the mean scale to 1, so compare each stamp
        # against the truth expressed the same way.
        mean_x = float(np.exp(np.mean(np.log([t[0] for t in truth]))))
        mean_y = float(np.exp(np.mean(np.log([t[1] for t in truth]))))
        for index, (scale_x, scale_y) in enumerate(truth):
            parts = measure_design.decompose(warps[index])
            self.assertAlmostEqual(parts['scale_x'], scale_x / mean_x, delta=0.002)
            self.assertAlmostEqual(parts['scale_y'], scale_y / mean_y, delta=0.002)
            self.assertAlmostEqual(parts['rotation_deg'], 0.0, delta=0.15)
            self.assertAlmostEqual(parts['shear'], 0.0, places=5)

        # The design rectangle itself, in each stamp's own pixels. The
        # measurand is the frame centreline, and cv2.rectangle centres its
        # thickness on the given outline, so the truth is the separation of
        # those outlines and does not depend on the band thickness at all.
        frame_width = 360 - 2 * 40 - 1
        frame_height = 480 - 2 * 40 - 1
        for index, (scale_x, scale_y) in enumerate(truth):
            parts = measure_design.decompose(warps[index])
            width = rectangle['width_px'] * parts['scale_x']
            height = rectangle['height_px'] * parts['scale_y']
            self.assertAlmostEqual(width, frame_width * scale_x, delta=1.5)
            self.assertAlmostEqual(height, frame_height * scale_y, delta=1.5)


if __name__ == '__main__':
    unittest.main()
