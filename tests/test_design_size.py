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


def rule_profile(length, rules, noise=0.0, seed=0):
    """A side profile: Gaussian-ish dark lines on textured paper."""
    profile = np.zeros(length)
    position = np.arange(length, dtype=float)
    for centre, width, height in rules:
        profile += height * np.exp(-0.5 * ((position - centre)
                                           / (width / 2.355)) ** 2)
    if noise:
        profile += np.random.default_rng(seed).normal(0, noise, length)
    return profile


class PeakTest(unittest.TestCase):
    def test_prominence_ignores_an_added_constant(self):
        signal = np.array([0.0, 1.0, 0.3, 2.0, 0.1, 0.5])
        a, _ = measure_design.peak_prominence(signal, 3)
        b, _ = measure_design.peak_prominence(signal + 17.0, 3)
        self.assertAlmostEqual(a, b, places=9)

    def test_prominence_is_measured_to_the_key_saddle(self):
        # Peak at index 1 is lower than the one at 3, so its prominence is
        # measured down to the valley between them, not to zero.
        signal = np.array([0.0, 1.0, 0.4, 2.0, 0.0])
        prominence, saddle = measure_design.peak_prominence(signal, 1)
        self.assertAlmostEqual(prominence, 0.6, places=9)
        self.assertAlmostEqual(saddle, 0.4, places=9)

    def test_a_faint_rule_is_described_like_a_strong_one(self):
        """The 3K failure in miniature. Two rules of equal width, one printed
        at a third the strength. At a fixed darkness level the faint one is
        narrower and was rejected; at half prominence both measure the same."""
        strong = rule_profile(400, [(80.0, 9.0, 1.0)], noise=0.01, seed=1)
        faint = rule_profile(400, [(80.0, 9.0, 0.33)], noise=0.01, seed=1)
        a = measure_design.peak_candidates(strong, 0.5, 6.0)[0]
        b = measure_design.peak_candidates(faint, 0.5, 6.0)[0]
        self.assertAlmostEqual(a['width_px'], b['width_px'], delta=0.6)
        self.assertAlmostEqual(a['centre_px'], 80.0, delta=0.5)
        self.assertAlmostEqual(b['centre_px'], 80.0, delta=0.5)
        # Both are far above the noise, which is what licenses them.
        self.assertGreater(b['significance'], 20)

    def test_candidates_are_ordered_outermost_first(self):
        profile = rule_profile(400, [(30.0, 6.0, 1.0), (70.0, 18.0, 1.4)],
                               noise=0.01, seed=2)
        found = measure_design.peak_candidates(profile, 0.5, 6.0)
        self.assertGreaterEqual(len(found), 2)
        self.assertAlmostEqual(found[0]['centre_px'], 30.0, delta=0.6)
        self.assertAlmostEqual(found[1]['centre_px'], 70.0, delta=0.8)
        self.assertLess(found[0]['width_px'], found[1]['width_px'])


class SelectionTest(unittest.TestCase):
    def side(self, entries):
        return [{'centre_px': c, 'width_px': w, 'prominence': 1.0,
                 'significance': s} for c, w, s in entries]

    def test_rejects_a_family_that_mixes_two_different_features(self):
        """3K and 70K exactly: the thin rule found on two sides and the thick
        band on the other two. Outermost-first alone would take it; the
        four-side width agreement is what refuses."""
        thin, thick = (6.0, 30.0), (18.0, 25.0)
        candidates = {
            'left': self.side([(86.0, *thin), (105.0, *thick)]),
            'top': self.side([(94.0, *thin), (114.0, *thick)]),
            # On these two the thin rule is fainter, so it sits second in the
            # candidate list, but it is still the outermost one.
            'right': self.side([(88.0, 6.0, 12.0), (107.0, *thick)]),
            'bottom': self.side([(93.0, 6.0, 14.0), (104.0, *thick)]),
        }
        start = {side: 0 for side in candidates}
        choice = measure_design.choose_consistent_rule(candidates, start, 1.5)
        for side in candidates:
            self.assertAlmostEqual(choice[side][1]['width_px'], 6.0, places=6)

        inward = {side: index + 1 for side, (index, _) in choice.items()}
        second = measure_design.choose_consistent_rule(candidates, inward, 1.5)
        for side in candidates:
            self.assertAlmostEqual(second[side][1]['width_px'], 18.0, places=6)

    def test_skips_a_spurious_peak_no_other_side_corroborates(self):
        candidates = {
            'left': self.side([(40.0, 17.0, 9.0), (86.0, 6.0, 30.0)]),
            'right': self.side([(88.0, 6.0, 28.0)]),
            'top': self.side([(94.0, 6.0, 31.0)]),
            'bottom': self.side([(93.0, 6.0, 27.0)]),
        }
        start = {side: 0 for side in candidates}
        choice = measure_design.choose_consistent_rule(candidates, start, 1.5)
        self.assertAlmostEqual(choice['left'][1]['centre_px'], 86.0, places=6)
        self.assertEqual(choice['left'][0], 1)

    def test_steps_all_four_sides_in_when_one_side_cannot_see_the_outermost(
            self):
        """70K exactly: the quorum clips the canvas through the left outer
        rule, so the left list begins at the thick band. No family containing
        the outer rule is available, and the thick band is reported for all
        four rather than a left/right mixture."""
        candidates = {
            'left': self.side([(90.7, 14.2, 81.0), (173.0, 19.2, 56.0)]),
            'right': self.side([(81.0, 8.2, 66.0), (100.7, 15.3, 81.0)]),
            'top': self.side([(85.6, 8.1, 65.0), (105.2, 14.3, 69.0)]),
            'bottom': self.side([(84.2, 8.1, 60.0), (103.6, 14.2, 64.0)]),
        }
        start = {side: 0 for side in candidates}
        choice = measure_design.choose_consistent_rule(candidates, start, 1.5)
        centres = {side: entry['centre_px']
                   for side, (_, entry) in choice.items()}
        self.assertEqual(centres, {'left': 90.7, 'right': 100.7,
                                   'top': 105.2, 'bottom': 103.6})

    def test_reports_nothing_when_a_side_has_no_candidate(self):
        candidates = {'left': self.side([(86.0, 6.0, 30.0)]), 'right': [],
                      'top': self.side([(94.0, 6.0, 31.0)]),
                      'bottom': self.side([(93.0, 6.0, 27.0)])}
        start = {side: 0 for side in candidates}
        self.assertIsNone(
            measure_design.choose_consistent_rule(candidates, start, 1.5))


class BandTest(unittest.TestCase):
    def test_thinly_covered_lines_cannot_look_like_a_rule(self):
        """The 70K margin artefact: a dark band carried by one or two stamps
        is indistinguishable from a rule until a quorum is required."""
        darkness = np.zeros((40, 40), np.float32)
        darkness[:, :6] = 9.0           # dark, but only two stamps reach it
        darkness[:, 10:16] = 1.0        # the real rule, reached by all ten
        coverage = np.full((40, 40), 10.0, np.float32)
        coverage[:, :6] = 2.0
        loose, loose_valid = measure_design.masked_profile(
            darkness, coverage, 0, span=1.0, min_stamps=1)
        strict, strict_valid = measure_design.masked_profile(
            darkness, coverage, 0, span=1.0, min_stamps=6)
        self.assertAlmostEqual(float(loose[2]), 9.0, places=5)
        self.assertTrue(bool(loose_valid[2]))
        # Under the quorum the artefact is not merely darkened, it is marked
        # invalid, so the rule search never reaches it.
        self.assertFalse(bool(strict_valid[2]))
        self.assertTrue(bool(strict_valid[12]))
        self.assertAlmostEqual(float(strict[12]), 1.0, places=5)


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

        rules, _ = measure_design.frame_rectangle(
            golden, coverage, span=0.5, min_stamps=0.6 * len(truth),
            search_span=0.35, min_significance=6.0, width_tolerance=1.8)
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
