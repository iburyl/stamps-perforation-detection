import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = ROOT / "design_components"
if str(COMPONENTS) not in sys.path:
    sys.path.insert(0, str(COMPONENTS))

import background_ink
import golden_components
import ink_palette


def unit(vector):
    value = np.asarray(vector, np.float64)
    return value / np.linalg.norm(value)


def render_density(density, paper):
    linear = background_ink.to_linear(np.asarray(paper, np.float64))[None, None, :]
    return background_ink.to_srgb(linear * 10.0 ** (-density)).astype(np.uint8)


class PaletteTests(unittest.TestCase):
    def test_achromatic_mode_does_not_replace_coloured_mode(self):
        neutral = unit([1.0, 1.0, 1.0])
        coloured = unit([0.90, 0.55, 0.12])
        directions = np.vstack(
            [np.repeat(neutral[None, :], 3000, axis=0),
             np.repeat(coloured[None, :], 7000, axis=0)]
        )

        palette = ink_palette.find_palette(directions)

        self.assertEqual(palette["count"], 2)
        self.assertEqual(int(np.sum(palette["achromatic"])), 1)
        chromatic = palette["rays"][~palette["achromatic"]][0]
        self.assertGreater(float(chromatic @ coloured), 0.999)

    def test_second_stage_does_not_rediscover_achromatic_ink(self):
        neutral = unit([1.0, 1.0, 1.0])
        coloured = unit([0.90, 0.55, 0.12])
        directions = np.vstack(
            [np.repeat(neutral[None, :], 4000, axis=0),
             np.repeat(coloured[None, :], 6000, axis=0)]
        )

        palette = ink_palette.find_coloured_palette(directions)

        self.assertEqual(palette["count"], 1)
        self.assertFalse(palette["achromatic"].any())
        self.assertGreater(float(palette["rays"][0] @ coloured), 0.999)


class AchromaticLayerTests(unittest.TestCase):
    def make_collection(self):
        count, height, width = 7, 90, 120
        paper = np.array([205.0, 198.0, 178.0])
        neutral = unit([1.0, 1.0, 1.0])
        coloured = unit([0.90, 0.55, 0.12])
        images = []
        for index in range(count):
            density = np.zeros((height, width, 3), np.float64)
            density[20:70, 15:43] = 1.0 * neutral
            density[25:65, 76:105] = 0.8 * coloured
            # A black cancellation moves between impressions and must not become
            # part of the repeatable achromatic plate.
            x0 = 48 + 3 * index
            density[5:17, x0:x0 + 4] = 1.4 * neutral
            images.append(render_density(density, paper))
        images = np.stack(images)
        valid = np.ones((count, height, width), bool)
        baseline = np.median(images, axis=0).astype(np.uint8)
        return golden_components.Collection(
            scan=Path("synthetic.png"),
            images=images,
            valid=valid,
            numbers=list(range(1, count + 1)),
            papers=np.repeat(paper[None, :], count, axis=0),
            paper=paper,
            baseline=baseline,
            covered=np.ones((height, width), bool),
            corner=(0, 0),
        ), neutral, coloured

    def test_repeatable_neutral_design_is_kept_without_swallowing_colour(self):
        collection, neutral, coloured = self.make_collection()
        palette = {
            "rays": np.stack([neutral, coloured]),
            "shares": np.array([0.5, 0.5]),
            "count": 2,
            "separation_deg": 30.0,
            "achromatic": np.array([True, False]),
            "achromatic_added": np.array([False, False]),
        }
        palette = golden_components.validate_achromatic_palette(collection, palette)

        support = palette["achromatic_supports"][0]
        self.assertIsNotNone(support)
        self.assertTrue(support[40, 25])
        self.assertFalse(support[10, 60])

        layers, readable = golden_components.colour_layers(collection, palette)
        self.assertGreater(float(np.median(layers[0, :, 30:60, 20:38])), 0.7)
        self.assertLess(float(np.max(layers[0, :, 5:17, 48:74])), 0.05)
        self.assertGreater(float(np.median(layers[1, :, 30:60, 82:100])), 0.5)
        self.assertTrue(readable[1, :, 40, 90].all())

    def test_achromatic_mask_keeps_black_and_suppresses_colour(self):
        height, width = 80, 120
        paper = np.array([205.0, 198.0, 178.0])
        neutral = unit([1.0, 1.0, 1.0])
        coloured = unit([0.95, 0.28, 0.10])
        density = np.zeros((height, width, 3), np.float64)
        density[15:65, 10:48] = 1.0 * neutral
        density[15:65, 72:110] = 1.0 * coloured
        image = render_density(density, paper)

        layer = golden_components.achromatic_layer(
            image, paper, np.ones((height, width), bool), neutral
        )

        black = float(np.median(layer[25:55, 20:38]))
        colour = float(np.median(layer[25:55, 82:100]))
        self.assertGreater(black, 0.8)
        self.assertLess(colour, 0.25 * black)
        self.assertEqual(len(np.unique(layer[layer > 0])), 1)

    def test_achromatic_mask_rejects_faint_neutral_background(self):
        height, width = 80, 140
        paper = np.array([205.0, 198.0, 178.0])
        neutral = unit([1.0, 1.0, 1.0])
        almost_neutral = unit([0.54, 0.63, 0.55])
        density = np.zeros((height, width, 3), np.float64)
        density[10:70, 10:52] = 1.1 * neutral
        density[5:75, 65:135] = 0.35 * almost_neutral
        image = render_density(density, paper)

        layer = golden_components.achromatic_layer(
            image, paper, np.ones((height, width), bool), neutral
        )

        self.assertGreater(float(np.median(layer[20:60, 20:42])), 0.9)
        self.assertEqual(float(np.max(layer[15:65, 75:125])), 0.0)

    def test_blurred_refinement_falls_back_to_averaged_layer(self):
        baseline = np.zeros((100, 120), np.float32)
        baseline[20:80, 25:95] = 1.0
        refined = cv2.GaussianBlur(baseline, (0, 0), 2.0)
        covered = np.ones(baseline.shape, bool)

        chosen, seen, _, before, after, improved = (
            golden_components.select_achromatic_golden(
                refined, baseline, covered, covered, None
            )
        )

        self.assertFalse(improved)
        self.assertLess(after, before)
        np.testing.assert_array_equal(chosen, baseline)
        np.testing.assert_array_equal(seen, covered)

    def test_incomplete_refinement_cannot_hide_missing_coverage(self):
        baseline = np.zeros((100, 120), np.float32)
        baseline[20:80, 25:95] = 1.0
        seen = np.ones(baseline.shape, bool)
        seen[:, :20] = False
        covered = np.ones(baseline.shape, bool)

        chosen, chosen_seen, _, _, _, improved = (
            golden_components.select_achromatic_golden(
                baseline.copy(), baseline, seen, covered, None
            )
        )

        self.assertFalse(improved)
        np.testing.assert_array_equal(chosen, baseline)
        np.testing.assert_array_equal(chosen_seen, covered)

    def test_local_colour_estimates_follow_each_stamp_variant(self):
        count, height, width = 6, 100, 140
        paper = np.array([205.0, 198.0, 178.0])
        mauve = unit([0.43, 0.76, 0.50])
        blue = unit([0.08, 0.50, 0.86])
        images = []
        for index in range(count):
            density = np.zeros((height, width, 3), np.float64)
            density[30:70, 40:100] = 0.35 * (mauve if index < 3 else blue)
            images.append(render_density(density, paper))
        images = np.stack(images)
        valid = np.ones((count, height, width), bool)
        collection = golden_components.Collection(
            scan=Path("variants.png"),
            images=images,
            valid=valid,
            numbers=list(range(count)),
            papers=np.repeat(paper[None, :], count, axis=0),
            paper=paper,
            baseline=np.median(images, axis=0).astype(np.uint8),
            covered=np.ones((height, width), bool),
            corner=(0, 0),
        )

        rays, evidence = golden_components.local_colour_estimates(
            collection, valid, 25.0
        )

        self.assertTrue(np.isfinite(rays).all())
        self.assertTrue((evidence > 200).all())
        self.assertTrue(np.all(rays[:3] @ mauve > rays[:3] @ blue))
        self.assertTrue(np.all(rays[3:] @ blue > rays[3:] @ mauve))

    def test_structure_correlation_distinguishes_duplicate_from_separate_plate(self):
        shape = (180, 220)
        first = np.zeros(shape, np.float32)
        duplicate = np.zeros(shape, np.float32)
        separate = np.zeros(shape, np.float32)
        cv2.rectangle(first, (35, 25), (185, 155), 1.0, 5)
        cv2.rectangle(duplicate, (35, 25), (185, 155), 0.6, 9)
        for x in range(20, 210, 18):
            cv2.line(separate, (x, 15), (x, 165), 1.0, 2)
        valid = np.ones(shape, bool)

        same = golden_components.layer_structure_correlation(
            first, duplicate, valid, 4.0
        )
        different = golden_components.layer_structure_correlation(
            first, separate, valid, 4.0
        )

        self.assertGreater(same, 0.8)
        self.assertLess(different, 0.3)


class ColourPreservationTests(unittest.TestCase):
    @staticmethod
    def layers(strengths):
        height, width = 40, 50
        layers = np.zeros((len(strengths), height, width), np.float32)
        masks = np.ones_like(layers, bool)
        for index, strength in enumerate(strengths):
            layers[index, 8:32, 10:40] = strength
        return layers, masks

    def test_separated_well_preserved_group_has_adaptive_size(self):
        # Seven good impressions deliberately guard against a hidden top-five
        # policy; the weak group is separated by water-loss-like attenuation.
        layers, masks = self.layers([1.0] * 7 + [0.25] * 5)

        selected, scores, boundary = golden_components.colour_preservation_selection(
            layers, masks
        )

        np.testing.assert_array_equal(
            selected, np.array([True] * 7 + [False] * 5)
        )
        self.assertIsNotNone(boundary)
        self.assertTrue(np.allclose(scores[:7], 1.0))

    def test_smooth_quality_gradient_is_not_cut_arbitrarily(self):
        layers, masks = self.layers([1.0, 0.9, 0.8, 0.7])

        selected, _, boundary = golden_components.colour_preservation_selection(
            layers, masks
        )

        self.assertTrue(selected.all())
        self.assertIsNone(boundary)

    def test_single_impression_is_kept_as_best_effort(self):
        layers, masks = self.layers([0.45])

        selected, scores, boundary = golden_components.colour_preservation_selection(
            layers, masks
        )

        np.testing.assert_array_equal(selected, np.array([True]))
        np.testing.assert_allclose(scores, np.array([1.0]))
        self.assertIsNone(boundary)

    def test_single_impression_completes_the_colour_pass(self):
        height, width = 90, 120
        paper = np.array([205.0, 198.0, 178.0])
        neutral = unit([1.0, 1.0, 1.0])
        coloured = unit([0.90, 0.55, 0.12])
        density = np.zeros((height, width, 3), np.float64)
        density[12:78, 8:38] = 1.0 * neutral
        density[25:65, 62:108] = 0.8 * coloured
        image = render_density(density, paper)
        valid = np.ones((1, height, width), bool)
        collection = golden_components.Collection(
            scan=Path("one.png"),
            images=image[None, ...],
            valid=valid,
            numbers=[1],
            papers=paper[None, :],
            paper=paper,
            baseline=image,
            covered=np.ones((height, width), bool),
            corner=(0, 0),
        )
        palette = {
            "rays": np.stack([neutral, coloured]),
            "shares": np.array([0.5, 0.5]),
            "count": 2,
            "separation_deg": 30.0,
            "achromatic": np.array([True, False]),
            "achromatic_added": np.array([False, False]),
            "achromatic_supports": [None, None],
            "achromatic_thresholds": [0.6, None],
            "local_colour_rays": coloured[None, :],
        }

        passes = golden_components.colour_passes(
            collection, palette, golden_components.Settings(iterations=1)
        )

        self.assertEqual(len(passes), 2)
        self.assertGreater(float(passes[1].golden.max()), 0.5)
        self.assertEqual(len(passes[1].comparison_goldens), 1)
        self.assertEqual(
            passes[1].comparison_notes[0],
            "colour from best single stamp #1 only",
        )
        np.testing.assert_allclose(passes[1].comparison_rays[0], coloured, atol=1e-6)

    def test_screen_selection_falls_back_to_stronger_singleton(self):
        values = {
            frozenset({0}): 8.0,
            frozenset({1}): 3.0,
            frozenset({2}): 2.0,
            frozenset({0, 1}): 4.0,
            frozenset({0, 2}): 4.5,
            frozenset({1, 2}): 4.0,
            frozenset({0, 1, 2}): 5.0,
        }

        chosen, trail, alone = golden_components.choose_impressions(
            lambda members: values[frozenset(members)], 3
        )

        self.assertEqual(chosen, [0])
        self.assertEqual(trail[-1], 8.0)
        self.assertEqual(alone, [8.0, 3.0, 2.0])

    def test_shared_screen_corner_follows_colour_evidence(self):
        size = golden_components.background_ink.SCREEN_WINDOW
        fields = [np.zeros((size + 80, size + 100), bool) for _ in range(3)]
        for field in fields:
            field[40 : 40 + size, 70 : 70 + size] = True

        corner = golden_components.shared_screen_corner(fields)

        self.assertEqual(corner, (40, 70))

    def test_screen_geometry_preserves_best_colour_and_only_fills_its_gap(self):
        shape = (20, 24)
        best = np.full(shape, 0.8, np.float32)
        peer = np.full(shape, 0.4, np.float32)
        best_mask = np.ones(shape, bool)
        best_mask[6:12, 8:15] = False
        entries = [
            {"source_index": 0, "coverage": best, "readable": best_mask},
            {"source_index": 1, "coverage": peer, "readable": np.ones(shape, bool)},
        ]

        merged, seen, source = golden_components.best_preserving_colour_merge(
            entries,
            np.zeros((2, 2), np.float32),
            np.array([0, 1]),
            np.array([1.0, 0.6]),
        )

        self.assertEqual(source, 0)
        np.testing.assert_allclose(merged[best_mask], 0.8)
        np.testing.assert_allclose(merged[~best_mask], 0.4)
        self.assertTrue(seen.all())

    def test_background_projection_is_not_counted_twice(self):
        main_ray = unit([0.82, 0.50, 0.28])
        background_ray = unit([0.60, 0.66, 0.45])
        main_amount = np.array([[0.2, 0.8], [1.1, 0.4]], np.float32)
        background_amount = np.array([[0.5, 0.1], [0.3, 0.7]], np.float32)
        along_main = (
            main_amount
            + background_amount * float(background_ray @ main_ray)
        )

        corrected = golden_components.remove_background_projection(
            along_main, background_amount, main_ray, background_ray
        )

        np.testing.assert_allclose(corrected, main_amount, atol=1e-6)


class ScreenCellRegistrationTests(unittest.TestCase):
    def test_resolves_whole_period_and_along_ruling_offsets(self):
        height = width = 320
        pattern = np.zeros((height, width), np.float32)
        random = np.random.default_rng(7)
        for row, y in enumerate(range(18, height - 18, 9)):
            for x in range(18, width - 18, 18):
                jitter = int(random.integers(-3, 4))
                strength = float(random.uniform(0.7, 1.3))
                cv2.circle(pattern, (x + jitter, y), 2, strength, -1)
        pattern = cv2.GaussianBlur(pattern, (0, 0), 0.65)
        expected = np.array([[0.0, 0.0], [5.0, 9.0], [-4.0, -9.0]], np.float32)
        layers = np.stack([
            golden_components.shift_image(pattern, -shift, cv2.INTER_CUBIC)
            for shift in expected
        ])
        masks = np.ones_like(layers, bool)
        phase = np.zeros((len(layers), 2), np.float32)
        wave = np.array([0.0, 1.0 / 9.0], np.float64)

        resolved = golden_components.resolve_screen_cells(
            layers, masks, phase, (32, 32), wave, 12.0, strongest=0
        )

        for name in ("anchor_shifts", "consensus_shifts"):
            relative = resolved[name] - resolved[name][0]
            np.testing.assert_allclose(relative, expected, atol=1.25)

    def test_single_impression_keeps_phase_as_best_effort(self):
        layer = np.zeros((320, 320), np.float32)
        layer[40:280:9, 30:290:17] = 1.0
        layers = layer[None, ...]
        masks = np.ones_like(layers, bool)
        phase = np.array([[0.4, -1.7]], np.float32)

        resolved = golden_components.resolve_screen_cells(
            layers, masks, phase, (32, 32), np.array([0.0, 1.0 / 9.0]),
            12.0, strongest=0,
        )

        self.assertEqual(resolved["pair_count"], 0)
        np.testing.assert_allclose(resolved["anchor_shifts"], phase)
        np.testing.assert_allclose(resolved["consensus_shifts"], phase)

    def test_held_out_score_rejects_a_neighbouring_period(self):
        height = width = 800
        pattern = np.zeros((height, width), np.float32)
        random = np.random.default_rng(17)
        for row, y in enumerate(range(16, height - 16, 9)):
            for x in range(16, width - 16, 18):
                jitter = int(random.integers(-3, 4))
                cv2.circle(pattern, (x + jitter, y), 2, 1.0, -1)
        pattern = cv2.GaussianBlur(pattern, (0, 0), 0.65)
        true_shifts = np.array([[0.0, 0.0], [4.0, 2.0], [-3.0, -2.0]], np.float32)
        layers = np.stack([
            golden_components.shift_image(pattern, -shift, cv2.INTER_CUBIC)
            for shift in true_shifts
        ])
        masks = np.ones_like(layers, bool)
        wrong = true_shifts.copy()
        wrong[2, 1] += 9.0

        correct_score = golden_components.held_out_cell_score(
            layers, masks, true_shifts, (96, 96), 9.0
        )
        wrong_score = golden_components.held_out_cell_score(
            layers, masks, wrong, (96, 96), 9.0
        )

        self.assertGreater(correct_score, wrong_score + 0.05)


if __name__ == "__main__":
    unittest.main()
