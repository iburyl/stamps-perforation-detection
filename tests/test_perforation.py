"""Geometry tests with known hole pitch, valley depth, rotation and damage."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from perforation import (circular_arc_count, draw_measurement, fit_arc_lattice,
                         fit_extreme_arc_lattice,
                         exclude_points_outside_corners,
                         measure_stamp, refine_edge, refine_edge_from_arc_lattice,
                         recover_missing_side, recover_side_by_parallel_scan,
                         recover_side_by_sequential_kmeans_lab,
                         recovery_targets, measure_profile,
                         select_cross_support_candidate)
from segment_stamps import (average_perforation, summary_row, write_perf_json,
                            perforation_label, reconcile_perforation,
                            refine_orientation_from_measurement,
                            _analyze_stamp_measurement)


class PerforationTests(unittest.TestCase):
    def sample(self, angle=0, damaged=False, perforated=True, hinge=False,
               paper_color=(125, 180, 200), collect_trials=False,
               primary_only=False):
        mask = np.zeros((900, 900), np.uint8)
        cv2.rectangle(mask, (250, 170), (650, 730), 255, -1)
        if perforated:
            for i, x in enumerate(range(262, 645, 28)):
                for y in (170, 730):
                    if not damaged or i % 5 != 2:
                        cv2.circle(mask, (x, y), 8, 0, -1)
            for i, y in enumerate(range(186, 725, 32)):
                for x in (250, 650):
                    if not damaged or i % 6 != 2:
                        cv2.circle(mask, (x, y), 8, 0, -1)
        if damaged:
            cv2.line(mask, (235, 370), (665, 580), 0, 6)
        unrotated_mask = mask.copy()
        matrix = cv2.getRotationMatrix2D((450, 450), -angle, 1)
        mask = cv2.warpAffine(mask, matrix, (900, 900), flags=cv2.INTER_LINEAR)
        gray = np.uint8(30+mask.astype(float)*190/255)
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if hinge:
            # A grey hinge is brighter than the normal background threshold,
            # nearly as bright as the paper, and covers most top perforations.
            image = np.full((900, 900, 3), 30, np.uint8)
            cv2.rectangle(image, (290, 150), (645, 205), (175, 175, 175), -1)
            image[unrotated_mask > 0] = paper_color
            image = cv2.warpAffine(image, matrix, (900, 900), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=(30, 30, 30))
        orientation = dict(angle_deg=angle, width_px=400., height_px=560.,
                           center_x=450., center_y=450.)
        return measure_stamp(
            image, orientation, 100, collect_trials=collect_trials,
            primary_only=primary_only,
        )

    def test_diagnostic_run_forces_last_fallback_and_records_it(self):
        forced_trial = {
            'status': 'review',
            'reason': 'forced diagnostic trial',
            'method': 'sequential_kmeans_lab',
            'slope': 0.,
            'intercept': 10.,
            'pitch_px': 20.,
            'geometry_source': 'circle_arcs',
            'refinement_fits': [
                {'model': 'circle', 'rms_px': .1} for _ in range(5)
            ],
        }
        with (patch('perforation.recover_missing_side', return_value=None),
              patch('perforation.recover_side_by_parallel_scan',
                    return_value=None),
              patch('perforation.recover_side_by_sequential_kmeans_lab',
                    return_value=forced_trial) as sequential):
            result = self.sample(collect_trials=True)

        self.assertGreaterEqual(sequential.call_count, 4)
        for side in ('top', 'bottom', 'left', 'right'):
            self.assertIn('cross_support', result['algorithm_trials'][side])
            self.assertNotIn('brightness', result['algorithm_trials'][side])
            primary = result['algorithm_trials'][side]['cross_support']
            self.assertEqual(len(primary['profile_candidates']), 14)
            debug = result['debug_cross_support'][side]
            self.assertEqual(len(debug['profile_candidates']), 14)
            self.assertEqual(set(debug['clusterings']), {3, 4})
            for clusters in (3, 4):
                self.assertEqual(
                    debug['clusterings'][clusters]['centers_lab'].shape,
                    (clusters, 3),
                )
            self.assertIn(primary['profile_map'], {
                'brightness', 'k30', 'k31',
                'k400', 'k401', 'k410', 'k411',
            })
            trial = result['algorithm_trials'][side][
                'sequential_kmeans_lab'
            ]
            self.assertEqual(
                trial['method'], 'sequential_kmeans_lab',
            )
            self.assertEqual(trial['circular_arc_count'], 5)
            self.assertTrue(trial['reliable'])

    def test_known_dimensions_pitch_and_rotation(self):
        for angle in (-6, 0, 5):
            for damaged in (False, True):
                with self.subTest(angle=angle, damaged=damaged):
                    result = self.sample(angle, damaged)
                    self.assertIn(result['status'], ('ok', 'review'))
                    self.assertAlmostEqual(result['valley_width_px'], 384, delta=3)
                    self.assertAlmostEqual(result['valley_height_px'], 544, delta=3)
                    for side, pitch in [('top', 28), ('bottom', 28), ('left', 32), ('right', 32)]:
                        edge = result['sides'][side]
                        self.assertGreaterEqual(edge.get('count', 0), 5)
                        self.assertEqual(edge['method'], 'cross_support')
                        self.assertAlmostEqual(edge['pitch_px'], pitch, delta=0.5)

    def test_primary_method_runs_one_arc_pass_per_side(self):
        with patch('perforation.refine_edge', wraps=refine_edge) as arc_pass:
            result = self.sample()

        self.assertEqual(arc_pass.call_count, 4)
        self.assertEqual(
            {edge['method'] for edge in result['sides'].values()},
            {'cross_support'},
        )

    def test_primary_only_measurement_does_not_run_fallbacks(self):
        with (patch('perforation.adaptive_color_edge') as adaptive,
              patch('perforation.recover_missing_side') as edge_recovery,
              patch('perforation.recover_side_by_parallel_scan') as scan_recovery,
              patch('perforation.recover_side_by_sequential_kmeans_lab') as sequential):
            self.sample(perforated=False, primary_only=True)

        adaptive.assert_not_called()
        edge_recovery.assert_not_called()
        scan_recovery.assert_not_called()
        sequential.assert_not_called()

    def test_cross_support_bonus_is_symmetric_and_pre_arc(self):
        brightness = {
            'map': 'brightness', 'score': 10.0, 'pitch_px': 64.0,
            'line_center': 100.0,
        }
        kmeans = {
            'map': 'k31', 'score': 9.5, 'pitch_px': 64.5,
            'line_center': 103.0,
        }
        unsupported = {
            'map': 'k400', 'score': 10.4, 'pitch_px': 32.0,
            'line_center': 150.0,
        }

        chosen = select_cross_support_candidate(
            [brightness, kmeans, unsupported],
        )

        self.assertIs(chosen, brightness)
        self.assertTrue(brightness['cross_supported'])
        self.assertTrue(kmeans['cross_supported'])
        self.assertFalse(unsupported['cross_supported'])
        self.assertAlmostEqual(brightness['adjusted_score'], 11.0)

    def test_straight_edges_not_perforation(self):
        result = self.sample(perforated=False)
        self.assertEqual(result['status'], 'unavailable')
        self.assertNotIn('valley_width_px', result)

    def test_hinge_behind_colored_paper(self):
        for angle in (-6, 0, 5):
            for color in ((125, 180, 200), (200, 180, 125)):
                with self.subTest(angle=angle, color=color):
                    result = self.sample(angle=angle, hinge=True, paper_color=color)
                    top = result['sides']['top']
                    self.assertIn(top['method'], (
                        'cross_support', 'adaptive_lab_a', 'adaptive_lab_b',
                    ))
                    self.assertGreaterEqual(top['count'], 6)
                    self.assertAlmostEqual(top['pitch_px'], 28, delta=0.5)
                    self.assertAlmostEqual(result['valley_height_px'], 544, delta=3)
                    points = top['points_image'][top['accepted']]
                    # Transform back: the points must lie on paper valley bases,
                    # not on the straight top edge of the hinge (y=150).
                    matrix = cv2.getRotationMatrix2D((450, 450), angle, 1)
                    restored = points @ matrix[:, :2].T + matrix[:, 2]
                    np.testing.assert_allclose(restored[:, 1], 178, atol=3)

    def test_hinge_does_not_invent_perforations_on_straight_paper(self):
        result = self.sample(perforated=False, hinge=True)
        self.assertEqual(result['status'], 'unavailable')

    def test_1k_stamp_3_right_hinge_uses_cross_support(self):
        path = (Path(__file__).resolve().parent / 'fixtures'
                / '1k_stamp_3_hinge.png')
        image = cv2.imread(str(path))
        self.assertIsNotNone(image, f'missing test fixture: {path}')
        orientation = {
            'angle_deg': 0.3758811950683594,
            'width_px': 923.7013009914681,
            'height_px': 1209.7041932449097,
            'center_x': 632.9213353648538,
            'center_y': 778.5192327695399,
        }
        result = measure_stamp(image, orientation, 72.0)
        right = result['sides']['right']
        self.assertEqual(right['method'], 'cross_support')
        self.assertEqual(right['profile_map'], 'k400')
        self.assertEqual(right['profile_ridge'], 'stack')
        self.assertGreaterEqual(circular_arc_count(right), 5)
        self.assertAlmostEqual(right['pitch_px'], 63.3, delta=1.0)

    def test_empty_profile_and_no_orientation(self):
        self.assertEqual(measure_profile(np.arange(100), np.full(100, np.nan), 100)['status'], 'unavailable')
        self.assertEqual(measure_stamp(None, None, 100)['status'], 'unavailable')

    def test_orientation_is_refined_from_consistent_perforation_lines(self):
        orientation = {
            'angle_deg': 0., 'width_px': 400., 'height_px': 560.,
            'center_x': 300., 'center_y': 350., 'corners': np.zeros((4, 2)),
            'status': 'review', 'edge_spread_deg': 2.,
        }
        sides = {}
        for index, (side, angle) in enumerate(zip(
                ('top', 'bottom', 'left', 'right'),
                (-2.0, -2.2, -1.9, 8.0))):
            radians = np.deg2rad(angle + (90 if side in ('left', 'right') else 0))
            direction = np.array([np.cos(radians), np.sin(radians)]) * 100
            sides[side] = {
                'line_image': np.array([[0., 0.], direction]),
                'geometry_source': 'circle_arcs', 'count': 8,
                'status': 'ok', 'method': 'brightness',
            }
        refined = refine_orientation_from_measurement(
            orientation, {'sides': sides}
        )
        self.assertAlmostEqual(refined['angle_deg'], -2.033, delta=0.1)
        self.assertEqual(set(refined['orientation_inlier_sides']),
                         {'top', 'bottom', 'left'})
        self.assertNotIn('right', refined['orientation_inlier_sides'])

    def test_orientation_probe_runs_full_pass_when_primary_is_incomplete(self):
        orientation = {
            'angle_deg': 0., 'width_px': 400., 'height_px': 560.,
            'center_x': 300., 'center_y': 350.,
        }
        primary = {'status': 'review', 'sides': {}}
        final = {'status': 'ok', 'sides': {}}
        consensus = dict(
            orientation,
            orientation_source='perforation_line_consensus',
            orientation_correction_deg=1.,
            status='ok',
        )
        with (patch('segment_stamps.measure_stamp',
                    side_effect=(primary, final)) as measured,
              patch('segment_stamps.refine_orientation_from_measurement',
                    return_value=consensus)):
            _, _, result = _analyze_stamp_measurement(
                np.zeros((10, 10, 3), np.uint8), 0, orientation, 100., False,
            )

        self.assertIs(result, final)
        self.assertTrue(measured.call_args_list[0].kwargs['primary_only'])
        self.assertNotIn('primary_only', measured.call_args_list[1].kwargs)

    def test_orientation_probe_falls_back_to_legacy_coarse_pass_without_consensus(self):
        orientation = {
            'angle_deg': 0., 'width_px': 400., 'height_px': 560.,
            'center_x': 300., 'center_y': 350.,
        }
        primary = {'status': 'review', 'sides': {}}
        recovered = {'status': 'review', 'sides': {}}
        refined = dict(
            orientation,
            orientation_source='perforation_line_consensus',
            orientation_correction_deg=2.,
        )
        final = {'status': 'ok', 'sides': {}}
        with (patch('segment_stamps.measure_stamp',
                    side_effect=(primary, recovered, final)) as measured,
              patch('segment_stamps.refine_orientation_from_measurement',
                    side_effect=(orientation, refined))):
            _, actual_orientation, result = _analyze_stamp_measurement(
                np.zeros((10, 10, 3), np.uint8), 0, orientation, 100., False,
            )

        self.assertIs(actual_orientation, refined)
        self.assertIs(result, final)
        self.assertEqual(measured.call_count, 3)
        self.assertTrue(measured.call_args_list[0].kwargs['primary_only'])
        self.assertNotIn('primary_only', measured.call_args_list[1].kwargs)
        self.assertNotIn('primary_only', measured.call_args_list[2].kwargs)

    def test_per_image_json_contains_summary_and_detailed_hole_ids(self):
        result = self.sample()
        reconcile_perforation(result, ('top', 'bottom'), 1200)
        reconcile_perforation(result, ('left', 'right'), 1200)
        # The system TEMP can be read-only under a Windows sandbox. Keep the
        # test hermetic by creating its disposable output beside the tests.
        with tempfile.TemporaryDirectory(
                dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / 'data_scan_perf.json'
            orientation = dict(angle_deg=0., width_px=400., height_px=560.,
                               center_x=450., center_y=450., corners=np.zeros((4, 2)))
            write_perf_json(path, 'scan.png', [(250, 170, 650, 730)],
                            [orientation], [result], 1200, {})
            payload = json.loads(path.read_text(encoding='utf-8'))
            row = payload['stamps'][0]['summary']
            self.assertEqual(row['source'], 'scan.png')
            self.assertEqual(len(payload['stamps']), 1)
            self.assertAlmostEqual(float(row['width_mm']),
                                   result['valley_width_px'] * 25.4 / 1200, places=3)
            self.assertFalse(any('point' in name for name in row))
            holes = payload['stamps'][0]['measurement']['sides']['top']['holes']
            self.assertEqual([hole['id'] for hole in holes], list(range(1, len(holes) + 1)))

    def test_circular_arc_refinement_has_distinct_drawing_color(self):
        image = np.zeros((40, 40, 3), np.uint8)
        result = {'sides': {'top': {
            'points_image': np.array([[10., 10.], [25., 25.]]),
            'accepted': np.array([True, True]),
            'refinement_fits': [{'model': 'circle'}, None],
            'line_image': np.array([[0., 35.], [39., 35.]]),
        }}}
        draw_measurement(image, result, thickness=2)
        np.testing.assert_array_equal(image[10, 10], [255, 0, 255])
        np.testing.assert_array_equal(image[25, 25], [0, 120, 255])

    def test_rejected_circle_is_hollow_and_does_not_support_side(self):
        image = np.zeros((40, 40, 3), np.uint8)
        edge = {
            'points_image': np.array([[10., 10.], [25., 25.]]),
            'accepted': np.array([True, False]),
            'refinement_fits': [{'model': 'circle'}, {'model': 'circle'}],
            'line_image': np.array([[0., 35.], [39., 35.]]),
        }
        draw_measurement(image, {'sides': {'top': edge}}, thickness=2)
        np.testing.assert_array_equal(image[10, 10], [255, 0, 255])
        np.testing.assert_array_equal(image[25, 25], [0, 0, 0])
        self.assertGreater(int(image[25, 21, 0]), 0)
        self.assertEqual(circular_arc_count(edge), 1)

    def test_arc_outlier_is_excluded_before_lattice_fit(self):
        points = np.array([[20., 10.], [40., 10.], [60., 10.],
                           [80., 10.], [100., 10.], [120., 24.]])
        edge = {'status': 'ok', 'pitch_px': 20., 'amplitude_px': 8.}
        fits = [{'point': point.copy(), 'curve': np.empty((0, 2)),
                 'model': 'circle', 'rms_px': .1} for point in points]
        with patch('perforation.refine_valleys',
                   side_effect=lambda signal, positions, depth, seeds, pitch:
                   [None] * len(seeds)):
            success = refine_edge_from_arc_lattice(
                edge, np.zeros((40, 141)), np.arange(141.),
                np.full(141, 10.), points, fits)
        self.assertTrue(success)
        self.assertEqual(edge['accepted'].sum(), 5)
        self.assertFalse(edge['accepted'][5])
        self.assertAlmostEqual(edge['pitch_px'], 20.)
        self.assertAlmostEqual(edge['slope'], 0.)

    def test_corner_arc_beyond_adjacent_edge_is_excluded(self):
        valleys = np.column_stack([
            np.arange(0., 91., 15.), np.full(7, 10.)
        ])
        top = {
            'status': 'ok', 'line': (0., 10.), 'slope': 0.,
            'intercept': 10., 'pitch_px': 15., 'amplitude_px': 8.,
            'geometry_source': 'circle_arcs', 'valleys': valleys,
            'accepted': np.ones(7, bool),
        }
        sides = {
            'top': top,
            'bottom': {'status': 'unavailable'},
            'left': {'line': (0., 10.)},
            'right': {'line': (0., 80.)},
        }
        contexts = {'top': {'positions': np.arange(101.)}}

        exclude_points_outside_corners(sides, contexts)

        np.testing.assert_array_equal(
            top['accepted'], [False, True, True, True, True, True, False]
        )
        np.testing.assert_array_equal(
            top['inside_corner_bounds'],
            [False, True, True, True, True, True, False],
        )
        self.assertEqual(top['count'], 5)
        self.assertAlmostEqual(top['pitch_px'], 15.)

    def test_bottom_refit_does_not_collapse_vertical_corner_bounds(self):
        def measured_side(line, slope, intercept, valleys, pitch=15.):
            return {
                'line': line, 'slope': slope, 'intercept': intercept,
                'pitch_px': pitch, 'amplitude_px': 8.,
                'geometry_source': 'circle_arcs', 'valleys': valleys,
                'accepted': np.ones(len(valleys), bool),
            }

        horizontal = np.column_stack([
            [0., 20., 35., 50., 65., 80.], np.full(6, 9.),
        ])
        vertical = np.column_stack([
            [10., 25., 40., 55., 70., 85., 90.], np.full(7, 10.),
        ])
        sides = {
            'top': {'line': (0., 10.)},
            # Patch y=90, while side-local inward y=9.  Trimming x=0 forces
            # this flipped side through the refit branch that exposed the bug.
            'bottom': measured_side((0., 90.), 0., 9., horizontal),
            'left': measured_side((0., 10.), 0., 10., vertical),
            'right': measured_side((0., 90.), 0., 9., vertical),
        }
        context = {'positions': np.arange(100.), 'work_shape': (100, 100)}
        contexts = {name: dict(context) for name in sides}

        exclude_points_outside_corners(sides, contexts)

        self.assertEqual(sides['bottom']['accepted'].sum(), 5)
        self.assertEqual(sides['left']['accepted'].sum(), 7)
        self.assertEqual(sides['right']['accepted'].sum(), 7)
        self.assertAlmostEqual(sides['bottom']['line_patch'][1], 90.)

    def test_point_outside_corner_bounds_is_not_drawn(self):
        image = np.zeros((40, 40, 3), np.uint8)
        edge = {
            'points_image': np.array([[10., 10.], [25., 25.]]),
            'accepted': np.array([False, True]),
            'inside_corner_bounds': np.array([False, True]),
            'refinement_fits': [{'model': 'circle'}, {'model': 'circle'}],
            'line_image': np.array([[0., 35.], [39., 35.]]),
        }
        draw_measurement(image, {'sides': {'top': edge}}, thickness=2)
        np.testing.assert_array_equal(image[10, 10], [0, 0, 0])
        np.testing.assert_array_equal(image[25, 25], [255, 0, 255])

    def test_final_gauge_uses_only_arc_lattice_pitch(self):
        measurement = {'sides': {
            'top': {'pitch_px': 20., 'geometry_source': 'initial_peaks'},
            'bottom': {'pitch_px': 25., 'geometry_source': 'circle_arcs', 'count': 5},
            'left': {'pitch_px': 30., 'geometry_source': 'initial_peaks'},
            'right': {},
        }}
        self.assertIsNone(
            average_perforation(measurement, ('top', 'bottom'), 1200)
        )
        self.assertIsNone(
            average_perforation(measurement, ('left', 'right'), 1200)
        )
        self.assertEqual(perforation_label(measurement, 1200), '-- x --')

    def test_opposite_sides_reject_one_false_half_pitch_arc(self):
        top_x = np.array([
            214.20, 243.27, 306.26, 371.21, 437.89, 497.88, 568.78,
            630.48, 695.65, 761.81, 825.14, 889.54, 948.68,
        ])
        top_points = np.column_stack([top_x, np.full(len(top_x), 180.)])
        circle_fits = [
            {'point': point.copy(), 'curve': np.empty((0, 2)),
             'model': 'circle', 'rms_px': .2}
            for point in top_points
        ]
        measurement = {'sides': {
            'top': {
                'pitch_px': 32.2206, 'geometry_source': 'circle_arcs',
                'count': len(top_x), 'slope': 0., 'valleys': top_points,
                'accepted': np.ones(len(top_x), bool),
                'refinement_fits': circle_fits,
            },
            'bottom': {
                'pitch_px': 64.5888, 'geometry_source': 'circle_arcs',
                'count': 7, 'method': 'brightness',
            },
        }}
        gauge = average_perforation(measurement, ('top', 'bottom'), 1200)
        self.assertIsInstance(gauge, float)
        self.assertAlmostEqual(gauge, 14.63, delta=.1)
        self.assertFalse(measurement['sides']['top']['accepted'][0])
        self.assertAlmostEqual(measurement['sides']['top']['pitch_px'], 64.4,
                               delta=1.)

    def test_discordant_final_sides_are_reported_separately(self):
        measurement = {'sides': {
            'top': {'pitch_px': 60., 'geometry_source': 'circle_arcs',
                    'count': 7},
            'bottom': {'pitch_px': 70., 'geometry_source': 'circle_arcs',
                       'count': 7},
            'left': {}, 'right': {},
        }}
        gauges = average_perforation(measurement, ('top', 'bottom'), 1200)
        self.assertIsInstance(gauges, tuple)
        self.assertGreater(abs(gauges[0] - gauges[1]), .25)
        self.assertRegex(perforation_label(measurement, 1200),
                         r'^\([0-9.]+/[0-9.]+\) x --$')

    def test_arc_lattice_recovers_base_period_across_large_gaps(self):
        points = np.array([396.83, 459.65, 841.34, 1033.38, 1161.01])
        fitted = fit_arc_lattice(points, np.arange(140., 1321.))
        self.assertIsNotNone(fitted)
        pitch, _, chosen, lattice = fitted
        self.assertEqual(len(chosen), 5)
        self.assertAlmostEqual(pitch, 63.6, delta=1.0)
        self.assertEqual(len(np.unique(lattice)), 5)

    def test_arc_lattice_prefers_fourteen_intervals_over_shorter_alias(self):
        # Five accepted arcs on 7K #1: the large gap contains nine intervals.
        points = np.array([198.0686, 392.2235, 454.1873, 1008.1788, 1072.4859])
        for reference in (None, 62.36):
            with self.subTest(reference=reference):
                fitted = fit_arc_lattice(points, np.arange(140., 1321.), reference)
                self.assertIsNotNone(fitted)
                pitch, _, _, indices = fitted
                self.assertEqual(int(np.ptp(indices)), 14)
                self.assertAlmostEqual(pitch, 62.11, delta=0.1)

    def test_arc_lattice_uses_every_circle_and_rejects_double_period(self):
        # 1K #11 bottom: a sparse subset supports ~133 px, while all seven
        # circular arcs support the actual ~66 px lattice.
        points = np.array([
            4527.2, 4667.6, 4731.0, 4859.5, 4930.6, 5066.0, 5192.0,
        ])
        fitted = fit_extreme_arc_lattice(points, np.array([4450., 5280.]))
        self.assertIsNotNone(fitted)
        pitch, _, chosen, indices = fitted
        self.assertEqual(len(chosen), len(points))
        self.assertAlmostEqual(pitch, 66.45, delta=.2)
        np.testing.assert_array_equal(indices, [0, 2, 3, 5, 6, 8, 10])

    def test_valid_parity_breaking_circle_is_not_removed(self):
        # 14K #4: omitting the fifth circle permits a false ~130 px period.
        # Since all circles already fit ~65 px, leave-one-out must not fire.
        points = np.array([
            3466.8, 3587.7, 3718.9, 3854.3, 3916.2, 4113.5,
        ])
        fitted = fit_extreme_arc_lattice(points, np.array([3400., 4180.]))
        self.assertIsNotNone(fitted)
        pitch, _, chosen, _ = fitted
        self.assertEqual(len(chosen), len(points))
        self.assertAlmostEqual(pitch, 65.0, delta=.2)

    def test_opposite_side_reconciliation_halves_double_period(self):
        def side(points, pitch):
            valleys = np.column_stack([points, np.full(len(points), 20.)])
            return {
                'pitch_px': pitch, 'geometry_source': 'circle_arcs',
                'count': len(points), 'slope': 0., 'valleys': valleys,
                'accepted': np.ones(len(points), bool),
                'refinement_fits': [
                    {'point': point.copy(), 'curve': np.empty((0, 2)),
                     'model': 'circle', 'rms_px': .2}
                    for point in valleys
                ],
            }

        points = np.array([
            4527.2, 4667.6, 4731.0, 4859.5, 4930.6, 5066.0, 5192.0,
        ])
        measurement = {'sides': {
            'top': side(points, 66.45),
            'bottom': side(points, 132.9),
        }}
        gauge = average_perforation(measurement, ('top', 'bottom'), 1200)
        self.assertIsInstance(gauge, float)
        self.assertAlmostEqual(gauge, 14.22, delta=.1)
        self.assertAlmostEqual(
            measurement['sides']['bottom']['pitch_px'], 66.45, delta=.2,
        )

    def test_discordant_recovered_side_does_not_bias_gauge(self):
        measurement = {'sides': {
            'left': {'pitch_px': 62.36, 'geometry_source': 'circle_arcs',
                     'count': 11, 'status': 'review', 'method': 'brightness'},
            'right': {'pitch_px': 67.82, 'geometry_source': 'circle_arcs',
                      'count': 5, 'status': 'review',
                      'method': 'parallel_edge_arc_recovery',
                      'spacing_rms_px': 5.13},
        }}
        self.assertAlmostEqual(
            average_perforation(measurement, ('left', 'right'), 1200),
            24000 / (25.4 * 62.36), places=6)
        self.assertEqual(perforation_label(measurement, 1200), '-- x 15.15')

    def test_five_arcs_complete_lattice_and_exclude_other_points_from_line(self):
        valleys = np.column_stack([np.arange(20., 101., 20.), np.full(5, 10.)])
        edge = {
            'valleys': valleys,
            'accepted': np.ones(5, bool),
            'pitch_px': 20.,
            'slope': 0.,
            'intercept': 10.,
            'amplitude_px': 8.,
            'autocorrelation': 0.8,
            'status': 'ok',
        }
        def fake_refinement(signal, positions, depth, seeds, pitch):
            if len(seeds) == 7:
                return [
                    ({'point': seed.copy(), 'curve': np.empty((0, 2)),
                      'model': 'circle', 'rms_px': 0.1}
                     if 20 <= seed[0] <= 100 else None)
                    for seed in seeds
                ]
            return [
                {'point': seed.copy(), 'curve': np.empty((0, 2)),
                 'model': 'circle', 'rms_px': 0.1}
                for seed in seeds
            ]

        with patch('perforation.refine_valleys', side_effect=fake_refinement) as mocked:
            refine_edge(edge, np.zeros((30, 121)), np.arange(121.),
                        np.full(121, 10.), )

        self.assertEqual(mocked.call_count, 2)
        first_pass_seeds = mocked.call_args_list[0].args[3]
        np.testing.assert_allclose(first_pass_seeds[:, 0],
                                   [20, 40, 60, 80, 100, 0, 120], atol=1e-9)
        predicted = mocked.call_args_list[1].args[3]
        np.testing.assert_allclose(predicted[:, 0], [0, 120], atol=1e-9)
        self.assertEqual(edge['accepted'].sum(), 7)
        self.assertTrue(all(
            fit is not None and fit['model'] == 'circle'
            for fit, accepted in zip(edge['refinement_fits'], edge['accepted'])
            if accepted
        ))
        self.assertAlmostEqual(edge['pitch_px'], 20.)
        self.assertAlmostEqual(edge['slope'], 0.)

    def test_missing_side_is_recovered_from_parallel_and_adjacent_edges(self):
        def adjacent(side):
            valleys = np.column_stack([np.arange(30., 111., 20.), np.full(5, 10.)])
            return {
                'status': 'ok', 'method': 'brightness', 'line': (0., 10.),
                'slope': 0., 'intercept': 10., 'pitch_px': 20.,
                'geometry_source': 'circle_arcs',
                'valleys': valleys, 'accepted': np.ones(5, bool),
                'refinement_fits': [
                    {'point': point.copy(), 'curve': np.empty((0, 2)),
                     'model': 'circle', 'rms_px': 0.1}
                    for point in valleys
                ],
            }

        positions = np.arange(10., 141.)
        context = {'positions': positions, 'signal': np.zeros((150, 150)),
                   'depth': np.full(len(positions), 10.), 'work_shape': (150, 150)}
        sides = {
            'top': adjacent('top'), 'bottom': adjacent('bottom'),
            'left': {'status': 'unavailable'},
            # No opposite edge: the two adjacent arc-supported sides suffice.
            'right': {'status': 'unavailable'},
        }
        contexts = {name: dict(context) for name in sides}

        def fake_refinement(signal, positions, depth, seeds, pitch):
            if len(seeds) < 5:
                return [None] * len(seeds)
            return [
                {'point': seed.copy(), 'curve': np.empty((0, 2)),
                 'model': 'circle', 'rms_px': 0.1}
                for seed in seeds
            ]

        with patch('perforation.refine_valleys', side_effect=fake_refinement):
            recovered = recover_missing_side('left', sides, contexts)

        self.assertIsNotNone(recovered)
        self.assertEqual(recovered['method'], 'parallel_edge_arc_recovery')
        self.assertEqual(recovered['status'], 'review')
        self.assertGreaterEqual(recovered['count'], 5)
        self.assertAlmostEqual(recovered['slope'], 0.)

    def test_missing_side_is_recovered_from_opposite_and_one_adjacent_edge(self):
        def reliable(valleys):
            return {
                'status': 'ok', 'method': 'brightness', 'line': (0., 10.),
                'slope': 0., 'intercept': 10., 'pitch_px': 20.,
                'geometry_source': 'circle_arcs',
                'valleys': valleys, 'accepted': np.ones(len(valleys), bool),
                'refinement_fits': [
                    {'point': point.copy(), 'curve': np.empty((0, 2)),
                     'model': 'circle', 'rms_px': 0.1}
                    for point in valleys
                ],
            }

        horizontal = np.column_stack([
            np.arange(30., 111., 20.), np.full(5, 10.)
        ])
        vertical = np.column_stack([
            np.arange(30., 111., 20.), np.full(5, 10.)
        ])
        sides = {
            'top': {'status': 'unavailable'},
            'bottom': reliable(horizontal),
            'left': reliable(vertical),
            'right': {'status': 'unavailable'},
        }
        positions = np.arange(10., 141.)
        context = {
            'positions': positions, 'signal': np.zeros((150, 150)),
            'depth': np.full(len(positions), 39.), 'work_shape': (150, 150),
        }
        contexts = {name: dict(context) for name in sides}

        def fake_refinement(signal, positions, depth, seeds, pitch):
            if len(seeds) < 5:
                return [None] * len(seeds)
            return [
                {'point': seed.copy(), 'curve': np.empty((0, 2)),
                 'model': 'circle', 'rms_px': 0.1}
                for seed in seeds
            ]

        with patch('perforation.refine_valleys', side_effect=fake_refinement):
            recovered = recover_missing_side('right', sides, contexts)

        self.assertIsNotNone(recovered)
        self.assertEqual(recovered['method'], 'parallel_edge_arc_recovery')
        self.assertGreaterEqual(recovered['count'], 5)
        self.assertAlmostEqual(recovered['slope'], -sides['left']['line'][0])

    def test_parallel_scan_can_follow_plausible_weak_target_slope(self):
        points = np.column_stack([
            np.arange(20., 121., 20.), np.full(6, 10.),
        ])
        circles = [
            {'point': point.copy(), 'curve': np.empty((0, 2)),
             'model': 'circle', 'rms_px': .1}
            for point in points
        ]
        sides = {
            'top': {
                'line': (0., 10.), 'pitch_px': 20.,
                'geometry_source': 'circle_arcs', 'accepted': np.ones(6, bool),
                'refinement_fits': circles,
            },
            'bottom': {'slope': .03, 'intercept': 8.},
        }
        positions = np.arange(10., 131.)
        context = {
            'positions': positions, 'signal': np.zeros((80, 150)),
            'normal_search': (0, 35), 'expected_normal': 10.,
            'work_shape': (80, 150),
        }

        def fake_refinement(signal, positions, depth, seeds, pitch):
            slope = np.polyfit(positions, depth, 1)[0]
            if abs(slope - .03) > 1e-4:
                return [None] * len(seeds)
            return [
                {'point': seed.copy(), 'curve': np.empty((0, 2)),
                 'model': 'circle', 'rms_px': .1}
                for seed in seeds
            ]

        with patch('perforation.refine_valleys', side_effect=fake_refinement):
            recovered = recover_side_by_parallel_scan(
                'bottom', sides, {'bottom': context}
            )

        self.assertIsNotNone(recovered)
        self.assertAlmostEqual(recovered['slope'], .03, places=4)
        self.assertGreaterEqual(circular_arc_count(recovered), 5)

    def test_arc_poor_and_nonparallel_sides_are_recovery_targets(self):
        def edge(count, slope=0., status='ok'):
            return {
                'status': status,
                'line': (slope, 10.),
                'geometry_source': 'circle_arcs',
                'coverage': 0.8,
                'refinement_fits': [
                    {'model': 'circle', 'rms_px': 0.1}
                    for _ in range(count)
                ],
            }

        sides = {
            'top': edge(7),
            'bottom': edge(4),
            'left': edge(6),
            'right': edge(6),
        }
        self.assertEqual(recovery_targets(sides), {'bottom'})

        # With five magenta points on both sides, a 3-degree disagreement still
        # rebuilds the less strongly supported member, not both sides.
        sides['bottom'] = edge(5, np.tan(np.deg2rad(3)))
        self.assertEqual(recovery_targets(sides), {'bottom'})


if __name__ == '__main__':
    unittest.main()
