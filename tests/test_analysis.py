import json
from pathlib import Path
import random
import tempfile
import unittest

from PIL import Image

from analysis.analyze_perforation import (
    attach_design_clusters,
    circular_summary,
    cluster_diagnostics,
    corner_records_for_stamp,
    draw_design_spacing_change,
    draw_design_spacing_points,
    draw_design_source_chart,
    draw_diagnostics,
    find_inputs,
    kmeans,
    line_intersection,
    load_dataset,
    natural_key,
    order_clusters,
    outlier_indices,
    ALGORITHM_OUTLINE_COLORS,
    split_spacing_count_outliers,
    source_labels,
    source_colors,
)
from perforation import PHASE_COLORS


class AnalysisTests(unittest.TestCase):
    def test_kmeans_finds_two_groups_with_stable_order(self):
        rng = random.Random(11)
        points = [
            (1.0 + rng.uniform(-0.05, 0.05), 2.0 + rng.uniform(-0.05, 0.05))
            for _ in range(20)
        ] + [
            (4.0 + rng.uniform(-0.05, 0.05), 5.0 + rng.uniform(-0.05, 0.05))
            for _ in range(20)
        ]
        _, centers, labels = kmeans(points, 2, seed=3, restarts=12)
        centers, labels = order_clusters(centers, labels)
        self.assertAlmostEqual(centers[0][0], 1.0, delta=0.03)
        self.assertAlmostEqual(centers[1][1], 5.0, delta=0.03)
        self.assertEqual(labels.count(0), 20)
        self.assertEqual(labels.count(1), 20)

    def test_gap_statistic_selects_two_clear_groups(self):
        rng = random.Random(19)
        points = [
            (rng.gauss(-2, 0.15), rng.gauss(-2, 0.15)) for _ in range(35)
        ] + [
            (rng.gauss(2, 0.15), rng.gauss(2, 0.15)) for _ in range(35)
        ]
        _, selected = cluster_diagnostics(points, max_k=4, references=40)
        self.assertEqual(selected, 2)

    def test_single_cluster_diagnostics_have_a_valid_chart(self):
        diagnostics, selected = cluster_diagnostics([(3.0, 4.0)] * 4,
                                                     max_k=6, references=4)
        self.assertEqual(selected, 1)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            output = Path(directory) / "diagnostics.png"
            draw_diagnostics(diagnostics, selected, output)
            with Image.open(output) as image:
                self.assertEqual(image.format, "PNG")
                self.assertGreater(image.width, 100)
                self.assertGreater(image.height, 100)

    def test_design_charts_are_pngs(self):
        rows = [
            {
                "result_file": r"D:\results\1K_perf.json",
                "source": "1K.jpg", "stamp": str(index),
                "design_width_mm": 16.2 + index*0.01,
                "design_height_mm": 22.1 + index*0.02,
                "point": (16.2 + index*0.01, 22.1 + index*0.02),
                "cluster": index % 2,
                "horizontal_design_gap_mm": 2.1 + index*0.03,
                "vertical_design_gap_mm": 2.8 + index*0.02,
            }
            for index in range(1, 5)
        ]
        colors = source_colors([rows[0]["result_file"]])
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            source_output = Path(directory) / "sources.png"
            cluster_output = Path(directory) / "clusters.png"
            points_output = Path(directory) / "points.png"
            draw_design_source_chart(rows, source_output, colors)
            draw_design_spacing_change(rows, 2, cluster_output, colors)
            draw_design_spacing_points(rows, 2, points_output, colors)
            for output in (source_output, cluster_output, points_output):
                with Image.open(output) as image:
                    self.assertEqual(image.format, "PNG")
                    self.assertGreater(image.width, 100)
                    self.assertGreater(image.height, 100)

    def test_non_modal_hole_counts_are_excluded_from_spacing(self):
        rows = [
            {
                "result_file": r"D:\results\1K_perf.json",
                "stamp": str(index), "cluster": 0,
                "frame_holes_across_width": width,
                "frame_holes_across_height": 19,
                "horizontal_design_gap_mm": 1.5,
                "vertical_design_gap_mm": 2.0,
            }
            for index, width in enumerate((14, 14, 15), start=1)
        ]
        inliers, outliers = split_spacing_count_outliers(rows)
        self.assertEqual([row["stamp"] for row in inliers], ["1", "2"])
        self.assertEqual([row["stamp"] for row in outliers], ["3"])
        self.assertEqual(outliers[0]["expected_frame_holes_across_width"], 14)
        self.assertTrue(outliers[0]["spacing_count_outlier"])
        tied = [{**row, "cluster": 1} for row in rows[:1] + rows[2:]]
        tied_inliers, tied_outliers = split_spacing_count_outliers(tied)
        self.assertEqual(tied_inliers, [])
        self.assertEqual(len(tied_outliers), 2)
        self.assertTrue(all(row["expected_frame_holes_across_width"] is None
                            for row in tied_outliers))

    def test_line_intersection(self):
        point = line_intersection(((0, 1), (2, 1)), ((1, 0), (1, 2)))
        self.assertEqual(point, (1.0, 1.0))

    def test_circular_summary_handles_phase_wrap(self):
        mean, resultant = circular_summary([0.98, 0.02, 1.01, -0.01])
        self.assertTrue(mean < 0.03 or mean > 0.97)
        self.assertGreater(resultant, 0.98)

    def test_corner_phase_difference_is_circular(self):
        def side(line, pitch, point):
            return {
                "line_image": line,
                "pitch_px": pitch,
                "holes": [{
                    "point_image": point,
                    "refinement": {"model": "circle"},
                }],
            }

        sides = {
            "top": side(((0, 0), (20, 0)), 10, (9.8, 0)),
            "left": side(((0, 0), (0, 20)), 10, (0, 10.2)),
        }
        records = corner_records_for_stamp("sample.png", "1", sides)
        self.assertEqual(len(records), 1)
        self.assertAlmostEqual(records[0]["horizontal_phase"], 0.98)
        self.assertAlmostEqual(records[0]["vertical_phase"], 0.02)
        self.assertAlmostEqual(records[0]["phase_mismatch"], 0.04)

    def test_input_discovery_and_dataset_loading_are_generic(self):
        payload = {
            "format": "stamp-perforation-results",
            "source": "anything.png",
            "dpi": 100,
            "stamps": [{
                "stamp": 7,
                "summary": {
                    "width_px": 100,
                    "height_px": 200,
                    "horizontal_perforation_per_20mm": 12.25,
                    "vertical_perforation_per_20mm": 12.5,
                },
                "measurement": {
                    "status": "ok",
                    "sides": {
                        name: {"phase": phase}
                        for name, phase in (("top", 1), ("right", 3),
                                            ("bottom", 1), ("left", 2))
                    },
                },
            }],
        }
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / "nested" / "anything_perf.json"
            path.parent.mkdir()
            path.write_text(json.dumps(payload), encoding="utf-8")
            design_path = path.with_name("anything_design.json")
            design_path.write_text(json.dumps({
                "format": "stamp-design-size",
                "source": "anything.png",
                "stamps": [{
                    "stamp": 7,
                    "design_width_mm": 16.25,
                    "design_height_mm": 22.5,
                }],
            }), encoding="utf-8")
            paths = find_inputs([directory])
            dataset = load_dataset(paths)
        self.assertEqual(len(paths), 1)
        self.assertEqual(len(dataset.rows), 1)
        self.assertEqual(dataset.rows[0]["source"], "anything.png")
        self.assertEqual(dataset.rows[0]["point"], (12.25, 12.5))
        self.assertEqual(dataset.rows[0]["worst_edge_algorithm"], 3)
        self.assertEqual(len(dataset.design_rows), 1)
        self.assertEqual(dataset.design_rows[0]["point"], (16.25, 22.5))
        joined = attach_design_clusters(dataset.design_rows, dataset.rows, [1])
        self.assertEqual(joined[0]["cluster"], 1)
        self.assertEqual(joined[0]["frame_holes_across_width"], 17)
        self.assertEqual(joined[0]["frame_holes_across_height"], 33)
        self.assertAlmostEqual(joined[0]["hole_center_width_mm"], 16*20/12.25)
        self.assertAlmostEqual(joined[0]["hole_center_height_mm"], 32*20/12.5)
        self.assertAlmostEqual(joined[0]["horizontal_design_gap_mm"],
                               16*20/12.25 - 16.25)
        self.assertAlmostEqual(joined[0]["vertical_design_gap_mm"], 51.2 - 22.5)

    def test_source_labels_drop_suffix_and_sort_numbers_naturally(self):
        paths = [
            r"D:\results\14K_perf.json",
            r"D:\results\2K_perf.json",
            r"D:\results\1K_perf.json",
        ]
        labels = source_labels(paths)
        ordered = sorted(paths, key=lambda path: natural_key(labels[path]))
        self.assertEqual([labels[path] for path in ordered], ["1K", "2K", "14K"])

    def test_outliers_are_farthest_from_their_assigned_centroid(self):
        rows = [
            {"point": (0.1, 0.0), "result_file": "1K_perf.json", "stamp": "1"},
            {"point": (3.0, 0.0), "result_file": "1K_perf.json", "stamp": "2"},
            {"point": (0.2, 0.0), "result_file": "1K_perf.json", "stamp": "3"},
        ]
        self.assertEqual(outlier_indices(rows, [(0.0, 0.0)], [0, 0, 0], count=2), [1, 2])

    def test_algorithm_outline_colors_match_detector_colors(self):
        converted = {
            phase: f"#{red:02x}{green:02x}{blue:02x}"
            for phase, (blue, green, red) in PHASE_COLORS.items()
        }
        self.assertEqual(ALGORITHM_OUTLINE_COLORS, converted)


if __name__ == "__main__":
    unittest.main()
