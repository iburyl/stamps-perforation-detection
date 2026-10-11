#!/usr/bin/env python3
"""Analyze perforation clusters, frame-versus-line evidence and printed-design sizes."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import colorsys
import csv
from dataclasses import dataclass
import glob
import json
import math
from pathlib import Path
import random
import re
import statistics
import sys

from PIL import Image, ImageDraw, ImageFont


HORIZONTAL = "horizontal_perforation_per_20mm"
VERTICAL = "vertical_perforation_per_20mm"
DESIGN_WIDTH = "design_width_mm"
DESIGN_HEIGHT = "design_height_mm"
HOLE_CENTER_WIDTH = "hole_center_width_mm"
HOLE_CENTER_HEIGHT = "hole_center_height_mm"
HORIZONTAL_DESIGN_GAP = "horizontal_design_gap_mm"
VERTICAL_DESIGN_GAP = "vertical_design_gap_mm"
WIDTH_HOLE_COUNT = "frame_holes_across_width"
HEIGHT_HOLE_COUNT = "frame_holes_across_height"
EXPECTED_WIDTH_HOLE_COUNT = "expected_frame_holes_across_width"
EXPECTED_HEIGHT_HOLE_COUNT = "expected_frame_holes_across_height"
SPACING_COUNT_OUTLIER = "spacing_count_outlier"
SIDES_AT_CORNERS = (
    ("top", "left"),
    ("top", "right"),
    ("bottom", "left"),
    ("bottom", "right"),
)
PASTEL_PALETTE = (
    "#7297c5", "#d88482", "#76ad88", "#a083c5",
    "#d99a65", "#67a9b8", "#8588c4", "#c77d96",
)
ALGORITHM_OUTLINE_COLORS = {
    1: "#00be00",  # green: main method
    2: "#ffff00",  # yellow: first fallback
    3: "#ffa500",  # orange: second fallback
    4: "#ff0000",  # red: last resort
}
ALGORITHM_LEGEND = (
    (1, "Algorithm 1: cross support"),
    (2, "Algorithm 2: adaptive colour"),
    (3, "Algorithm 3: geometric recovery"),
    (4, "Algorithm 4: sequential K-means Lab"),
)
SUMMARY_FIELDS = (
    "result_file", "source", "stamp", "status", "worst_edge_algorithm",
    "width_px", "height_px", "width_mm", "height_mm",
    WIDTH_HOLE_COUNT, HEIGHT_HOLE_COUNT, HOLE_CENTER_WIDTH, HOLE_CENTER_HEIGHT,
    HORIZONTAL, VERTICAL,
)


@dataclass
class Dataset:
    json_paths: list[Path]
    rows: list[dict]
    incomplete: list[dict]
    corner_records: list[dict]
    design_paths: list[Path]
    design_rows: list[dict]
    errors: list[str]
    total_stamps: int


def find_inputs(specifications, recursive=True):
    """Expand files, directories and shell-independent wildcard patterns."""
    paths = []
    for specification in specifications:
        path = Path(specification).expanduser()
        if path.is_dir():
            iterator = path.rglob("*_perf.json") if recursive else path.glob("*_perf.json")
            paths.extend(iterator)
        elif path.is_file():
            paths.append(path)
        else:
            paths.extend(Path(match) for match in glob.glob(str(path), recursive=recursive))
    unique = {}
    for path in paths:
        if path.is_file():
            resolved = path.resolve()
            unique[str(resolved).casefold()] = resolved
    return sorted(unique.values(), key=lambda item: str(item).casefold())


def finite_positive(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def sample_sd(values):
    values = list(values)
    return statistics.stdev(values) if len(values) > 1 else 0.0


def line_intersection(first, second):
    (x1, y1), (x2, y2) = first
    (x3, y3), (x4, y4) = second
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) < 1e-9:
        return None
    return (
        ((x1 * y2 - y1 * x2) * (x3 - x4)
         - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator,
        ((x1 * y2 - y1 * x2) * (y3 - y4)
         - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator,
    )


def refined_hole_points(side):
    points = []
    for hole in side.get("holes", []):
        refinement = hole.get("refinement")
        point = hole.get("point_image")
        if (isinstance(refinement, dict)
                and refinement.get("model") == "circle"
                and isinstance(point, (list, tuple)) and len(point) == 2):
            points.append((float(point[0]), float(point[1])))
    return points


def frame_hole_geometry(valley_dimension_mm, gauge):
    """Infer a frame's whole hole count and extreme-centre span from gauge."""
    valley_dimension_mm = finite_positive(valley_dimension_mm)
    gauge = finite_positive(gauge)
    if valley_dimension_mm is None or gauge is None:
        return None, None
    pitch_mm = 20.0 / gauge
    intervals = max(1, math.ceil(valley_dimension_mm / pitch_mm - 1e-9))
    return intervals + 1, intervals * pitch_mm


def corner_records_for_stamp(source, stamp_id, sides, max_distance=1.25):
    records = []
    for horizontal_name, vertical_name in SIDES_AT_CORNERS:
        horizontal = sides.get(horizontal_name, {})
        vertical = sides.get(vertical_name, {})
        horizontal_pitch = finite_positive(horizontal.get("pitch_px"))
        vertical_pitch = finite_positive(vertical.get("pitch_px"))
        if not (horizontal.get("line_image") and vertical.get("line_image")
                and horizontal_pitch and vertical_pitch):
            continue
        corner = line_intersection(horizontal["line_image"], vertical["line_image"])
        if corner is None:
            continue
        nearest = []
        normalized_distances = []
        for side, pitch in ((horizontal, horizontal_pitch), (vertical, vertical_pitch)):
            candidates = refined_hole_points(side)
            if not candidates:
                break
            point = min(candidates, key=lambda item: math.dist(item, corner))
            nearest.append(point)
            normalized_distances.append(math.dist(point, corner) / pitch)
        if len(normalized_distances) != 2 or max(normalized_distances) > max_distance:
            continue
        horizontal_phase = normalized_distances[0] % 1.0
        vertical_phase = normalized_distances[1] % 1.0
        raw_difference = abs(horizontal_phase - vertical_phase)
        records.append({
            "source": source,
            "stamp": str(stamp_id),
            "corner": f"{horizontal_name}-{vertical_name}",
            "horizontal_phase": horizontal_phase,
            "vertical_phase": vertical_phase,
            "phase_mismatch": min(raw_difference, 1.0 - raw_difference),
            "center_separation": (
                math.dist(nearest[0], nearest[1])
                / math.sqrt(horizontal_pitch * vertical_pitch)
            ),
        })
    return records


def load_dataset(paths):
    rows, incomplete, corners, errors = [], [], [], []
    valid_paths = []
    total_stamps = 0
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("format") != "stamp-perforation-results":
                raise ValueError("unsupported JSON format")
            stamps = payload.get("stamps")
            if not isinstance(stamps, list):
                raise ValueError("'stamps' is not a list")
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
            errors.append(f"{path}: {error}")
            continue
        valid_paths.append(path)
        source = str(payload.get("source") or path.name.removesuffix("_perf.json"))
        dpi = finite_positive(payload.get("dpi"))
        for stamp in stamps:
            total_stamps += 1
            stamp_id = str(stamp.get("stamp", ""))
            summary = stamp.get("summary")
            measurement = stamp.get("measurement") or {}
            sides = measurement.get("sides") or {}
            phases = [
                side.get("phase")
                for side in sides.values()
                if isinstance(side, dict) and isinstance(side.get("phase"), int)
            ] if isinstance(sides, dict) else []
            if not isinstance(summary, dict):
                incomplete.append({"source": source, "stamp": stamp_id,
                                   "reason": "missing summary"})
                continue
            horizontal = finite_positive(summary.get(HORIZONTAL))
            vertical = finite_positive(summary.get(VERTICAL))
            row = {field: summary.get(field) for field in SUMMARY_FIELDS}
            row.update({
                "result_file": str(path),
                "source": source,
                "stamp": stamp_id,
                "status": str(measurement.get("status") or "unknown"),
                "worst_edge_algorithm": max(phases) if phases else None,
            })
            width_mm = finite_positive(summary.get("width_mm"))
            height_mm = finite_positive(summary.get("height_mm"))
            if dpi is not None:
                width_px = (finite_positive(summary.get("width_px"))
                            or finite_positive(measurement.get("valley_width_px")))
                height_px = (finite_positive(summary.get("height_px"))
                             or finite_positive(measurement.get("valley_height_px")))
                width_mm = width_mm or (width_px * 25.4 / dpi if width_px else None)
                height_mm = height_mm or (height_px * 25.4 / dpi if height_px else None)
            width_holes, center_width = frame_hole_geometry(width_mm, horizontal)
            height_holes, center_height = frame_hole_geometry(height_mm, vertical)
            row[WIDTH_HOLE_COUNT] = width_holes
            row[HEIGHT_HOLE_COUNT] = height_holes
            row[HOLE_CENTER_WIDTH] = center_width
            row[HOLE_CENTER_HEIGHT] = center_height
            if horizontal is None or vertical is None:
                incomplete.append({"source": source, "stamp": stamp_id,
                                   "reason": "incomplete gauge pair"})
            else:
                row[HORIZONTAL] = horizontal
                row[VERTICAL] = vertical
                row["point"] = (horizontal, vertical)
                rows.append(row)
            if isinstance(sides, dict):
                corners.extend(corner_records_for_stamp(source, stamp_id, sides))

    design_paths, design_rows = [], []
    for perf_path in valid_paths:
        suffix = "_perf.json"
        if not perf_path.name.casefold().endswith(suffix):
            continue
        design_path = perf_path.with_name(perf_path.name[:-len(suffix)] + "_design.json")
        if not design_path.is_file():
            continue
        try:
            payload = json.loads(design_path.read_text(encoding="utf-8"))
            if payload.get("format") != "stamp-design-size":
                raise ValueError("unsupported design JSON format")
            stamps = payload.get("stamps")
            if not isinstance(stamps, list):
                raise ValueError("'stamps' is not a list")
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
            errors.append(f"{design_path}: {error}")
            continue
        design_paths.append(design_path)
        source = str(payload.get("source") or design_path.name[:-len("_design.json")])
        for stamp in stamps:
            width = finite_positive(stamp.get(DESIGN_WIDTH))
            height = finite_positive(stamp.get(DESIGN_HEIGHT))
            if width is None or height is None:
                continue
            design_rows.append({
                "design_file": str(design_path),
                "result_file": str(perf_path),
                "source": source,
                "stamp": str(stamp.get("stamp", "")),
                "registration_status": str(
                    stamp.get("registration_status") or "unknown"),
                "registration_reasons": "; ".join(
                    str(reason) for reason in
                    (stamp.get("registration_reasons") or [])),
                DESIGN_WIDTH: width,
                DESIGN_HEIGHT: height,
                "point": (width, height),
            })
    return Dataset(valid_paths, rows, incomplete, corners,
                   design_paths, design_rows, errors, total_stamps)


def squared_distance(first, second):
    return sum((left - right) ** 2 for left, right in zip(first, second))


def kmeans(points, k, seed=1729, restarts=32):
    """Deterministic multi-start K-means with k-means++ initialization."""
    unique_count = len(set(points))
    if not 1 <= k <= unique_count:
        raise ValueError(f"cannot fit k={k} to {unique_count} distinct points")
    best = None
    for restart in range(restarts):
        rng = random.Random(seed + restart * 1009)
        centers = [points[rng.randrange(len(points))]]
        while len(centers) < k:
            distances = [min(squared_distance(point, center) for center in centers)
                         for point in points]
            total = sum(distances)
            if total <= 0:
                break
            target = rng.random() * total
            running = 0.0
            for point, distance in zip(points, distances):
                running += distance
                if running >= target:
                    centers.append(point)
                    break
        if len(centers) != k:
            continue
        labels = None
        for _ in range(150):
            new_labels = [min(range(k), key=lambda index: squared_distance(point, centers[index]))
                          for point in points]
            if len(set(new_labels)) != k:
                break
            new_centers = []
            for group in range(k):
                members = [point for point, label in zip(points, new_labels) if label == group]
                new_centers.append(tuple(
                    sum(point[dimension] for point in members) / len(members)
                    for dimension in range(len(points[0]))
                ))
            if (labels == new_labels
                    and max(squared_distance(a, b) for a, b in zip(centers, new_centers)) < 1e-18):
                centers, labels = new_centers, new_labels
                break
            centers, labels = new_centers, new_labels
        if labels is None or len(set(labels)) != k:
            continue
        inertia = sum(squared_distance(point, centers[label])
                      for point, label in zip(points, labels))
        if best is None or inertia < best[0]:
            best = inertia, centers, labels
    if best is None:
        raise ValueError(f"K-means failed for k={k}")
    return best


def order_clusters(centers, labels):
    order = sorted(range(len(centers)), key=lambda index: (sum(centers[index]), centers[index]))
    mapping = {old: new for new, old in enumerate(order)}
    return [centers[index] for index in order], [mapping[label] for label in labels]


def outlier_indices(rows, centers, labels, count=6):
    """Indices farthest from their assigned centroid, in descending order."""
    ranked = sorted(
        range(len(rows)),
        key=lambda index: (
            squared_distance(rows[index]["point"], centers[labels[index]]),
            source_name(rows[index]["result_file"]),
            str(rows[index]["stamp"]),
        ),
        reverse=True,
    )
    return ranked[:min(count, len(ranked))]


def silhouette(points, labels):
    groups = sorted(set(labels))
    if len(groups) < 2:
        return None
    values = []
    for index, point in enumerate(points):
        own = [math.sqrt(squared_distance(point, other))
               for other_index, other in enumerate(points)
               if other_index != index and labels[other_index] == labels[index]]
        if not own:
            values.append(0.0)
            continue
        within = statistics.mean(own)
        nearest_other = min(
            statistics.mean(
                math.sqrt(squared_distance(point, other))
                for other, label in zip(points, labels) if label == group
            )
            for group in groups if group != labels[index]
        )
        scale = max(within, nearest_other)
        values.append((nearest_other - within) / scale if scale else 0.0)
    return statistics.mean(values)


def cluster_diagnostics(points, max_k=6, references=250):
    max_k = min(max_k, len(set(points)), max(1, len(points) - 1))
    observed = []
    for k in range(1, max_k + 1):
        inertia, centers, labels = kmeans(points, k, seed=4000 + k, restarts=48)
        observed.append({
            "k": k,
            "inertia": inertia,
            "centers": centers,
            "labels": labels,
            "silhouette": silhouette(points, labels),
        })
    low = tuple(min(point[dimension] for point in points) for dimension in range(2))
    high = tuple(max(point[dimension] for point in points) for dimension in range(2))
    rng = random.Random(918273)
    reference_logs = [[] for _ in range(max_k)]
    for sample_index in range(references):
        sample = [tuple(rng.uniform(low[d], high[d]) for d in range(2)) for _ in points]
        for k in range(1, max_k + 1):
            inertia, _, _ = kmeans(sample, k,
                                   seed=200000 + sample_index * 17 + k, restarts=6)
            reference_logs[k - 1].append(math.log(max(inertia, 1e-300)))
    for result, logs in zip(observed, reference_logs):
        result["gap"] = statistics.mean(logs) - math.log(max(result["inertia"], 1e-300))
        result["gap_se"] = (statistics.stdev(logs) * math.sqrt(1 + 1 / references)
                            if len(logs) > 1 else 0.0)
    selected = max_k
    for left, right in zip(observed, observed[1:]):
        if left["gap"] >= right["gap"] - right["gap_se"]:
            selected = left["k"]
            break
    return observed, selected


def circular_summary(values):
    angles = [2 * math.pi * (value % 1.0) for value in values]
    cosine = sum(math.cos(angle) for angle in angles)
    sine = sum(math.sin(angle) for angle in angles)
    resultant = math.hypot(cosine, sine) / len(angles)
    mean_phase = (math.atan2(sine, cosine) / (2 * math.pi)) % 1.0
    return mean_phase, resultant


def corner_statistics(records, simulations=10000, bootstraps=5000):
    if not records:
        return None
    horizontal = [row["horizontal_phase"] for row in records]
    vertical = [row["vertical_phase"] for row in records]
    horizontal_mean, horizontal_resultant = circular_summary(horizontal)
    vertical_mean, vertical_resultant = circular_summary(vertical)
    observed_concentration = (len(horizontal) * horizontal_resultant ** 2
                              + len(vertical) * vertical_resultant ** 2)
    phase_groups = defaultdict(list)
    by_stamp = defaultdict(list)
    for row in records:
        stamp_key = (row["source"], row["stamp"])
        phase_groups[(stamp_key, "horizontal")].append(row["horizontal_phase"])
        phase_groups[(stamp_key, "vertical")].append(row["vertical_phase"])
        by_stamp[stamp_key].append(row["phase_mismatch"])
    rng = random.Random(7319921)
    extreme = 0
    for _ in range(simulations):
        simulated = {"horizontal": [], "vertical": []}
        for (_, orientation), values in phase_groups.items():
            shift = rng.random()
            simulated[orientation].extend((value + shift) % 1.0 for value in values)
        _, horizontal_r = circular_summary(simulated["horizontal"])
        _, vertical_r = circular_summary(simulated["vertical"])
        score = len(horizontal) * horizontal_r ** 2 + len(vertical) * vertical_r ** 2
        extreme += score >= observed_concentration
    uniformity_p = (extreme + 1) / (simulations + 1)

    stamp_groups = list(by_stamp.values())
    medians, aligned_rates = [], []
    for _ in range(bootstraps):
        sample = [stamp_groups[rng.randrange(len(stamp_groups))] for _ in stamp_groups]
        values = [value for group in sample for value in group]
        medians.append(statistics.median(values))
        aligned_rates.append(sum(value <= 0.25 for value in values) / len(values))
    medians.sort()
    aligned_rates.sort()

    def interval(values):
        return values[int(0.025 * len(values))], values[max(0, int(0.975 * len(values)) - 1)]

    mismatches = [row["phase_mismatch"] for row in records]
    shared = sum(row["center_separation"] <= 0.25 for row in records)
    if len(records) < 12 or len(by_stamp) < 4:
        classification = "insufficient-data"
    elif (uniformity_p < 0.05
          and horizontal_resultant >= 0.30
          and vertical_resultant >= 0.30):
        classification = "frame"
    elif uniformity_p >= 0.05:
        classification = "linear-compatible"
    else:
        classification = "indeterminate"
    return {
        "classification": classification,
        "corners": len(records),
        "stamps": len(by_stamp),
        "median_mismatch": statistics.median(mismatches),
        "mean_mismatch": statistics.mean(mismatches),
        "median_ci": interval(medians),
        "aligned": sum(value <= 0.25 for value in mismatches),
        "aligned_rate": sum(value <= 0.25 for value in mismatches) / len(mismatches),
        "aligned_ci": interval(aligned_rates),
        "shared": shared,
        "shared_rate": shared / len(records),
        "horizontal_phase_mean": horizontal_mean,
        "vertical_phase_mean": vertical_mean,
        "horizontal_resultant": horizontal_resultant,
        "vertical_resultant": vertical_resultant,
        "uniformity_p": uniformity_p,
    }


def nice_bounds(values, padding=0.07):
    minimum, maximum = min(values), max(values)
    span = maximum - minimum
    pad = max(span * padding, 0.01)
    return minimum - pad, maximum + pad


def quarter_bounds(values):
    """Bounds aligned to 0.25 so every visible quarter-step can be gridded."""
    minimum = math.floor((min(values) - 0.02) * 4) / 4
    maximum = math.ceil((max(values) + 0.02) * 4) / 4
    if maximum <= minimum:
        minimum -= 0.25
        maximum += 0.25
    return minimum, maximum


def load_font(size, bold=False):
    candidates = (
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def source_colors(keys):
    """Return a distinct color for every input JSON, beyond the fixed palette too."""
    colors = {}
    for index, key in enumerate(keys):
        if index < len(PASTEL_PALETTE):
            colors[key] = PASTEL_PALETTE[index]
        else:
            hue = (index * 0.618033988749895) % 1.0
            red, green, blue = colorsys.hsv_to_rgb(hue, 0.42, 0.82)
            colors[key] = f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}"
    return colors


def source_name(key):
    name = Path(key).name
    suffix = "_perf.json"
    return name[:-len(suffix)] if name.casefold().endswith(suffix) else Path(name).stem


def natural_key(value):
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", value)
        if part
    )


def source_labels(keys):
    names = {key: source_name(key) for key in keys}
    basenames = Counter(name.casefold() for name in names.values())
    labels = {}
    for key in keys:
        path = Path(key)
        name = names[key]
        labels[key] = (name if basenames[name.casefold()] == 1
                       else str(Path(path.parent.name) / name))
    return labels


def draw_marker(draw, x, y, color, cluster, radius=7, outline="#202020"):
    """Draw cluster by shape; the point color is reserved for its source JSON."""
    shape = cluster % 6
    if shape == 0:
        draw.ellipse((x-radius, y-radius, x+radius, y+radius),
                     fill=color, outline=outline, width=2)
    elif shape == 1:
        draw.polygon(((x, y-radius-1), (x+radius+1, y),
                      (x, y+radius+1), (x-radius-1, y)),
                     fill=color, outline=outline, width=2)
    elif shape == 2:
        draw.rectangle((x-radius, y-radius, x+radius, y+radius),
                       fill=color, outline=outline, width=2)
    elif shape == 3:
        draw.polygon(((x, y-radius-2), (x+radius+2, y+radius),
                      (x-radius-2, y+radius)), fill=color, outline=outline, width=2)
    elif shape == 4:
        draw.polygon(((x-radius-2, y-radius), (x+radius+2, y-radius),
                      (x, y+radius+2)), fill=color, outline=outline, width=2)
    else:
        draw.line((x-radius, y-radius, x+radius, y+radius), fill=outline, width=7)
        draw.line((x-radius, y+radius, x+radius, y-radius), fill=outline, width=7)
        draw.line((x-radius, y-radius, x+radius, y+radius), fill=color, width=4)
        draw.line((x-radius, y+radius, x+radius, y-radius), fill=color, width=4)


def draw_dashed_line(draw, first, second, fill, width=2, pieces=80):
    for index in range(pieces):
        if index % 2:
            continue
        start = index / pieces
        end = min(1.0, (index + 1) / pieces)
        draw.line((
            first[0] + (second[0]-first[0]) * start,
            first[1] + (second[1]-first[1]) * start,
            first[0] + (second[0]-first[0]) * end,
            first[1] + (second[1]-first[1]) * end,
        ), fill=fill, width=width)


def draw_outlier_labels(draw, rows, centers, labels, screen, font, plot_box, count=6):
    """Label the observations farthest from their assigned cluster centroid."""
    left, top, right, bottom = plot_box
    placed = []
    offsets = (
        (14, -28), (14, 10), (-14, -28), (-14, 10),
        (24, -9), (-24, -9), (10, -48), (-10, 28),
    )

    def overlaps(first, second, padding=4):
        return not (first[2]+padding < second[0] or second[2]+padding < first[0]
                    or first[3]+padding < second[1] or second[3]+padding < first[1])

    for row_index in outlier_indices(rows, centers, labels, count):
        row = rows[row_index]
        point_x, point_y = screen(row["point"])
        text = f"{source_name(row['result_file'])}:{row['stamp']}"
        text_box = draw.textbbox((0, 0), text, font=font)
        text_width = text_box[2]-text_box[0]
        text_height = text_box[3]-text_box[1]
        selected = None
        for offset_x, offset_y in offsets:
            label_left = point_x+offset_x if offset_x >= 0 else point_x+offset_x-text_width-8
            label_top = point_y+offset_y
            candidate = (label_left-4, label_top-3,
                         label_left+text_width+4, label_top+text_height+5)
            if (candidate[0] >= left+3 and candidate[1] >= top+3
                    and candidate[2] <= right-3 and candidate[3] <= bottom-3
                    and not any(overlaps(candidate, other) for other in placed)):
                selected = candidate
                break
        if selected is None:
            label_left = min(max(point_x+14, left+4), right-text_width-12)
            label_top = min(max(point_y-28, top+4), bottom-text_height-8)
            selected = (label_left-4, label_top-3,
                        label_left+text_width+4, label_top+text_height+5)
        else:
            label_left = selected[0]+4
            label_top = selected[1]+3
        placed.append(selected)
        algorithm_color = ALGORITHM_OUTLINE_COLORS.get(
            row.get("worst_edge_algorithm"), "#888888"
        )
        label_center = ((selected[0]+selected[2])/2, (selected[1]+selected[3])/2)
        draw.line((point_x, point_y, *label_center), fill=algorithm_color, width=2)
        draw.rounded_rectangle(selected, radius=4, fill=(255, 255, 255, 235),
                               outline=algorithm_color, width=2)
        draw.text((label_left, label_top), text, font=font, fill="#202020")


def draw_cluster_chart(rows, centers, labels, output, colors=None):
    width, height = 1500, 980
    left, top, right, bottom = 150, 115, 1085, 820
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = load_font(30, True)
    label_font = load_font(22)
    tick_font = load_font(17)
    small_font = load_font(16)
    points = [row["point"] for row in rows]
    x_min, x_max = quarter_bounds([point[0] for point in points]
                                  + [center[0] for center in centers])
    y_min, y_max = quarter_bounds([point[1] for point in points]
                                  + [center[1] for center in centers])

    def screen(point):
        return (left + (point[0] - x_min) * (right - left) / (x_max - x_min),
                bottom - (point[1] - y_min) * (bottom - top) / (y_max - y_min))

    draw.text((width/2, 43), f"Perforation-gauge clusters, k={len(centers)}",
              anchor="mm", font=title_font, fill="#202020")
    draw.text((width/2, 82),
              f"n={len(rows)}; fill = source, shape = cluster, outline = edge-detection algorithm",
              anchor="mm", font=small_font, fill="#555555")
    draw.rectangle((left, top, right, bottom), outline="#303030", width=2)

    for quarter in range(round(x_min * 4), round(x_max * 4) + 1):
        x_value = quarter / 4
        x, _ = screen((x_value, y_min))
        draw.line((x, top, x, bottom), fill="#dedede", width=1)
        draw.text((x, bottom+12), f"{x_value:.2f}", anchor="mt",
                  font=tick_font, fill="#303030")
    for quarter in range(round(y_min * 4), round(y_max * 4) + 1):
        y_value = quarter / 4
        _, y = screen((x_min, y_value))
        draw.line((left, y, right, y), fill="#dedede", width=1)
        draw.text((left-12, y), f"{y_value:.2f}", anchor="rm",
                  font=tick_font, fill="#303030")

    # Restore the old two-cluster nearest-centroid boundary when k=2.
    if len(centers) == 2:
        first, second = centers
        dx, dy = second[0]-first[0], second[1]-first[1]
        rhs = (second[0]**2 + second[1]**2 - first[0]**2 - first[1]**2) / 2
        intersections = []
        if abs(dy) > 1e-12:
            for x_value in (x_min, x_max):
                y_value = (rhs - dx*x_value) / dy
                if y_min <= y_value <= y_max:
                    intersections.append((x_value, y_value))
        if abs(dx) > 1e-12:
            for y_value in (y_min, y_max):
                x_value = (rhs - dy*y_value) / dx
                if x_min <= x_value <= x_max:
                    intersections.append((x_value, y_value))
        if len(intersections) >= 2:
            pair = max(((a, b) for a in intersections for b in intersections),
                       key=lambda item: squared_distance(*item))
            draw_dashed_line(draw, screen(pair[0]), screen(pair[1]), "#555555", 2)

    sources = sorted({row["result_file"] for row in rows},
                     key=lambda key: natural_key(source_name(key)))
    colors = colors or source_colors(sources)
    display_names = source_labels(sources)
    for row, label in zip(rows, labels):
        x, y = screen(row["point"])
        outline = ALGORITHM_OUTLINE_COLORS.get(
            row.get("worst_edge_algorithm"), "#888888"
        )
        draw_marker(draw, x, y, colors[row["result_file"]], label, outline=outline)
    for index, center in enumerate(centers):
        x, y = screen(center)
        draw.ellipse((x-15, y-15, x+15, y+15), fill="white", outline="black", width=3)
        draw.line((x-10, y-10, x+10, y+10), fill="black", width=5)
        draw.line((x-10, y+10, x+10, y-10), fill="black", width=5)
        draw.text((x+17, y-17), f"C{index+1}", font=small_font, fill="#202020")

    draw_outlier_labels(draw, rows, centers, labels, screen, small_font,
                        (left, top, right, bottom), count=6)

    draw.text(((left+right)/2, 900), "Horizontal gauge (holes per 20 mm)",
              anchor="mm", font=label_font, fill="#202020")
    vertical = Image.new("RGBA", (500, 42), (255, 255, 255, 0))
    ImageDraw.Draw(vertical).text((250, 21), "Vertical gauge (holes per 20 mm)",
                                  anchor="mm", font=label_font, fill="#202020")
    vertical = vertical.rotate(90, expand=True)
    image.paste(vertical, (25, 215), vertical)

    legend_x, legend_y = 1130, 130
    draw.text((legend_x, legend_y-35), "Source", font=label_font, fill="#202020")
    for index, source in enumerate(sources):
        y = legend_y + index*37
        draw.ellipse((legend_x, y-7, legend_x+14, y+7),
                     fill=colors[source], outline="#777777")
        draw.text((legend_x+25, y), display_names[source], anchor="lm",
                  font=tick_font, fill="#202020")
    cluster_y = legend_y + len(sources)*37 + 30
    draw.text((legend_x, cluster_y), "Cluster", font=label_font, fill="#202020")
    counts = Counter(labels)
    for index in range(len(centers)):
        y = cluster_y + 40 + index*39
        draw_marker(draw, legend_x+7, y, "#b0b0b0", index, outline="#777777")
        draw.text((legend_x+25, y), f"C{index+1}, n={counts[index]}", anchor="lm",
                  font=tick_font, fill="#202020")
    phase_y = cluster_y + 40 + len(centers)*39 + 28
    draw.text((legend_x, phase_y), "Edge detection", font=label_font, fill="#202020")
    for index, (algorithm, name) in enumerate(ALGORITHM_LEGEND):
        y = phase_y + 40 + index*35
        draw.ellipse((legend_x, y-7, legend_x+14, y+7), fill="white",
                     outline=ALGORITHM_OUTLINE_COLORS[algorithm], width=3)
        draw.text((legend_x+25, y), name, anchor="lm", font=small_font, fill="#202020")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG")


def attach_design_clusters(design_rows, perforation_rows, labels):
    assignments = {
        (row["result_file"].casefold(), str(row["stamp"])): (row, label)
        for row, label in zip(perforation_rows, labels)
    }
    joined = []
    for row in design_rows:
        match = assignments.get((row["result_file"].casefold(), str(row["stamp"])))
        if match is None:
            continue
        perforation, label = match
        item = {**row, "cluster": label}
        for field in (WIDTH_HOLE_COUNT, HEIGHT_HOLE_COUNT,
                      HOLE_CENTER_WIDTH, HOLE_CENTER_HEIGHT):
            item[field] = perforation.get(field)
        if item.get(HOLE_CENTER_WIDTH) is not None:
            item[HORIZONTAL_DESIGN_GAP] = (
                item[HOLE_CENTER_WIDTH] - item[DESIGN_WIDTH]
            )
        if item.get(HOLE_CENTER_HEIGHT) is not None:
            item[VERTICAL_DESIGN_GAP] = (
                item[HOLE_CENTER_HEIGHT] - item[DESIGN_HEIGHT]
            )
        joined.append(item)
    return joined


def usable_design_spacing_rows(rows):
    return [
        row for row in rows
        if row.get(HORIZONTAL_DESIGN_GAP) is not None
        and row.get(VERTICAL_DESIGN_GAP) is not None
        and math.isfinite(row[HORIZONTAL_DESIGN_GAP])
        and math.isfinite(row[VERTICAL_DESIGN_GAP])
    ]


def split_spacing_count_outliers(rows):
    """Keep the unique modal hole-count pair in each Source/cluster group."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["result_file"], row["cluster"])].append(row)
    inliers, outliers = [], []
    for members in grouped.values():
        frequencies = Counter(
            (row[WIDTH_HOLE_COUNT], row[HEIGHT_HOLE_COUNT]) for row in members
        )
        highest = max(frequencies.values())
        modes = sorted(pair for pair, count in frequencies.items() if count == highest)
        expected = modes[0] if len(modes) == 1 else None
        for row in members:
            pair = (row[WIDTH_HOLE_COUNT], row[HEIGHT_HOLE_COUNT])
            item = {
                **row,
                EXPECTED_WIDTH_HOLE_COUNT: expected[0] if expected else None,
                EXPECTED_HEIGHT_HOLE_COUNT: expected[1] if expected else None,
                SPACING_COUNT_OUTLIER: expected is None or pair != expected,
            }
            if item[SPACING_COUNT_OUTLIER]:
                item["spacing_count_outlier_reason"] = (
                    "no unique modal hole count" if expected is None
                    else "non-modal hole count"
                )
                outliers.append(item)
            else:
                inliers.append(item)
    sort_key = lambda row: (
        natural_key(source_name(row["result_file"])), row["cluster"],
        natural_key(str(row["stamp"])),
    )
    return sorted(inliers, key=sort_key), sorted(outliers, key=sort_key)


def design_spacing_summaries(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["result_file"], row["cluster"])].append(row)
    return [
        {
            "result_file": source,
            "source": source_name(source),
            "cluster": cluster,
            "count": len(members),
            "mean_horizontal_gap_mm": statistics.mean(
                row[HORIZONTAL_DESIGN_GAP] for row in members
            ),
            "mean_vertical_gap_mm": statistics.mean(
                row[VERTICAL_DESIGN_GAP] for row in members
            ),
            "width_hole_count": members[0][WIDTH_HOLE_COUNT],
            "height_hole_count": members[0][HEIGHT_HOLE_COUNT],
        }
        for (source, cluster), members in sorted(
            grouped.items(),
            key=lambda item: (natural_key(source_name(item[0][0])), item[0][1]),
        )
    ]


def design_spacing_changes(summaries):
    by_source = defaultdict(list)
    for summary in summaries:
        by_source[summary["result_file"]].append(summary)
    changes = []
    for source in sorted(by_source, key=lambda key: natural_key(source_name(key))):
        ordered = sorted(by_source[source], key=lambda item: item["cluster"])
        for first, second in zip(ordered, ordered[1:]):
            changes.append({
                "source": source_name(source),
                "from_cluster": first["cluster"],
                "to_cluster": second["cluster"],
                "delta_horizontal_gap_mm": (
                    second["mean_horizontal_gap_mm"] - first["mean_horizontal_gap_mm"]
                ),
                "delta_vertical_gap_mm": (
                    second["mean_vertical_gap_mm"] - first["mean_vertical_gap_mm"]
                ),
            })
    return changes


def draw_design_source_chart(rows, output, colors):
    width, height = 1400, 920
    left, top, right, bottom = 150, 115, 1035, 785
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = load_font(30, True)
    label_font = load_font(22)
    tick_font = load_font(17)
    small_font = load_font(16)
    x_min, x_max = nice_bounds([row[DESIGN_WIDTH] for row in rows], 0.08)
    y_min, y_max = nice_bounds([row[DESIGN_HEIGHT] for row in rows], 0.08)

    def screen(point):
        return (left + (point[0]-x_min)*(right-left)/(x_max-x_min),
                bottom - (point[1]-y_min)*(bottom-top)/(y_max-y_min))

    draw.text((width/2, 43), "Printed-design dimensions by source",
              anchor="mm", font=title_font, fill="#202020")
    draw.text((width/2, 82), f"n={len(rows)} measured designs; crosses = source means",
              anchor="mm", font=small_font, fill="#555555")
    draw.rectangle((left, top, right, bottom), outline="#303030", width=2)
    for index in range(6):
        x_value = x_min + (x_max-x_min)*index/5
        y_value = y_min + (y_max-y_min)*index/5
        x, _ = screen((x_value, y_min))
        _, y = screen((x_min, y_value))
        draw.line((x, top, x, bottom), fill="#dedede")
        draw.line((left, y, right, y), fill="#dedede")
        draw.text((x, bottom+12), f"{x_value:.2f}", anchor="mt",
                  font=tick_font, fill="#303030")
        draw.text((left-12, y), f"{y_value:.2f}", anchor="rm",
                  font=tick_font, fill="#303030")

    sources = sorted({row["result_file"] for row in rows},
                     key=lambda key: natural_key(source_name(key)))
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["result_file"]].append(row)
        x, y = screen(row["point"])
        draw.ellipse((x-6, y-6, x+6, y+6), fill=colors[row["result_file"]],
                     outline="#555555", width=1)
    means = {}
    for source in sources:
        members = grouped[source]
        mean = (statistics.mean(row[DESIGN_WIDTH] for row in members),
                statistics.mean(row[DESIGN_HEIGHT] for row in members))
        means[source] = mean
        x, y = screen(mean)
        draw.ellipse((x-12, y-12, x+12, y+12), fill="white",
                     outline=colors[source], width=3)
        draw.line((x-8, y-8, x+8, y+8), fill=colors[source], width=4)
        draw.line((x-8, y+8, x+8, y-8), fill=colors[source], width=4)

    draw.text(((left+right)/2, 875), "Design width (mm)", anchor="mm",
              font=label_font, fill="#202020")
    vertical = Image.new("RGBA", (330, 40), (255, 255, 255, 0))
    ImageDraw.Draw(vertical).text((165, 20), "Design height (mm)", anchor="mm",
                                  font=label_font, fill="#202020")
    vertical = vertical.rotate(90, expand=True)
    image.paste(vertical, (35, 285), vertical)
    legend_x, legend_y = 1085, 125
    draw.text((legend_x, legend_y-35), "Source", font=label_font, fill="#202020")
    for index, source in enumerate(sources):
        y = legend_y + index*48
        draw.ellipse((legend_x, y-7, legend_x+14, y+7),
                     fill=colors[source], outline="#777777")
        draw.text((legend_x+25, y), source_name(source), anchor="lm",
                  font=tick_font, fill="#202020")
        mean = means[source]
        draw.text((legend_x+25, y+19),
                  f"mean {mean[0]:.3f} × {mean[1]:.3f} mm; n={len(grouped[source])}",
                  anchor="lm", font=small_font, fill="#555555")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG")


def draw_design_spacing_change(rows, cluster_count, output, colors):
    """Plot estimated edge-to-edge design gaps by Source and perforation cluster."""
    sources = sorted({row["result_file"] for row in rows},
                     key=lambda key: natural_key(source_name(key)))
    width = 1760
    top, row_step = 175, 66
    bottom = top + max(1, len(sources)-1)*row_step + 35
    height = bottom + 145
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = load_font(30, True)
    label_font = load_font(20)
    tick_font = load_font(16)
    small_font = load_font(15)
    draw.text((width/2, 42), "Estimated spacing between neighboring designs",
              anchor="mm", font=title_font, fill="#202020")
    draw.text((width/2, 82),
              "Hole-count-consistent means; lines show within-Source cluster change",
              anchor="mm", font=small_font, fill="#555555")

    panels = (
        (HORIZONTAL_DESIGN_GAP, "Horizontal gap (mm)", 245, 915),
        (VERTICAL_DESIGN_GAP, "Vertical gap (mm)", 1035, 1705),
    )
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["result_file"]][row["cluster"]].append(row)

    for source_index, source in enumerate(sources):
        y = top + source_index*row_step
        draw.text((225, y), source_name(source), anchor="rm",
                  font=label_font, fill="#202020")

    for field, panel_title, left, right in panels:
        values = [
            statistics.mean(row[field] for row in members if row.get(field) is not None)
            for clusters in grouped.values()
            for members in clusters.values()
            if any(row.get(field) is not None for row in members)
        ]
        x_min, x_max = nice_bounds(values, 0.12)

        def screen_x(value):
            return left + (value-x_min)*(right-left)/(x_max-x_min)

        draw.text(((left+right)/2, 128), panel_title, anchor="mm",
                  font=label_font, fill="#202020")
        draw.rectangle((left, top-30, right, bottom), outline="#505050", width=2)
        for tick in range(6):
            value = x_min + (x_max-x_min)*tick/5
            x = screen_x(value)
            draw.line((x, top-30, x, bottom), fill="#dedede")
            draw.text((x, bottom+12), f"{value:.2f}", anchor="mt",
                      font=tick_font, fill="#303030")
        for source_index, source in enumerate(sources):
            y = top + source_index*row_step
            draw.line((left, y, right, y), fill="#ededed")
            means = []
            for cluster, members in sorted(grouped[source].items()):
                usable = [row[field] for row in members if row.get(field) is not None]
                if usable:
                    means.append((cluster, statistics.mean(usable), len(usable)))
            for (_, first, _), (_, second, _) in zip(means, means[1:]):
                draw.line((screen_x(first), y, screen_x(second), y),
                          fill=colors[source], width=4)
            for cluster, mean, count in means:
                x = screen_x(mean)
                draw_marker(draw, x, y, colors[source], cluster,
                            radius=10, outline="#333333")
                label_y = y-15 if cluster % 2 == 0 else y+15
                draw.text((x, label_y), f"{mean:.2f}", anchor="ms" if cluster % 2 == 0 else "ma",
                          font=small_font, fill="#333333")

    legend_y = height-42
    legend_width = cluster_count*150
    start_x = (width-legend_width)/2
    for cluster in range(cluster_count):
        x = start_x + cluster*150
        draw_marker(draw, x, legend_y, "#b0b0b0", cluster,
                    radius=7, outline="#555555")
        draw.text((x+18, legend_y), f"Cluster C{cluster+1}", anchor="lm",
                  font=small_font, fill="#202020")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG")


def draw_design_spacing_points(rows, cluster_count, output, colors):
    """Plot every count-consistent stamp as a jittered spacing observation."""
    sources = sorted({row["result_file"] for row in rows},
                     key=lambda key: natural_key(source_name(key)))
    width = 1760
    top, row_step = 175, 66
    bottom = top + max(1, len(sources)-1)*row_step + 35
    height = bottom + 145
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = load_font(30, True)
    label_font = load_font(20)
    tick_font = load_font(16)
    small_font = load_font(15)
    draw.text((width/2, 42), "Per-stamp estimated spacing between neighboring designs",
              anchor="mm", font=title_font, fill="#202020")
    draw.text((width/2, 82),
              f"n={len(rows)} count-consistent stamps; vertical jitter is visual only",
              anchor="mm", font=small_font, fill="#555555")

    panels = (
        (HORIZONTAL_DESIGN_GAP, "Horizontal gap (mm)", 245, 915),
        (VERTICAL_DESIGN_GAP, "Vertical gap (mm)", 1035, 1705),
    )
    grouped = defaultdict(list)
    jitter = {}
    for source in sources:
        members = sorted(
            (row for row in rows if row["result_file"] == source),
            key=lambda row: natural_key(str(row["stamp"])),
        )
        grouped[source] = members
        for index, row in enumerate(members):
            jitter[(source, str(row["stamp"]), row["cluster"])] = (index*11 % 31) - 15

    for source_index, source in enumerate(sources):
        y = top + source_index*row_step
        draw.text((225, y), source_name(source), anchor="rm",
                  font=label_font, fill="#202020")

    for field, panel_title, left, right in panels:
        values = [row[field] for row in rows]
        x_min, x_max = nice_bounds(values, 0.07)

        def screen_x(value):
            return left + (value-x_min)*(right-left)/(x_max-x_min)

        draw.text(((left+right)/2, 128), panel_title, anchor="mm",
                  font=label_font, fill="#202020")
        draw.rectangle((left, top-30, right, bottom), outline="#505050", width=2)
        for tick in range(6):
            value = x_min + (x_max-x_min)*tick/5
            x = screen_x(value)
            draw.line((x, top-30, x, bottom), fill="#dedede")
            draw.text((x, bottom+12), f"{value:.2f}", anchor="mt",
                      font=tick_font, fill="#303030")
        for source_index in range(len(sources)-1):
            separator_y = top + (source_index+0.5)*row_step
            draw.line((left, separator_y, right, separator_y), fill="#dddddd")
        for source_index, source in enumerate(sources):
            center_y = top + source_index*row_step
            for row in grouped[source]:
                y = center_y + jitter[(source, str(row["stamp"]), row["cluster"])]
                draw_marker(draw, screen_x(row[field]), y, colors[source],
                            row["cluster"], radius=6, outline="#333333")

    legend_y = height-42
    legend_width = cluster_count*150
    start_x = (width-legend_width)/2
    for cluster in range(cluster_count):
        x = start_x + cluster*150
        draw_marker(draw, x, legend_y, "#b0b0b0", cluster,
                    radius=7, outline="#555555")
        draw.text((x+18, legend_y), f"Cluster C{cluster+1}", anchor="lm",
                  font=small_font, fill="#202020")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG")


def draw_diagnostics(results, selected, output):
    width, height = 1320, 700
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = load_font(28, True)
    label_font = load_font(20)
    tick_font = load_font(16)
    draw.text((width/2, 38), "Cluster-count diagnostics",
              anchor="mm", font=title_font, fill="#202020")

    def panel(box, values, errors, title, note):
        left, top, right, bottom = box
        present = [(value, error) for value, error in zip(values, errors) if value is not None]
        if not present:
            draw.text(((left+right)/2, top-38), title, anchor="mm",
                      font=label_font, fill="#202020")
            draw.rectangle(box, fill="#f9fafb", outline="#303030", width=2)
            draw.text(((left+right)/2, (top+bottom)/2), "not defined for k=1",
                      anchor="mm", font=tick_font, fill="#555555")
            return
        y_min, y_max = nice_bounds([value-error for value, error in present]
                                   + [value+error for value, error in present], 0.12)
        draw.rectangle(box, outline="#303030", width=2)
        draw.text(((left+right)/2, top-38), title, anchor="mm",
                  font=label_font, fill="#202020")

        def screen(index, value):
            x = left + index * (right-left) / max(1, len(values)-1)
            y = bottom - (value-y_min) * (bottom-top) / (y_max-y_min)
            return x, y

        for index in range(5):
            value = y_min + (y_max-y_min)*index/4
            _, y = screen(0, value)
            draw.line((left, y, right, y), fill="#e5e5e5")
            draw.text((left-10, y), f"{value:.2f}", anchor="rm",
                      font=tick_font, fill="#333333")
        selected_x, _ = screen(selected-1, y_min)
        draw.rectangle((selected_x-25, top, selected_x+25, bottom),
                       fill=(44, 160, 44, 20))
        previous = None
        for index, (value, error) in enumerate(zip(values, errors)):
            if value is None:
                continue
            x, y = screen(index, value)
            if previous:
                draw.line((*previous, x, y), fill="#2563eb", width=3)
            if error:
                _, low_y = screen(index, value-error)
                _, high_y = screen(index, value+error)
                draw.line((x, low_y, x, high_y), fill="#2563eb", width=2)
                draw.line((x-6, low_y, x+6, low_y), fill="#2563eb", width=2)
                draw.line((x-6, high_y, x+6, high_y), fill="#2563eb", width=2)
            color = "#16a34a" if index+1 == selected else "#2563eb"
            draw.ellipse((x-6, y-6, x+6, y+6), fill=color)
            draw.text((x, bottom+13), str(index+1), anchor="mt",
                      font=tick_font, fill="#303030")
            previous = (x, y)
        draw.text(((left+right)/2, bottom+48), "k", anchor="mm",
                  font=label_font, fill="#202020")
        draw.text((left+8, top+8), note, font=tick_font, fill="#555555")

    panel((100, 125, 620, 585), [result["gap"] for result in results],
          [result["gap_se"] for result in results], "Gap statistic (±1 SE)", "higher is better")
    panel((760, 125, 1280, 585), [result["silhouette"] for result in results],
          [0.0] * len(results), "Mean silhouette", "higher is better")
    draw.text((width/2, 665), f"Selected k={selected} by the Gap 1-SE rule",
              anchor="mm", font=label_font, fill="#15803d")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG")


def draw_corner_chart(records, stats, output):
    width, height = 1180, 850
    left, top, right, bottom = 150, 110, 820, 755
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = load_font(28, True)
    label_font = load_font(20)
    tick_font = load_font(16)
    draw.text((width/2, 40), "Corner-hole phase consistency",
              anchor="mm", font=title_font, fill="#202020")
    draw.text((width/2, 77), "Nearest fitted hole modulo one side pitch",
              anchor="mm", font=tick_font, fill="#555555")

    def screen(x, y):
        return left + x * (right-left), bottom - y * (bottom-top)

    draw.polygon((screen(0, 0), screen(.25, 0), screen(1, .75),
                  screen(1, 1), screen(.75, 1), screen(0, .25)),
                 fill="#e5e7eb")
    draw.polygon((screen(0, 1), screen(.25, 1), screen(0, .75)), fill="#e5e7eb")
    draw.polygon((screen(1, 0), screen(.75, 0), screen(1, .25)), fill="#e5e7eb")
    draw.rectangle((left, top, right, bottom), outline="#303030", width=2)
    for index in range(6):
        value = index / 5
        x, _ = screen(value, 0)
        _, y = screen(0, value)
        draw.line((x, top, x, bottom), fill="#d1d5db")
        draw.line((left, y, right, y), fill="#d1d5db")
        draw.text((x, bottom+12), f"{value:.1f}", anchor="mt",
                  font=tick_font, fill="#303030")
        draw.text((left-12, y), f"{value:.1f}", anchor="rm",
                  font=tick_font, fill="#303030")
    draw.line((*screen(0, 0), *screen(1, 1)), fill="#4b5563", width=3)
    for row in records:
        x, y = screen(row["horizontal_phase"], row["vertical_phase"])
        draw.ellipse((x-4, y-4, x+4, y+4), fill="#2563eb", outline="white")
    mean_x, mean_y = screen(stats["horizontal_phase_mean"], stats["vertical_phase_mean"])
    draw_dashed_line(draw, (mean_x, top), (mean_x, bottom), "#111111")
    draw_dashed_line(draw, (left, mean_y), (right, mean_y), "#111111")
    draw.text(((left+right)/2, 825), "Horizontal-side phase",
              anchor="mm", font=label_font, fill="#202020")
    vertical = Image.new("RGBA", (360, 35), (255, 255, 255, 0))
    ImageDraw.Draw(vertical).text((180, 18), "Vertical-side phase",
                                  anchor="mm", font=label_font, fill="#202020")
    vertical = vertical.rotate(90, expand=True)
    image.paste(vertical, (35, 255), vertical)
    summary_x = 875
    draw.text((summary_x, 125), "Summary", font=label_font, fill="#202020")
    lines = (
        f"Corners: {stats['corners']}",
        f"Stamps: {stats['stamps']}",
        f"Median |Δ|: {stats['median_mismatch']:.3f}",
        f"Within ±0.25: {stats['aligned_rate']:.1%}",
        f"R horizontal: {stats['horizontal_resultant']:.3f}",
        f"R vertical: {stats['vertical_resultant']:.3f}",
        f"Uniform null p: {format_p(stats['uniformity_p'])}",
    )
    for index, line in enumerate(lines):
        draw.text((summary_x, 175+index*42), line, font=tick_font, fill="#303030")
    draw.text((summary_x, 510), "Diagonal: equal phases", font=tick_font, fill="#555555")
    draw.text((summary_x, 545), "Grey: circular |Δ| ≤ 0.25", font=tick_font, fill="#555555")
    draw.text((summary_x, 580), "Dashed cross: mean phases", font=tick_font, fill="#555555")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG")


def write_summary(rows, output):
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in SUMMARY_FIELDS})


def write_design_summary(design_rows, perforation_rows, labels, output):
    joined = attach_design_clusters(design_rows, perforation_rows, labels)
    usable = usable_design_spacing_rows(joined)
    spacing_inliers, spacing_outliers = split_spacing_count_outliers(usable)
    classified = {
        (row["result_file"].casefold(), str(row["stamp"])): row
        for row in spacing_inliers + spacing_outliers
    }
    assignments = {
        (row["result_file"].casefold(), str(row["stamp"])): row
        for row in joined
    }
    fields = ("design_file", "result_file", "source", "stamp",
              "registration_status", "registration_reasons",
              DESIGN_WIDTH, DESIGN_HEIGHT, "perforation_cluster",
              WIDTH_HOLE_COUNT, HEIGHT_HOLE_COUNT,
              HOLE_CENTER_WIDTH, HOLE_CENTER_HEIGHT,
              HORIZONTAL_DESIGN_GAP, VERTICAL_DESIGN_GAP,
              EXPECTED_WIDTH_HOLE_COUNT, EXPECTED_HEIGHT_HOLE_COUNT,
              SPACING_COUNT_OUTLIER, "spacing_count_outlier_reason")
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in design_rows:
            match = assignments.get((row["result_file"].casefold(), str(row["stamp"])))
            key = (row["result_file"].casefold(), str(row["stamp"]))
            source = classified.get(key) or match or row
            item = {field: source.get(field) for field in fields}
            item["perforation_cluster"] = "" if match is None else f"C{match['cluster']+1}"
            writer.writerow(item)


def outlier_records(rows, centers, labels, count=6):
    records = []
    for index in outlier_indices(rows, centers, labels, count):
        row = rows[index]
        records.append({
            "source": source_name(row["result_file"]),
            "stamp": row["stamp"],
            "cluster": f"C{labels[index]+1}",
            "distance_to_centroid": math.sqrt(
                squared_distance(row["point"], centers[labels[index]])
            ),
            "worst_edge_algorithm": row.get("worst_edge_algorithm"),
        })
    return records


def write_assignments(rows, centers, labels, output):
    fields = ("result_file", "source", "stamp", "status", "worst_edge_algorithm",
              HORIZONTAL, VERTICAL, "cluster", "distance_to_centroid", "outlier")
    marked = set(outlier_indices(rows, centers, labels))
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, (row, label) in enumerate(zip(rows, labels)):
            item = {field: row.get(field) for field in fields}
            item["cluster"] = f"C{label + 1}"
            item["distance_to_centroid"] = math.sqrt(
                squared_distance(row["point"], centers[label])
            )
            item["outlier"] = index in marked
            writer.writerow(item)


def format_p(value):
    return "< 0.001" if value < 0.001 else f"{value:.3f}"


def frame_conclusion(stats):
    if stats is None or stats["classification"] == "insufficient-data":
        return ("There are too few usable corners for classification; at least 12 corners "
                "from 4 stamps are required.")
    if stats["classification"] == "frame":
        return ("Hole phases are concentrated at a repeatable position in both orientations. "
                "This supports frame perforation over independent line-perforator passes.")
    if stats["classification"] == "linear-compatible":
        return ("The data do not reject uniform corner phase. This is compatible with line "
                "perforation and provides no positive evidence for frame perforation.")
    return ("The corner-phase result is indeterminate: it does not meet the decision rule "
            "for either a frame-supported or line-compatible result.")


def write_report(path, dataset, diagnostics, selected_k, centers, labels, corner_stats,
                 references, simulations):
    counts = Counter(labels)
    statuses = Counter(row["status"] for row in dataset.rows)
    outliers = outlier_records(dataset.rows, centers, labels)
    design_with_clusters = attach_design_clusters(dataset.design_rows, dataset.rows, labels)
    spacing_candidates = usable_design_spacing_rows(design_with_clusters)
    spacing_rows, spacing_count_outliers = split_spacing_count_outliers(spacing_candidates)
    spacing_summaries = design_spacing_summaries(spacing_rows)
    spacing_changes = design_spacing_changes(spacing_summaries)
    lines = [
        "# Perforation and printed-design analysis",
        "",
        "## Result",
        "",
        f"- Cluster count selected by the Gap 1-SE rule: **{selected_k}**.",
        f"- Frame versus line: **{corner_stats['classification'] if corner_stats else 'insufficient-data'}**. "
        + frame_conclusion(corner_stats),
        "",
        "## Input coverage",
        "",
        f"Result files read: **{len(dataset.json_paths)}**; stamps found: **{dataset.total_stamps}**; "
        f"complete gauge pairs: **{len(dataset.rows)}**; incomplete pairs: **{len(dataset.incomplete)}**.",
        "",
        f"Design-size files read: **{len(dataset.design_paths)}**; measured designs: "
        f"**{len(dataset.design_rows)}**; matched to a perforation cluster: "
        f"**{len(design_with_clusters)}**; hole-count outliers excluded from spacing: "
        f"**{len(spacing_count_outliers)}**.",
        "",
        "Measurement status among complete pairs: "
        + ", ".join(f"`{name}` — {count}" for name, count in sorted(statuses.items())) + ".",
        "",
        "Machine-readable measurements: [summary.csv](summary.csv).",
        "",
        "## Gauge clusters",
        "",
        "![Selected perforation-gauge clusters](perforation_clusters.png)",
        "",
        "Each source JSON has its own color; marker shape denotes the selected cluster. "
        "Marker outline denotes the worst edge-detection algorithm, using the same colors as the "
        "annotated stamp images. Grid lines are placed at every visible 0.25-gauge boundary.",
        "",
        "Clustering uses the horizontal and vertical gauges directly. Cluster names are ordered "
        "by increasing sum of their two centroid coordinates and have no predefined catalogue meaning.",
        "",
        "| Cluster | n | Horizontal centroid | Vertical centroid |",
        "|---:|---:|---:|---:|",
    ]
    for index, center in enumerate(centers):
        lines.append(f"| C{index+1} | {counts[index]} | {center[0]:.4f} | {center[1]:.4f} |")
    lines += [
        "",
        "The six observations farthest from their assigned centroid are labelled on the chart:",
        "",
        "| Observation | Cluster | Distance to centroid | Worst edge algorithm |",
        "|---|---:|---:|---:|",
    ]
    for outlier in outliers:
        lines.append(
            f"| {outlier['source']}:{outlier['stamp']} | {outlier['cluster']} | "
            f"{outlier['distance_to_centroid']:.4f} | "
            f"{outlier['worst_edge_algorithm'] or '—'} |"
        )
    lines += [
        "",
        "Per-stamp assignments: [cluster_assignments.csv](cluster_assignments.csv).",
        "",
        "### Choosing the number of clusters",
        "",
        "![Cluster-count diagnostics](cluster_count_diagnostics.png)",
        "",
        "| k | Within-cluster sum of squares | Gap | SE(Gap) | Silhouette |",
        "|---:|---:|---:|---:|---:|",
    ]
    for result in diagnostics:
        silhouette_text = "—" if result["silhouette"] is None else f"{result['silhouette']:.3f}"
        lines.append(f"| {result['k']} | {result['inertia']:.4f} | {result['gap']:.3f} | "
                     f"{result['gap_se']:.3f} | {silhouette_text} |")
    lines += [
        "",
        f"The Gap statistic compares the observed clustering with **{references}** uniform "
        "reference samples over the same two-dimensional range. The standard 1-SE rule chooses "
        "the smallest k for which `Gap(k) ≥ Gap(k+1) − SE(k+1)`.",
        "",
    ]
    if dataset.design_rows:
        design_groups = defaultdict(list)
        for row in dataset.design_rows:
            design_groups[row["result_file"]].append(row)
        lines += [
            "## Printed-design dimensions by source",
            "",
            "![Printed-design dimensions by source](design_sizes_by_source.png)",
            "",
            "Each point is one measured printed design. Large crosses mark the mean for each "
            "Source; both dimensions are in millimetres.",
            "",
            "The `±` interval is one sample standard deviation and describes the observed "
            "within-Source spread; it is not a confidence interval.",
            "",
            "| Source | n | Width, mean ± SD | Height, mean ± SD |",
            "|---|---:|---:|---:|",
        ]
        for source in sorted(design_groups, key=lambda key: natural_key(source_name(key))):
            members = design_groups[source]
            widths = [row[DESIGN_WIDTH] for row in members]
            heights = [row[DESIGN_HEIGHT] for row in members]
            lines.append(
                f"| {source_name(source)} | {len(members)} | "
                f"{statistics.mean(widths):.4f} ± {sample_sd(widths):.4f} mm | "
                f"{statistics.mean(heights):.4f} ± {sample_sd(heights):.4f} mm |"
            )
        lines += ["", "Measurements and cluster joins: [design_summary.csv](design_summary.csv).", ""]
        if spacing_rows or spacing_count_outliers:
            lines += [
                "## Estimated spacing between neighboring designs",
                "",
            ]
            if spacing_rows:
                lines += [
                    "![Estimated spacing between neighboring designs]"
                    "(design_spacing_by_perforation_cluster.png)",
                    "",
                ]
            lines += [
                "Under the frame-perforation hypothesis, each side contains a whole number of "
                "equally spaced holes. The pitch is obtained from the measured gauge. The "
                "smallest whole number of pitch intervals that spans the measured valley-line "
                "dimension determines both the hole count and the distance between the centres "
                "of the extreme holes:",
                "",
                "`pitch = 20 / gauge`",
                "",
                "`interval count = ceil(valley-line dimension / pitch)`",
                "",
                "`hole count = interval count + 1`",
                "",
                "`extreme-hole-centre dimension = interval count × pitch`",
                "",
                "`horizontal design gap = hole-centre width − design width`",
                "",
                "`vertical design gap = hole-centre height − design height`",
                "",
                "The chart shows cluster means for every Source. A connecting line makes the "
                "change from one perforation cluster to the next explicit. Within each "
                "Source/cluster group, the unique modal width/height hole-count pair is required. "
                "Other estimates are listed below and excluded from the chart and all means.",
                "",
                "No fitted hole radius is used in this calculation.",
                "",
                "| Source | Cluster | n | Width holes | Height holes | Mean horizontal gap | Mean vertical gap |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
            for summary in spacing_summaries:
                lines.append(
                    f"| {summary['source']} | C{summary['cluster']+1} | "
                    f"{summary['count']} | "
                    f"{summary['width_hole_count']} | "
                    f"{summary['height_hole_count']} | "
                    f"{summary['mean_horizontal_gap_mm']:.4f} mm | "
                    f"{summary['mean_vertical_gap_mm']:.4f} mm |"
                )
            if spacing_rows:
                lines += [
                    "",
                    "### Per-stamp distributions",
                    "",
                    "![Per-stamp estimated spacing between neighboring designs]"
                    "(design_spacing_points_by_perforation_cluster.png)",
                    "",
                    "Each marker is one stamp; color denotes Source and shape denotes the "
                    "perforation cluster. Vertical jitter has no measurement meaning and only "
                    "keeps coincident points visible.",
                    "",
                    "The cluster ranges overlap substantially in the per-stamp observations. "
                    "Design spacing can therefore provide supporting context, but it must not "
                    "be used as a determining factor on its own.",
                ]
            if spacing_count_outliers:
                lines += [
                    "",
                    "### Excluded hole-count outliers",
                    "",
                    "| Observation | Cluster | Estimated holes (W × H) | Expected holes (W × H) | Reason |",
                    "|---|---:|---:|---:|---|",
                ]
                for row in spacing_count_outliers:
                    expected = (
                        f"{row[EXPECTED_WIDTH_HOLE_COUNT]} × {row[EXPECTED_HEIGHT_HOLE_COUNT]}"
                        if row[EXPECTED_WIDTH_HOLE_COUNT] is not None else "no unique mode"
                    )
                    lines.append(
                        f"| {source_name(row['result_file'])}:{row['stamp']} | "
                        f"C{row['cluster']+1} | {row[WIDTH_HOLE_COUNT]} × "
                        f"{row[HEIGHT_HOLE_COUNT]} | {expected} | "
                        f"{row['spacing_count_outlier_reason']} |"
                    )
            if spacing_changes:
                lines += [
                    "",
                    "### Change between consecutive clusters",
                    "",
                    "| Source | Transition | Δ horizontal gap | Δ vertical gap |",
                    "|---|---:|---:|---:|",
                ]
                for change in spacing_changes:
                    lines.append(
                        f"| {change['source']} | C{change['from_cluster']+1} → "
                        f"C{change['to_cluster']+1} | "
                        f"{change['delta_horizontal_gap_mm']:+.4f} mm | "
                        f"{change['delta_vertical_gap_mm']:+.4f} mm |"
                    )
            lines += [
                "",
                "These are hypothesis-based repeat-spacing estimates, not direct measurements "
                "of a physical pair of neighboring stamps. They assume a complete frame with "
                "a whole number of pitch intervals and that the extreme hole centres define the "
                "repeated stamp spacing.",
                "",
            ]
    lines += ["## Frame or line perforation", ""]
    if corner_stats:
        lines += [
            "![Corner-hole phase consistency](corner_alignment.png)",
            "",
            "For each usable corner, the fitted baselines of adjacent sides are intersected. "
            "The distance from that intersection to the nearest circle-fitted hole on each side "
            "is reduced modulo the side pitch. A frame tends to repeat a common phase; independent "
            "line-perforator passes permit phase to vary.",
            "",
            "| Metric | Result |",
            "|---|---:|",
            f"| Usable corners / stamps | {corner_stats['corners']} / {corner_stats['stamps']} |",
            f"| Median circular `|Δ phase|` | {corner_stats['median_mismatch']:.3f} pitch |",
            f"| 95% stamp-bootstrap interval | {corner_stats['median_ci'][0]:.3f}…{corner_stats['median_ci'][1]:.3f} |",
            f"| Corners with `|Δ phase| ≤ 0.25` | {corner_stats['aligned_rate']:.1%} |",
            f"| 95% stamp-bootstrap interval | {corner_stats['aligned_ci'][0]:.1%}…{corner_stats['aligned_ci'][1]:.1%} |",
            f"| Coincident corner-hole centres (≤0.25 pitch) | {corner_stats['shared_rate']:.1%} |",
            f"| Circular concentration R, horizontal / vertical | {corner_stats['horizontal_resultant']:.3f} / {corner_stats['vertical_resultant']:.3f} |",
            f"| Uniform-phase null test | p {format_p(corner_stats['uniformity_p'])} |",
            "",
            "**Conclusion:** " + frame_conclusion(corner_stats),
            "",
            f"The null distribution uses **{simulations}** simulations. One random circular shift "
            "is applied per stamp and orientation, preserving dependencies among the corners of "
            "one stamp while removing collection-wide phase locking. `R=0` is uniform and `R=1` "
            "is perfect phase concentration.",
        ]
    else:
        lines += [frame_conclusion(None)]
    lines += [
        "",
        "### Interpretation limit",
        "",
        "This is an automated geometric test of detector output, not expert philatelic attribution. "
        "Paper damage, partial corner holes and circle-fit errors can change the result. Inspect "
        "representative source-image corners before treating the classification as final.",
    ]
    if dataset.errors:
        lines += ["", "## Files not processed", ""]
        lines.extend(f"- `{error}`" for error in dataset.errors)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()
                if key not in {"labels"}}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def default_output(inputs):
    if len(inputs) == 1:
        path = Path(inputs[0]).expanduser()
        if path.is_dir():
            return path / "perforation-analysis"
        if path.is_file():
            return path.parent / "perforation-analysis"
    return Path.cwd() / "perforation-analysis"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("inputs", nargs="+",
                        help="*_perf.json file, directory, or wildcard pattern")
    parser.add_argument("-o", "--output-dir", type=Path,
                        help="report directory (default: beside a single input, otherwise current directory)")
    parser.add_argument("--no-recursive", action="store_true",
                        help="do not search input directories recursively")
    parser.add_argument("--max-k", type=int, default=6,
                        help="largest cluster count to test")
    parser.add_argument("--gap-references", type=int, default=250,
                        help="uniform reference samples for the Gap statistic")
    parser.add_argument("--phase-simulations", type=int, default=10000,
                        help="simulations for the frame-versus-line phase test")
    parser.add_argument("--bootstrap-samples", type=int, default=5000,
                        help="stamp-level bootstrap samples for corner intervals")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.max_k < 1 or args.gap_references < 2:
        raise SystemExit("--max-k must be positive and --gap-references must be at least 2")
    if args.phase_simulations < 1 or args.bootstrap_samples < 1:
        raise SystemExit("simulation and bootstrap counts must be positive")
    paths = find_inputs(args.inputs, recursive=not args.no_recursive)
    if not paths:
        raise SystemExit("no input files matched; expected *_perf.json results")
    dataset = load_dataset(paths)
    if len(dataset.rows) < 3:
        raise SystemExit("at least three complete horizontal/vertical gauge pairs are required")
    output = (args.output_dir or default_output(args.inputs)).resolve()
    output.mkdir(parents=True, exist_ok=True)
    points = [row["point"] for row in dataset.rows]
    diagnostics, selected_k = cluster_diagnostics(
        points, max_k=args.max_k, references=args.gap_references,
    )
    selected_result = next(result for result in diagnostics if result["k"] == selected_k)
    centers, labels = order_clusters(selected_result["centers"], selected_result["labels"])
    all_sources = sorted(
        {row["result_file"] for row in dataset.rows + dataset.design_rows},
        key=lambda key: natural_key(source_name(key)),
    )
    colors = source_colors(all_sources)
    design_with_clusters = attach_design_clusters(dataset.design_rows, dataset.rows, labels)
    spacing_candidates = usable_design_spacing_rows(design_with_clusters)
    spacing_rows, spacing_count_outliers = split_spacing_count_outliers(spacing_candidates)
    spacing_summaries = design_spacing_summaries(spacing_rows)
    spacing_changes = design_spacing_changes(spacing_summaries)
    corner_stats = corner_statistics(
        dataset.corner_records,
        simulations=args.phase_simulations,
        bootstraps=args.bootstrap_samples,
    )
    write_summary(dataset.rows, output / "summary.csv")
    write_assignments(dataset.rows, centers, labels, output / "cluster_assignments.csv")
    draw_cluster_chart(dataset.rows, centers, labels,
                       output / "perforation_clusters.png", colors)
    draw_diagnostics(diagnostics, selected_k, output / "cluster_count_diagnostics.png")
    if dataset.design_rows:
        write_design_summary(dataset.design_rows, dataset.rows, labels,
                             output / "design_summary.csv")
        draw_design_source_chart(dataset.design_rows,
                                 output / "design_sizes_by_source.png", colors)
    legacy_design_chart = output / "design_size_by_perforation_cluster.png"
    if legacy_design_chart.exists():
        legacy_design_chart.unlink()
    spacing_chart = output / "design_spacing_by_perforation_cluster.png"
    if spacing_chart.exists():
        spacing_chart.unlink()
    spacing_points_chart = output / "design_spacing_points_by_perforation_cluster.png"
    if spacing_points_chart.exists():
        spacing_points_chart.unlink()
    if spacing_rows:
        draw_design_spacing_change(
            spacing_rows, selected_k,
            spacing_chart, colors,
        )
        draw_design_spacing_points(
            spacing_rows, selected_k,
            spacing_points_chart, colors,
        )
    if corner_stats:
        draw_corner_chart(dataset.corner_records, corner_stats, output / "corner_alignment.png")
    write_report(output / "report.md", dataset, diagnostics, selected_k, centers, labels,
                 corner_stats, args.gap_references, args.phase_simulations)
    machine_result = {
        "format": "stamp-perforation-analysis",
        "version": 1,
        "inputs": [str(path) for path in dataset.json_paths],
        "coverage": {
            "result_files": len(dataset.json_paths),
            "stamps": dataset.total_stamps,
            "complete_gauge_pairs": len(dataset.rows),
            "incomplete_gauge_pairs": len(dataset.incomplete),
            "design_files": len(dataset.design_paths),
            "measured_designs": len(dataset.design_rows),
            "designs_with_perforation_cluster": len(design_with_clusters),
            "designs_with_spacing_estimate": len(spacing_rows),
            "design_spacing_count_outliers": len(spacing_count_outliers),
            "errors": dataset.errors,
        },
        "clustering": {
            "selected_k": selected_k,
            "centers": centers,
            "counts": [labels.count(index) for index in range(selected_k)],
            "outliers": outlier_records(dataset.rows, centers, labels),
            "diagnostics": json_ready(diagnostics),
        },
        "design_size": {
            "files": [str(path) for path in dataset.design_paths],
            "measurements": len(dataset.design_rows),
            "matched_to_perforation_cluster": len(design_with_clusters),
            "by_source": [
                {
                    "source": source_name(source),
                    "count": len(members),
                    "mean_width_mm": statistics.mean(
                        row[DESIGN_WIDTH] for row in members
                    ),
                    "mean_height_mm": statistics.mean(
                        row[DESIGN_HEIGHT] for row in members
                    ),
                    "sd_width_mm": sample_sd(
                        row[DESIGN_WIDTH] for row in members
                    ),
                    "sd_height_mm": sample_sd(
                        row[DESIGN_HEIGHT] for row in members
                    ),
                }
                for source, members in sorted(
                    ((source, [row for row in dataset.design_rows
                               if row["result_file"] == source])
                     for source in {row["result_file"] for row in dataset.design_rows}),
                    key=lambda item: natural_key(source_name(item[0])),
                )
            ],
            "spacing_estimate": {
                "assumption": (
                    "a complete frame has a whole number of pitch intervals; "
                    "the extreme hole centres define stamp repeat"
                ),
                "method": "ceil(valley-line dimension / (20 / gauge)) whole pitch intervals",
                "measurements": len(spacing_rows),
                "candidate_measurements": len(spacing_candidates),
                "excluded_hole_count_outliers": [
                    {
                        "source": source_name(row["result_file"]),
                        "stamp": row["stamp"],
                        "cluster": row["cluster"],
                        "width_hole_count": row[WIDTH_HOLE_COUNT],
                        "height_hole_count": row[HEIGHT_HOLE_COUNT],
                        "expected_width_hole_count": row[EXPECTED_WIDTH_HOLE_COUNT],
                        "expected_height_hole_count": row[EXPECTED_HEIGHT_HOLE_COUNT],
                        "reason": row["spacing_count_outlier_reason"],
                    }
                    for row in spacing_count_outliers
                ],
                "by_source_and_cluster": spacing_summaries,
                "changes_between_consecutive_clusters": spacing_changes,
            },
        },
        "frame_vs_line": json_ready(corner_stats),
    }
    (output / "analysis.json").write_text(
        json.dumps(machine_result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(f"Report: {output / 'report.md'}")
    print(f"Measurements: {len(dataset.rows)} complete, {len(dataset.incomplete)} incomplete")
    print(f"Design sizes: {len(dataset.design_rows)} measured, "
          f"{len(design_with_clusters)} matched to a perforation cluster")
    print(f"Clusters: k={selected_k}; " + "; ".join(
        f"C{index+1}=({center[0]:.4f}, {center[1]:.4f}), n={labels.count(index)}"
        for index, center in enumerate(centers)
    ))
    print("Frame versus line: "
          + (corner_stats["classification"] if corner_stats else "insufficient-data"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
