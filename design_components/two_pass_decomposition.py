"""BASELINE EXPERIMENT: separation of two printing passes in registered scans.

This is deliberately isolated from the production pipeline.  It consumes the
``data_*_perf.json`` and ``data_*_design.json`` files already produced by the current
pipeline, reconstructs colour stamp patches in the common design frame, and
groups printed pixels by their *motion signature* across the collection.

The result is diagnostic, not a production measurement.  Dense optical flow
is noisy on flat paper and cancellation marks, so only repeatable design edges
seed the two groups.  Labels are then propagated into the bodies of printed
strokes.  Colour is measured only after the motion-only split has been made.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np


SIDES = ("top", "bottom", "left", "right")


def valley_quad(stamp: dict) -> np.ndarray | None:
    """Return the four perforation valley-line intersections."""
    sides = stamp.get("measurement", {}).get("sides", {})
    lines: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in SIDES:
        points = (sides.get(name) or {}).get("line_image")
        if not points:
            return None
        (x0, y0), (x1, y1) = points
        origin = np.array([x0, y0], np.float64)
        direction = np.array([x1 - x0, y1 - y0], np.float64)
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            return None
        lines[name] = origin, direction / norm

    def intersect(first: str, second: str) -> np.ndarray | None:
        p1, d1 = lines[first]
        p2, d2 = lines[second]
        matrix = np.array([d1, -d2]).T
        if abs(np.linalg.det(matrix)) < 1e-9:
            return None
        t = np.linalg.solve(matrix, p2 - p1)
        return p1 + t[0] * d1

    corners = [
        intersect("top", "left"),
        intersect("top", "right"),
        intersect("bottom", "right"),
        intersect("bottom", "left"),
    ]
    if any(corner is None for corner in corners):
        return None
    return np.asarray(corners, np.float64)


def quad_angle_deg(quad: np.ndarray) -> float:
    horizontal = (quad[1] - quad[0]) + (quad[2] - quad[3])
    return float(np.degrees(np.arctan2(horizontal[1], horizontal[0])))


def placement_matrix(quad: np.ndarray, canvas: tuple[int, int]) -> np.ndarray:
    height, width = canvas
    centre = quad.mean(axis=0)
    matrix = cv2.getRotationMatrix2D(
        (float(centre[0]), float(centre[1])), quad_angle_deg(quad), 1.0
    )
    matrix[0, 2] += width / 2.0 - centre[0]
    matrix[1, 2] += height / 2.0 - centre[1]
    return matrix.astype(np.float64)


def result_warp(result: dict) -> np.ndarray:
    """Rebuild the no-shear reference-to-stamp warp stored in design JSON."""
    angle = math.radians(float(result["residual_rotation_deg"]))
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)],
         [math.sin(angle), math.cos(angle)]],
        np.float64,
    )
    linear = rotation @ np.diag(
        [float(result["scale_x"]), float(result["scale_y"])]
    )
    return np.column_stack(
        [linear, [float(result["offset_x_px"]), float(result["offset_y_px"])]]
    )


def affine3(matrix: np.ndarray) -> np.ndarray:
    return np.vstack([matrix, [0.0, 0.0, 1.0]])


def apply_affine(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:, :2].T + matrix[:, 2]


def aligned_collection(
    scan_path: Path,
    perf: dict,
    design: dict,
    downsample: int,
    minimum: int = 5,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Render usable stamps directly into the common reference frame.

    The historical two-motion experiments require five observations by
    default.  Production callers that implement their own small-collection
    fallbacks may explicitly lower ``minimum``.
    """
    scan = cv2.imread(str(scan_path), cv2.IMREAD_COLOR)
    if scan is None:
        raise RuntimeError(f"Cannot read scan: {scan_path}")

    full_canvas = (
        int(design["golden"]["canvas_height_px"]),
        int(design["golden"]["canvas_width_px"]),
    )
    height = full_canvas[0] // downsample
    width = full_canvas[1] // downsample
    by_number = {int(item["stamp"]): item for item in design["stamps"]}
    patches: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    numbers: list[int] = []

    for stamp in perf["stamps"]:
        number = int(stamp["stamp"])
        result = by_number.get(number)
        quad = valley_quad(stamp)
        if result is None or quad is None:
            continue
        placement = placement_matrix(quad, full_canvas)
        warp = result_warp(result)

        # Destination (reference) -> source (scan) sampling transform.
        reference_to_scan = np.linalg.inv(affine3(placement)) @ affine3(warp)
        small_to_full = np.diag([downsample, downsample, 1.0])
        sample = reference_to_scan @ small_to_full
        aligned = cv2.warpAffine(
            scan,
            sample[:2].astype(np.float32),
            (width, height),
            flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        in_patch = apply_affine(placement, quad)
        in_reference = apply_affine(cv2.invertAffineTransform(warp), in_patch)
        polygon = np.round(in_reference / downsample).astype(np.int32)
        valid = np.zeros((height, width), np.uint8)
        cv2.fillConvexPoly(valid, polygon, 255)
        valid = cv2.erode(valid, np.ones((3, 3), np.uint8))

        patches.append(aligned)
        masks.append(valid)
        numbers.append(number)

    if len(patches) < minimum:
        raise RuntimeError(
            f"Need at least {minimum} aligned stamp(s); "
            f"found {len(patches)}"
        )
    return np.stack(patches), np.stack(masks) > 0, numbers


def masked_colour_median(images: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = images.astype(np.float32)
    covered = valid.any(axis=0)
    reference = np.zeros(values.shape[1:], np.float32)
    columns = values[:, covered, :]
    columns[~valid[:, covered]] = np.nan
    reference[covered] = np.nanmedian(columns, axis=0)
    return reference.astype(np.float32)


def luminance_bgr(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)


def photometric_match(
    moving: np.ndarray, reference: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    """Fit moving grey values to the reference before optical flow."""
    x = moving[valid].astype(np.float64)
    y = reference[valid].astype(np.float64)
    if x.size < 1000:
        return moving
    keep = np.ones(x.size, bool)
    slope, intercept = 1.0, 0.0
    for _ in range(3):
        matrix = np.column_stack([x[keep], np.ones(keep.sum())])
        slope, intercept = np.linalg.lstsq(matrix, y[keep], rcond=None)[0]
        residual = y - (slope * x + intercept)
        centre = np.median(residual[keep])
        sigma = 1.4826 * np.median(np.abs(residual[keep] - centre)) + 1e-6
        keep = np.abs(residual - centre) < 4.0 * sigma
    return np.clip(slope * moving + intercept, 0.0, 255.0).astype(np.float32)


def flow_stack(
    images: np.ndarray,
    valid: np.ndarray,
    reference_bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense reference-to-stamp flow, with cancellation-contaminated pixels masked."""
    reference = luminance_bgr(reference_bgr)
    quorum = valid.sum(axis=0) >= max(3, math.ceil(0.6 * len(images)))
    flows: list[np.ndarray] = []
    flow_valid: list[np.ndarray] = []
    matched_greys: list[np.ndarray] = []

    for image, stamp_valid in zip(images, valid):
        grey = luminance_bgr(image)
        fit_mask = stamp_valid & quorum
        matched = photometric_match(grey, reference, fit_mask)
        residual = np.abs(matched - reference)
        centre = np.median(residual[fit_mask])
        sigma = 1.4826 * np.median(np.abs(residual[fit_mask] - centre)) + 1e-6
        # A generous cutoff keeps genuine misregistration edges while removing
        # the much stronger black cancellation strokes.
        clean = fit_mask & (residual < max(28.0, centre + 7.0 * sigma))

        flow = cv2.calcOpticalFlowFarneback(
            reference.astype(np.uint8),
            matched.astype(np.uint8),
            None,
            pyr_scale=0.5,
            levels=4,
            winsize=25,
            iterations=5,
            poly_n=7,
            poly_sigma=1.5,
            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
        )
        # Remove any remaining whole-stamp displacement.  Only differential
        # motion between the two impressions is evidence for separation.
        for axis in range(2):
            flow[:, :, axis] -= np.median(flow[:, :, axis][clean])
            flow[:, :, axis] = cv2.GaussianBlur(flow[:, :, axis], (0, 0), 1.2)
        flows.append(flow.astype(np.float32))
        flow_valid.append(clean)
        matched_greys.append(matched)

    return np.stack(flows), np.stack(flow_valid), np.stack(matched_greys)


def kmeans_two(
    values: np.ndarray, rng: np.random.Generator, attempts: int = 8
) -> tuple[np.ndarray, np.ndarray, float]:
    """Small deterministic two-means, returning labels, centres and inertia."""
    best: tuple[np.ndarray, np.ndarray, float] | None = None
    count = len(values)
    for _ in range(attempts):
        first = int(rng.integers(count))
        distances = np.sum((values - values[first]) ** 2, axis=1)
        second = int(np.argmax(distances))
        centres = np.stack([values[first], values[second]]).astype(np.float64)
        labels = np.zeros(count, np.uint8)
        for _ in range(40):
            d0 = np.sum((values - centres[0]) ** 2, axis=1)
            d1 = np.sum((values - centres[1]) ** 2, axis=1)
            updated = (d1 < d0).astype(np.uint8)
            if np.array_equal(updated, labels):
                break
            labels = updated
            if labels.min() == labels.max():
                break
            centres = np.stack(
                [values[labels == group].mean(axis=0) for group in (0, 1)]
            )
        inertia = float(
            np.sum((values - centres[labels]) ** 2) / max(1, count)
        )
        if labels.min() != labels.max() and (best is None or inertia < best[2]):
            best = labels, centres.astype(np.float32), inertia
    if best is None:
        raise RuntimeError("Motion clustering collapsed to one group")
    return best


def motion_partition(
    flows: np.ndarray,
    flow_valid: np.ndarray,
    reference_bgr: np.ndarray,
    rng: np.random.Generator,
    max_samples: int,
) -> dict:
    """Cluster edge pixels using their displacement vector across all stamps."""
    grey = luminance_bgr(reference_bgr)
    paper_level = float(np.percentile(grey[grey > 0], 92))
    darkness = np.clip((paper_level - grey) / max(paper_level, 1.0), 0.0, 1.0)
    gx = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    enough = flow_valid.sum(axis=0) >= math.ceil(0.65 * len(flows))
    ink = darkness > np.percentile(darkness[enough], 35)
    edge_threshold = np.percentile(gradient[ink & enough], 55)
    candidates = ink & enough & (gradient >= edge_threshold)
    yy, xx = np.nonzero(candidates)
    if len(xx) < 1000:
        raise RuntimeError("Too few repeatable printed edges for motion clustering")

    # One row per pixel; columns are dx,dy for every stamp. Missing observations
    # are set to the neutral displacement after per-dimension centring.
    features = flows[:, yy, xx, :].transpose(1, 0, 2).reshape(len(xx), -1)
    observations = flow_valid[:, yy, xx].T
    expanded = np.repeat(observations, 2, axis=1)
    centres = np.zeros(features.shape[1], np.float32)
    for column in range(features.shape[1]):
        seen = expanded[:, column]
        centres[column] = np.median(features[seen, column]) if seen.any() else 0.0
        features[~seen, column] = centres[column]
    features -= centres
    common_scale = 1.4826 * np.median(np.abs(features)) + 1e-4
    standard = np.clip(features / common_scale, -8.0, 8.0)

    if len(standard) > max_samples:
        sample_index = rng.choice(len(standard), max_samples, replace=False)
    else:
        sample_index = np.arange(len(standard))
    _, cluster_centres, inertia = kmeans_two(standard[sample_index], rng)
    d0 = np.sum((standard - cluster_centres[0]) ** 2, axis=1)
    d1 = np.sum((standard - cluster_centres[1]) ** 2, axis=1)
    labels = (d1 < d0).astype(np.uint8)
    confidence = np.abs(d1 - d0) / (d0 + d1 + 1e-6)

    # Only high-confidence edges seed propagation into stroke interiors.
    seed = np.full(grey.shape, 255, np.uint8)
    reliable = np.zeros(len(confidence), bool)
    for group in (0, 1):
        members = labels == group
        if members.any():
            reliable[members] = confidence[members] >= np.percentile(
                confidence[members], 35
            )
    seed[yy[reliable], xx[reliable]] = labels[reliable]
    seeds0 = seed == 0
    seeds1 = seed == 1
    if seeds0.sum() < 100 or seeds1.sum() < 100:
        raise RuntimeError("One motion group has too few reliable design edges")
    source0 = np.ones(grey.shape, np.uint8)
    source1 = np.ones(grey.shape, np.uint8)
    source0[seeds0] = 0
    source1[seeds1] = 0
    distance0 = cv2.distanceTransform(source0, cv2.DIST_L2, 5)
    distance1 = cv2.distanceTransform(source1, cv2.DIST_L2, 5)
    assignment = (distance1 < distance0).astype(np.uint8)
    assignment[~ink] = 255

    # Translate labels into per-stamp motion centres and a transparent quality
    # score. These are calculated from raw flow, not standardised features.
    shifts = np.zeros((len(flows), 2, 2), np.float64)
    scatters = np.zeros((len(flows), 2), np.float64)
    for stamp_index in range(len(flows)):
        for group in (0, 1):
            use = reliable & (labels == group) & observations[:, stamp_index]
            vectors = flows[stamp_index, yy[use], xx[use], :]
            shifts[stamp_index, group] = np.median(vectors, axis=0)
            residual = np.linalg.norm(vectors - shifts[stamp_index, group], axis=1)
            scatters[stamp_index, group] = 1.4826 * np.median(
                np.abs(residual - np.median(residual))
            )

    centroid_distance = float(np.linalg.norm(cluster_centres[1] - cluster_centres[0]))
    pooled_rms = math.sqrt(max(inertia, 1e-12))
    return {
        "assignment": assignment,
        "darkness": darkness,
        "gradient": gradient,
        "candidate_count": int(len(xx)),
        "shifts": shifts,
        "scatters": scatters,
        "cluster_separation": centroid_distance / pooled_rms,
        "feature_scale_px": float(common_scale),
    }


def group_spread(assignment: np.ndarray, group: int, darkness: np.ndarray) -> float:
    yy, xx = np.nonzero((assignment == group) & (darkness > 0.05))
    if len(xx) < 10:
        return 0.0
    points = np.column_stack([xx, yy]).astype(np.float64)
    covariance = np.cov(points, rowvar=False)
    return float(math.sqrt(max(np.linalg.det(covariance), 0.0)))


def colour_partition(
    reference_bgr: np.ndarray,
    darkness: np.ndarray,
    rng: np.random.Generator,
    max_samples: int,
) -> dict:
    """Independent two-ink control partition from optical-density direction.

    Density magnitude is deliberately discarded: two impressions of the same
    ink at different strengths should have the same direction from paper in
    RGB optical-density space.  Conversely, blue/red or blue/yellow inks make
    two well-separated directions.  This partition is not used to claim
    motion-only separation; it is the independent check of that claim.
    """
    grey = luminance_bgr(reference_bgr)
    paper_region = darkness <= np.percentile(darkness[darkness >= 0], 28)
    paper_pixels = reference_bgr[paper_region]
    paper = np.percentile(paper_pixels, 90, axis=0).astype(np.float64)
    optical_density = -np.log(
        np.clip((reference_bgr.astype(np.float64) + 2.0) /
                (paper[None, None, :] + 2.0), 1e-3, 1.5)
    )
    optical_density = np.clip(optical_density, 0.0, None)
    magnitude = np.linalg.norm(optical_density, axis=2)
    usable = (grey > 0) & (magnitude > np.percentile(magnitude[grey > 0], 38))
    yy, xx = np.nonzero(usable)
    directions = optical_density[yy, xx]
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-8)
    if len(directions) > max_samples:
        chosen = rng.choice(len(directions), max_samples, replace=False)
    else:
        chosen = np.arange(len(directions))
    _, centres, inertia = kmeans_two(directions[chosen], rng)
    centres /= np.maximum(np.linalg.norm(centres, axis=1, keepdims=True), 1e-8)
    d0 = np.sum((directions - centres[0]) ** 2, axis=1)
    d1 = np.sum((directions - centres[1]) ** 2, axis=1)
    labels = (d1 < d0).astype(np.uint8)
    assignment = np.full(grey.shape, 255, np.uint8)
    assignment[yy, xx] = labels
    separation = float(np.linalg.norm(centres[1] - centres[0]))
    quality = separation / math.sqrt(max(inertia, 1e-12))
    return {
        "assignment": assignment,
        "paper_bgr": paper,
        "od_direction_centres_bgr": centres,
        "direction_separation": separation,
        "cluster_separation": quality,
        "usable_pixels": int(len(xx)),
        # About 7 degrees between optical-density rays. Below this the two-ray
        # model becomes ill-conditioned and tends to split one ink merely by
        # density or scanner non-linearity.  The spatial coherence check is
        # still essential for subtle pairs such as 2K.
        "distinct": bool(separation >= 0.12 and quality >= 0.7),
    }


def balanced_partition_agreement(
    first: np.ndarray, second: np.ndarray
) -> tuple[float, bool, int]:
    common = (first < 2) & (second < 2)
    count = int(common.sum())
    if count == 0:
        return float("nan"), False, 0
    a = first[common]
    b = second[common]

    def balanced(flip: bool) -> float:
        predicted = 1 - a if flip else a
        scores = []
        for group in (0, 1):
            where = b == group
            if where.any():
                scores.append(float(np.mean(predicted[where] == group)))
        return float(np.mean(scores))

    direct = balanced(False)
    flipped = balanced(True)
    return (flipped, True, count) if flipped > direct else (direct, False, count)


def shifts_for_partition(
    flows: np.ndarray,
    flow_valid: np.ndarray,
    assignment: np.ndarray,
    gradient: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Robust per-stamp translations for an externally supplied partition."""
    shifts = np.zeros((len(flows), 2, 2), np.float64)
    scatters = np.zeros((len(flows), 2), np.float64)
    printed = assignment < 2
    edge_cutoff = np.percentile(gradient[printed], 45)
    for stamp_index in range(len(flows)):
        for group in (0, 1):
            use = (
                (assignment == group)
                & (gradient >= edge_cutoff)
                & flow_valid[stamp_index]
            )
            vectors = flows[stamp_index][use]
            if len(vectors) < 100:
                continue
            centre = np.median(vectors, axis=0)
            residual = np.linalg.norm(vectors - centre, axis=1)
            trim = residual <= np.percentile(residual, 85)
            vectors = vectors[trim]
            centre = np.median(vectors, axis=0)
            shifts[stamp_index, group] = centre
            residual = np.linalg.norm(vectors - centre, axis=1)
            scatters[stamp_index, group] = 1.4826 * np.median(
                np.abs(residual - np.median(residual))
            )
    return shifts, scatters


def bgr_hex(colour: np.ndarray) -> str:
    b, g, r = np.clip(np.round(colour), 0, 255).astype(np.uint8)
    return f"#{r:02X}{g:02X}{b:02X}"


def representative_colours(
    images: np.ndarray,
    valid: np.ndarray,
    assignment: np.ndarray,
    darkness: np.ndarray,
    shifts: np.ndarray,
) -> list[dict]:
    """Estimate scanned paper and effective solid colour of each motion layer."""
    rows: list[dict] = []
    height, width = assignment.shape
    base_masks = [(assignment == group).astype(np.uint8) for group in (0, 1)]
    kernel = np.ones((3, 3), np.uint8)

    for index, (image, stamp_valid) in enumerate(zip(images, valid)):
        moved: list[np.ndarray] = []
        for group in (0, 1):
            dx, dy = shifts[index, group]
            matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], np.float32)
            mask = cv2.warpAffine(
                base_masks[group], matrix, (width, height),
                flags=cv2.INTER_NEAREST, borderValue=0,
            ) > 0
            moved.append(mask)

        combined = cv2.warpAffine(
            darkness,
            np.array([[1.0, 0.0, shifts[index].mean(axis=0)[0]],
                      [0.0, 1.0, shifts[index].mean(axis=0)[1]]], np.float32),
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderValue=0,
        )
        paper_mask = stamp_valid & (combined < np.percentile(combined[stamp_valid], 30))
        paper_pixels = image[paper_mask].astype(np.float64)
        if len(paper_pixels) < 100:
            paper_pixels = image[stamp_valid].astype(np.float64)
        # Upper channel percentiles are less biased by fibre shadows and pale ink.
        paper = np.percentile(paper_pixels, 88, axis=0)

        colours: list[np.ndarray] = []
        directions: list[np.ndarray] = []
        strengths: list[float] = []
        for group in (0, 1):
            other = cv2.dilate(moved[1 - group].astype(np.uint8), kernel) > 0
            pure = moved[group] & ~other & stamp_valid
            pixels = image[pure].astype(np.float64)
            if len(pixels) < 100:
                pixels = image[moved[group] & stamp_valid].astype(np.float64)
            optical_density = -np.log(
                np.clip((pixels + 2.0) / (paper[None, :] + 2.0), 1e-3, 1.5)
            )
            optical_density = np.clip(optical_density, 0.0, None)
            magnitude = np.linalg.norm(optical_density, axis=1)
            keep = magnitude > np.percentile(magnitude, 45)
            unit = optical_density[keep] / np.maximum(magnitude[keep, None], 1e-8)
            direction = np.median(unit, axis=0)
            direction /= max(np.linalg.norm(direction), 1e-8)
            projection = optical_density @ direction
            strength = float(np.percentile(projection, 82))
            colour = (paper + 2.0) * np.exp(-direction * strength) - 2.0
            colours.append(np.clip(colour, 0.0, 255.0))
            directions.append(direction)
            strengths.append(strength)

        rows.append({
            "paper_bgr": paper.tolist(),
            "paper_hex": bgr_hex(paper),
            "group_colours_bgr": [colour.tolist() for colour in colours],
            "group_colours_hex": [bgr_hex(colour) for colour in colours],
            "group_od_directions": [direction.tolist() for direction in directions],
            "group_od_strengths": strengths,
        })
    return rows


def median_bgr_hex(rows: list[dict], key: str, group: int | None = None) -> str:
    if group is None:
        values = np.array([row[key] for row in rows], np.float64)
    else:
        values = np.array([row[key][group] for row in rows], np.float64)
    return bgr_hex(np.median(values, axis=0))


def render_layer(
    assignment: np.ndarray,
    darkness: np.ndarray,
    group: int,
    paper_bgr: np.ndarray,
    ink_bgr: np.ndarray,
) -> np.ndarray:
    alpha = np.where(assignment == group, darkness, 0.0)[:, :, None]
    alpha = np.clip(alpha / max(float(np.percentile(alpha[alpha > 0], 92)), 1e-6), 0, 1)
    output = paper_bgr[None, None, :] * (1.0 - alpha) + ink_bgr[None, None, :] * alpha
    return np.clip(output, 0, 255).astype(np.uint8)


def write_outputs(
    output: Path,
    scan_path: Path,
    numbers: list[int],
    reference_bgr: np.ndarray,
    partition: dict,
    colour_control: dict,
    flows: np.ndarray,
    flow_valid: np.ndarray,
    colours: list[dict],
    downsample: int,
    force_colour_control: bool,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    motion_assignment = partition["assignment"]
    colour_assignment = colour_control["assignment"]
    agreement, motion_flipped, agreement_pixels = balanced_partition_agreement(
        motion_assignment, colour_assignment
    )
    # When ink rays are genuinely distinct, colour is an independent and much
    # better-conditioned source of the spatial masks. Motion remains the test:
    # agreement says whether microshifts alone rediscovered those same masks.
    if colour_control["distinct"] or force_colour_control:
        assignment = colour_assignment
        selected_method = (
            "colour_control" if colour_control["distinct"]
            else "colour_control_forced"
        )
        shifts, scatters = shifts_for_partition(
            flows, flow_valid, assignment, partition["gradient"]
        )
    else:
        assignment = motion_assignment
        selected_method = "motion_only_unvalidated"
        shifts = partition["shifts"]
        scatters = partition["scatters"]
    darkness = partition["darkness"]

    spreads = [group_spread(assignment, group, darkness) for group in (0, 1)]
    denomination_match = re.match(r"(\d+)", scan_path.stem)
    denomination = int(denomination_match.group(1)) if denomination_match else None
    if denomination is not None and denomination <= 7:
        # In 1--7K the first impression is the pale background and the second
        # is the denser outline. Spatial extent alone is ambiguous because the
        # outline includes the outer frame and therefore spans the whole stamp.
        group_density = [
            float(np.median(darkness[assignment == group])) for group in (0, 1)
        ]
        stage1_group = int(np.argmin(group_density))
        mapping_reason = "lower-density group (background)"
    else:
        # In 14--70K the first pass is the broad pattern and the second is the
        # spatially localised coat of arms.
        stage1_group = int(np.argmax(spreads))
        mapping_reason = "broader group (pattern)"
    stage2_group = 1 - stage1_group
    order = [stage1_group, stage2_group]

    paper_bgr = np.median(np.array([row["paper_bgr"] for row in colours]), axis=0)
    ink_bgr = [
        np.median(np.array([row["group_colours_bgr"][group] for row in colours]), axis=0)
        for group in (0, 1)
    ]

    reference_path = output / "registered_colour_median.png"
    cv2.imwrite(str(reference_path), np.clip(reference_bgr, 0, 255).astype(np.uint8))
    stage_paths: list[Path] = []
    for stage_number, group in enumerate(order, 1):
        path = output / f"stage{stage_number}.png"
        cv2.imwrite(
            str(path),
            render_layer(assignment, darkness, group, paper_bgr, ink_bgr[group]),
        )
        stage_paths.append(path)

    palette = [np.array([255, 190, 30]), np.array([35, 70, 255])]
    overlay_paths = {}
    for name, source in (
        ("motion_only_partition", motion_assignment),
        ("colour_partition", colour_assignment),
        ("selected_partition", assignment),
    ):
        overlay = np.clip(reference_bgr * 0.42, 0, 255).astype(np.uint8)
        source_order = order
        if name == "motion_only_partition" and motion_flipped:
            source_order = [1 - order[0], 1 - order[1]]
        for stage_number, group in enumerate(source_order):
            where = source == group
            overlay[where] = np.clip(
                0.35 * overlay[where] + 0.65 * palette[stage_number], 0, 255
            ).astype(np.uint8)
        path = output / f"{name}.png"
        cv2.imwrite(str(path), overlay)
        overlay_paths[name] = path

    relative = shifts[:, stage2_group] - shifts[:, stage1_group]
    relative_full = relative * downsample
    rms_relative = float(np.sqrt(np.mean(np.sum(relative_full ** 2, axis=1))))
    variation = float(np.sqrt(np.sum(np.var(relative_full, axis=0, ddof=1))))
    median_scatter = float(np.median(scatters) * downsample)
    evidence_ratio = variation / max(median_scatter, 1e-6)
    if colour_control["distinct"] and agreement >= 0.80 and evidence_ratio >= 1.0:
        verdict = "strong"
    elif colour_control["distinct"] and agreement >= 0.68 and evidence_ratio >= 0.55:
        verdict = "tentative"
    else:
        verdict = "insufficient"

    stamp_rows = []
    for index, number in enumerate(numbers):
        colour_row = colours[index]
        stamp_rows.append({
            "stamp": number,
            "stage1_shift_x_px": float(shifts[index, stage1_group, 0] * downsample),
            "stage1_shift_y_px": float(shifts[index, stage1_group, 1] * downsample),
            "stage2_shift_x_px": float(shifts[index, stage2_group, 0] * downsample),
            "stage2_shift_y_px": float(shifts[index, stage2_group, 1] * downsample),
            "relative_shift_x_px": float(relative_full[index, 0]),
            "relative_shift_y_px": float(relative_full[index, 1]),
            "paper_hex": colour_row["paper_hex"],
            "stage1_hex": colour_row["group_colours_hex"][stage1_group],
            "stage2_hex": colour_row["group_colours_hex"][stage2_group],
            "stage1_od_direction_bgr": colour_row["group_od_directions"][stage1_group],
            "stage2_od_direction_bgr": colour_row["group_od_directions"][stage2_group],
            "stage1_od_strength": colour_row["group_od_strengths"][stage1_group],
            "stage2_od_strength": colour_row["group_od_strengths"][stage2_group],
        })

    payload = {
        "format": "experimental-two-pass-decomposition",
        "version": 1,
        "source": scan_path.name,
        "method": "motion-only clustering of dense optical-flow signatures",
        "selected_output_method": selected_method,
        "downsample": downsample,
        "stamps": len(numbers),
        "stage_mapping": {
            "stage1": "background for 1--7K; pattern for 14--70K",
            "stage2": "edges for 1--7K; coat of arms for 14--70K",
            "mapping_reason": mapping_reason,
            "stage1_internal_group": stage1_group,
            "stage2_internal_group": stage2_group,
        },
        "diagnostics": {
            "motion_candidate_pixels": partition["candidate_count"],
            "cluster_separation": partition["cluster_separation"],
            "relative_shift_rms_px": rms_relative,
            "relative_shift_variation_px": variation,
            "median_within_group_scatter_px": median_scatter,
            "variation_to_scatter_ratio": evidence_ratio,
            "microshift_evidence": verdict,
            "group_spatial_spread": spreads,
            "colour_control_distinct": colour_control["distinct"],
            "colour_direction_separation": colour_control["direction_separation"],
            "colour_cluster_separation": colour_control["cluster_separation"],
            "motion_vs_colour_balanced_agreement": agreement,
            "motion_vs_colour_common_pixels": agreement_pixels,
        },
        "collection_colours": {
            "paper_hex": bgr_hex(paper_bgr),
            "stage1_hex": bgr_hex(ink_bgr[stage1_group]),
            "stage2_hex": bgr_hex(ink_bgr[stage2_group]),
            "reliable": bool(selected_method == "colour_control" or verdict != "insufficient"),
            "note": (
                "Scanner-space effective colours, not calibrated physical ink "
                "reflectance; absolute values depend on scanner/profile/exposure. "
                "When reliable is false, the forced two-group colours are "
                "diagnostic only and must not be interpreted as two inks."
            ),
        },
        "outputs": {
            "registered_colour_median": reference_path.name,
            "motion_only_partition": overlay_paths["motion_only_partition"].name,
            "colour_partition": overlay_paths["colour_partition"].name,
            "selected_partition": overlay_paths["selected_partition"].name,
            "stage1": stage_paths[0].name,
            "stage2": stage_paths[1].name,
        },
        "per_stamp": stamp_rows,
    }
    json_path = output / "result.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output / "per_stamp.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stamp_rows[0]))
        writer.writeheader()
        writer.writerows(stamp_rows)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Experimentally split a registered stamp design into two printing "
            "passes using differential micro-movements across stamps."
        )
    )
    parser.add_argument("scan", type=Path, help="Original scan")
    parser.add_argument("--perf", type=Path, help="Perforation JSON")
    parser.add_argument("--design", type=Path, help="Design-size JSON")
    parser.add_argument("--output", type=Path, help="Output directory")
    parser.add_argument(
        "--downsample", type=int, default=2,
        help="Optical-flow scale reduction (default: 2)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=60000,
        help="Maximum edge pixels used to fit the two clusters",
    )
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument(
        "--force-colour-control", action="store_true",
        help=(
            "Use the two-ray colour partition even when its collection-wide "
            "separation is below the reliability threshold"
        ),
    )
    parser.add_argument(
        "--save-stamps", default="",
        help="Comma-separated stamp numbers whose aligned images are saved",
    )
    args = parser.parse_args()
    if args.downsample < 1:
        parser.error("--downsample must be positive")

    perf_path = args.perf or args.scan.with_name("data_" + args.scan.stem + "_perf.json")
    design_path = args.design or args.scan.with_name("data_" + args.scan.stem + "_design.json")
    output = args.output or args.scan.with_name(args.scan.stem + "_two_pass")
    perf = json.loads(perf_path.read_text(encoding="utf-8"))
    design = json.loads(design_path.read_text(encoding="utf-8"))

    print(f"Loading and aligning {args.scan.name} ...")
    images, valid, numbers = aligned_collection(
        args.scan, perf, design, args.downsample
    )
    print(f"  {len(numbers)} stamps, analysis canvas {images.shape[2]}x{images.shape[1]}")
    reference_bgr = masked_colour_median(images, valid)
    print("Estimating dense per-stamp micro-movements ...")
    flows, flow_valid, _ = flow_stack(images, valid, reference_bgr)
    print("Clustering design pixels by motion signature ...")
    partition = motion_partition(
        flows,
        flow_valid,
        reference_bgr,
        np.random.default_rng(args.seed),
        args.max_samples,
    )
    colour_control = colour_partition(
        reference_bgr,
        partition["darkness"],
        np.random.default_rng(args.seed + 1),
        args.max_samples,
    )
    if colour_control["distinct"] or args.force_colour_control:
        selected_assignment = colour_control["assignment"]
        selected_shifts, _ = shifts_for_partition(
            flows, flow_valid, selected_assignment, partition["gradient"]
        )
    else:
        selected_assignment = partition["assignment"]
        selected_shifts = partition["shifts"]
    colours = representative_colours(
        images,
        valid,
        selected_assignment,
        partition["darkness"],
        selected_shifts,
    )
    payload = write_outputs(
        output,
        args.scan,
        numbers,
        reference_bgr,
        partition,
        colour_control,
        flows,
        flow_valid,
        colours,
        args.downsample,
        args.force_colour_control,
    )
    if args.save_stamps:
        requested = {int(value.strip()) for value in args.save_stamps.split(",") if value.strip()}
        stamp_dir = output / "aligned_stamps"
        stamp_dir.mkdir(parents=True, exist_ok=True)
        for image, number in zip(images, numbers):
            if number in requested:
                cv2.imwrite(str(stamp_dir / f"stamp_{number}.png"), image)
    diagnostics = payload["diagnostics"]
    colours_out = payload["collection_colours"]
    print(
        f"Evidence: {diagnostics['microshift_evidence']} "
        f"(cluster separation {diagnostics['cluster_separation']:.2f}, "
        f"relative-shift variation {diagnostics['relative_shift_variation_px']:.2f}px, "
        f"noise {diagnostics['median_within_group_scatter_px']:.2f}px)"
    )
    print(
        f"Colours: stage 1 {colours_out['stage1_hex']}, "
        f"stage 2 {colours_out['stage2_hex']}, paper {colours_out['paper_hex']}"
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
