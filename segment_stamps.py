#!/usr/bin/env python3

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory
from pathlib import Path

import cv2
import numpy as np

from edge_profiles import profile_depth
from perforation import (PHASE_NAMES, PHASE_SEED_SOURCES, SIDES,
                         boundary_profile, draw_measurement,
                         fit_extreme_arc_lattice, measure_stamp, method_phase,
                         phase_color)
from refine_perforation import fit_arc, fit_parabola


def _reading_order(boxes, typical_height):
    """Stable numbering only; rows are not used to decide what is a stamp."""
    remaining = list(boxes)
    ordered = []
    band = max(1.0, 0.45 * typical_height)
    while remaining:
        anchor = min(remaining, key=lambda b: ((b[1] + b[3]) / 2, b[0]))
        anchor_y = (anchor[1] + anchor[3]) / 2
        same_line = [
            box for box in remaining
            if abs((box[1] + box[3]) / 2 - anchor_y) <= band
        ]
        same_line.sort(key=lambda b: (b[0], b[1]))
        ordered.extend(same_line)
        selected = {id(box) for box in same_line}
        remaining = [box for box in remaining if id(box) not in selected]
    return ordered


def detect_stamps_2d(
    image,
    max_processing_width=3000,
    background_delta=35,
):
    """Detect stamp-shaped 2-D components in the complete scan.

    Detection does not assume a single backing-paper rectangle and does not
    project the image into rows.  Several nearby thresholds make pale and dark
    stamps available as proposals; overlapping proposals are consolidated in
    two dimensions.  A component about twice the normal stamp width/height is
    split, which retains touching pairs.
    """
    image_h, image_w = image.shape[:2]
    scale = min(1.0, max_processing_width / image_w)
    if scale < 1.0:
        work = cv2.resize(image, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA)
    else:
        work = image.copy()
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    work_h, work_w = gray.shape

    # The lower percentile samples the black mounting sheet even when a ruler
    # divides it into separate regions.  Avoid allowing a very bright exterior
    # to determine the threshold.
    backing_level = float(np.percentile(gray, 20))
    deltas = sorted({
        max(12.0, background_delta - 10.0),
        max(15.0, background_delta - 5.0),
        float(background_delta),
        min(70.0, background_delta + 10.0),
    })

    component_sets = []
    orientation_foregrounds = []
    for delta in deltas:
        foreground = (gray > min(250.0, backing_level + delta)).astype(np.uint8)
        orientation_foregrounds.append(foreground.copy())
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            foreground, 8
        )
        components = []
        border_labels = set()
        for label in range(1, count):
            x, y, width, height, area = [int(v) for v in stats[label]]
            if x == 0 or y == 0 or x + width == work_w or y + height == work_h:
                border_labels.add(label)
                continue
            if area < max(100, work_w * work_h * 0.00015):
                continue
            components.append({
                'box': (x, y, x + width - 1, y + height - 1),
                'width': width,
                'height': height,
                'area': area,
                'center': tuple(float(v) for v in centroids[label]),
            })
        if border_labels:
            foreground[np.isin(labels, list(border_labels))] = 0
        component_sets.append((foreground, components))

    # Estimate the scale from compact, stamp-sized components rather than from
    # their position.  Broad limits are relative to processing width, so scans
    # at other pixel dimensions remain supported.
    reference_components = component_sets[min(1, len(component_sets) - 1)][1]
    scale_candidates = [
        component for component in reference_components
        if (0.055 * work_w <= component['width'] <= 0.14 * work_w
            and 0.075 * work_w <= component['height'] <= 0.18 * work_w
            and 0.28 <= component['width'] / component['height'] <= 1.35
            and component['area'] >= 0.22 * component['width'] * component['height'])
    ]
    if not scale_candidates:
        return [], np.zeros((image_h, image_w), dtype=np.uint8)
    typical_width = float(np.median([c['width'] for c in scale_candidates]))
    typical_height = float(np.median([c['height'] for c in scale_candidates]))
    typical_area = float(np.median([c['area'] for c in scale_candidates]))

    proposals = []
    for level, (_, components) in enumerate(component_sets):
        for component in components:
            width = component['width']
            height = component['height']
            area = component['area']
            if area < 0.16 * typical_area:
                continue
            if not (0.28 * typical_width <= width <= 3.2 * typical_width
                    and 0.28 * typical_height <= height <= 3.2 * typical_height):
                continue
            if area < 0.14 * width * height:
                continue

            nx = max(1, int(round(width / typical_width))) if width > 1.5 * typical_width else 1
            ny = max(1, int(round(height / typical_height))) if height > 1.5 * typical_height else 1
            if nx * ny > 4:
                continue
            x0, y0, x1, y1 = component['box']
            x_edges = np.linspace(x0, x1 + 1, nx + 1)
            y_edges = np.linspace(y0, y1 + 1, ny + 1)
            for iy in range(ny):
                for ix in range(nx):
                    sx0, sx1 = x_edges[ix], x_edges[ix + 1] - 1
                    sy0, sy1 = y_edges[iy], y_edges[iy + 1] - 1
                    cx = (sx0 + sx1) / 2
                    cy = (sy0 + sy1) / 2
                    proposal_w = max(typical_width, sx1 - sx0 + 1)
                    proposal_h = max(typical_height, sy1 - sy0 + 1)
                    proposals.append({
                        'box': (
                            int(round(cx - proposal_w / 2)),
                            int(round(cy - proposal_h / 2)),
                            int(round(cx + proposal_w / 2)),
                            int(round(cy + proposal_h / 2)),
                        ),
                        'score': area / max(1.0, nx * ny),
                        'level': level,
                    })

    def intersection_over_union(first, second):
        ax0, ay0, ax1, ay1 = first
        bx0, by0, bx1, by1 = second
        width = max(0, min(ax1, bx1) - max(ax0, bx0) + 1)
        height = max(0, min(ay1, by1) - max(ay0, by0) + 1)
        intersection = width * height
        union = ((ax1 - ax0 + 1) * (ay1 - ay0 + 1)
                 + (bx1 - bx0 + 1) * (by1 - by0 + 1) - intersection)
        return intersection / union if union else 0.0

    # Non-maximum suppression merges threshold variants and fragments of one
    # heavily cancelled stamp, but does not merge neighbouring stamps.
    selected = []
    for proposal in sorted(proposals, key=lambda item: item['score'], reverse=True):
        box = proposal['box']
        supporting_levels = {
            candidate['level'] for candidate in proposals
            if intersection_over_union(box, candidate['box']) >= 0.25
        }
        if len(supporting_levels) < 2:
            continue
        if any(intersection_over_union(box, old) >= 0.25 for old in selected):
            continue
        x0, y0, x1, y1 = box
        selected.append((
            max(0, x0), max(0, y0),
            min(work_w - 1, x1), min(work_h - 1, y1),
        ))

    selected = _reading_order(selected, typical_height)
    boxes = []
    for x0, y0, x1, y1 in selected:
        boxes.append((
            int(round(x0 / scale)),
            int(round(y0 / scale)),
            min(image_w - 1, int(round((x1 + 1) / scale)) - 1),
            min(image_h - 1, int(round((y1 + 1) / scale)) - 1),
        ))

    # The most inclusive threshold is used only for local orientation fitting;
    # it cannot introduce additional detected objects.
    debug_foreground = orientation_foregrounds[0] * 255
    if scale < 1.0:
        debug_mask = cv2.resize(debug_foreground, (image_w, image_h),
                                interpolation=cv2.INTER_NEAREST)
    else:
        debug_mask = debug_foreground
    return boxes, debug_mask


def estimate_orientation(mask, box):
    """Fit paper edges; angle is clockwise from image horizontal, modulo 90.

    Central edge sections exclude corners; robust line fits suppress perforation
    and damage. Dimensions describe a fitted outer envelope, not paper in mm.
    Coordinates are relative to the backing-paper crop.
    """
    x0, y0, x1, y1 = box
    local = mask[y0:y1 + 1, x0:x1 + 1]
    # Cancellation ink can split the threshold mask into disconnected pieces.
    # Close narrow gaps locally, with explicit black padding at crop boundaries.
    kernel_size = max(3, int(round(min(local.shape) * 0.04)) | 1)
    padded = cv2.copyMakeBorder(local, kernel_size, kernel_size,
                               kernel_size, kernel_size, cv2.BORDER_CONSTANT, value=0)
    closed = cv2.morphologyEx(padded, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                       (kernel_size, kernel_size)))
    local = closed[kernel_size:-kernel_size, kernel_size:-kernel_size]
    contours, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 0.2 * local.size:
        return None
    points = contour.reshape(-1, 2).astype(np.float32)
    rect = cv2.minAreaRect(points)
    initial = (rect[2] + 45.0) % 90.0 - 45.0

    def axes(angle):
        radians = np.deg2rad(angle)
        return np.array([[np.cos(radians), np.sin(radians)],
                         [-np.sin(radians), np.cos(radians)]])

    projected = points @ axes(initial).T
    lo, hi = projected.min(axis=0), projected.max(axis=0)
    extent = hi - lo
    if min(extent) < 10:
        return None
    angles = []
    for normal in (0, 1):
        along = 1 - normal
        central = ((projected[:, along] > lo[along] + 0.15 * extent[along]) &
                   (projected[:, along] < hi[along] - 0.15 * extent[along]))
        for side in (lo[normal], hi[normal]):
            selected = points[central &
                              (np.abs(projected[:, normal] - side) < 0.08 * extent[normal])]
            if len(selected) < 20:
                continue
            vx, vy, _, _ = cv2.fitLine(selected, cv2.DIST_HUBER, 0, 0.001, 0.001).ravel()
            angle = (np.rad2deg(np.arctan2(vy, vx)) + 45.0) % 90.0 - 45.0
            # Unwrap near the initial angle for stamps close to +/-45 degrees.
            angles.append(initial + (angle - initial + 45.0) % 90.0 - 45.0)
    angle = float(np.median(angles)) if angles else initial
    spread = float(max(angles) - min(angles)) if len(angles) >= 2 else None
    projected = points @ axes(angle).T
    # Quantiles ignore isolated fibres while following the outer tooth envelope.
    lo, hi = np.percentile(projected, [1, 99], axis=0)
    corners = np.array([[lo[0], lo[1]], [hi[0], lo[1]],
                        [hi[0], hi[1]], [lo[0], hi[1]]]) @ axes(angle)
    corners += [x0, y0]
    center = corners.mean(axis=0)
    return {
        'angle_deg': (angle + 45.0) % 90.0 - 45.0,
        'width_px': float(hi[0] - lo[0]),
        'height_px': float(hi[1] - lo[1]),
        'center_x': float(center[0]), 'center_y': float(center[1]),
        'corners': corners,
        'edge_spread_deg': spread,
        'status': 'ok' if len(angles) >= 3 and spread <= 2.0 else 'review',
    }


def refine_orientation_from_measurement(orientation, measurement,
                                        consensus_tolerance_deg=1.5):
    """Refine the coarse angle from perforation base lines in image space.

    The coarse contour angle only has to make the first rectification usable.
    Circle-supported perforation lines are a much cleaner orientation signal:
    all four sides of a rigid stamp have the same angle modulo 90 degrees.
    Require at least two agreeing sides so one damaged edge cannot rotate the
    second pass on its own.
    """
    if orientation is None:
        return None
    coarse_angle = float(orientation['angle_deg'])
    candidates = []
    for side in SIDES:
        edge = measurement.get('sides', {}).get(side, {})
        line = edge.get('line_image')
        if (line is None or edge.get('geometry_source') != 'circle_arcs'
                or edge.get('count', 0) < 5):
            continue
        line = np.asarray(line, dtype=float)
        delta = line[1] - line[0]
        if not np.isfinite(delta).all() or np.linalg.norm(delta) < 1:
            continue
        raw = (np.rad2deg(np.arctan2(delta[1], delta[0])) + 45.0) % 90.0 - 45.0
        angle = coarse_angle + (raw - coarse_angle + 45.0) % 90.0 - 45.0
        weight = float(min(20, edge.get('count', 5)))
        if edge.get('status') == 'ok':
            weight *= 1.15
        if edge.get('method') == 'parallel_edge_arc_recovery':
            weight *= 0.85
        candidates.append({'side': side, 'angle_deg': float(angle), 'weight': weight})
    if len(candidates) < 2:
        return dict(orientation)

    # Find the strongest mutually agreeing cluster rather than averaging an
    # isolated false edge into the result.
    best = None
    for center in [item['angle_deg'] for item in candidates]:
        cluster = [
            item for item in candidates
            if abs(item['angle_deg'] - center) <= consensus_tolerance_deg
        ]
        score = (len(cluster), sum(item['weight'] for item in cluster))
        if best is None or score > best[0]:
            best = (score, cluster)
    inliers = best[1]
    if len(inliers) < 2:
        return dict(orientation)
    weights = np.array([item['weight'] for item in inliers], dtype=float)
    angles = np.array([item['angle_deg'] for item in inliers], dtype=float)
    refined_angle = float(np.average(angles, weights=weights))
    # A very large jump means the coarse warp did not provide a trustworthy
    # first measurement; keep it reviewable rather than applying it silently.
    if abs(refined_angle - coarse_angle) > 10.0:
        return dict(orientation)

    refined = dict(orientation)
    refined['angle_deg'] = (refined_angle + 45.0) % 90.0 - 45.0
    refined['coarse_angle_deg'] = coarse_angle
    refined['orientation_correction_deg'] = refined['angle_deg'] - coarse_angle
    refined['orientation_source'] = 'perforation_line_consensus'
    refined['orientation_edge_angles_deg'] = {
        item['side']: item['angle_deg'] for item in candidates
    }
    refined['orientation_inlier_sides'] = [item['side'] for item in inliers]
    refined['edge_spread_deg'] = float(np.ptp(angles))
    refined['status'] = 'ok' if len(inliers) >= 3 and np.ptp(angles) <= 1.0 else 'review'
    width, height = refined['width_px'], refined['height_px']
    center = np.array([refined['center_x'], refined['center_y']], dtype=float)
    radians = np.deg2rad(refined['angle_deg'])
    rotation = np.array([[np.cos(radians), np.sin(radians)],
                         [-np.sin(radians), np.cos(radians)]])
    local_corners = np.array([
        [-width / 2, -height / 2], [width / 2, -height / 2],
        [width / 2, height / 2], [-width / 2, height / 2],
    ])
    refined['corners'] = local_corners @ rotation + center
    return refined


def perforation_label(measurement, dpi):
    """Horizontal x vertical gauge, retaining discordant opposite sides."""
    values = []
    for names in (('top', 'bottom'), ('left', 'right')):
        gauge = average_perforation(measurement, names, dpi)
        if gauge is None:
            values.append('--')
        elif isinstance(gauge, tuple):
            values.append(f'({gauge[0]:.2f}/{gauge[1]:.2f})')
        else:
            values.append(f'{gauge:.2f}')
    return ' x '.join(values)


PERFORATION_LIMITS = (5.0, 22.0)
PERFORATION_PRIMARY_LIMITS = (8.0, 17.0)
PERFORATION_AGREEMENT = 0.25


def _edge_gauge(edge, dpi):
    """Return an arc-supported gauge inside the absolute admissible range."""
    pitch = edge.get('pitch_px', 0)
    if (not dpi or pitch <= 0
            or edge.get('geometry_source') != 'circle_arcs'
            or edge.get('count', 0) < 5):
        return None
    gauge = 20 * dpi / (25.4 * pitch)
    return gauge if PERFORATION_LIMITS[0] <= gauge <= PERFORATION_LIMITS[1] else None


def _refit_circle_pitch(edge, reference_pitch, dpi):
    """Refit a side near a target pitch using every in-bounds circle fit."""
    points = np.asarray(edge.get('valleys', []), dtype=float).reshape(-1, 2)
    fits = edge.get('refinement_fits', [])
    inside = np.asarray(edge.get('inside_corner_bounds',
                                 np.ones(len(points), bool)), dtype=bool)
    circle_indices = np.array([
        index for index in range(min(len(points), len(fits), len(inside)))
        if inside[index]
        and fits[index] is not None and fits[index].get('model') == 'circle'
    ], dtype=int)
    if len(circle_indices) < 5:
        return None
    values = points[circle_indices, 0]
    all_along = points[:, 0] if len(points) else values
    positions = np.array([np.min(all_along), np.max(all_along)], dtype=float)
    slope = float(edge.get('slope', 0.0))
    axis_reference = (None if reference_pitch is None else
                      reference_pitch / np.sqrt(1 + slope * slope))
    lattice_fit = fit_extreme_arc_lattice(values, positions, axis_reference)
    if lattice_fit is None:
        return None
    _, _, chosen, lattice = lattice_fit
    selected = circle_indices[chosen]
    spacing = np.polyfit(lattice, points[selected, 0], 1)
    pitch = float(spacing[0] * np.sqrt(1 + slope * slope))
    gauge = 20 * dpi / (25.4 * pitch)
    if not (PERFORATION_LIMITS[0] <= gauge <= PERFORATION_LIMITS[1]):
        return None
    residual = points[selected, 0] - np.polyval(spacing, lattice)
    mask = np.zeros(len(points), dtype=bool)
    mask[selected] = True
    result = {
        'accepted': mask,
        'pitch_px': pitch,
        'count': int(mask.sum()),
        'refined_count': int(mask.sum()),
        'spacing_rms_px': float(np.sqrt(np.mean(residual ** 2))),
        'gauge': float(gauge),
    }
    omitted = np.setdiff1d(circle_indices, selected)
    if len(omitted) == 1:
        result['excluded_point_index'] = int(omitted[0])
    return result


def reconcile_perforation(measurement, side_names, dpi):
    """Reconcile opposite sides, including exact pitch harmonics."""
    if not dpi:
        return
    sides = [measurement.get('sides', {}).get(name, {}) for name in side_names]
    # This is deliberately a final-period pass.  It uses all already detected
    # in-bounds circles, but does not feed back into edge or orientation search.
    for edge in sides:
        refit = _refit_circle_pitch(edge, None, dpi)
        if refit is not None:
            edge.update({key: value for key, value in refit.items()
                         if key != 'gauge'})
    valid = [_edge_gauge(edge, dpi) for edge in sides]
    if (valid[0] is not None and valid[1] is not None
            and abs(valid[0] - valid[1]) <= PERFORATION_AGREEMENT):
        return

    candidates = []
    # First try the other side's period directly.  The lattice fitter may
    # discard one uniquely incompatible circle, but never a convenient subset.
    for side_index, edge in enumerate(sides):
        other_index = 1 - side_index
        other_pitch = sides[other_index].get('pitch_px', 0)
        other_gauge = valid[other_index]
        if other_pitch <= 0 or other_gauge is None:
            continue
        refit = _refit_circle_pitch(edge, float(other_pitch), dpi)
        if refit is None:
            continue
        difference = abs(refit['gauge'] - other_gauge)
        if difference > PERFORATION_AGREEMENT:
            continue
        primary = (PERFORATION_PRIMARY_LIMITS[0] <= refit['gauge']
                   <= PERFORATION_PRIMARY_LIMITS[1]
                   and PERFORATION_PRIMARY_LIMITS[0] <= other_gauge
                   <= PERFORATION_PRIMARY_LIMITS[1])
        candidates.append((not primary, difference,
                           refit['spacing_rms_px'], side_index, refit))

    # If one side landed on every second (or third) hole, test the smaller
    # period only after the two independently measured sides disagree.
    if valid[0] is not None and valid[1] is not None:
        larger_index = int(sides[1].get('pitch_px', 0)
                           > sides[0].get('pitch_px', 0))
        other_index = 1 - larger_index
        other_gauge = valid[other_index]
        larger_pitch = float(sides[larger_index]['pitch_px'])
        for divisor in (2, 3):
            target_pitch = larger_pitch / divisor
            implied_gauge = 20 * dpi / (25.4 * target_pitch)
            if abs(implied_gauge - other_gauge) > PERFORATION_AGREEMENT:
                continue
            refit = _refit_circle_pitch(
                sides[larger_index], target_pitch, dpi,
            )
            if (refit is None
                    or abs(refit['gauge'] - other_gauge)
                    > PERFORATION_AGREEMENT):
                # Sparse points may contain only every second/third hole, so a
                # target-period refit can remain underdetermined.  The exact
                # harmonic agreement itself is the requested final evidence.
                refit = {
                    'pitch_px': target_pitch,
                    'gauge': implied_gauge,
                    'harmonic_divisor': divisor,
                    'spacing_rms_px': float(
                        sides[larger_index].get('spacing_rms_px', 0.0)
                    ),
                }
            difference = abs(refit['gauge'] - other_gauge)
            candidates.append((False, difference,
                               refit['spacing_rms_px'], larger_index, refit))
    if candidates:
        _, _, _, side_index, refit = min(
            candidates, key=lambda candidate: candidate[:3],
        )
        sides[side_index].update({
            key: value for key, value in refit.items() if key != 'gauge'
        })


def average_perforation(measurement, side_names, dpi):
    """Return one gauge, or both side values when they cannot be reconciled."""
    if dpi is None:
        return None
    reconcile_perforation(measurement, side_names, dpi)
    sides = [measurement['sides'].get(name, {}) for name in side_names]
    available = [(side, _edge_gauge(side, dpi)) for side in sides]
    available = [(side, gauge) for side, gauge in available if gauge is not None]
    native = [side for side in available
              if side[0].get('method') != 'parallel_edge_arc_recovery']
    if native:
        # Do not let an ambiguous reconstruction move a measured gauge.
        available = [
            side for side in available
            if side[0].get('method') != 'parallel_edge_arc_recovery'
            or (side[0].get('spacing_rms_px', float('inf')) / side[0]['pitch_px'] < 0.06
                and all(abs(side[0]['pitch_px'] / other[0]['pitch_px'] - 1) <= 0.04
                        for other in native))
        ]
    if not available:
        return None
    gauges = [gauge for _, gauge in available]
    if len(gauges) == 1:
        return float(gauges[0])
    if abs(gauges[0] - gauges[1]) <= PERFORATION_AGREEMENT:
        return float(np.mean(gauges))
    return tuple(float(gauge) for gauge in gauges)


def summary_row(input_path, number, measurement, dpi):
    """Build the compact per-stamp row embedded in the results JSON."""
    width_px = measurement.get('valley_width_px')
    height_px = measurement.get('valley_height_px')
    def summary_gauge(value):
        if isinstance(value, tuple):
            return f'({value[0]:.4f}/{value[1]:.4f})'
        return value

    row = {
        'source': Path(input_path).name,
        'stamp': number,
        'width_px': width_px,
        'height_px': height_px,
        'width_mm': width_px * 25.4 / dpi if width_px is not None and dpi else None,
        'height_mm': height_px * 25.4 / dpi if height_px is not None and dpi else None,
        'horizontal_perforation_per_20mm': summary_gauge(
            average_perforation(measurement, ('top', 'bottom'), dpi)
        ),
        'vertical_perforation_per_20mm': summary_gauge(
            average_perforation(measurement, ('left', 'right'), dpi)
        ),
    }
    return {
        key: round(value, 4) if isinstance(value, float) else value
        for key, value in row.items()
    }


def _json_value(value):
    """Convert NumPy-rich measurement structures to portable JSON values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()
                if not key.startswith('debug_')}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _hole_indices(edge):
    """Map public hole ids to candidates in physical order along a side."""
    points = edge.get('valleys', np.empty((0, 2)))
    initial = edge.get('initial_valleys', points)
    fits = edge.get('refinement_fits', [])
    count = max(len(points), len(initial), len(fits))
    def along(index):
        if index < len(initial):
            return float(initial[index][0])
        if index < len(points):
            return float(points[index][0])
        return float('inf')
    return sorted(range(count), key=along)


def measurement_details(measurement):
    """Serialize all numeric results and give every attempted hole a stable id.

    This block is written for downstream aggregation and for agent-driven
    debugging, not for direct human reading: ``status``, ``reason``, ``phase``
    and ``seed_source`` are included so a consumer can tell which phase of the
    fallback ladder produced each side and whether its arcs were seeded from
    the image or from a pitch hypothesis.
    """
    details = {
        key: _json_value(value)
        for key, value in measurement.items()
        if key != 'sides' and not key.startswith('debug_')
    }
    details['sides'] = {}
    for side in SIDES:
        edge = measurement.get('sides', {}).get(side, {
            'status': 'unavailable', 'reason': 'no orientation'
        })
        side_result = {
            key: _json_value(value)
            for key, value in edge.items()
            if key not in {
                'valleys', 'initial_valleys', 'points_image',
                'initial_points_image', 'accepted', 'inside_corner_bounds',
                'refinement_fits',
            } and not key.startswith('debug_')
        }
        phase = method_phase(edge)
        side_result['phase'] = phase
        side_result['phase_name'] = PHASE_NAMES.get(phase)
        side_result['seed_source'] = PHASE_SEED_SOURCES.get(phase)
        points = edge.get('points_image', np.empty((0, 2)))
        initial = edge.get('initial_points_image', points)
        accepted = edge.get('accepted', np.zeros(len(points), dtype=bool))
        inside = edge.get('inside_corner_bounds', np.ones(len(points), dtype=bool))
        fits = edge.get('refinement_fits', [None] * len(points))
        holes = []
        for public_id, index in enumerate(_hole_indices(edge), 1):
            fit = fits[index] if index < len(fits) else None
            hole = {
                'id': public_id,
                'accepted': bool(accepted[index]) if index < len(accepted) else False,
                'inside_corner_bounds': bool(inside[index]) if index < len(inside) else True,
                'point_image': _json_value(points[index]) if index < len(points) else None,
                'initial_point_image': (
                    _json_value(initial[index]) if index < len(initial) else None
                ),
                'refinement': None,
            }
            if fit is not None and fit.get('model') == 'circle':
                hole['category'] = ('perforation' if hole['accepted']
                                    else 'ignored_perforation')
            else:
                hole['category'] = 'approximation'
            if fit is not None:
                hole['refinement'] = {
                    key: _json_value(value)
                    for key, value in fit.items()
                    if key != 'curve_image'
                }
                if 'curve_image' in fit:
                    hole['refinement']['curve_image'] = _json_value(fit['curve_image'])
            holes.append(hole)
        side_result['holes'] = holes
        details['sides'][side] = side_result
    return details


def write_perf_json(path, input_path, boxes, orientations, measurements, dpi,
                    parameters, coarse_orientations=None):
    def public_orientation(value):
        # ``status`` is retained: a weak angle consensus explains a weak
        # measurement, and that link is wanted for automated debugging.
        return None if value is None else _json_value(value)

    stamps = []
    for number, (box, orientation, measurement) in enumerate(
            zip(boxes, orientations, measurements), 1):
        if measurement is None:
            continue
        stamp = {
            'stamp': number,
            'bbox': {'x0': box[0], 'y0': box[1], 'x1': box[2], 'y1': box[3]},
            'orientation': public_orientation(orientation),
            'summary': summary_row(input_path, number, measurement, dpi),
            'measurement': measurement_details(measurement),
        }
        if coarse_orientations is not None:
            stamp['coarse_orientation'] = public_orientation(
                coarse_orientations[number - 1]
            )
        stamps.append(stamp)
    payload = {
        'format': 'stamp-perforation-results',
        'version': 1,
        'source': Path(input_path).name,
        'source_path': str(Path(input_path).resolve()),
        'coordinate_system': 'original image pixels; x right, y down',
        'dpi': dpi,
        'parameters': parameters,
        'stamps': stamps,
    }
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding='utf-8',
    )


SIDE_ALIASES = {
    't': 'top', 'top': 'top',
    'b': 'bottom', 'bottom': 'bottom',
    'l': 'left', 'left': 'left',
    'r': 'right', 'right': 'right',
}


def parse_side(value):
    side = SIDE_ALIASES.get(value.lower())
    if side is None:
        raise argparse.ArgumentTypeError(
            'side must be t/top, b/bottom, l/left, or r/right'
        )
    return side


def _write_image(path, image):
    path = Path(path)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f'Cannot write image: {path}')


def _display_gray(values):
    """Make an arbitrary numeric analysis array visible as an 8-bit image."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(values.shape, dtype=np.uint8)
    low, high = np.percentile(values[finite], [1, 99])
    if high <= low:
        high = low + 1
    result = np.clip((values - low) * 255 / (high - low), 0, 255)
    result[~finite] = 0
    return result.astype(np.uint8)


def analysis_context(image, orientation, measurement, side, threshold,
                     edge=None):
    """Rebuild the exact side-oriented arrays supplied to edge analysis.

    ``edge`` selects whose input signal is rebuilt and defaults to the side's
    published edge. Pass a shadow-trial edge to inspect a phase that ran but
    was not selected, or ``{}`` to force the raw threshold input.
    """
    angle = np.deg2rad(orientation['angle_deg'])
    basis = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
    width, height = orientation['width_px'], orientation['height_px']
    margin = int(np.ceil(min(width, height) * 0.12))
    shape = (int(np.ceil(width)) + 2 * margin,
             int(np.ceil(height)) + 2 * margin)
    origin = (np.array([orientation['center_x'], orientation['center_y']])
              - basis @ np.array([shape[0] / 2, shape[1] / 2]))
    transform = np.column_stack([basis, origin])
    patch = cv2.warpAffine(
        image, transform, shape,
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )
    gray = cv2.GaussianBlur(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), (3, 3), 0.6)
    binary = (gray > threshold).astype(np.uint8)
    vertical = side in ('left', 'right')
    flipped = side in ('bottom', 'right')
    work_binary = binary.T if vertical else binary
    work_gray = gray.T if vertical else gray
    if flipped:
        work_binary = work_binary[::-1]
        work_gray = work_gray[::-1]
    along_length = height if vertical else width
    normal_length = width if vertical else height
    start = int(margin + along_length * 0.01)
    end = int(margin + along_length * 0.99)
    n0 = max(0, int(margin - normal_length * 0.055))
    n1 = min(work_binary.shape[0] - 3, int(margin + normal_length * 0.13))
    if edge is None:
        edge = measurement.get('sides', {}).get(side, {})
    signal = work_gray.astype(float)
    analysis_binary = work_binary
    if (edge.get('method') == 'cross_support'
            and edge.get('debug_mask_band') is not None):
        analysis_binary = np.zeros_like(work_binary)
        analysis_binary[n0:n1 + 3, start:end] = edge['debug_mask_band']
    elif edge.get('method', '').startswith('adaptive_lab_'):
        lab = cv2.cvtColor(cv2.GaussianBlur(patch, (5, 5), 1.0), cv2.COLOR_BGR2LAB)
        color_work = lab.transpose(1, 0, 2) if vertical else lab
        if flipped:
            color_work = color_work[::-1]
        channel = 1 if edge['method'].endswith('_a') else 2
        polarity = edge['color_polarity']
        signal = color_work[:, :, channel].astype(float) * polarity
        analysis_binary = ((signal > edge['color_threshold'] * polarity)
                           & (work_gray > threshold)).astype(np.uint8)
        n1 = min(n1, int(margin + normal_length * 0.07))
    if edge.get('method') == 'cross_support' and edge.get('profile_ridge'):
        depth = profile_depth(
            analysis_binary[n0:n1 + 3, start:end], n0,
            edge['profile_ridge'],
        )
    else:
        depth = boundary_profile(analysis_binary, n0, n1, start, end)
    return {
        'patch': patch,
        'signal': signal,
        'binary': analysis_binary,
        'positions': np.arange(start, end, dtype=float),
        'depth': depth,
        'start': start, 'end': end, 'n0': n0, 'n1': n1,
        'edge': edge,
    }


def _phase_banner(image, text, color):
    """Prefix a panel with its phase, so a sheet of images is self-describing."""
    scale, thickness = 0.62, 2
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
    bar = np.full((size[1] + 24, image.shape[1], 3), 245, np.uint8)
    cv2.putText(bar, text, (12, size[1] + 11), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)
    return np.vstack([bar, image])


def _numbered_side_image(context):
    start, end, n0, n1 = (context[key] for key in ('start', 'end', 'n0', 'n1'))
    image = cv2.cvtColor(
        _display_gray(context['signal'][n0:n1 + 3, start:end]),
        cv2.COLOR_GRAY2BGR,
    )
    depth = context['depth']
    for i in range(len(depth) - 1):
        if np.isfinite(depth[i]) and np.isfinite(depth[i + 1]):
            cv2.line(image, (i, int(round(depth[i] - n0))),
                     (i + 1, int(round(depth[i + 1] - n0))),
                     (255, 220, 0), 1, cv2.LINE_AA)
    edge = context['edge']
    points = edge.get('valleys', np.empty((0, 2)))
    initial = edge.get('initial_valleys', points)
    accepted = edge.get('accepted', np.zeros(len(points), dtype=bool))
    fits = edge.get('refinement_fits', [None] * len(points))
    scale = max(1.0, 1800 / max(1, image.shape[1]))
    shown = cv2.resize(image, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_CUBIC)
    for public_id, index in enumerate(_hole_indices(edge), 1):
        point = points[index] if index < len(points) else initial[index]
        x = int(round((point[0] - start) * scale))
        y = int(round((point[1] - n0) * scale))
        fit = fits[index] if index < len(fits) else None
        is_accepted = bool(accepted[index]) if index < len(accepted) else False
        if fit is not None and fit.get('model') == 'circle':
            color = (255, 0, 255)
        else:
            color = (0, 120, 255)
        cv2.circle(shown, (x, y), 7, color, 2, cv2.LINE_AA)
        cv2.putText(shown, str(public_id), (x + 8, max(18, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return shown


def _profile_image(context):
    width, height = 1800, 520
    margin = 60
    canvas = np.full((height, width, 3), 250, np.uint8)
    depth = context['depth']
    valid = np.isfinite(depth)
    if valid.any():
        low, high = np.percentile(depth[valid], [1, 99])
        if high <= low:
            high = low + 1
        xs = np.linspace(margin, width - margin, len(depth)).astype(int)
        ys = np.zeros(len(depth), dtype=int)
        ys[valid] = (
            margin + (depth[valid] - low) * (height - 2 * margin) / (high - low)
        ).astype(int)
        previous = None
        for index in range(len(depth)):
            if not valid[index]:
                previous = None
                continue
            point = (xs[index], int(np.clip(ys[index], margin, height - margin)))
            if previous is not None:
                cv2.line(canvas, previous, point, (80, 80, 80), 1, cv2.LINE_AA)
            previous = point
        edge = context['edge']
        points = edge.get('valleys', np.empty((0, 2)))
        accepted = edge.get('accepted', np.zeros(len(points), dtype=bool))
        fits = edge.get('refinement_fits', [None] * len(points))
        positions = context['positions']
        public_ids = {
            index: public_id for public_id, index in enumerate(_hole_indices(edge), 1)
        }
        for index, point in enumerate(points):
            x = int(np.interp(point[0], [positions[0], positions[-1]],
                              [margin, width - margin]))
            y = int(np.interp(point[1], [low, high], [margin, height - margin]))
            fit = fits[index] if index < len(fits) else None
            color = ((255, 0, 255) if fit is not None and fit.get('model') == 'circle'
                     else (0, 100, 230))
            cv2.circle(canvas, (x, y), 6, color, -1, cv2.LINE_AA)
            cv2.putText(canvas, str(public_ids.get(index, index + 1)), (x + 7, y - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.putText(canvas, 'boundary depth (inside is down); numbered hole candidates',
                (margin, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 30, 30), 2,
                cv2.LINE_AA)
    return canvas


def _labelled_edge_strip(image, title, width=600, height=210):
    """Put a compact title above one side-oriented diagnostic strip."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    canvas = np.full((height, width, 3), 245, np.uint8)
    resized = cv2.resize(
        image, (width, height - 46),
        interpolation=(cv2.INTER_NEAREST if image.dtype == np.uint8
                       and len(np.unique(image.reshape(-1, image.shape[-1]),
                                         axis=0)) <= 8
                       else cv2.INTER_AREA),
    )
    canvas[46:] = resized
    cv2.putText(canvas, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, (25, 25, 25), 1, cv2.LINE_AA)
    return canvas


def _candidate_profile_panel(candidate, context, winner=False):
    """Draw one pre-arc binary profile and every valley guess it produced."""
    width, height = 900, 290
    title_height, strip_height = 62, 132
    chart_top, chart_bottom = 207, height - 14
    canvas = np.full((height, width, 3), 245, np.uint8)
    mask = cv2.resize(candidate['mask'] * 255, (width, strip_height),
                      interpolation=cv2.INTER_NEAREST)
    canvas[title_height:title_height + strip_height] = cv2.cvtColor(
        mask, cv2.COLOR_GRAY2BGR,
    )
    cv2.rectangle(canvas, (0, chart_top), (width - 1, chart_bottom),
                  (30, 30, 30), -1)

    start, end, n0, n1 = (context[key]
                           for key in ('start', 'end', 'n0', 'n1'))
    depth = candidate['depth']
    profile = candidate.get('profile', candidate.get('edge', {}))

    def x_coordinate(position):
        return int(round((position - start) * (width - 1)
                         / max(1, end - start - 1)))

    def strip_point(position, value):
        return (
            x_coordinate(position),
            title_height + int(round(
                (value - n0) * (strip_height - 1) / max(1, n1 + 2 - n0)
            )),
        )

    def chart_point(position, value):
        return (
            x_coordinate(position),
            chart_top + int(round(
                (value - n0) * (chart_bottom - chart_top)
                / max(1, n1 + 2 - n0)
            )),
        )

    valid = np.isfinite(depth)
    for projector in (strip_point, chart_point):
        previous = None
        for position, value, is_valid in zip(context['positions'], depth, valid):
            if not is_valid:
                previous = None
                continue
            point = projector(position, value)
            if previous is not None:
                cv2.line(canvas, previous, point, (0, 215, 255), 1,
                         cv2.LINE_AA)
            previous = point

    accepted = profile.get('accepted', [])
    guesses = profile.get('valleys', np.empty((0, 2)))
    for index, point in enumerate(guesses):
        color = ((255, 255, 0)
                 if index < len(accepted) and accepted[index]
                 else (255, 0, 255))
        for projector in (strip_point, chart_point):
            cv2.drawMarker(canvas, projector(point[0], point[1]), color,
                           cv2.MARKER_CROSS, 9, 2, cv2.LINE_AA)

    pitch = profile.get('pitch_px')
    pitch_text = '--' if pitch is None else f'{pitch:.2f}'
    bonus = 'yes' if candidate.get('cross_supported') else 'no'
    winner_text = 'WINNER | ' if winner else ''
    title = (
        f'{winner_text}{candidate["map"]}/{candidate["ridge"]}  '
        f'status={profile.get("status", "unavailable")}  '
        f'guess={len(guesses)}  pitch={pitch_text}  '
        f'score={candidate.get("score", -1):.3f}  '
        f'adj={candidate.get("adjusted_score", -1):.3f}  cross={bonus}'
    )
    cv2.putText(canvas, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.47, (20, 20, 20), 1, cv2.LINE_AA)
    centers = candidate.get('centers_L')
    subtitle = (
        'yellow=profile; cyan=accepted guess; magenta=rejected guess'
        + (f'; sorted centroid L={centers}' if centers is not None else '')
    )
    cv2.putText(canvas, subtitle, (10, 49), cv2.FONT_HERSHEY_SIMPLEX,
                0.39, (75, 75, 75), 1, cv2.LINE_AA)
    if winner:
        cv2.rectangle(canvas, (2, 2), (width - 3, height - 3),
                      (0, 0, 255), 4)
    return canvas


def _cross_support_profiles_image(context, debug, selected_edge):
    """Render exact k-means colours and all fourteen pre-arc candidates."""
    start, end, n0, n1 = (context[key]
                           for key in ('start', 'end', 'n0', 'n1'))
    side = debug.get('side')
    patch = context['patch']
    work_color = patch.transpose(1, 0, 2) if side in ('left', 'right') else patch
    if side in ('bottom', 'right'):
        work_color = work_color[::-1]
    original = work_color[n0:n1 + 3, start:end]

    overview = [_labelled_edge_strip(original, 'Original colour edge strip')]
    for k in (3, 4):
        clustering = debug.get('clusterings', {}).get(k)
        if clustering is None:
            continue
        centers = np.clip(np.rint(clustering['centers_lab']), 0, 255).astype(
            np.uint8,
        )
        center_bgr = cv2.cvtColor(centers[None], cv2.COLOR_LAB2BGR)[0]
        quantized = center_bgr[clustering['labels']]
        sorted_l = sorted(round(float(value), 1)
                          for value in clustering['centers_lab'][:, 0])
        overview.append(_labelled_edge_strip(
            quantized, f'k={k} exact centroid colours; sorted L={sorted_l}',
        ))
    while len(overview) < 3:
        overview.append(np.full_like(overview[0], 245))
    overview_image = np.hstack(overview[:3])

    candidates = debug.get('profile_candidates', [])
    by_key = {
        (candidate['map'], candidate['ridge']): candidate
        for candidate in candidates
    }
    map_order = ('brightness', 'k30', 'k31',
                 'k400', 'k401', 'k410', 'k411')
    winner_key = (selected_edge.get('profile_map'),
                  selected_edge.get('profile_ridge'))
    rows = []
    for name in map_order:
        panels = []
        for ridge in ('stack', 'close'):
            candidate = by_key.get((name, ridge))
            if candidate is None:
                panels.append(_message_image(
                    [f'{name}/{ridge}', 'candidate not available'],
                    width=900, height=290,
                ))
            else:
                panels.append(_candidate_profile_panel(
                    candidate, context, winner=(name, ridge) == winner_key,
                ))
        rows.append(np.hstack(panels))
    profiles = np.vstack(rows)
    heading = _message_image([
        'PHASE 1 - exact k-means colours, all 14 pre-arc profiles and guess points',
        (f'Selected: {winner_key[0]}/{winner_key[1]}; '
         'red border=winner; candidates are shown before circle fitting'),
    ], width=profiles.shape[1], height=125)
    return np.vstack([heading, overview_image, profiles])


def _message_image(lines, width=1200, height=360):
    image = np.full((height, width, 3), 245, np.uint8)
    for index, line in enumerate(lines):
        cv2.putText(image, str(line), (35, 65 + index * 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (25, 25, 25), 2,
                    cv2.LINE_AA)
    return image


def save_spot_diagnostics(directory, context, spot, prefix='spot'):
    edge = context['edge']
    seeds = edge.get('initial_valleys', edge.get('valleys', np.empty((0, 2))))
    order = _hole_indices(edge)
    if spot < 1 or spot > len(order):
        available = f'available identifiers: 1..{len(order)}' if len(order) else 'no hole candidates'
        _write_image(directory / f'{prefix}_{spot:02}_not_available.png', _message_image([
            f'Hole candidate {spot} was not attempted on this side.', available,
            f"Side status: {edge.get('status', 'unavailable')}",
            f"Reason: {edge.get('reason', '')}",
        ]))
        return
    internal_index = order[spot - 1]
    source_points = seeds if internal_index < len(seeds) else edge.get('valleys', seeds)
    seed = np.asarray(source_points[internal_index], dtype=float)
    pitch = edge.get('pitch_px')
    if not pitch or not np.isfinite(pitch):
        points = edge.get('valleys', np.empty((0, 2)))
        pitch = float(np.median(np.diff(np.sort(points[:, 0])))) if len(points) >= 2 else 30.0
    signal = context['signal'].astype(np.float32)
    smooth = cv2.GaussianBlur(signal, (0, 0), 1.2)
    gradient = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3) / 8
    positions, depth = context['positions'], context['depth']
    valid = np.isfinite(depth)
    if valid.sum() < 3:
        _write_image(directory / f'{prefix}_{spot:02}_insufficient_profile.png',
                     _message_image([f'Hole candidate {spot}', 'Insufficient boundary profile.']))
        return
    prior = np.interp(positions, positions[valid], depth[valid])
    panels = []
    for fraction in (0.23, 0.30, 0.37):
        for offset in (-0.12, 0, 0.12):
            x0 = max(int(positions[0]), int(seed[0] + pitch * (offset - fraction)))
            x1 = min(int(positions[-1]), int(seed[0] + pitch * (offset + fraction)))
            points = []
            for x in range(x0, x1 + 1):
                initial = prior[x - int(positions[0])]
                if abs(initial - seed[1]) > pitch * 0.6:
                    continue
                y0 = max(2, int(initial - pitch * 0.10))
                y1 = min(signal.shape[0] - 3, int(initial + pitch * 0.18))
                if y1 <= y0:
                    continue
                yy = np.arange(y0, y1 + 1)
                score = gradient[yy, x] * np.exp(
                    -0.5 * ((yy - initial) / (pitch * 0.15)) ** 2
                )
                y = int(yy[np.argmax(score)])
                if y in (y0, y1) or gradient[y, x] < 1.0:
                    continue
                g0, g1, g2 = gradient[y - 1:y + 2, x]
                denominator = g0 - 2 * g1 + g2
                shift = (np.clip(0.5 * (g0 - g2) / denominator, -0.5, 0.5)
                         if denominator < -1e-6 else 0)
                points.append((x, y + shift))
            points = np.asarray(points, dtype=float).reshape(-1, 2)
            circle = fit_arc(points, seed, pitch) if len(points) >= max(11, (x1 - x0) * 0.65) else None
            parabola = fit_parabola(points, seed, pitch) if len(points) >= max(11, (x1 - x0) * 0.65) else None
            view_x0 = max(0, int(seed[0] - pitch * 0.48))
            view_x1 = min(signal.shape[1], int(seed[0] + pitch * 0.48) + 1)
            view_y0 = max(0, int(seed[1] - pitch * 0.30))
            view_y1 = min(signal.shape[0], int(seed[1] + pitch * 0.30) + 1)
            panel = cv2.cvtColor(_display_gray(signal[view_y0:view_y1, view_x0:view_x1]),
                                 cv2.COLOR_GRAY2BGR)
            factor = min(6.0, 330 / max(1, panel.shape[1]), 240 / max(1, panel.shape[0]))
            panel = cv2.resize(panel, None, fx=factor, fy=factor,
                               interpolation=cv2.INTER_NEAREST)
            def project(point):
                return (int(round((point[0] - view_x0) * factor)),
                        int(round((point[1] - view_y0) * factor)))
            cv2.rectangle(panel, project((x0, view_y0)), project((x1, view_y1 - 1)),
                          (255, 180, 0), 2)
            for point in points:
                cv2.circle(panel, project(point), 2, (0, 255, 255), -1)
            for fit, color in ((parabola, (0, 200, 0)), (circle, (255, 0, 255))):
                if fit is None:
                    continue
                curve = fit['curve']
                for a, b in zip(curve[:-1], curve[1:]):
                    cv2.line(panel, project(a), project(b), color, 2, cv2.LINE_AA)
            cv2.drawMarker(panel, project(seed), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
            label = (f'w={fraction:.2f}p shift={offset:+.2f}p '
                     f'pts={len(points)} C={"Y" if circle else "N"} '
                     f'P={"Y" if parabola else "N"}')
            cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, 31),
                          (245, 245, 245), -1)
            cv2.putText(panel, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.47, (0, 0, 0), 1, cv2.LINE_AA)
            panel = cv2.resize(panel, (420, 300), interpolation=cv2.INTER_AREA)
            panels.append(panel)
    sheet = np.vstack([np.hstack(panels[row:row + 3]) for row in (0, 3, 6)])
    fits = edge.get('refinement_fits', [])
    final_fit = fits[internal_index] if internal_index < len(fits) else None
    accepted = edge.get('accepted', [])
    status = ('FOUND' if final_fit is not None else 'NOT FOUND')
    accepted_text = bool(accepted[internal_index]) if internal_index < len(accepted) else False
    header = _message_image([
        f'Hole candidate {spot}: {status}; accepted={accepted_text}',
        f'Initial seed s={seed[0]:.2f}, d={seed[1]:.2f}; pitch={pitch:.2f}px',
        'Orange: working approximation; magenta: accepted circle; hollow magenta: ignored circle.',
    ], width=sheet.shape[1], height=190)
    _write_image(directory / f'{prefix}_{spot:02}_all_search_windows.png',
                 np.vstack([header, sheet]))
    gx0 = max(0, int(seed[0] - pitch))
    gx1 = min(gradient.shape[1], int(seed[0] + pitch) + 1)
    gy0 = max(0, int(seed[1] - pitch * 0.6))
    gy1 = min(gradient.shape[0], int(seed[1] + pitch * 0.6) + 1)
    heat = cv2.applyColorMap(_display_gray(gradient[gy0:gy1, gx0:gx1]), cv2.COLORMAP_TURBO)
    heat = cv2.resize(heat, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
    _write_image(directory / f'{prefix}_{spot:02}_normal_gradient.png', heat)


def _stage_outline_image(image, box, stamp_number, orientation=None):
    """Crop the original scan and draw the boundaries known at one early stage."""
    x0, y0, x1, y1 = box
    extent_points = np.array([[x0, y0], [x1, y1]], dtype=float)
    if orientation is not None:
        extent_points = np.vstack([extent_points, orientation['corners']])
    min_xy = extent_points.min(axis=0)
    max_xy = extent_points.max(axis=0)
    padding = max(30, int(round(max(max_xy - min_xy) * 0.12)))
    image_h, image_w = image.shape[:2]
    crop_x0 = max(0, int(np.floor(min_xy[0])) - padding)
    crop_y0 = max(0, int(np.floor(min_xy[1])) - padding)
    crop_x1 = min(image_w, int(np.ceil(max_xy[0])) + padding + 1)
    crop_y1 = min(image_h, int(np.ceil(max_xy[1])) + padding + 1)
    result = image[crop_y0:crop_y1, crop_x0:crop_x1].copy()
    offset = np.array([crop_x0, crop_y0])
    thickness = max(2, int(round(max(result.shape[:2]) / 600)))
    detection_a = tuple(np.rint(np.array([x0, y0]) - offset).astype(int))
    detection_b = tuple(np.rint(np.array([x1, y1]) - offset).astype(int))
    if orientation is None:
        cv2.rectangle(result, detection_a, detection_b, (0, 0, 255), thickness,
                      cv2.LINE_AA)
        legend = f'STAGE 1  stamp {stamp_number}: detection bbox (red)'
    else:
        corners = np.rint(np.asarray(orientation['corners']) - offset).astype(np.int32)
        orientation_color = ((0, 210, 0) if orientation.get('status') == 'ok'
                             else (0, 220, 255))
        cv2.polylines(result, [corners], True, orientation_color,
                      thickness, cv2.LINE_AA)
        for number, corner in enumerate(corners, 1):
            point = tuple(corner)
            cv2.circle(result, point, max(4, thickness * 2), orientation_color, -1,
                       cv2.LINE_AA)
            cv2.putText(result, str(number), (point[0] + 7, point[1] - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, orientation_color,
                        max(1, thickness), cv2.LINE_AA)
        center = corners.astype(float).mean(axis=0)
        horizontal_start = (corners[0] + corners[3]) / 2
        horizontal_end = (corners[1] + corners[2]) / 2
        vertical_start = (corners[0] + corners[1]) / 2
        vertical_end = (corners[3] + corners[2]) / 2
        for start, end, color in (
                (horizontal_start, horizontal_end, (255, 160, 0)),
                (vertical_start, vertical_end, (255, 0, 255))):
            # Short central axis segments make the fitted rotation unambiguous
            # without hiding the stamp artwork.
            axis_start = center + 0.16 * (start - center)
            axis_end = center + 0.16 * (end - center)
            cv2.arrowedLine(result, tuple(np.rint(axis_start).astype(int)),
                            tuple(np.rint(axis_end).astype(int)), color,
                            thickness, cv2.LINE_AA, tipLength=0.18)
        # Draw the stage-1 box last and dashed: a solid orientation envelope can
        # otherwise cover it almost completely when both estimates agree.
        dx0, dy0 = detection_a
        dx1, dy1 = detection_b
        detection_corners = [(dx0, dy0), (dx1, dy0), (dx1, dy1), (dx0, dy1)]
        dash, gap = max(10, thickness * 5), max(7, thickness * 3)
        for first, second in zip(detection_corners,
                                 detection_corners[1:] + detection_corners[:1]):
            vector = np.asarray(second, dtype=float) - np.asarray(first, dtype=float)
            length = float(np.linalg.norm(vector))
            if length == 0:
                continue
            direction = vector / length
            for position in np.arange(0, length, dash + gap):
                segment_start = np.asarray(first) + direction * position
                segment_end = np.asarray(first) + direction * min(position + dash, length)
                cv2.line(result, tuple(np.rint(segment_start).astype(int)),
                         tuple(np.rint(segment_end).astype(int)), (0, 0, 255),
                         thickness, cv2.LINE_AA)
        if orientation.get('orientation_source') == 'perforation_line_consensus':
            legend = (
                f'STAGE 2B  stamp {stamp_number}: refined '
                f'{orientation["coarse_angle_deg"]:+.3f} -> '
                f'{orientation["angle_deg"]:+.3f} deg; '
                f'orientation={"green" if orientation.get("status") == "ok" else "yellow"}; '
                f'status={orientation.get("status", "unknown")}'
            )
        else:
            legend = (
                f'STAGE 2A  stamp {stamp_number}: coarse orientation; '
                f'angle={orientation["angle_deg"]:+.3f} deg; '
                f'orientation={"green" if orientation.get("status") == "ok" else "yellow"}; '
                f'status={orientation.get("status", "unknown")}'
            )
    font_scale = max(0.6, min(1.1, result.shape[1] / 1300))
    text_size = cv2.getTextSize(legend, cv2.FONT_HERSHEY_SIMPLEX,
                                font_scale, thickness)[0]
    cv2.rectangle(result, (0, 0),
                  (min(result.shape[1] - 1, text_size[0] + 24), text_size[1] + 24),
                  (245, 245, 245), -1)
    cv2.putText(result, legend, (12, text_size[1] + 10),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (20, 20, 20),
                thickness, cv2.LINE_AA)
    return result


def _side_input_panels(context):
    """The grey/colour signal and the threshold mask actually fed to a phase."""
    start, end, n0, n1 = (context[key] for key in ('start', 'end', 'n0', 'n1'))
    return (_display_gray(context['signal'][n0:n1 + 3, start:end]),
            context['binary'][n0:n1 + 3, start:end] * 255)


def save_details(directory, image, box, stamp_number, coarse_orientation,
                 orientation, measurement, side, threshold, spot=None):
    """Write diagnostic images for one stamp side, grouped by algorithm phase.

    Every file name starts with the phase that produced it, so sorting the
    directory walks the algorithm in order:

        0  stamp localization and rectification
        1  one-pass cross-support (green on the annotated scan)
        2  adaptive colour, first fallback (yellow)
        3  geometric recovery, second fallback (orange)
        4  sequential k=3/4 Lab arc fitting, last resort (red)

    A phase that did not run for this side leaves no images, so the set of
    files present is itself a record of how far the ladder was climbed. Each
    panel is banner-labelled with its phase and whether it was selected.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    trials = measurement.get('debug_algorithm_edges', {}).get(side, {})
    selected = measurement.get('sides', {}).get(side, {})
    selected_phase = method_phase(selected)

    def banner(phase, edge):
        mark = ('SELECTED' if phase == selected_phase and
                edge.get('method') == selected.get('method') else 'not selected')
        return (f'PHASE {phase} - {PHASE_NAMES[phase]}'
                f" - {edge.get('method', 'n/a')} - {mark}"
                f" - status={edge.get('status', 'unavailable')}")

    def write(phase, name, panel, edge=None):
        if edge is not None and panel.ndim == 3:
            panel = _phase_banner(panel, banner(phase, edge), phase_color(phase))
        _write_image(directory / f'{phase}_{name}.png', panel)

    # Phase 0: where the stamp is and how it was rectified.
    write(0, 'detection_bbox', _stage_outline_image(image, box, stamp_number))
    write(0, 'orientation_coarse',
          _stage_outline_image(image, box, stamp_number, coarse_orientation))
    write(0, 'orientation_refined',
          _stage_outline_image(image, box, stamp_number, orientation))
    gray_context = analysis_context(image, orientation, measurement, side,
                                    threshold, edge={})
    write(0, 'rectified_stamp', gray_context['patch'])

    # Phase 1: fourteen cheap profiles select one binary boundary, followed by
    # the primary method's only greyscale arc pass.
    primary_edge = trials.get('cross_support', selected)
    primary = analysis_context(image, orientation, measurement, side,
                               threshold, edge=primary_edge)
    primary['edge'] = primary_edge
    signal, mask = _side_input_panels(primary)
    write(1, 'input_gray', signal)
    write(1, 'selected_binary_map', mask)
    cross_debug = measurement.get('debug_cross_support', {}).get(side)
    if cross_debug and cross_debug.get('profile_candidates'):
        cross_debug = dict(cross_debug, side=side)
        write(1, 'all_profiles_and_guesses',
              _cross_support_profiles_image(
                  primary, cross_debug, primary_edge,
              ), primary_edge)
    write(1, 'cross_support_profile', _profile_image(primary), primary_edge)
    write(1, 'holes_numbered', _numbered_side_image(primary), primary_edge)

    # Phase 2: adaptive Lab colour boundary, still image-seeded.
    colour_edge = trials.get('adaptive_color')
    if colour_edge and 'color_threshold' in colour_edge:
        colour = analysis_context(image, orientation, measurement, side,
                                  threshold, edge=colour_edge)
        colour['edge'] = colour_edge
        signal, mask = _side_input_panels(colour)
        write(2, 'input_colour', signal)
        write(2, 'input_threshold', mask)
        write(2, 'colour_profile', _profile_image(colour), colour_edge)
        write(2, 'holes_numbered', _numbered_side_image(colour), colour_edge)

    # Phase 3: geometric reconstruction from neighbouring sides. Both variants
    # are recorded when both ran, since they are tried in order.
    for name, key in (('edge_recovery', 'parallel_edge_recovery'),
                      ('normal_scan', 'parallel_normal_scan')):
        recovery_edge = trials.get(key)
        if not recovery_edge:
            continue
        recovery = dict(gray_context, edge=recovery_edge)
        write(3, f'{name}_holes_numbered', _numbered_side_image(recovery),
              recovery_edge)

    # Phase 4: sequential k=3/4 profiles, with each candidate fitted in Lab
    # L/a/b order until five post-corner circular arcs survive.
    sequential_edge = trials.get('sequential_kmeans_lab')
    if sequential_edge:
        sequential = dict(gray_context, edge=sequential_edge)
        write(4, 'sequential_kmeans_lab_holes_numbered',
              _numbered_side_image(sequential), sequential_edge)

    if spot is not None:
        # Drill into one hole of whichever phase was actually selected.
        context = analysis_context(image, orientation, measurement, side,
                                   threshold)
        save_spot_diagnostics(directory, context, spot,
                              prefix=f'{selected_phase or 1}_spot')


def save_stamp_trial_details(directory, image, box, stamp_number,
                             coarse_orientation, orientation, measurement,
                             side=None):
    """Save compact diagnostics for a selected stamp without normal outputs."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    _write_image(directory / '0_detection_bbox.png',
                 _stage_outline_image(image, box, stamp_number))
    _write_image(directory / '0_orientation_coarse.png',
                 _stage_outline_image(image, box, stamp_number,
                                      coarse_orientation))
    _write_image(directory / '0_orientation_refined.png',
                 _stage_outline_image(image, box, stamp_number, orientation))
    trials = measurement.get('algorithm_trials', {})
    payload = {
        'stamp': stamp_number,
        'side_filter': side,
        'coarse_orientation': _json_value(coarse_orientation),
        'refined_orientation': _json_value(orientation),
        'measurement_status': measurement.get('status'),
        'selected_phase': {
            name: method_phase(edge)
            for name, edge in measurement.get('sides', {}).items()
            if side is None or name == side
        },
        'algorithm_trials': (
            {side: trials.get(side, {})} if side is not None else trials
        ),
        'selected_sides': {
            name: measurement_details({'sides': {name: edge}})['sides'][name]
            for name, edge in measurement.get('sides', {}).items()
            if side is None or name == side
        },
    }
    (directory / 'algorithm_trials.json').write_text(
        json.dumps(_json_value(payload), ensure_ascii=False, indent=2,
                   allow_nan=False),
        encoding='utf-8',
    )


_SHARED_IMAGE = None
_SHARED_IMAGE_MEMORY = None


def _init_measurement_worker(memory_name, shape, dtype_name):
    """Attach a worker to the parent's read-only scan buffer."""
    global _SHARED_IMAGE, _SHARED_IMAGE_MEMORY
    _SHARED_IMAGE_MEMORY = shared_memory.SharedMemory(name=memory_name)
    _SHARED_IMAGE = np.ndarray(
        tuple(shape), dtype=np.dtype(dtype_name),
        buffer=_SHARED_IMAGE_MEMORY.buf,
    )
    cv2.setNumThreads(1)


def _analyze_stamp_measurement(image, index, coarse_orientation, threshold,
                               collect_trials):
    # Try to establish the orientation from primary edges before paying for
    # the recovery ladder. If they do not form a consensus, retain the legacy
    # full coarse-angle pass: recovered edges can be essential orientation
    # evidence on difficult stamps (14K #6 is one such case).
    primary_measurement = measure_stamp(
        image, coarse_orientation, threshold, primary_only=True,
    )
    refined_orientation = refine_orientation_from_measurement(
        coarse_orientation, primary_measurement,
    )
    if (refined_orientation is not None
            and refined_orientation.get('orientation_source')
            == 'perforation_line_consensus'
            and refined_orientation.get('status') == 'ok'):
        first_measurement = primary_measurement
    else:
        first_measurement = measure_stamp(
            image, coarse_orientation, threshold,
        )
        refined_orientation = refine_orientation_from_measurement(
            coarse_orientation, first_measurement,
        )
    if collect_trials:
        measurement = measure_stamp(
            image, refined_orientation, threshold, collect_trials=True,
        )
    elif (refined_orientation is not None
          and (abs(refined_orientation.get(
              'orientation_correction_deg', 0,
          )) > 0.05
               or first_measurement.get('status') != 'ok')):
        measurement = measure_stamp(image, refined_orientation, threshold)
    else:
        measurement = first_measurement
    return index, refined_orientation, measurement


def _analyze_shared_stamp(task):
    """Process-pool entry point; the full scan is not pickled per task."""
    index, coarse_orientation, threshold, collect_trials = task
    return _analyze_stamp_measurement(
        _SHARED_IMAGE, index, coarse_orientation, threshold, collect_trials,
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Detect postage stamps placed on "
            "dark backing paper in a scanner image."
        )
    )

    parser.add_argument(
        "input",
        help="Input scan",
    )

    parser.add_argument(
        "--stamp-delta",
        type=float,
        default=35,
        help=(
            "How much brighter than the backing "
            "paper a pixel must be to count as "
            "stamp foreground "
            "(default: 35)"
        ),
    )

    parser.add_argument(
        "--processing-width",
        type=int,
        default=3000,
        help=(
            "Maximum image width used for "
            "detection (default: 3000)"
        ),
    )
    parser.add_argument(
        '--workers', type=int, default=os.cpu_count() or 1,
        help='Parallel stamp workers (default: logical CPU count)',
    )

    parser.add_argument('--dpi', type=float, help='Verified scan DPI for mm and perforations per 20 mm')
    parser.add_argument(
        '--stamp', type=int, metavar='STAMP',
        help='Analyze only this 1-based stamp number',
    )
    parser.add_argument(
        '--side', type=parse_side, metavar='SIDE',
        help='With --stamp, save diagnostics for t/top, b/bottom, l/left, or r/right',
    )
    parser.add_argument(
        '--spot', type=int, metavar='HOLE',
        help='With --stamp and --side, save deeper images for this hole candidate',
    )
    args = parser.parse_args()
    if args.dpi is not None and (not np.isfinite(args.dpi) or args.dpi <= 0):
        parser.error('--dpi must be finite and positive')
    if args.workers < 1:
        parser.error('--workers must be positive')
    if args.stamp is not None and args.stamp < 1:
        parser.error('--stamp must be a positive stamp number')
    if args.side is not None and args.stamp is None:
        parser.error('--side requires --stamp')
    if args.spot is not None and (
            args.stamp is None or args.side is None or args.spot < 1):
        parser.error('--spot requires --stamp and --side and must be positive')
    input_path = Path(args.input)
    output_path = input_path.with_name(input_path.stem + '_perf.jpg')
    perf_path = input_path.with_name('data_' + input_path.stem + '_perf.json')
    print(f"Image: {input_path}")

    image = cv2.imread(
        args.input,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(
            f"Cannot read image: "
            f"{args.input}"
        )

    # Detect stamp-shaped components across the complete scan.
    boxes, mask = detect_stamps_2d(
        image,
        max_processing_width=(
            args.processing_width
        ),
        background_delta=(
            args.stamp_delta
        ),
    )
    print(f"Found stamps: {len(boxes)}")

    #
    # Draw result.
    #

    if args.stamp is not None and args.stamp > len(boxes):
        parser.error(
            f'--stamp {args.stamp} is out of range; '
            f'the scan contains {len(boxes)} stamp(s)'
        )
    selected_indices = list(
        [args.stamp - 1] if args.stamp is not None else range(len(boxes))
    )
    coarse_orientations = [None] * len(boxes)
    for index in selected_indices:
        coarse_orientations[index] = estimate_orientation(mask, boxes[index])
    orientations = [None] * len(boxes)
    sample_scale = min(1.0, 1500 / image.shape[1])
    sample = cv2.resize(image, None, fx=sample_scale, fy=sample_scale,
                        interpolation=cv2.INTER_AREA)
    threshold = min(250, float(np.percentile(
        cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY), 10
    )) + args.stamp_delta)
    measurements = [None] * len(boxes)
    worker_count = min(args.workers, max(1, len(selected_indices)))
    if worker_count == 1:
        analyzed = (
            _analyze_stamp_measurement(
                image, index, coarse_orientations[index], threshold,
                args.stamp is not None,
            )
            for index in selected_indices
        )
        for index, refined, measurement in analyzed:
            orientations[index] = refined
            measurements[index] = measurement
    else:
        # Windows workers are spawned, so passing ``image`` normally would
        # pickle a full scan for every stamp. Copy it once into shared memory;
        # workers attach read-only and return only their compact measurements.
        image_memory = shared_memory.SharedMemory(
            create=True, size=image.nbytes,
        )
        shared_image = np.ndarray(
            image.shape, dtype=image.dtype, buffer=image_memory.buf,
        )
        shared_image[:] = image
        del shared_image
        tasks = [
            (index, coarse_orientations[index], threshold, False)
            for index in selected_indices
        ]
        try:
            with ProcessPoolExecutor(
                    max_workers=worker_count,
                    initializer=_init_measurement_worker,
                    initargs=(image_memory.name, image.shape, image.dtype.str),
            ) as executor:
                for index, refined, measurement in executor.map(
                        _analyze_shared_stamp, tasks):
                    orientations[index] = refined
                    measurements[index] = measurement
        finally:
            image_memory.close()
            image_memory.unlink()
    for measurement in measurements:
        if measurement is None:
            continue
        reconcile_perforation(measurement, ('top', 'bottom'), args.dpi)
        reconcile_perforation(measurement, ('left', 'right'), args.dpi)
    result = image.copy() if args.stamp is None else None

    image_h, image_w = (
        image.shape[:2]
    )

    thickness = max(
        2,
        round(
            max(
                image_h,
                image_w,
            )
            / 2500
        ),
    )

    font_scale = max(
        0.7,
        max(
            image_h,
            image_w,
        )
        / 5000,
    )

    #
    # Edge lines and the stamp label are coloured by phase: green for the main
    # method through red for the last resort.  The label takes the worst phase
    # on the stamp, so a sheet can be triaged without zooming in.
    #

    for i, (
        x0,
        y0,
        x1,
        y1,
    ) in enumerate(
        boxes,
        start=1,
    ):

        if result is None:
            break
        if measurements[i - 1] is None:
            continue
        draw_measurement(result, measurements[i - 1], np.array([0, 0]), thickness)
        label_y = max(
            thickness * 4 + 5,
            y0 - thickness * 3,
        )

        phases = [
            method_phase(edge)
            for edge in measurements[i - 1].get('sides', {}).values()
        ]
        measured = [phase for phase in phases if phase is not None]
        # A side that produced nothing is worse than any fallback that did.
        label_color = (phase_color(max(measured)) if measured
                       and len(measured) == len(phases) else (180, 180, 180))

        label = f"{i}: {perforation_label(measurements[i - 1], args.dpi)}"
        # Keep labels inside each stamp's horizontal allocation.
        label_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0]
        label_scale = font_scale * min(1.0, (x1-x0+1)/max(1, label_width))
        cv2.putText(
            result,
            label,
            (
                x0,
                label_y,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            label_scale,
            label_color,
            thickness,
            cv2.LINE_AA,
        )

    if args.stamp is None:
        if not cv2.imwrite(
            str(output_path),
            result,
        ):
            raise RuntimeError(
                f"Cannot write image: "
                f"{output_path}"
            )

        write_perf_json(
            perf_path, input_path, boxes, orientations, measurements, args.dpi,
            parameters={
                'stamp_delta': args.stamp_delta,
                'processing_width': args.processing_width,
                'workers': worker_count,
                'selected_stamp': args.stamp,
            },
            coarse_orientations=coarse_orientations,
        )
    details_path = None
    if args.side is not None:
        orientation = orientations[args.stamp - 1]
        if orientation is None:
            raise RuntimeError(
                f'Stamp {args.stamp} has no usable orientation for diagnostics'
            )
        details_path = input_path.with_name(
            input_path.stem + '_details'
        ) / f'stamp_{args.stamp:02}_{args.side}'
        save_details(
            details_path, image, boxes[args.stamp - 1], args.stamp,
            coarse_orientations[args.stamp - 1], orientation,
            measurements[args.stamp - 1],
            args.side, threshold, args.spot,
        )
        save_stamp_trial_details(
            details_path, image, boxes[args.stamp - 1], args.stamp,
            coarse_orientations[args.stamp - 1], orientation,
            measurements[args.stamp - 1], args.side,
        )
    elif args.stamp is not None:
        orientation = orientations[args.stamp - 1]
        if orientation is None:
            raise RuntimeError(
                f'Stamp {args.stamp} has no usable orientation for diagnostics'
            )
        details_path = input_path.with_name(
            input_path.stem + '_details'
        ) / f'stamp_{args.stamp:02}'
        save_stamp_trial_details(
            details_path, image, boxes[args.stamp - 1], args.stamp,
            coarse_orientations[args.stamp - 1], orientation,
            measurements[args.stamp - 1],
        )

    for index in selected_indices:
        number = index + 1
        measurement = measurements[index]
        label = (perforation_label(measurement, args.dpi)
                 if measurement is not None else '-- x --')
        measured_sides = (measurement.get('sides', {})
                          if measurement is not None else {})
        sides = ' '.join(
            f"{side}={measured_sides.get(side, {}).get('count', 0)}"
            for side in ('top', 'bottom', 'left', 'right')
        )
        print(
            f"Stamp {number}: "
            f"perforation={label}; "
            f"reliable points: {sides}"
        )


if __name__ == "__main__":
    main()
