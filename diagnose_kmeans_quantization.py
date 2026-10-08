"""Compare brightness perforation measurement with k-means edge masks.

This is an intentionally isolated diagnostic.  It runs no adaptive-colour,
parallel-edge or parallel-scan recovery methods.  K-means is used
only to create a boundary profile; circle arcs are always fitted on the normal
grey signal.  The partition variants fix the darkest cluster as background and
the lightest as paper, then exhaustively assign every intermediate cluster to
one of those two classes before any arcs are fitted.  Premature boundary points
on the dark side of each candidate's fitted line are corrected using the binary
light-majority rule before the profile is scored.
"""
import argparse
import copy
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from kmeans_brightness.ridges import line_majority_depth
from perforation import (SIDES, add_edge_image_geometry, boundary_profile,
                         circular_arc_count, exclude_points_outside_corners,
                         measure_profile, refine_edge)
from segment_stamps import detect_stamps_2d, estimate_orientation


VARIANTS = (
    'brightness',
    'k3_original_arcs',
    'k4_original_arcs',
    'k3_partition_search',
    'k4_partition_search',
)


def parse_stamp_range(value):
    stamps = []
    for part in value.split(','):
        part = part.strip()
        if '-' in part:
            first, last = (int(item) for item in part.split('-', 1))
            stamps.extend(range(first, last + 1))
        elif part:
            stamps.append(int(part))
    if not stamps or min(stamps) < 1:
        raise argparse.ArgumentTypeError('stamps must be positive, e.g. 1-7')
    return sorted(set(stamps))


def prepare_stamp(image, orientation, threshold):
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
        image, transform, shape, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )
    gray = cv2.GaussianBlur(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY),
                            (3, 3), 0.6)
    binary = (gray > threshold).astype(np.uint8)
    lab = cv2.cvtColor(cv2.GaussianBlur(patch, (5, 5), 1.0),
                       cv2.COLOR_BGR2LAB)
    contexts = {}
    for side in SIDES:
        vertical = side in ('left', 'right')
        flipped = side in ('bottom', 'right')
        work = binary.T if vertical else binary
        gray_work = gray.T if vertical else gray
        lab_work = lab.transpose(1, 0, 2) if vertical else lab
        if flipped:
            work = work[::-1]
            gray_work = gray_work[::-1]
            lab_work = lab_work[::-1]
        along_length = height if vertical else width
        normal_length = width if vertical else height
        start = int(margin + along_length * 0.01)
        end = int(margin + along_length * 0.99)
        n0 = max(0, int(margin - normal_length * 0.055))
        n1 = min(work.shape[0] - 3,
                 int(margin + normal_length * 0.13))
        positions = np.arange(start, end, dtype=float)
        contexts[side] = {
            'positions': positions,
            'gray': gray_work,
            'lab': lab_work,
            'binary': work,
            'depth': boundary_profile(work, n0, n1, start, end),
            'start': start,
            'end': end,
            'n0': n0,
            'n1': n1,
            'work_shape': work.shape,
            'short_size': min(width, height),
        }
    return basis, origin, contexts


def kmeans_signal(context, clusters):
    """Return a k-level paper-directed signal and its first-edge profile."""
    start, end = context['start'], context['end']
    n0, n1 = context['n0'], context['n1']
    band = context['lab'][n0:n1 + 3, start:end].astype(np.float32)
    flat = band.reshape(-1, 3)
    # Training on at most 50k evenly distributed pixels keeps the diagnostic
    # deterministic and prevents large stamps from dominating run time.
    stride = max(1, len(flat) // 50000)
    training = flat[::stride][:50000].copy()
    cv2.setRNGSeed(41000 + clusters)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                80, 0.2)
    _, _, centers = cv2.kmeans(
        training, clusters, None, criteria, 5, cv2.KMEANS_PP_CENTERS,
    )
    distances = np.sum((flat[:, None, :] - centers[None, :, :]) ** 2,
                       axis=2)
    labels = np.argmin(distances, axis=1).reshape(band.shape[:2])

    band_height = labels.shape[0]
    outer_stop = max(1, int(round(band_height * 0.22)))
    inner_start = min(band_height - 1, int(round(band_height * 0.68)))
    outer_counts = np.bincount(labels[:outer_stop].ravel(),
                               minlength=clusters)
    inner_counts = np.bincount(labels[inner_start:].ravel(),
                               minlength=clusters)
    background_label = int(np.argmax(outer_counts))
    paper_order = np.argsort(inner_counts)[::-1]
    paper_label = next(
        int(label) for label in paper_order if label != background_label
    )

    # Project Lab along the background-to-paper centroid direction.  Cluster
    # centres become exactly k intensity levels, ordered so paper is brighter.
    direction = centers[paper_label] - centers[background_label]
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        direction = np.array([1., 0., 0.], dtype=np.float32)
    else:
        direction /= norm
    projections = centers @ direction
    low, high = float(np.min(projections)), float(np.max(projections))
    levels = ((projections - low) * (255. / max(high - low, 1e-6)))
    quantized_band = levels[labels].astype(np.float32)
    midpoint = float(
        (levels[background_label] + levels[paper_label]) / 2
    )
    mask_band = (quantized_band > midpoint).astype(np.uint8)

    quantized_signal = context['gray'].astype(np.float32).copy()
    quantized_signal[n0:n1 + 3, start:end] = quantized_band
    quantized_binary = np.zeros(context['binary'].shape, dtype=np.uint8)
    quantized_binary[n0:n1 + 3, start:end] = mask_band
    depth = boundary_profile(
        quantized_binary, n0, n1, start, end,
    )
    metadata = {
        'clusters': clusters,
        'background_label': background_label,
        'paper_label': paper_label,
        'centers_lab': centers.tolist(),
        'levels': levels.tolist(),
    }
    cluster_data = {'labels': labels, 'centers': centers}
    return quantized_signal, depth, metadata, cluster_data


def profile_score(edge):
    """Rank a boundary profile without looking at any fitted circle arcs."""
    if 'valleys' not in edge:
        return (0, 0, 0., 0., -np.inf, -np.inf, 0)
    pitch = max(float(edge.get('pitch_px', 0.)), 1e-6)
    amplitude = max(float(edge.get('amplitude_px', 0.)), 1e-6)
    coverage = float(edge.get('coverage', 0.))
    autocorrelation = float(edge.get('autocorrelation', 0.))
    return (
        1,
        int(edge.get('status') == 'ok'),
        autocorrelation * coverage,
        autocorrelation,
        coverage,
        -float(edge.get('spacing_rms_px', np.inf)) / pitch,
        -float(edge.get('depth_rms_px', np.inf)) / amplitude,
        int(edge.get('count', 0)),
    )


def smooth_dark_background_profile(context, mask_band, depth):
    """Suppress only premature k-means boundaries outside the fitted edge.

    ``boundary_profile`` records the first sustained light run in every column.
    A small light island in the dark background can therefore pull the curve
    outward (towards a smaller normal coordinate).  For such columns, find the
    outermost light pixel for which light pixels are already the strict majority
    on the way to the fitted straight edge.  Genuine valleys on the inward side
    of the line are deliberately left untouched.

    If the majority test has no solution, follow the specified conservative
    fallback and use the first light pixel below the original curve.
    """
    return line_majority_depth(
        mask_band, int(context['n0']), depth=depth, return_metadata=True,
    )


def best_partition_profile(context, clusters, cluster_data):
    """Try every dark/light assignment of the intermediate clusters."""
    labels = cluster_data['labels']
    centers = cluster_data['centers']
    darkest = int(np.argmin(centers[:, 0]))
    lightest = int(np.argmax(centers[:, 0]))
    intermediate = [
        label for label in range(clusters)
        if label not in (darkest, lightest)
    ]
    start, end = context['start'], context['end']
    n0, n1 = context['n0'], context['n1']
    candidates = []
    for assignment in range(1 << len(intermediate)):
        light_labels = {lightest}
        for bit, label in enumerate(intermediate):
            if assignment & (1 << bit):
                light_labels.add(label)
        mask_band = np.isin(labels, sorted(light_labels)).astype(np.uint8)
        binary = np.zeros(context['binary'].shape, dtype=np.uint8)
        binary[n0:n1 + 3, start:end] = mask_band
        raw_depth = boundary_profile(binary, n0, n1, start, end)
        depth, smoothing = smooth_dark_background_profile(
            context, mask_band, raw_depth,
        )
        edge = measure_profile(
            context['positions'], depth, context['short_size'],
        )
        score = profile_score(edge)
        candidates.append({
            'assignment': assignment,
            'light_labels': sorted(light_labels),
            'dark_labels': [
                label for label in range(clusters)
                if label not in light_labels
            ],
            'raw_depth': raw_depth,
            'depth': depth,
            'edge': edge,
            'score': score,
            'smoothing': smoothing,
        })
    selected = max(candidates, key=lambda item: item['score'])
    metadata = {
        'darkest_label': darkest,
        'lightest_label': lightest,
        'intermediate_labels': intermediate,
        'selected_assignment': selected['assignment'],
        'selected_light_labels': selected['light_labels'],
        'selected_dark_labels': selected['dark_labels'],
        'selected_smoothing': selected['smoothing'],
        'candidates': [
            {
                'assignment': candidate['assignment'],
                'light_labels': candidate['light_labels'],
                'dark_labels': candidate['dark_labels'],
                'smoothing': candidate['smoothing'],
                'profile': edge_summary(candidate['edge']),
            }
            for candidate in candidates
        ],
    }
    return selected['depth'], selected['edge'], metadata


def fit_variant(context, depth, arc_signal, method):
    edge = measure_profile(
        context['positions'], depth, context['short_size'],
    )
    edge['method'] = method
    if 'valleys' in edge:
        refine_edge(edge, arc_signal, context['positions'], depth)
    return edge


def refine_profile_variant(context, profile_edge, depth, method):
    edge = copy.deepcopy(profile_edge)
    edge['method'] = method
    if 'valleys' in edge:
        refine_edge(edge, context['gray'], context['positions'], depth)
    return edge


def edge_summary(edge):
    def number(name):
        value = edge.get(name)
        return float(value) if value is not None and np.isfinite(value) else None
    arc_count = int(circular_arc_count(edge))
    reliable = bool(
        edge.get('geometry_source') == 'circle_arcs' and arc_count >= 5
    )
    raw_pitch = number('pitch_px')
    return {
        'status': edge.get('status', 'unavailable'),
        'reason': edge.get('reason', ''),
        'method': edge.get('method'),
        'count': int(edge.get('count', 0)),
        'circular_arc_count': arc_count,
        'reliable': reliable,
        # Preserve a profile-only hypothesis for debugging, but do not expose
        # it as a measured pitch unless five fitted circular arcs support it.
        'pitch_px': raw_pitch if reliable else None,
        'profile_pitch_px': raw_pitch,
        'coverage': number('coverage'),
        'spacing_rms_px': number('spacing_rms_px'),
        'depth_rms_px': number('depth_rms_px'),
        'autocorrelation': number('autocorrelation'),
        'geometry_source': edge.get('geometry_source'),
    }


def side_panel(signal, context, edge, title):
    start, end = context['start'], context['end']
    n0, n1 = context['n0'], context['n1']
    crop = signal[n0:n1 + 3, start:end].astype(np.float32)
    finite = crop[np.isfinite(crop)]
    if finite.size:
        low, high = np.percentile(finite, [1, 99])
        crop = np.clip((crop - low) * 255 / max(high - low, 1), 0, 255)
    panel = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    fits = edge.get('refinement_fits', [])
    accepted = edge.get('accepted', np.zeros(len(fits), dtype=bool))
    for index, fit in enumerate(fits):
        if fit is None or fit.get('model') != 'circle':
            continue
        point = np.rint(np.asarray(fit['point']) - [start, n0]).astype(int)
        colour = (255, 0, 255) if (index < len(accepted)
                                    and accepted[index]) else (0, 120, 255)
        cv2.circle(panel, tuple(point), 3, colour, -1, cv2.LINE_AA)
    if 'slope' in edge and 'intercept' in edge:
        x0, x1 = float(start), float(end - 1)
        points = np.rint([
            [x0 - start, edge['slope'] * x0 + edge['intercept'] - n0],
            [x1 - start, edge['slope'] * x1 + edge['intercept'] - n0],
        ]).astype(int)
        cv2.line(panel, tuple(points[0]), tuple(points[1]),
                 (0, 220, 0), 1, cv2.LINE_AA)
    panel = cv2.resize(panel, (500, 145), interpolation=cv2.INTER_AREA)
    canvas = np.full((184, 500, 3), 245, dtype=np.uint8)
    canvas[39:] = panel
    pitch = edge.get('pitch_px')
    pitch_text = '--' if pitch is None else f'{pitch:.2f}'
    label = (f'{title} | {edge.get("status", "unavailable")} | '
             f'arcs={circular_arc_count(edge)} pitch={pitch_text}')
    cv2.putText(canvas, label, (7, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.47, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def measure_diagnostic(image, orientation, threshold):
    basis, origin, contexts = prepare_stamp(image, orientation, threshold)
    variants = {name: {} for name in VARIANTS}
    display_signals = {side: {} for side in SIDES}
    clustering = {side: {} for side in SIDES}
    for side in SIDES:
        context = contexts[side]
        baseline = fit_variant(
            context, context['depth'], context['gray'], 'brightness',
        )
        variants['brightness'][side] = baseline
        display_signals[side]['brightness'] = context['gray']
        for clusters in (3, 4):
            _, depth, metadata, cluster_data = kmeans_signal(
                context, clusters,
            )
            name = f'k{clusters}_original_arcs'
            edge = fit_variant(context, depth, context['gray'], name)
            variants[name][side] = edge
            display_signals[side][name] = context['gray']

            partition_depth, profile_edge, partition_metadata = (
                best_partition_profile(context, clusters, cluster_data)
            )
            partition_name = f'k{clusters}_partition_search'
            partition_edge = refine_profile_variant(
                context, profile_edge, partition_depth, partition_name,
            )
            variants[partition_name][side] = partition_edge
            display_signals[side][partition_name] = context['gray']
            clustering[side][f'k{clusters}'] = {
                'centroid_projection': metadata,
                'partition_search': partition_metadata,
            }

    # Apply the normal corner exclusion independently to each variant.  There
    # is deliberately no fallback or cross-variant recovery here.
    for edges in variants.values():
        for side, edge in edges.items():
            if 'valleys' in edge:
                add_edge_image_geometry(
                    edge, side, contexts[side], basis, origin,
                )
        exclude_points_outside_corners(edges, contexts)
        for side, edge in edges.items():
            if 'valleys' in edge and 'line' in edge:
                add_edge_image_geometry(
                    edge, side, contexts[side], basis, origin,
                )
    panels = {side: {} for side in SIDES}
    for side in SIDES:
        for name in VARIANTS:
            panels[side][name] = side_panel(
                display_signals[side][name], contexts[side],
                variants[name][side], name,
            )
    return variants, panels, clustering


def write_contact_sheet(path, panels):
    side_rows = []
    for side in SIDES:
        row = np.hstack([panels[side][name] for name in VARIANTS])
        cv2.putText(row, side.upper(), (8, 178), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (20, 20, 20), 1, cv2.LINE_AA)
        side_rows.append(row)
    sheet = np.vstack(side_rows)
    cv2.imwrite(str(path), sheet)


def opposite_disagreement(sides, first, second):
    edges = [sides[name] for name in (first, second)]
    if any(edge.get('geometry_source') != 'circle_arcs'
           or circular_arc_count(edge) < 5 for edge in edges):
        return None
    pitches = [edge.get('pitch_px') for edge in edges]
    if any(value is None for value in pitches):
        return None
    return float(abs(pitches[0] - pitches[1]))


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('--stamps', type=parse_stamp_range, default='1-7')
    parser.add_argument('--stamp-delta', type=float, default=35.)
    parser.add_argument('--processing-width', type=int, default=3000)
    parser.add_argument('--output')
    args = parser.parse_args()
    if isinstance(args.stamps, str):
        args.stamps = parse_stamp_range(args.stamps)
    input_path = Path(args.input)
    output = (Path(args.output) if args.output else
              input_path.with_name(
                  input_path.stem + '_kmeans_partition_diagnostics'))
    output.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f'Cannot read {input_path}')
    boxes, mask = detect_stamps_2d(
        image, max_processing_width=args.processing_width,
        background_delta=args.stamp_delta,
    )
    if max(args.stamps) > len(boxes):
        raise RuntimeError(
            f'Requested stamp {max(args.stamps)}, found only {len(boxes)}'
        )
    sample_scale = min(1.0, 1500 / image.shape[1])
    sample = cv2.resize(image, None, fx=sample_scale, fy=sample_scale,
                        interpolation=cv2.INTER_AREA)
    threshold = min(250., float(np.percentile(
        cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY), 10,
    )) + args.stamp_delta)
    payload = {
        'input': str(input_path),
        'threshold': threshold,
        'stamps': {},
        'variants': list(VARIANTS),
        'excluded_methods': [
            'adaptive_color', 'parallel_edge_recovery',
            'parallel_normal_scan',
        ],
    }
    csv_rows = []
    for stamp_number in args.stamps:
        orientation = estimate_orientation(mask, boxes[stamp_number - 1])
        variants, panels, clustering = measure_diagnostic(
            image, orientation, threshold,
        )
        stamp_payload = {
            'orientation': json_value(orientation),
            'clustering': clustering,
            'variants': {},
        }
        for variant, sides in variants.items():
            stamp_payload['variants'][variant] = {
                'horizontal_pitch_difference_px': opposite_disagreement(
                    sides, 'top', 'bottom',
                ),
                'vertical_pitch_difference_px': opposite_disagreement(
                    sides, 'left', 'right',
                ),
                'sides': {
                    side: edge_summary(edge) for side, edge in sides.items()
                },
            }
            for side, edge in sides.items():
                row = {'stamp': stamp_number, 'variant': variant,
                       'side': side}
                row.update(edge_summary(edge))
                csv_rows.append(row)
        payload['stamps'][str(stamp_number)] = stamp_payload
        write_contact_sheet(output / f'stamp_{stamp_number:02}.png', panels)
        print(f'Finished stamp {stamp_number}')

    (output / 'report.json').write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding='utf-8',
    )
    with (output / 'report.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f'Report: {output / "report.json"}')


if __name__ == '__main__':
    main()
