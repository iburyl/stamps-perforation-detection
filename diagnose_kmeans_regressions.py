"""Render detailed profile diagnostics for combined-kmeans regressions."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from diagnose_brightness_vs_kmeans import measure_stamp
from diagnose_kmeans_quantization import (
    edge_summary, json_value, kmeans_signal, prepare_stamp, profile_score,
    side_panel, smooth_dark_background_profile,
)
from perforation import boundary_profile, measure_profile
from segment_stamps import detect_stamps_2d, estimate_orientation


DEFAULT_TARGETS = ((14, 'top'), (24, 'top'),
                   (27, 'left'), (21, 'bottom'))


def partition_candidates(context, clusters, cluster_data):
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
        candidates.append({
            'clusters': clusters,
            'assignment': assignment,
            'dark_labels': [
                label for label in range(clusters)
                if label not in light_labels
            ],
            'light_labels': sorted(light_labels),
            'mask_band': mask_band,
            'raw_depth': raw_depth,
            'depth': depth,
            'edge': edge,
            'score': profile_score(edge),
            'smoothing': smoothing,
        })
    return candidates


def cluster_colour_image(labels, centers):
    center_lab = np.clip(np.rint(centers), 0, 255).astype(np.uint8)[None]
    center_bgr = cv2.cvtColor(center_lab, cv2.COLOR_LAB2BGR)[0]
    return center_bgr[labels]


def resize_strip(image, width=700, height=180):
    interpolation = (cv2.INTER_NEAREST if image.ndim == 2
                     or len(np.unique(image.reshape(-1, image.shape[-1]
                        if image.ndim == 3 else 1), axis=0)) <= 8
                     else cv2.INTER_AREA)
    return cv2.resize(image, (width, height), interpolation=interpolation)


def labelled_strip(image, title, width=700, height=220):
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    canvas = np.full((height, width, 3), 245, np.uint8)
    canvas[40:] = resize_strip(image, width, height - 40)
    cv2.putText(canvas, title, (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def draw_profile_panel(context, depth, edge, title, mask_band=None,
                       final_edge=None, winner=False, raw_depth=None):
    width, height = 720, 330
    title_height, strip_height = 54, 122
    canvas = np.full((height, width, 3), 245, np.uint8)
    start, end = context['start'], context['end']
    n0, n1 = context['n0'], context['n1']
    if mask_band is None:
        strip = context['gray'][n0:n1 + 3, start:end]
        strip = cv2.cvtColor(strip.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        strip = cv2.cvtColor(mask_band * 255, cv2.COLOR_GRAY2BGR)
    strip = cv2.resize(strip, (width, strip_height),
                       interpolation=(cv2.INTER_NEAREST if mask_band is not None
                                      else cv2.INTER_AREA))
    canvas[title_height:title_height + strip_height] = strip

    def image_point(point):
        x = int(round((point[0] - start) * (width - 1) / max(1, end - start - 1)))
        y = int(round((point[1] - n0) * (strip_height - 1)
                      / max(1, n1 + 2 - n0))) + title_height
        return x, y

    finite = np.isfinite(depth)
    if raw_depth is not None:
        raw_finite = np.isfinite(raw_depth)
        last = None
        for position, value, valid in zip(
                context['positions'], raw_depth, raw_finite):
            if not valid:
                last = None
                continue
            point = image_point((position, value))
            if last is not None:
                cv2.line(canvas, last, point, (160, 160, 160), 1,
                         cv2.LINE_AA)
            last = point
    last = None
    for position, value, valid in zip(context['positions'], depth, finite):
        if not valid:
            last = None
            continue
        point = image_point((position, value))
        if last is not None:
            cv2.line(canvas, last, point, (0, 200, 255), 1, cv2.LINE_AA)
        last = point
    accepted = edge.get('accepted', [])
    for index, point in enumerate(edge.get('valleys', [])):
        colour = ((255, 255, 0) if index < len(accepted) and accepted[index]
                  else (0, 120, 255))
        cv2.circle(canvas, image_point(point), 3, colour, -1, cv2.LINE_AA)
    if final_edge is not None:
        final_accepted = final_edge.get('accepted', [])
        for index, fit in enumerate(final_edge.get('refinement_fits', [])):
            if (fit is not None and fit.get('model') == 'circle'
                    and index < len(final_accepted) and final_accepted[index]):
                cv2.circle(canvas, image_point(fit['point']), 4,
                           (255, 0, 255), -1, cv2.LINE_AA)

    chart_top = title_height + strip_height + 9
    chart_bottom = height - 13
    cv2.rectangle(canvas, (0, chart_top), (width - 1, chart_bottom),
                  (25, 25, 25), -1)
    last = None
    for position, value, valid in zip(context['positions'], depth, finite):
        if not valid:
            last = None
            continue
        x = int(round((position - start) * (width - 1)
                      / max(1, end - start - 1)))
        y = int(round(chart_top + (value - n0)
                      * (chart_bottom - chart_top) / max(1, n1 + 2 - n0)))
        point = (x, y)
        if last is not None:
            cv2.line(canvas, last, point, (0, 200, 255), 1, cv2.LINE_AA)
        last = point
    if 'slope' in edge and 'intercept' in edge:
        line_points = []
        for position in (start, end - 1):
            value = edge['slope'] * position + edge['intercept']
            x = int(round((position - start) * (width - 1)
                          / max(1, end - start - 1)))
            y = int(round(chart_top + (value - n0)
                          * (chart_bottom - chart_top)
                          / max(1, n1 + 2 - n0)))
            line_points.append((x, y))
        cv2.line(canvas, line_points[0], line_points[1],
                 (0, 220, 0), 1, cv2.LINE_AA)
    for index, point in enumerate(edge.get('valleys', [])):
        if index < len(accepted) and accepted[index]:
            x, strip_y = image_point(point)
            y = chart_top + int(round(
                (strip_y - title_height) * (chart_bottom - chart_top)
                / max(1, strip_height - 1)))
            cv2.circle(canvas, (x, y), 3, (255, 255, 0), -1, cv2.LINE_AA)

    summary = edge_summary(edge)
    pitch = summary['profile_pitch_px']
    pitch_text = '--' if pitch is None else f'{pitch:.2f}'
    prefix = 'WINNER | ' if winner else ''
    label = (f'{prefix}{title} | {summary["status"]} | '
             f'points={summary["count"]} pitch={pitch_text} '
             f'ac={summary["autocorrelation"] or 0:.3f} '
             f'cov={summary["coverage"] or 0:.3f}')
    cv2.putText(canvas, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.50, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(canvas,
                'grey=raw  yellow=smoothed profile  cyan=points  magenta=arcs',
                (8, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (70, 70, 70), 1, cv2.LINE_AA)
    if winner:
        cv2.rectangle(canvas, (2, 2), (width - 3, height - 3),
                      (0, 0, 255), 4)
    return canvas


def write_target(output, stamp_number, side, image, mask, boxes, threshold):
    orientation = estimate_orientation(mask, boxes[stamp_number - 1])
    variants, selection, _ = measure_stamp(image, orientation, threshold)
    _, _, contexts = prepare_stamp(image, orientation, threshold)
    context = contexts[side]
    start, end = context['start'], context['end']
    n0, n1 = context['n0'], context['n1']

    all_candidates = []
    cluster_images = {}
    centers_payload = {}
    for clusters in (3, 4):
        _, _, metadata, cluster_data = kmeans_signal(context, clusters)
        cluster_images[clusters] = cluster_colour_image(
            cluster_data['labels'], cluster_data['centers'],
        )
        centers_payload[f'k{clusters}'] = metadata['centers_lab']
        all_candidates.extend(
            partition_candidates(context, clusters, cluster_data)
        )
    winner = max(all_candidates, key=lambda item: item['score'])
    final_combined = variants['kmeans_3_4_partition_search'][side]
    final_brightness = variants['brightness'][side]

    directory = output / f'stamp_{stamp_number:02}_{side}'
    directory.mkdir(parents=True, exist_ok=True)
    source = context['gray'][n0:n1 + 3, start:end]
    overview = np.hstack([
        labelled_strip(source, 'Original grey edge strip'),
        labelled_strip(cluster_images[3], 'k=3 centroid colours'),
        labelled_strip(cluster_images[4], 'k=4 centroid colours'),
    ])
    cv2.imwrite(str(directory / '01_clustering.png'), overview)

    brightness_profile = measure_profile(
        context['positions'], context['depth'], context['short_size'],
    )
    brightness_panel = draw_profile_panel(
        context, context['depth'], brightness_profile,
        'brightness profile', final_edge=final_brightness,
    )
    cv2.imwrite(str(directory / '02_brightness_profile.png'),
                brightness_panel)

    profile_panels = []
    candidates_payload = []
    for candidate in all_candidates:
        is_winner = (candidate['clusters'] == winner['clusters']
                     and candidate['assignment'] == winner['assignment'])
        title = (
            f'k={candidate["clusters"]} assignment={candidate["assignment"]} '
            f'D={candidate["dark_labels"]} L={candidate["light_labels"]}'
        )
        profile_panels.append(draw_profile_panel(
            context, candidate['depth'], candidate['edge'], title,
            mask_band=candidate['mask_band'],
            final_edge=final_combined if is_winner else None,
            winner=is_winner,
            raw_depth=candidate['raw_depth'],
        ))
        candidates_payload.append({
            'clusters': candidate['clusters'],
            'assignment': candidate['assignment'],
            'dark_labels': candidate['dark_labels'],
            'light_labels': candidate['light_labels'],
            'smoothing': candidate['smoothing'],
            'profile': edge_summary(candidate['edge']),
            'winner': is_winner,
        })
    grid = np.vstack([
        np.hstack(profile_panels[0:2]),
        np.hstack(profile_panels[2:4]),
        np.hstack(profile_panels[4:6]),
    ])
    cv2.imwrite(str(directory / '03_all_kmeans_profiles.png'), grid)

    final_panel = np.hstack([
        side_panel(context['gray'], context, final_brightness,
                   'FINAL brightness'),
        side_panel(context['gray'], context, final_combined,
                   'FINAL combined k-means'),
    ])
    cv2.imwrite(str(directory / '04_final_arcs.png'), final_panel)
    payload = {
        'stamp': stamp_number,
        'side': side,
        'centers_lab': centers_payload,
        'winner': {
            'clusters': winner['clusters'],
            'assignment': winner['assignment'],
            'dark_labels': winner['dark_labels'],
            'light_labels': winner['light_labels'],
        },
        'brightness_final': edge_summary(final_brightness),
        'combined_final': edge_summary(final_combined),
        'profiles': candidates_payload,
        'selection_from_full_run': selection[side],
    }
    (directory / 'details.json').write_text(
        json.dumps(json_value(payload), ensure_ascii=False, indent=2,
                   allow_nan=False), encoding='utf-8',
    )
    print(f'Wrote {directory}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('--output')
    parser.add_argument('--stamp-delta', type=float, default=35.)
    parser.add_argument('--processing-width', type=int, default=3000)
    args = parser.parse_args()
    input_path = Path(args.input)
    output = (Path(args.output) if args.output else input_path.with_name(
        input_path.stem + '_brightness_vs_kmeans') / 'regressions')
    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f'Cannot read {input_path}')
    boxes, mask = detect_stamps_2d(
        image, max_processing_width=args.processing_width,
        background_delta=args.stamp_delta,
    )
    sample_scale = min(1.0, 1500 / image.shape[1])
    sample = cv2.resize(image, None, fx=sample_scale, fy=sample_scale,
                        interpolation=cv2.INTER_AREA)
    threshold = min(250., float(np.percentile(
        cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY), 10,
    )) + args.stamp_delta)
    for stamp_number, side in DEFAULT_TARGETS:
        write_target(output, stamp_number, side, image, mask, boxes, threshold)


if __name__ == '__main__':
    main()
