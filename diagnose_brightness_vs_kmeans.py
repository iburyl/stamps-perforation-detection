"""Compare brightness with one combined k=3/4 partition-search method.

For the combined method every dark/light partition is scored using only its
boundary profile after dark-background noise correction.  Exactly one winning
profile across both k values is then allowed to run the normal grey-image
circular-arc refinement.
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from diagnose_kmeans_quantization import (
    best_partition_profile, edge_summary, fit_variant, json_value,
    kmeans_signal, opposite_disagreement, prepare_stamp, profile_score,
    refine_profile_variant, side_panel,
)
from perforation import (SIDES, add_edge_image_geometry, circular_arc_count,
                         exclude_points_outside_corners)
from segment_stamps import detect_stamps_2d, estimate_orientation


VARIANTS = ('brightness', 'kmeans_3_4_partition_search')


def finalize_edges(edges, contexts, basis, origin):
    for side, edge in edges.items():
        if 'valleys' in edge:
            add_edge_image_geometry(edge, side, contexts[side], basis, origin)
    exclude_points_outside_corners(edges, contexts)
    for side, edge in edges.items():
        if 'valleys' in edge and 'line' in edge:
            add_edge_image_geometry(edge, side, contexts[side], basis, origin)


def measure_stamp(image, orientation, threshold):
    basis, origin, contexts = prepare_stamp(image, orientation, threshold)
    variants = {name: {} for name in VARIANTS}
    selection = {}
    panels = {side: {} for side in SIDES}
    for side in SIDES:
        context = contexts[side]
        variants['brightness'][side] = fit_variant(
            context, context['depth'], context['gray'], 'brightness',
        )
        finalists = []
        for clusters in (3, 4):
            _, _, centroid_metadata, cluster_data = kmeans_signal(
                context, clusters,
            )
            depth, profile_edge, partition_metadata = best_partition_profile(
                context, clusters, cluster_data,
            )
            finalists.append({
                'clusters': clusters,
                'depth': depth,
                'profile_edge': profile_edge,
                'score': profile_score(profile_edge),
                'centers_lab': centroid_metadata['centers_lab'],
                'partition_search': partition_metadata,
            })
        winner = max(finalists, key=lambda item: item['score'])
        combined = refine_profile_variant(
            context, winner['profile_edge'], winner['depth'],
            'kmeans_3_4_partition_search',
        )
        combined['selected_k'] = winner['clusters']
        combined['selected_assignment'] = winner[
            'partition_search'
        ]['selected_assignment']
        variants['kmeans_3_4_partition_search'][side] = combined
        selection[side] = {
            'selected_k': winner['clusters'],
            'selected_assignment': winner[
                'partition_search'
            ]['selected_assignment'],
            'selected_light_labels': winner[
                'partition_search'
            ]['selected_light_labels'],
            'selected_dark_labels': winner[
                'partition_search'
            ]['selected_dark_labels'],
            'finalists': [
                {
                    'clusters': candidate['clusters'],
                    'profile': edge_summary(candidate['profile_edge']),
                    'partition_search': candidate['partition_search'],
                }
                for candidate in finalists
            ],
        }

    for edges in variants.values():
        finalize_edges(edges, contexts, basis, origin)
    for side in SIDES:
        for name in VARIANTS:
            panels[side][name] = side_panel(
                contexts[side]['gray'], contexts[side],
                variants[name][side], name,
            )
    return variants, selection, panels


def write_contact_sheet(path, panels):
    rows = []
    for side in SIDES:
        row = np.hstack([panels[side][name] for name in VARIANTS])
        cv2.putText(row, side.upper(), (8, 178), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (20, 20, 20), 1, cv2.LINE_AA)
        rows.append(row)
    cv2.imwrite(str(path), np.vstack(rows))


def variant_summary(edges):
    return {
        'horizontal_pitch_difference_px': opposite_disagreement(
            edges, 'top', 'bottom',
        ),
        'vertical_pitch_difference_px': opposite_disagreement(
            edges, 'left', 'right',
        ),
        'sides': {
            side: edge_summary(edge) for side, edge in edges.items()
        },
    }


def comparison_summary(brightness, combined):
    rows = {}
    for side in SIDES:
        before = edge_summary(brightness[side])
        after = edge_summary(combined[side])
        pitch_delta = None
        if before['pitch_px'] is not None and after['pitch_px'] is not None:
            pitch_delta = float(after['pitch_px'] - before['pitch_px'])
        rows[side] = {
            'brightness_reliable': before['reliable'],
            'combined_reliable': after['reliable'],
            'reliability_change': (
                'gained' if after['reliable'] and not before['reliable']
                else 'lost' if before['reliable'] and not after['reliable']
                else 'unchanged'
            ),
            'arc_count_delta': (
                after['circular_arc_count'] - before['circular_arc_count']
            ),
            'pitch_delta_px': pitch_delta,
        }
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('--stamp-delta', type=float, default=35.)
    parser.add_argument('--processing-width', type=int, default=3000)
    parser.add_argument('--output')
    args = parser.parse_args()
    input_path = Path(args.input)
    output = (Path(args.output) if args.output else input_path.with_name(
        input_path.stem + '_brightness_vs_kmeans'))
    output.mkdir(parents=True, exist_ok=True)
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
    payload = {
        'input': str(input_path),
        'stamp_count': len(boxes),
        'threshold': threshold,
        'variants': list(VARIANTS),
        'combined_rule': (
            'score all k=3/4 dark-light profiles; fit grey-image arcs only '
            'for the single best profile'
        ),
        'excluded_methods': [
            'adaptive_color', 'parallel_edge_recovery',
            'parallel_normal_scan',
            'quantized_signal_arc_fitting',
        ],
        'stamps': {},
    }
    csv_rows = []
    for stamp_number, box in enumerate(boxes, start=1):
        orientation = estimate_orientation(mask, box)
        variants, selection, panels = measure_stamp(
            image, orientation, threshold,
        )
        summaries = {
            name: variant_summary(edges)
            for name, edges in variants.items()
        }
        comparison = comparison_summary(
            variants['brightness'],
            variants['kmeans_3_4_partition_search'],
        )
        payload['stamps'][str(stamp_number)] = {
            'orientation': json_value(orientation),
            'selection': selection,
            'variants': summaries,
            'comparison': comparison,
        }
        for name in VARIANTS:
            for side in SIDES:
                row = {
                    'stamp': stamp_number,
                    'variant': name,
                    'side': side,
                }
                row.update(summaries[name]['sides'][side])
                if name == 'kmeans_3_4_partition_search':
                    row['selected_k'] = selection[side]['selected_k']
                    row['selected_assignment'] = selection[side][
                        'selected_assignment'
                    ]
                else:
                    row['selected_k'] = None
                    row['selected_assignment'] = None
                csv_rows.append(row)
        write_contact_sheet(output / f'stamp_{stamp_number:02}.png', panels)
        print(f'Finished stamp {stamp_number}/{len(boxes)}', flush=True)
    (output / 'report.json').write_text(
        json.dumps(json_value(payload), ensure_ascii=False, indent=2,
                   allow_nan=False),
        encoding='utf-8',
    )
    with (output / 'report.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f'Report: {output / "report.json"}')


if __name__ == '__main__':
    main()
