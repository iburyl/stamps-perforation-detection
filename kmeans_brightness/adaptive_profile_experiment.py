"""Compare old and production stack/close profiles inside Adaptive Lab.

All profile and threshold alternatives are cheap.  Each method selects one
stable candidate before performing exactly one expensive arc-fitting pass in
the winning Lab a/b channel.  Arc evidence is corner-filtered before reporting.
"""
import copy
import json
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import experiment as E  # noqa: E402
import fallback_channel_experiment as X  # noqa: E402
import fallback_experiment as F  # noqa: E402
from edge_profiles import profile_depth  # noqa: E402
from perforation import boundary_profile, measure_profile, refine_edge  # noqa: E402

OUT = HERE / 'adaptive_profile_results.json'
METHODS = ('adaptive_old', 'adaptive_close', 'adaptive_stack', 'adaptive_best')


def adaptive_candidates(view, patch):
    vertical = view['side'] in ('left', 'right')
    normal = patch['width'] if vertical else patch['height']
    n1 = min(view['n1'], int(patch['margin'] + normal * 0.07))
    start, end = view['start'], view['end']
    region = view['lab'][view['n0']:n1 + 3, start:end]
    light = view['gray'][view['n0']:n1 + 3, start:end] > patch['threshold']
    candidates = []
    for channel, name in ((1, 'a'), (2, 'b')):
        values = region[:, :, channel][light]
        if len(values) < len(view['positions']) * 4:
            continue
        low, high = np.percentile(values, [5, 95])
        if high - low < 6:
            continue
        levels = np.unique(np.rint(np.linspace(low, high, 21))).astype(int)
        for polarity in (1, -1):
            for level in levels:
                selected = ((view['lab'][:, :, channel] > level)
                            if polarity == 1
                            else (view['lab'][:, :, channel] < level))
                binary = (selected & (view['gray'] > patch['threshold'])).astype(
                    np.uint8
                )
                band = binary[view['n0']:n1 + 3, start:end]
                for variant in ('old', 'close', 'stack'):
                    depth = (
                        boundary_profile(binary, view['n0'], n1, start, end)
                        if variant == 'old'
                        else profile_depth(band, view['n0'], variant)
                    )
                    edge = measure_profile(
                        view['positions'], depth, patch['short'],
                    )
                    if (edge.get('count', 0) < 6
                            or edge.get('coverage', 0) < 0.6
                            or edge.get('autocorrelation', 0) < 0.45
                            or edge['spacing_rms_px'] / edge['pitch_px'] > 0.08):
                        continue
                    candidates.append({
                        'edge': edge, 'depth': depth, 'channel': name,
                        'channel_index': channel, 'polarity': polarity,
                        'threshold': int(level), 'variant': variant,
                        'score': float(
                            edge['count'] * edge['coverage']
                            * edge['autocorrelation']
                        ),
                    })

    center = float(np.mean(view['positions']))
    stable = []
    for candidate in candidates:
        edge = candidate['edge']
        peers = [
            other for other in candidates
            if other is not candidate
            and other['channel'] == candidate['channel']
            and other['polarity'] == candidate['polarity']
            and other['variant'] == candidate['variant']
            and 0 < abs(other['threshold'] - candidate['threshold']) <= 3
            and abs(other['edge']['pitch_px'] / edge['pitch_px'] - 1) < 0.04
            and abs(
                (other['edge']['slope'] - edge['slope']) * center
                + other['edge']['intercept'] - edge['intercept']
            ) < max(3, patch['short'] * 0.006)
        ]
        if peers:
            stable.append(candidate)
    return stable


def select(candidates, variants):
    pool = [candidate for candidate in candidates
            if candidate['variant'] in variants]
    return max(pool, key=lambda candidate: candidate['score'], default=None)


def fit(candidate, view, patch, dpi, side, saved_sides, views):
    if candidate is None:
        return X.unavailable()
    edge = copy.deepcopy(candidate['edge'])
    edge['reference_pitch_px'] = (
        edge['pitch_px'] / np.sqrt(1 + edge['slope'] ** 2)
    )
    signal = (
        view['lab'][:, :, candidate['channel_index']].astype(float)
        * candidate['polarity']
    )
    refine_edge(edge, signal, view['positions'], candidate['depth'])
    edge = X.corner_filter(edge, side, saved_sides, views, patch)
    item = {
        'map': f"adaptive_lab_{candidate['channel']}",
        'ridge': candidate['variant'],
    }
    result = X.summarize(
        edge, item, dpi, candidate['channel'], candidate['polarity'],
    )
    result.update(
        arc_fits=1, profiles_tried=1,
        color_threshold=candidate['threshold'],
        profile_score=candidate['score'],
    )
    return result


def run_scan(report_path):
    report = json.loads(report_path.read_text())
    image_path = report_path.with_name(report['source'])
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    threshold = F.scan_threshold(image, report['parameters']['stamp_delta'])
    dpi = float(report['dpi'])
    rows = []
    for stamp in report['stamps']:
        saved_sides = stamp['measurement']['sides']
        problem_sides = [
            side for side in E.SIDES
            if saved_sides[side].get('method') != 'cross_support'
            or saved_sides[side].get('pitch_px') is None
        ]
        if not problem_sides:
            continue
        patch = E.patch_context(image, stamp['orientation'], threshold)
        views = {side: E.side_view(patch, side) for side in E.SIDES}
        for side in problem_sides:
            views[side]['side'] = side
            candidates = adaptive_candidates(views[side], patch)
            choices = {
                'adaptive_old': select(candidates, {'old'}),
                'adaptive_close': select(candidates, {'close'}),
                'adaptive_stack': select(candidates, {'stack'}),
                'adaptive_best': select(candidates, {'close', 'stack'}),
            }
            methods = {
                method: fit(
                    choice, views[side], patch, dpi, side, saved_sides, views,
                )
                for method, choice in choices.items()
            }
            opposite = saved_sides[F.OPPOSITE[side]]
            rows.append({
                'scan': report['source'], 'stamp': stamp['stamp'],
                'side': side, 'current_method': saved_sides[side].get('method'),
                'opposite_gauge': F.gauge(opposite.get('pitch_px'), dpi),
                'stable_candidates': len(candidates),
                'methods': methods,
            })
    return rows


def main():
    paths = sorted(F.DATA.glob('data_*_perf.json'))
    rows = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for path, result in zip(paths, pool.map(run_scan, paths)):
            rows.extend(result)
            print(f'{path.name}: {len(result)} problem sides', flush=True)
    OUT.write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(f'-> {OUT} ({len(rows)} sides)')


if __name__ == '__main__':
    main()
