"""Compare four fallback families on sides rejected by cross-support.

The experiment uses no manual measurements.  Every arc result is passed
through the production corner filter before it can stop a search.  Lab a/b
channels have no intrinsic inward-positive direction, so their polarity is
chosen cheaply from the median colour change across the candidate boundary;
this keeps each channel to one expensive arc pass.
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

import candidate_search_experiment as C  # noqa: E402
import experiment as E  # noqa: E402
import fallback_experiment as F  # noqa: E402
from perforation import (  # noqa: E402
    adaptive_color_edge,
    add_edge_image_geometry,
    boundary_profile,
    circular_arc_count,
    exclude_points_outside_corners,
    refine_edge,
)

OUT = HERE / 'fallback_channel_results.json'
METHODS = ('cross_lab', 'sequential_k34', 'sequential_k34_lab',
           'adaptive_lab')
CHANNELS = (('L', 0), ('a', 1), ('b', 2))


def candidate_key(item):
    return (item['map'], item['ridge']) if item is not None else None


def reliable(edge):
    return bool(
        edge.get('geometry_source') == 'circle_arcs'
        and circular_arc_count(edge) >= 5
        and edge.get('pitch_px') is not None
    )


def evidence_key(result):
    return (
        result.get('reliable', False), result.get('arcs') or 0,
        result.get('status') == 'ok', result.get('gauge') is not None,
    )


def unavailable():
    return {
        'status': 'unavailable', 'reliable': False,
        'geometry_source': None, 'arcs': 0, 'pitch_px': None,
        'gauge': None, 'pick': None, 'ridge': None, 'channel': None,
        'polarity': None, 'arc_fits': 0, 'profiles_tried': 0,
    }


def orient_signal(signal, positions, depth, pitch):
    """Make the median outside-to-inside transition a positive gradient."""
    valid = np.isfinite(depth)
    if valid.sum() < 3:
        return signal, 1
    prior = np.interp(positions, positions[valid], depth[valid])
    columns = np.rint(positions).astype(int)
    gap = max(2, int(round(pitch * 0.08)))
    outer = np.rint(prior - gap).astype(int)
    inner = np.rint(prior + gap).astype(int)
    keep = (
        (columns >= 0) & (columns < signal.shape[1])
        & (outer >= 0) & (inner < signal.shape[0])
    )
    if keep.sum() < 3:
        return signal, 1
    change = np.median(
        signal[inner[keep], columns[keep]]
        - signal[outer[keep], columns[keep]]
    )
    polarity = 1 if change >= 0 else -1
    return signal.astype(float) * polarity, polarity


def corner_filter(edge, side, saved_sides, views, patch):
    if 'valleys' not in edge:
        return edge
    add_edge_image_geometry(
        edge, side, views[side]['context'], patch['basis'], patch['origin'],
    )
    sides = {}
    for name in E.SIDES:
        if name == side:
            sides[name] = edge
            continue
        line = saved_sides[name].get('line_patch')
        sides[name] = ({'line_patch': tuple(line)} if line is not None else {})
    exclude_points_outside_corners(
        sides, {name: views[name]['context'] for name in E.SIDES},
    )
    return sides[side]


def summarize(edge, item, dpi, channel, polarity):
    value = E.summarize(edge, {
        'pick': item['map'] if item is not None else None,
        'ridge': item['ridge'] if item is not None else None,
    })
    value['reliable'] = reliable(edge)
    value['gauge'] = F.gauge(value.get('pitch_px'), dpi)
    value['channel'] = channel
    value['polarity'] = polarity
    return value


def fit_item(view, item, dpi, side, saved_sides, views, patch,
             channel='gray'):
    if item is None or item['score'] <= 0:
        return unavailable()
    edge = copy.deepcopy(item['candidate']['profile'])
    if 'valleys' not in edge:
        return unavailable()
    edge['reference_pitch_px'] = (
        edge['pitch_px'] / np.sqrt(1 + edge['slope'] ** 2)
    )
    polarity = 1
    if channel == 'gray':
        signal = view['gray']
    else:
        index = dict(CHANNELS)[channel]
        signal, polarity = orient_signal(
            view['lab'][:, :, index], view['positions'],
            item['candidate']['depth'], edge['pitch_px'],
        )
    refine_edge(
        edge, signal, view['positions'], item['candidate']['depth'],
    )
    edge = corner_filter(edge, side, saved_sides, views, patch)
    return summarize(edge, item, dpi, channel, polarity)


def ordered_k34(candidates, primary=None, include_primary=False):
    primary_key = candidate_key(primary)
    pool = []
    for ridge in C.RIDGES:
        pool.extend(sorted(
            (item for item in candidates if item['ridge'] == ridge
             and (include_primary or candidate_key(item) != primary_key)),
            key=lambda item: item['score'], reverse=True,
        ))
    if (include_primary and primary is not None
            and primary['map'] != 'brightness'):
        pool = [primary] + [
            item for item in pool if candidate_key(item) != primary_key
        ]
    return pool


def try_profiles(view, candidates, channels, dpi, side, saved_sides,
                 views, patch):
    tried = []
    arc_fits = 0
    profiles_tried = 0
    for item in candidates:
        if item['score'] <= 0:
            continue
        profiles_tried += 1
        for channel in channels:
            arc_fits += 1
            result = fit_item(
                view, item, dpi, side, saved_sides, views, patch, channel,
            )
            tried.append(result)
            if result['reliable']:
                result['arc_fits'] = arc_fits
                result['profiles_tried'] = profiles_tried
                return result
    if not tried:
        return unavailable()
    result = max(tried, key=evidence_key)
    result['arc_fits'] = arc_fits
    result['profiles_tried'] = profiles_tried
    return result


def adaptive(view, patch, dpi, side, saved_sides, views):
    vertical = side in ('left', 'right')
    normal = patch['width'] if vertical else patch['height']
    color_n1 = min(
        view['n1'], int(patch['margin'] + normal * 0.07),
    )
    edge = adaptive_color_edge(
        view['lab'], view['gray'], patch['threshold'], view['positions'],
        view['n0'], color_n1, patch['short'],
    )
    if edge is None:
        return unavailable()
    channel = 'a' if edge['method'].endswith('_a') else 'b'
    index = 1 if channel == 'a' else 2
    polarity = edge['color_polarity']
    signal = view['lab'][:, :, index].astype(float) * polarity
    selected = (
        (signal > edge['color_threshold'] * polarity)
        & (view['gray'] > patch['threshold'])
    )
    depth = boundary_profile(
        selected.astype(np.uint8), view['n0'], color_n1,
        view['start'], view['end'],
    )
    refine_edge(edge, signal, view['positions'], depth)
    edge = corner_filter(edge, side, saved_sides, views, patch)
    item = {'map': edge['method'], 'ridge': 'adaptive'}
    result = summarize(edge, item, dpi, channel, polarity)
    result['arc_fits'] = 1
    result['profiles_tried'] = 1
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
            view = views[side]
            all14 = F.profile_pool(
                view, patch['short'], (3, 4), include_brightness=True,
            )
            k34 = [item for item in all14 if item['map'] != 'brightness']
            primary = F.cross_choice(all14)
            methods = {
                'cross_lab': try_profiles(
                    view, [primary] if primary is not None else [],
                    ('L', 'a', 'b'), dpi, side, saved_sides, views, patch,
                ),
                'sequential_k34': try_profiles(
                    view, ordered_k34(k34, primary, include_primary=False),
                    ('gray',), dpi, side, saved_sides, views, patch,
                ),
                'sequential_k34_lab': try_profiles(
                    view, ordered_k34(k34, primary, include_primary=True),
                    ('L', 'a', 'b'), dpi, side, saved_sides, views, patch,
                ),
                'adaptive_lab': adaptive(
                    view, patch, dpi, side, saved_sides, views,
                ),
            }
            opposite = saved_sides[F.OPPOSITE[side]]
            rows.append({
                'scan': report['source'], 'stamp': stamp['stamp'],
                'side': side, 'current_method': saved_sides[side].get('method'),
                'opposite_gauge': F.gauge(opposite.get('pitch_px'), dpi),
                'methods': methods,
            })
    return rows


def main():
    paths = sorted(F.DATA.glob('*_perf.json'))
    rows = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for path, result in zip(paths, pool.map(run_scan, paths)):
            rows.extend(result)
            print(f'{path.name}: {len(result)} problem sides', flush=True)
    OUT.write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(f'-> {OUT} ({len(rows)} sides)')


if __name__ == '__main__':
    main()
