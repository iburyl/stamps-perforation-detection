"""Evaluate fallback candidates only on sides rejected by cross-support.

No ruler measurements are read.  A candidate is judged by circular-arc
evidence, agreement with the already published opposite side, and distance
from the two axis-appropriate theoretical gauge values.  Results are
pre-corner: the experiment asks whether a fallback deserves entry into the
production geometry pass, not whether it is already safe to publish.
"""
import copy
import json
import math
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import arc_strategy_experiment as A  # noqa: E402
import candidate_search_experiment as C  # noqa: E402
import experiment as E  # noqa: E402
from perforation import select_cross_support_candidate  # noqa: E402

DATA = pathlib.Path(r'D:\workspace\IX vs X\IX-X')
OUT = HERE / 'fallback_results.json'
SIDES = ('top', 'bottom', 'left', 'right')
OPPOSITE = {'top': 'bottom', 'bottom': 'top',
            'left': 'right', 'right': 'left'}


def scan_threshold(image, delta):
    scale = min(1.0, 1500 / image.shape[1])
    sample = cv2.resize(image, None, fx=scale, fy=scale,
                        interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    return min(250, float(np.percentile(gray, 10)) + delta)


def gauge(pitch, dpi):
    return 20 * dpi / (25.4 * pitch) if pitch else None


def profile_pool(view, short, ks, include_brightness=False):
    maps = []
    if include_brightness:
        maps.append((
            'brightness', None,
            view['work'][view['n0']:view['n1'] + 3,
                         view['start']:view['end']],
        ))
    for k in ks:
        items, _, _ = E.split_masks(view, k)
        maps.extend((
            f"k{k}{''.join(str(value) for value in item['assignment'])}",
            k, item['mask'],
        ) for item in items)

    candidates = []
    center = float(np.mean(view['positions']))
    for ridge in C.RIDGES:
        for name, k, mask in maps:
            candidate = A.profile(view, short, name, k, mask, ridge)
            profile = candidate['profile']
            candidates.append({
                'candidate': candidate,
                'map': name, 'ridge': ridge,
                'score': float(candidate['score']),
                'pitch_px': profile.get('pitch_px'),
                'line_center': (
                    float(profile['slope'] * center + profile['intercept'])
                    if 'slope' in profile and 'intercept' in profile else None
                ),
                'cross_supported': False,
                'adjusted_score': float(candidate['score']),
            })
    return candidates


def fit(view, item, dpi):
    if item is None:
        return unavailable()
    edge = A.fit(view, item['candidate'], 'pitch_lock', None)
    summary = E.summarize(edge, {
        'pick': item['map'], 'ridge': item['ridge'],
        'profile_score': item['score'],
    })
    summary['reliable'] = bool(
        summary['geometry_source'] == 'circle_arcs'
        and (summary['arcs'] or 0) >= 5
    )
    summary['gauge'] = gauge(summary['pitch_px'], dpi)
    return summary


def unavailable():
    return {
        'status': 'unavailable', 'reliable': False,
        'geometry_source': None, 'arcs': 0, 'pitch_px': None,
        'gauge': None, 'pick': None, 'ridge': None,
    }


def evidence_key(result):
    return (
        result['geometry_source'] == 'circle_arcs', result['arcs'] or 0,
        result['status'] == 'ok', result['gauge'] is not None,
    )


def candidate_key(item):
    return (item['map'], item['ridge']) if item is not None else None


def sequential(view, candidates, dpi, strong_arcs=7, exclude=None):
    exclude = set(exclude or ())
    ordered = []
    for ridge in C.RIDGES:
        ordered.extend(sorted(
            (item for item in candidates
             if item['ridge'] == ridge
             and candidate_key(item) not in exclude),
            key=lambda item: item['score'], reverse=True,
        ))
    tried = []
    arc_fits = 0
    for item in ordered:
        arc_fits += int(item['score'] > 0)
        result = fit(view, item, dpi)
        tried.append(result)
        if (result['geometry_source'] == 'circle_arcs'
                and (result['arcs'] or 0) >= strong_arcs):
            result['fits'] = len(tried)
            result['arc_fits'] = arc_fits
            return result
    if not tried:
        return unavailable()
    result = max(tried, key=evidence_key)
    result['fits'] = len(tried)
    result['arc_fits'] = arc_fits
    return result


def one_pass(view, item, dpi):
    result = fit(view, item, dpi)
    result['fits'] = int(item is not None)
    result['arc_fits'] = int(item is not None and item['score'] > 0)
    return result


def cross_choice(candidates):
    usable = [item for item in candidates
              if item['score'] > 0 and item['pitch_px'] is not None
              and item['line_center'] is not None]
    return select_cross_support_candidate(usable) if usable else None


def opposite_choice(candidates, opposite_pitch):
    usable = [item for item in candidates
              if item['score'] > 0 and item['pitch_px'] is not None]
    if not usable or opposite_pitch is None:
        return None
    matching = [item for item in usable
                if abs(math.log(item['pitch_px'] / opposite_pitch)) <= .05]
    return max(matching, key=lambda item: item['score']) if matching else None


def second_cross_choice(candidates):
    chosen = cross_choice(candidates)
    if chosen is None:
        return None
    remaining = [item for item in candidates if item is not chosen]
    return max(remaining, key=lambda item: item['adjusted_score'],
               default=None)


def best_choice(candidates, exclude=None):
    exclude = set(exclude or ())
    return max(
        (item for item in candidates
         if item['score'] > 0 and candidate_key(item) not in exclude),
        key=lambda item: item['score'], default=None,
    )


def current_summary(side, dpi):
    pitch = side.get('pitch_px')
    holes = side.get('holes', [])
    arcs = sum(hole.get('accepted')
               and hole.get('category') == 'perforation' for hole in holes)
    return {
        'status': side.get('status'), 'reliable': pitch is not None,
        'geometry_source': side.get('geometry_source'), 'arcs': arcs,
        'pitch_px': pitch, 'gauge': gauge(pitch, dpi),
        'pick': side.get('profile_map'), 'ridge': side.get('profile_ridge'),
        'method': side.get('method'), 'fits': None, 'arc_fits': None,
    }


def run_scan(report_path):
    report = json.loads(report_path.read_text())
    image_path = report_path.with_name(report['source'])
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    threshold = scan_threshold(image, report['parameters']['stamp_delta'])
    dpi = float(report['dpi'])
    rows = []
    for stamp in report['stamps']:
        final_sides = stamp['measurement']['sides']
        problem_sides = [
            side for side in SIDES
            if final_sides[side].get('method') != 'cross_support'
            or final_sides[side].get('pitch_px') is None
        ]
        if not problem_sides:
            continue
        patch = E.patch_context(image, stamp['orientation'], threshold)
        for side in problem_sides:
            view = E.side_view(patch, side)
            k34 = profile_pool(view, patch['short'], (3, 4))
            k5 = profile_pool(view, patch['short'], (5,))
            all345 = profile_pool(
                view, patch['short'], (3, 4, 5), include_brightness=True,
            )
            current14 = [item for item in all345
                         if item['map'] == 'brightness'
                         or item['map'].startswith(('k3', 'k4'))]
            primary_choice = cross_choice(current14)
            primary_key = candidate_key(primary_choice)
            exclude_primary_kmeans = (
                {primary_key}
                if primary_key is not None
                and primary_choice['map'] != 'brightness' else set()
            )
            opposite = final_sides[OPPOSITE[side]]
            opposite_pitch = opposite.get('pitch_px')
            methods = {
                'current_final': current_summary(final_sides[side], dpi),
                'second_cross': one_pass(
                    view, second_cross_choice(current14), dpi,
                ),
                'next_kmeans': one_pass(
                    view, best_choice(k34, exclude_primary_kmeans), dpi,
                ),
                'opposite_k34': one_pass(
                    view, opposite_choice(k34, opposite_pitch), dpi,
                ),
                'sequential_k34': sequential(
                    view, k34, dpi, exclude=exclude_primary_kmeans,
                ),
                'cross_k5': one_pass(view, cross_choice(
                    profile_pool(view, patch['short'], (5,),
                                 include_brightness=True)), dpi),
                'best_k5': one_pass(view, best_choice(k5), dpi),
                'opposite_k5': one_pass(
                    view, opposite_choice(k5, opposite_pitch), dpi,
                ),
                'sequential_k5': sequential(view, k5, dpi),
                'sequential_k345': sequential(
                    view, k34 + k5, dpi,
                    exclude=exclude_primary_kmeans,
                ),
                'extended_cross_345': one_pass(
                    view, cross_choice(all345), dpi,
                ),
            }
            rows.append({
                'scan': report['source'], 'stamp': stamp['stamp'],
                'side': side, 'opposite': OPPOSITE[side],
                'opposite_method': opposite.get('method'),
                'opposite_pitch_px': opposite_pitch,
                'opposite_gauge': gauge(opposite_pitch, dpi),
                'methods': methods,
            })
    return rows


def main():
    paths = sorted(DATA.glob('data_*_perf.json'))
    rows = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for path, result in zip(paths, pool.map(run_scan, paths)):
            rows.extend(result)
            print(f'{path.name}: {len(result)} problem sides', flush=True)
    OUT.write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(f'-> {OUT} ({len(rows)} sides)')


if __name__ == '__main__':
    main()
