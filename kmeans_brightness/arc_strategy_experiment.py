"""Test whether the profile pitch should constrain the final arc lattice.

Only the three candidates used by the leading hybrid are measured:
brightness+close, kmeans34+close and kmeans34+stack.  Each is fitted normally
and with the profile pitch supplied as a 5% arc-lattice reference.
"""
import copy
import json
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import experiment as E  # noqa: E402
import ridges as R  # noqa: E402
from perforation import (add_edge_image_geometry,  # noqa: E402
                         exclude_points_outside_corners)

STRATEGIES = ('current', 'pitch_lock', 'brightness_prior', 'prior_lock')
CANDIDATES = ('brightness_raw', 'brightness_close', 'kmeans34_close',
              'kmeans34_stack')
OUT = HERE / 'arc_strategy_results.json'
_IMAGE = None


def profile(V, short, name, k, band, ridge):
    depth = R.depth_of(band, V['n0'], ridge)
    edge = E.measure_profile(V['positions'], depth, short)
    return {
        'name': name, 'k': k, 'ridge': ridge, 'depth': depth,
        'profile': edge,
        'score': E.profile_score(edge) if 'valleys' in edge else -1.,
    }


def side_profiles(V, short):
    brightness_band = V['work'][V['n0']:V['n1'] + 3,
                                V['start']:V['end']]
    brightness = profile(
        V, short, 'brightness', None, brightness_band, 'close',
    )
    brightness_raw = profile(
        V, short, 'brightness', None, brightness_band, 'raw',
    )
    splits = []
    masks = []
    for k in E.KS:
        items, _, _ = E.split_masks(V, k)
        masks.extend((k, item) for item in items)
    for ridge in ('close', 'stack'):
        pool = []
        for k, item in masks:
            name = f"k{k}{''.join(str(a) for a in item['assignment'])}"
            pool.append(profile(V, short, name, k, item['mask'], ridge))
        usable = [item for item in pool if item['score'] > 0]
        splits.append(max(usable, key=lambda item: item['score'])
                      if usable else None)
    return {
        'brightness_raw': brightness_raw,
        'brightness_close': brightness,
        'kmeans34_close': splits[0],
        'kmeans34_stack': splits[1],
    }


def fit(V, item, strategy, brightness_depth):
    if item is None:
        return {'status': 'unavailable', 'reason': 'no profile candidate'}
    edge = copy.deepcopy(item['profile'])
    if 'valleys' not in edge:
        return edge
    if strategy in ('pitch_lock', 'prior_lock'):
        edge['reference_pitch_px'] = (
            edge['pitch_px'] / np.sqrt(1 + edge['slope'] ** 2)
        )
    arc_depth = (brightness_depth
                 if strategy in ('brightness_prior', 'prior_lock')
                 else item['depth'])
    E.refine_edge(edge, V['gray'], V['positions'], arc_depth)
    return edge


def measure(image, orientation, threshold, strategies=None):
    P = E.patch_context(image, orientation, threshold)
    views = {side: E.side_view(P, side) for side in E.SIDES}
    contexts = {side: views[side]['context'] for side in E.SIDES}
    profiles = {
        side: side_profiles(views[side], P['short']) for side in E.SIDES
    }
    result = {}
    for strategy in (STRATEGIES if strategies is None else strategies):
        result[strategy] = {}
        for candidate in CANDIDATES:
            sides = {
                side: fit(
                    views[side], profiles[side][candidate], strategy,
                    profiles[side]['brightness_close']['depth'],
                )
                for side in E.SIDES
            }
            for side, edge in sides.items():
                if 'valleys' in edge:
                    add_edge_image_geometry(
                        edge, side, contexts[side], P['basis'], P['origin'],
                    )
            exclude_points_outside_corners(sides, contexts)
            for side, edge in sides.items():
                if 'valleys' in edge and 'line' in edge:
                    add_edge_image_geometry(
                        edge, side, contexts[side], P['basis'], P['origin'],
                    )
            result[strategy][candidate] = {
                side: E.summarize(sides[side], {
                    'pick': (profiles[side][candidate]['name']
                             if profiles[side][candidate] else None),
                })
                for side in E.SIDES
            }
    return result


def _init(path):
    global _IMAGE
    _IMAGE = cv2.imread(path)


def _run(task):
    index, orientation, threshold = task
    result = measure(_IMAGE, orientation, threshold)
    result['stamp'] = index + 1
    return result


def main():
    image = cv2.imread(E.SCAN)
    orientations, threshold = E.front_end(image)
    del image
    tasks = [(index, orientation, threshold)
             for index, orientation in enumerate(orientations)
             if orientation is not None]
    results = []
    with ProcessPoolExecutor(max_workers=10, initializer=_init,
                             initargs=(E.SCAN,)) as pool:
        for done, result in enumerate(pool.map(_run, tasks), 1):
            results.append(result)
            print(f'  stamp {result["stamp"]:>2} done '
                  f'({done}/{len(tasks)})', flush=True)
    results.sort(key=lambda item: item['stamp'])
    OUT.write_text(json.dumps(results, indent=1))
    print(f'-> {OUT}')


if __name__ == '__main__':
    main()
