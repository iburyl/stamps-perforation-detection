"""Four methods crossed with six ridge extractors on worst.jpg, no fallbacks.

experiment.py answered which binarization to measure. This answers how to
turn whichever band you chose into a depth profile, which the degradation
analysis pointed at: nearly every regression kept a correct intermediate
pitch and lost it in the arc pass, and the arc pass is seeded from the ridge.

Arc fitting happens once per distinct (candidate, ridge) pair, as it would in
a real run, so the cost here reflects the cost there.
"""
import copy
import json
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import experiment as E  # noqa: E402
import ridges as R  # noqa: E402
from experiment import METHODS, OUT, SCAN, SIDES  # noqa: E402
from perforation import (add_edge_image_geometry,  # noqa: E402
                         exclude_points_outside_corners, measure_profile)

_IMAGE = None


def side_bands(V, short):
    """Every candidate binarization of one side, as bands plus their names."""
    bands = [('brightness', None,
              V['work'][V['n0']:V['n1'] + 3, V['start']:V['end']])]
    for k in E.KS:
        masks, _, _ = E.split_masks(V, k)
        for item in masks:
            name = f"k{k}{''.join(str(a) for a in item['assignment'])}"
            bands.append((name, k, item['mask']))
    return bands


def side_grid(V, short):
    """Profile every candidate through every ridge, before any arc fitting."""
    grid = {}
    for name, k, band in side_bands(V, short):
        for variant in R.VARIANTS:
            depth = R.depth_of(band, V['n0'], variant)
            edge = measure_profile(V['positions'], depth, short)
            grid[(name, variant)] = {
                'name': name, 'k': k, 'ridge': variant,
                'depth': depth, 'profile': edge,
                'score': (E.profile_score(edge) if 'valleys' in edge
                          else -1.0),
            }
    return grid


def pick(grid, method, variant):
    """The candidate a method publishes for this side under this ridge."""
    if method == 'brightness':
        return grid[('brightness', variant)]
    pool = [item for (name, ridge), item in grid.items()
            if ridge == variant and name != 'brightness' and item['score'] > 0
            and (method == 'kmeans34' or item['k'] == int(method[-1]))]
    return max(pool, key=lambda item: item['score']) if pool else None


def measure_stamp_grid(image, orientation, threshold):
    P = E.patch_context(image, orientation, threshold)
    views = {side: E.side_view(P, side) for side in SIDES}
    contexts = {side: views[side]['context'] for side in SIDES}
    grids = {side: side_grid(views[side], P['short']) for side in SIDES}

    # One arc fit per distinct choice, reused by every method that makes it.
    fitted = {}

    def arc_fit(side, item):
        key = (side, item['name'], item['ridge'])
        if key not in fitted:
            edge = copy.deepcopy(item['profile'])
            if 'valleys' in edge:
                E.refine_edge(edge, views[side]['gray'],
                              views[side]['positions'], item['depth'])
            fitted[key] = edge
        return copy.deepcopy(fitted[key])

    result = {}
    for method in METHODS:
        result[method] = {}
        for variant in R.VARIANTS:
            sides, picked = {}, {}
            for side in SIDES:
                choice = pick(grids[side], method, variant)
                picked[side] = choice['name'] if choice else None
                edge = (arc_fit(side, choice) if choice else
                        {'status': 'unavailable', 'reason': 'no candidate'})
                sides[side] = edge
                if 'valleys' in edge:
                    add_edge_image_geometry(edge, side, contexts[side],
                                            P['basis'], P['origin'])
            exclude_points_outside_corners(sides, contexts)
            for side, edge in sides.items():
                if 'valleys' in edge and 'line' in edge:
                    add_edge_image_geometry(edge, side, contexts[side],
                                            P['basis'], P['origin'])
            result[method][variant] = {
                side: E.summarize(sides[side], {'pick': picked[side]})
                for side in SIDES
            }

    result['profiles'] = {
        side: [{'name': item['name'], 'ridge': item['ridge'],
                'score': round(float(item['score']), 3),
                'count': E._int(item['profile'].get('count', 0)),
                'pitch_px': (float(item['profile']['pitch_px'])
                             if 'pitch_px' in item['profile'] else None),
                'status': item['profile'].get('status')}
               for item in grids[side].values()]
        for side in SIDES
    }
    return result


def _init(path):
    global _IMAGE
    _IMAGE = cv2.imread(path)


def _run(task):
    index, orientation, threshold = task
    result = measure_stamp_grid(_IMAGE, orientation, threshold)
    result['stamp'] = index + 1
    return result


def main():
    image = cv2.imread(SCAN)
    orientations, threshold = E.front_end(image)
    print(f'{len(orientations)} stamps, threshold {threshold:.1f}')
    del image

    tasks = [(i, orientations[i], threshold) for i in range(len(orientations))
             if orientations[i] is not None]
    print(f'{len(tasks)} stamps x {len(METHODS)} methods x '
          f'{len(R.VARIANTS)} ridges')
    results = []
    with ProcessPoolExecutor(max_workers=10, initializer=_init,
                             initargs=(SCAN,)) as pool:
        for done, result in enumerate(pool.map(_run, tasks), 1):
            results.append(result)
            print(f'  stamp {result["stamp"]:>2} done ({done}/{len(tasks)})',
                  flush=True)
    results.sort(key=lambda r: r['stamp'])
    (OUT / 'ridge_results.json').write_text(json.dumps(
        results, indent=1,
        default=lambda v: v.item() if hasattr(v, 'item') else str(v)))
    print(f'-> {OUT / "ridge_results.json"}')


if __name__ == '__main__':
    main()
