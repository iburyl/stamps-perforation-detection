"""Evaluate selectors that are allowed exactly one expensive arc pass.

Every base candidate is fitted offline so selectors can be compared, but all
selection features are captured before ``refine_edge`` runs.  Brightness is a
seventh peer mask; it has no priority over the six k=3/4 partitions.
"""
import json
import pathlib
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import arc_strategy_experiment as A  # noqa: E402
import experiment as E  # noqa: E402
import ridges as R  # noqa: E402

OUT = HERE / 'one_pass_results.json'
RIDGES = ('stack', 'close')
_IMAGE = None


def profile_features(edge, center):
    pitch = edge.get('pitch_px')
    slope = edge.get('slope')
    intercept = edge.get('intercept')
    return {
        'status': edge.get('status'),
        'score': float(E.profile_score(edge)) if 'valleys' in edge else -1.,
        'pitch_px': float(pitch) if pitch is not None else None,
        'count': E._int(edge.get('count', 0)),
        'coverage': float(edge.get('coverage', 0.)),
        'autocorrelation': float(edge.get('autocorrelation', 0.)),
        'spacing_rms_px': (float(edge['spacing_rms_px'])
                           if 'spacing_rms_px' in edge else None),
        'depth_rms_px': (float(edge['depth_rms_px'])
                         if 'depth_rms_px' in edge else None),
        'slope': float(slope) if slope is not None else None,
        'line_center': (float(slope * center + intercept)
                        if slope is not None and intercept is not None
                        else None),
    }


def fitted_candidate(V, short, map_name, ridge, depth):
    profile = E.measure_profile(V['positions'], depth, short)
    item = {
        'name': map_name, 'k': None, 'ridge': ridge,
        'depth': depth, 'profile': profile,
        'score': E.profile_score(profile) if 'valleys' in profile else -1.,
    }
    edge = A.fit(V, item, 'pitch_lock', None)
    center = float(np.mean(V['positions']))
    return {
        'map': map_name,
        'ridge': ridge,
        'profile': profile_features(profile, center),
        'result': E.summarize(edge, {
            'pick': map_name, 'ridge': ridge,
            'source': f'{map_name}+{ridge}',
        }),
    }


def side_candidates(V, short):
    band = V['work'][V['n0']:V['n1'] + 3, V['start']:V['end']]
    maps = [('brightness', band)]
    for k in E.KS:
        items, _, _ = E.split_masks(V, k)
        maps.extend((
            f"k{k}{''.join(str(a) for a in item['assignment'])}",
            item['mask'],
        ) for item in items)

    candidates = []
    depths = {}
    for ridge in RIDGES:
        depths[ridge] = [
            R.depth_of(mask, V['n0'], ridge) for _, mask in maps
        ]
        candidates.extend(
            fitted_candidate(V, short, name, ridge, depth)
            for (name, _), depth in zip(maps, depths[ridge])
        )

        # Two deterministic seven-map fusions. Both still feed exactly one
        # profile to the eventual arc pass.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            median_depth = np.nanmedian(np.stack(depths[ridge]), axis=0)
        candidates.append(fitted_candidate(
            V, short, 'median7', ridge, median_depth,
        ))
        vote = (np.sum(np.stack([mask for _, mask in maps]), axis=0) >= 4)
        vote_depth = R.depth_of(vote.astype(np.uint8), V['n0'], ridge)
        candidates.append(fitted_candidate(
            V, short, 'vote4of7', ridge, vote_depth,
        ))
    return candidates


def measure(image, orientation, threshold):
    P = E.patch_context(image, orientation, threshold)
    return {
        'candidates': {
            side: side_candidates(E.side_view(P, side), P['short'])
            for side in E.SIDES
        },
    }


def _init(path):
    global _IMAGE
    _IMAGE = cv2.imread(path)


def _run(task):
    index, orientation, threshold = task
    result = measure(_IMAGE, orientation, threshold)
    result['stamp'] = index + 1
    return result


def run_scan(path, workers=10):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    orientations, threshold = E.front_end(image)
    del image
    tasks = [(index, orientation, threshold)
             for index, orientation in enumerate(orientations)
             if orientation is not None]
    results = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                             initargs=(str(path),)) as pool:
        for done, result in enumerate(pool.map(_run, tasks), 1):
            results.append(result)
            print(f'  stamp {result["stamp"]:>2} done '
                  f'({done}/{len(tasks)})', flush=True)
    return sorted(results, key=lambda item: item['stamp'])


def main():
    results = run_scan(E.SCAN)
    OUT.write_text(json.dumps(results, indent=1))
    print(f'-> {OUT}')


if __name__ == '__main__':
    main()
