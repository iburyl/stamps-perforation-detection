"""Fit ranked k-means partitions without using brightness as a candidate.

The cheap profile score is useful but is not a sufficient selector: a profile
can contain a convincing periodic signal and still leave no circular arcs.
This experiment therefore ranks all six k=3/4 partitions, tries ``stack``
first and ``close`` second, and stops logically at the first candidate with
enough circular evidence.  All candidates are fitted in the experiment so
different stopping thresholds can be evaluated from one run.
"""
import copy
import json
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import arc_strategy_experiment as A  # noqa: E402
import experiment as E  # noqa: E402
from perforation import (add_edge_image_geometry,  # noqa: E402
                         exclude_points_outside_corners)

OUT = HERE / 'candidate_search_results.json'
RIDGES = ('stack', 'close')
THRESHOLDS = tuple(range(5, 13))
_IMAGE = None


def side_candidates(V, short):
    masks = []
    for k in E.KS:
        items, _, _ = E.split_masks(V, k)
        masks.extend((k, item) for item in items)
    by_ridge = {}
    for ridge in RIDGES:
        pool = []
        for k, item in masks:
            name = f"k{k}{''.join(str(a) for a in item['assignment'])}"
            candidate = A.profile(
                V, short, name, k, item['mask'], ridge,
            )
            edge = A.fit(V, candidate, 'pitch_lock', None)
            pool.append({
                'candidate': candidate,
                'edge': edge,
                'summary': E.summarize(edge, {
                    'pick': name,
                    'ridge': ridge,
                    'profile_score': float(candidate['score']),
                    'profile_pitch_px': (
                        float(candidate['profile']['pitch_px'])
                        if 'pitch_px' in candidate['profile'] else None
                    ),
                }),
            })
        by_ridge[ridge] = sorted(
            pool, key=lambda item: item['candidate']['score'], reverse=True,
        )
    return by_ridge


def evidence_key(item):
    value = item['summary']
    return (
        value['geometry_source'] == 'circle_arcs', value['arcs'] or 0,
        value['status'] == 'ok', value['gauge'] is not None,
        -RIDGES.index(value['ridge']),
    )


def sequential(pool, strong_arcs):
    tried = []
    for ridge in RIDGES:
        for item in pool[ridge]:
            tried.append(item)
            value = item['summary']
            if (value['geometry_source'] == 'circle_arcs'
                    and (value['arcs'] or 0) >= strong_arcs):
                return item, len(tried)
    return max(tried, key=evidence_key), len(tried)


def finalize(chosen, views, contexts, P):
    sides = {
        side: copy.deepcopy(chosen[side]['edge']) for side in E.SIDES
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
    return {
        side: E.summarize(sides[side], {
            'pick': chosen[side]['summary']['pick'],
            'ridge': chosen[side]['summary']['ridge'],
            'source': (f"{chosen[side]['summary']['pick']}+"
                       f"{chosen[side]['summary']['ridge']}"),
            'profile_score': chosen[side]['summary']['profile_score'],
        })
        for side in E.SIDES
    }


def measure(image, orientation, threshold):
    P = E.patch_context(image, orientation, threshold)
    views = {side: E.side_view(P, side) for side in E.SIDES}
    contexts = {side: views[side]['context'] for side in E.SIDES}
    pools = {
        side: side_candidates(views[side], P['short']) for side in E.SIDES
    }
    result = {'candidates': {}}
    for side in E.SIDES:
        result['candidates'][side] = {
            ridge: [dict(item['summary'], rank=rank)
                    for rank, item in enumerate(pools[side][ridge], 1)]
            for ridge in RIDGES
        }
    for strong_arcs in THRESHOLDS:
        chosen, attempts = {}, {}
        for side in E.SIDES:
            chosen[side], attempts[side] = sequential(
                pools[side], strong_arcs,
            )
        result[f'sequential_{strong_arcs}'] = finalize(
            chosen, views, contexts, P,
        )
        result[f'sequential_{strong_arcs}_attempts'] = attempts

    oracle = {
        side: max(
            (item for ridge in RIDGES for item in pools[side][ridge]),
            key=evidence_key,
        )
        for side in E.SIDES
    }
    result['arc_oracle'] = finalize(oracle, views, contexts, P)
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
    if image is None:
        raise FileNotFoundError(E.SCAN)
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
