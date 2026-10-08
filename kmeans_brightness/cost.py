"""Per-stamp cost of each method, single core, as it would run in production.

The experiment deliberately fits arcs for all seven candidates so the
selection rule can be judged apart from the idea. A real run fits arcs once,
on the winner, so the extra cost of a k-means method is the clustering plus
the cheap profile of each split. This measures that.
"""
import copy
import pathlib
import statistics
import sys
import time

import cv2

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import experiment as E  # noqa: E402

STAMPS = 8


def brightness_only(V, short):
    depth = E.boundary_profile(V['work'], V['n0'], V['n1'], V['start'],
                               V['end'])
    edge = E.measure_profile(V['positions'], depth, short)
    if 'valleys' in edge:
        E.refine_edge(edge, V['gray'], V['positions'], depth)
    return edge


def kmeans_only(V, short, ks):
    """Profile every split, fit arcs once, on the winner."""
    best = None
    for k in ks:
        masks, _, _ = E.split_masks(V, k)
        for item in masks:
            depth = E.band_depth(V, item['mask'])
            edge = E.measure_profile(V['positions'], depth, short)
            score = E.profile_score(edge) if 'valleys' in edge else -1.0
            if score > 0 and (best is None or score > best[0]):
                best = (score, edge, depth)
    if best is None:
        return None
    _, edge, depth = best
    edge = copy.deepcopy(edge)
    E.refine_edge(edge, V['gray'], V['positions'], depth)
    return edge


def main():
    image = cv2.imread(E.SCAN)
    orientations, threshold = E.front_end(image)
    runs = {name: [] for name in ('brightness', 'kmeans3', 'kmeans4',
                                  'kmeans34')}
    for orientation in orientations[:STAMPS]:
        P = E.patch_context(image, orientation, threshold)
        views = [E.side_view(P, side) for side in E.SIDES]
        for name, work in (
                ('brightness', lambda V: brightness_only(V, P['short'])),
                ('kmeans3', lambda V: kmeans_only(V, P['short'], (3,))),
                ('kmeans4', lambda V: kmeans_only(V, P['short'], (4,))),
                ('kmeans34', lambda V: kmeans_only(V, P['short'], (3, 4)))):
            started = time.perf_counter()
            for V in views:
                work(V)
            runs[name].append(time.perf_counter() - started)

    print(f'{STAMPS} stamps, four sides each, one core\n')
    print(f'{"":<12}{"s/stamp":>10}{"vs brightness":>16}')
    base = statistics.median(runs['brightness'])
    for name, times in runs.items():
        median = statistics.median(times)
        print(f'{name:<12}{median:>10.2f}{median / base:>15.2f}x')


if __name__ == '__main__':
    main()
