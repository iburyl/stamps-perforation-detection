"""Four primary-path methods on worst.jpg, with no fallbacks of any kind.

    brightness   the existing global-threshold boundary
    kmeans3      best of the 2 black/white splits of a k=3 Lab clustering
    kmeans4      best of the 4 splits of a k=4 clustering
    kmeans34     best of all 6 splits, then one arc pass on the winner

A split fixes the darkest centroid to background and the brightest to paper,
and assigns each remaining centroid to one or the other. Every split yields a
boundary profile, measured exactly as the brightness path measures its own.
Selection happens on that profile, before any arc fitting, which is what keeps
the method cheap. All six are nonetheless carried through arc fitting here so
the selection rule can be judged separately from the idea.
"""
import copy
import itertools
import json
import os
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

OUT = pathlib.Path(__file__).resolve().parent
REPO = str(OUT.parent)
sys.path.insert(0, REPO)

from perforation import (SIDES, add_edge_image_geometry, boundary_profile,  # noqa: E402
                         circular_arc_count, exclude_points_outside_corners,
                         measure_profile, refine_edge, reliable_side)
from segment_stamps import detect_stamps_2d, estimate_orientation  # noqa: E402

# The worst-case test scan is 47 MB and lives outside the repository. Point
# STAMPS_WORST_SCAN at your own copy to reproduce these results elsewhere.
SCAN = os.environ.get('STAMPS_WORST_SCAN',
                      r'C:\Workspace\stamps\IX vs X\worst.jpg')
DPI = 1200.0
STAMP_DELTA = 35
KS = (3, 4)
METHODS = ('brightness', 'kmeans3', 'kmeans4', 'kmeans34')

_IMAGE = None


def gauge(pitch_px):
    return 20.0 * DPI / (25.4 * pitch_px) if pitch_px else None


def _int(value):
    return None if value is None else int(value)


# --------------------------------------------------------------------------
# Geometry, copied from measure_stamp so every method sees identical inputs.
# --------------------------------------------------------------------------

def patch_context(image, orientation, threshold):
    angle = np.deg2rad(orientation['angle_deg'])
    basis = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
    width, height = orientation['width_px'], orientation['height_px']
    margin = int(np.ceil(min(width, height) * 0.12))
    shape = (int(np.ceil(width)) + 2 * margin, int(np.ceil(height)) + 2 * margin)
    origin = (np.array([orientation['center_x'], orientation['center_y']])
              - basis @ np.array([shape[0] / 2, shape[1] / 2]))
    transform = np.column_stack([basis, origin])
    patch = cv2.warpAffine(image, transform, shape,
                           flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    gray = cv2.GaussianBlur(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), (3, 3), 0.6)
    return {
        'gray': gray,
        'binary': (gray > threshold).astype(np.uint8),
        'lab': cv2.cvtColor(cv2.GaussianBlur(patch, (5, 5), 1.0), cv2.COLOR_BGR2LAB),
        'basis': basis, 'origin': origin, 'margin': margin,
        'width': width, 'height': height, 'short': min(width, height),
        'threshold': threshold,
    }


def side_view(P, side):
    vertical = side in ('left', 'right')
    flipped = side in ('bottom', 'right')

    def plane(array):
        turned = array.T if vertical else array
        return turned[::-1] if flipped else turned

    def volume(array):
        turned = array.transpose(1, 0, 2) if vertical else array
        return turned[::-1] if flipped else turned

    work = plane(P['binary'])
    along = P['height'] if vertical else P['width']
    normal = P['width'] if vertical else P['height']
    margin = P['margin']
    start, end = int(margin + along * 0.01), int(margin + along * 0.99)
    n0 = max(0, int(margin - normal * 0.055))
    n1 = min(work.shape[0] - 3, int(margin + normal * 0.13))
    positions = np.arange(start, end, dtype=float)
    gray = plane(P['gray'])
    lab = volume(P['lab'])
    return {
        'work': work, 'gray': gray, 'lab': lab, 'positions': positions,
        'start': start, 'end': end, 'n0': n0, 'n1': n1,
        'context': {'positions': positions, 'signal': gray, 'lab_signal': lab,
                    'work_shape': work.shape, 'normal_search': (n0, n1),
                    'expected_normal': margin, 'collect_debug': False},
    }


def band_depth(V, mask_band):
    """Boundary profile from a mask covering only the edge band."""
    binary = np.zeros(V['work'].shape, np.uint8)
    binary[V['n0']:V['n1'] + 3, V['start']:V['end']] = mask_band
    return boundary_profile(binary, V['n0'], V['n1'], V['start'], V['end'])


# --------------------------------------------------------------------------
# Candidate boundaries.
# --------------------------------------------------------------------------

def profile_score(edge):
    """Rank candidate binarizations before any arc fitting is attempted."""
    return (edge.get('count', 0) * edge.get('coverage', 0.0)
            * edge.get('autocorrelation', 0.0))


def cluster_band(V, k):
    band = V['lab'][V['n0']:V['n1'] + 3, V['start']:V['end']].astype(np.float32)
    cv2.setRNGSeed(31003)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.25)
    _, labels, centers = cv2.kmeans(band.reshape(-1, 3), k, None, criteria, 5,
                                    cv2.KMEANS_PP_CENTERS)
    return labels.reshape(band.shape[:2]), centers


def split_masks(V, k):
    """Every assignment of the intermediate centroids to paper or background."""
    labels, centers = cluster_band(V, k)
    order = np.argsort(centers[:, 0])              # by Lab lightness
    bright, middles = int(order[-1]), [int(m) for m in order[1:-1]]
    out = []
    for assignment in itertools.product((0, 1), repeat=len(middles)):
        white = [bright] + [m for m, a in zip(middles, assignment) if a]
        out.append({
            'k': k, 'assignment': assignment,
            'mask': np.isin(labels, white).astype(np.uint8),
            'centers_L': [round(float(c), 1) for c in centers[order, 0]],
        })
    return out, labels, order


def side_candidates(V, short):
    """Brightness plus all six splits, each carried through to arc fitting.

    ``profile`` is the state after measure_profile, before any arc search:
    the ridge and the intermediate spots the selection rule actually sees.
    ``edge`` is the same candidate after the greyscale arc pass.
    """
    depth = boundary_profile(V['work'], V['n0'], V['n1'], V['start'], V['end'])
    edge = measure_profile(V['positions'], depth, short)
    edge['method'] = 'brightness'
    profile = copy.deepcopy(edge)
    if 'valleys' in edge:
        refine_edge(edge, V['gray'], V['positions'], depth)
    brightness = {'name': 'brightness', 'k': None, 'assignment': None,
                  'depth': depth, 'profile': profile, 'edge': edge,
                  'score': profile_score(profile) if 'valleys' in profile else -1.0,
                  'mask': V['work'][V['n0']:V['n1'] + 3, V['start']:V['end']]}

    splits = []
    for k in KS:
        masks, _, _ = split_masks(V, k)
        for item in masks:
            item_depth = band_depth(V, item['mask'])
            item_edge = measure_profile(V['positions'], item_depth, short)
            item_edge['method'] = f'kmeans{k}'
            item_profile = copy.deepcopy(item_edge)
            score = (profile_score(item_profile) if 'valleys' in item_profile
                     else -1.0)
            if 'valleys' in item_edge:
                refine_edge(item_edge, V['gray'], V['positions'], item_depth)
            splits.append({
                'name': f"k{k}{''.join(str(a) for a in item['assignment'])}",
                'k': k, 'assignment': item['assignment'],
                'centers_L': item['centers_L'], 'mask': item['mask'],
                'depth': item_depth, 'profile': item_profile,
                'edge': item_edge, 'score': score,
            })
    return brightness, splits


def select(brightness, splits, method):
    """The candidate a method publishes for this side."""
    if method == 'brightness':
        return brightness
    pool = [s for s in splits if s['score'] > 0
            and (method == 'kmeans34' or s['k'] == int(method[-1]))]
    return max(pool, key=lambda s: s['score']) if pool else None


# --------------------------------------------------------------------------
# One stamp.
# --------------------------------------------------------------------------

def summarize(edge, extra=None):
    out = {
        'status': edge.get('status'),
        'reliable': bool(reliable_side(edge)),
        'arcs': _int(circular_arc_count(edge)),
        'geometry_source': edge.get('geometry_source'),
        'pitch_px': float(edge['pitch_px']) if 'pitch_px' in edge else None,
        'gauge': gauge(edge.get('pitch_px')),
        'count': _int(edge.get('count', 0)),
        'coverage': float(edge.get('coverage', 0.0)),
        'autocorrelation': float(edge.get('autocorrelation', 0.0)),
    }
    out.update(extra or {})
    return out


def measure_stamp_methods(image, orientation, threshold):
    P = patch_context(image, orientation, threshold)
    views = {side: side_view(P, side) for side in SIDES}
    contexts = {side: views[side]['context'] for side in SIDES}
    candidates = {side: side_candidates(views[side], P['short'])
                  for side in SIDES}

    result = {'splits': {}, 'chosen': {}}
    for side in SIDES:
        brightness, splits = candidates[side]
        result['splits'][side] = [
            summarize(s['edge'], {
                'name': s['name'], 'k': s['k'],
                'assignment': list(s['assignment']),
                'score': round(float(s['score']), 3),
                'centers_L': s['centers_L'],
                'profile_pitch': (float(s['profile']['pitch_px'])
                                  if 'pitch_px' in s['profile'] else None),
                'profile_count': _int(s['profile'].get('count', 0)),
            })
            for s in splits
        ]

    for method in METHODS:
        sides, picked = {}, {}
        for side in SIDES:
            brightness, splits = candidates[side]
            choice = select(brightness, splits, method)
            picked[side] = choice['name'] if choice else None
            edge = (copy.deepcopy(choice['edge']) if choice else
                    {'status': 'unavailable', 'reason': 'no usable candidate'})
            sides[side] = edge
            if 'valleys' in edge:
                add_edge_image_geometry(edge, side, contexts[side], P['basis'],
                                        P['origin'])
        exclude_points_outside_corners(sides, contexts)
        for side, edge in sides.items():
            if 'valleys' in edge and 'line' in edge:
                add_edge_image_geometry(edge, side, contexts[side], P['basis'],
                                        P['origin'])
        result[method] = {side: summarize(sides[side], {'pick': picked[side]})
                          for side in SIDES}
        result['chosen'][method] = picked
    return result


def _init(path):
    global _IMAGE
    _IMAGE = cv2.imread(path)


def _run(task):
    index, orientation, threshold = task
    result = measure_stamp_methods(_IMAGE, orientation, threshold)
    result['stamp'] = index + 1
    return result


def front_end(image):
    """Detection, orientation and threshold, exactly as segment_stamps does."""
    boxes, mask = detect_stamps_2d(image, background_delta=STAMP_DELTA)
    orientations = [estimate_orientation(mask, box) for box in boxes]
    scale = min(1.0, 1500 / image.shape[1])
    sample = cv2.resize(image, None, fx=scale, fy=scale,
                        interpolation=cv2.INTER_AREA)
    threshold = min(250, float(np.percentile(
        cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY), 10)) + STAMP_DELTA)
    return orientations, threshold


def main():
    image = cv2.imread(SCAN)
    print(f'scan {image.shape}')
    orientations, threshold = front_end(image)
    print(f'detected {len(orientations)} stamps, threshold {threshold:.1f}')
    del image

    tasks = [(i, orientations[i], threshold) for i in range(len(orientations))
             if orientations[i] is not None]
    print(f'measuring {len(tasks)} stamps x {len(METHODS)} methods')
    results = []
    with ProcessPoolExecutor(max_workers=10, initializer=_init,
                             initargs=(SCAN,)) as pool:
        for done, result in enumerate(pool.map(_run, tasks), 1):
            results.append(result)
            print(f'  stamp {result["stamp"]:>2} done ({done}/{len(tasks)})',
                  flush=True)
    results.sort(key=lambda r: r['stamp'])
    (OUT / 'results.json').write_text(json.dumps(
        results, indent=1,
        default=lambda v: v.item() if hasattr(v, 'item') else str(v)))
    print(f'-> {OUT / "results.json"}')


if __name__ == '__main__':
    main()
