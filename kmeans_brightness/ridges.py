"""Interchangeable ways to turn one edge band into a boundary depth profile.

All of these take the band as a binary array whose first axis runs along the
normal, outward pixel first, and return one depth per column in the same
units as perforation.boundary_profile, so they are drop-in substitutes.

The baseline, ``raw``, assigns each column the first position where four
consecutive rows are paper. That is a minimum over a noisy set: a speck
outside the stamp can only move the answer outward, never inward, so the
error is one-sided and grows with how far out the band reaches. Each variant
below attacks that asymmetry from a different side, and ``stack`` applies all
of them together.
"""
import numpy as np

import cv2

VARIANTS = ('raw', 'area', 'conn', 'count', 'close', 'gate', 'stack')

# A perforation feature cannot be smaller than about a quarter of a millimetre
# across, which at 1200 dpi is a disk of roughly this area. Anything smaller
# is scanner noise or background texture whatever its brightness.
MIN_AREA = 120

# Narrower than a fifth of the shortest tooth the published gauge limits
# allow (gauge 22 is a 43 px pitch), so a real tooth flank always survives.
CLOSE_WINDOW = 9

# Matches the four-row run of the baseline: fewer paper pixels than this in a
# column is not evidence of paper at all.
MIN_PAPER = 4


def _first_run(band, n0):
    """The baseline, reimplemented on the band so variants can share it."""
    sustained = band[:-3] & band[1:-2] & band[2:-1] & band[3:]
    depth = sustained.argmax(axis=0).astype(float) + n0
    depth[~sustained.any(axis=0)] = np.nan
    return depth


def _drop_small(mask, min_area):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    small = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] < min_area]
    if not small:
        return mask
    return np.where(np.isin(labels, small), 0, mask).astype(np.uint8)


def area_filtered(band, min_area=MIN_AREA):
    """Remove paper specks and paper pinholes below the feature scale.

    Shape preserving in both directions: unlike an opening, the boundaries
    that survive do not move, so the valley bases keep their position and
    curvature and the measurand is untouched.
    """
    opened = _drop_small(band, min_area)
    return (1 - _drop_small(1 - opened, min_area)).astype(np.uint8)


def interior_connected(band, rows=3):
    """Keep only paper reachable from the stamp body.

    The innermost rows of the band lie inside the stamp, so a component that
    does not reach them is not stamp paper and cannot bound the edge. Specks
    outside the stamp then contribute nothing at all, which is the most direct
    attack on the one-sided bias. A mask so eroded that the teeth have broken
    away from the body would lose them, so the band is returned unchanged when
    the innermost rows hold no paper to grow from.
    """
    count, labels = cv2.connectedComponents(band, 8)
    seeds = set(np.unique(labels[-rows:])) - {0}
    if not seeds:
        return band
    return np.isin(labels, list(seeds)).astype(np.uint8)


def filled(band):
    """Make everything enclosed by paper count as paper.

    Counting paper only measures depth if the column really is a step, and it
    is not: a dark design near the edge falls below the threshold and leaves
    the stamp's own interior reading as background. Filling every background
    component that cannot reach the outer row of the band restores the step
    without touching the boundary itself.
    """
    outside = (1 - band).astype(np.uint8)
    count, labels = cv2.connectedComponents(outside, 4)
    reaching = set(np.unique(labels[0])) - {0}
    enclosed = (labels > 0) & ~np.isin(labels, list(reaching))
    return np.where(enclosed, 1, band).astype(np.uint8)


def count_depth(band, n0):
    """Depth from how much paper the column holds, not from where it starts.

    For a clean step column this is identical to the baseline. Under sparse
    noise outside the stamp the baseline is wrong by the speck's distance,
    while this is wrong by the speck's thickness, so the same information
    gives a far better breakdown point at the same cost. It does require the
    column to be a step, so callers pass it a filled band.
    """
    paper = band.sum(axis=0, dtype=np.int32)
    depth = (n0 + band.shape[0] - paper).astype(float)
    depth[paper < MIN_PAPER] = np.nan
    return depth


def _rank_filter(values, window, reducer):
    pad = window // 2
    padded = np.pad(values, pad, constant_values=np.nan)
    view = np.lib.stride_tricks.sliding_window_view(padded, window)
    out = np.full(len(values), np.nan)
    usable = ~np.isnan(view).all(axis=1)
    out[usable] = reducer(view[usable], axis=1)
    return out


def closed_depth(depth, window=CLOSE_WINDOW):
    """Morphological closing of the depth signal; removes shallow spikes only.

    Noise always makes a column look shallower, and the measurand is the
    innermost point, so the error and the signal sit at opposite extremes.
    A max filter followed by a min filter fills narrow dips and returns broad
    structure unchanged, which means it erases the noise and leaves the valley
    bases exactly where they were. Any symmetric smoother would instead pull
    those bases toward the mean.
    """
    return _rank_filter(_rank_filter(depth, window, np.nanmax),
                        window, np.nanmin)


def depth_of(band, n0, variant):
    """One band, one variant, one depth profile."""
    if variant == 'raw':
        return _first_run(band, n0)
    if variant == 'area':
        return _first_run(area_filtered(band), n0)
    if variant == 'conn':
        return _first_run(interior_connected(band), n0)
    if variant == 'count':
        return count_depth(filled(band), n0)
    if variant == 'close':
        return closed_depth(_first_run(band, n0))
    if variant == 'gate':
        # The two filters that never remove real paper: one works in the
        # image on noise too small to be perforation, the other works on the
        # profile on dips too narrow to be a tooth. They fail on different
        # sides, so together they cover more than either alone.
        return closed_depth(_first_run(area_filtered(band), n0))
    if variant == 'stack':
        cleaned = filled(interior_connected(area_filtered(band)))
        return closed_depth(count_depth(cleaned, n0))
    raise ValueError(f'unknown ridge variant: {variant}')
