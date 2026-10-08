"""Binary edge-band cleanup shared by production and experiments.

The normal axis is the first array axis, with the outermost pixel first.  A
returned depth is in the caller's full side-image coordinates.
"""
import cv2
import numpy as np

MIN_AREA = 120
CLOSE_WINDOW = 9
MIN_PAPER = 4


def first_run(band, n0):
    sustained = band[:-3] & band[1:-2] & band[2:-1] & band[3:]
    depth = sustained.argmax(axis=0).astype(float) + n0
    depth[~sustained.any(axis=0)] = np.nan
    return depth


def _drop_small(mask, min_area):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    small = [index for index in range(1, count)
             if stats[index, cv2.CC_STAT_AREA] < min_area]
    if not small:
        return mask
    return np.where(np.isin(labels, small), 0, mask).astype(np.uint8)


def area_filtered(band, min_area=MIN_AREA):
    opened = _drop_small(band, min_area)
    return (1 - _drop_small(1 - opened, min_area)).astype(np.uint8)


def interior_connected(band, rows=3):
    _, labels = cv2.connectedComponents(band, 8)
    seeds = set(np.unique(labels[-rows:])) - {0}
    if not seeds:
        return band
    return np.isin(labels, list(seeds)).astype(np.uint8)


def filled(band):
    outside = (1 - band).astype(np.uint8)
    _, labels = cv2.connectedComponents(outside, 4)
    reaching = set(np.unique(labels[0])) - {0}
    enclosed = (labels > 0) & ~np.isin(labels, list(reaching))
    return np.where(enclosed, 1, band).astype(np.uint8)


def count_depth(band, n0):
    paper = band.sum(axis=0, dtype=np.int32)
    depth = (n0 + band.shape[0] - paper).astype(float)
    depth[paper < MIN_PAPER] = np.nan
    return depth


def _rank_filter(values, window, reducer):
    pad = window // 2
    padded = np.pad(values, pad, constant_values=np.nan)
    view = np.lib.stride_tricks.sliding_window_view(padded, window)
    result = np.full(len(values), np.nan)
    usable = ~np.isnan(view).all(axis=1)
    result[usable] = reducer(view[usable], axis=1)
    return result


def closed_depth(depth, window=CLOSE_WINDOW):
    with np.errstate(all='ignore'):
        return _rank_filter(_rank_filter(depth, window, np.nanmax),
                            window, np.nanmin)


def profile_depth(band, n0, variant):
    """Return the selected production profile for one binary map."""
    band = np.asarray(band, dtype=np.uint8)
    if variant == 'close':
        return closed_depth(first_run(band, n0))
    if variant == 'stack':
        cleaned = filled(interior_connected(area_filtered(band)))
        return closed_depth(count_depth(cleaned, n0))
    raise ValueError(f'unknown production profile variant: {variant}')
