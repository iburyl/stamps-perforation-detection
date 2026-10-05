"""Measure perforation valleys from original-resolution image edge profiles.

Only OpenCV and NumPy are required. No closing/dilation is applied to the
measurement image. Pixel coordinates refer to the input image supplied by caller.
"""
import copy
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
from refine_perforation import refine_valleys

SIDES = ('top', 'bottom', 'left', 'right')


def _algorithm_trial(edge, elapsed_seconds, reason=None):
    """Return a compact, JSON-friendly record for a shadow algorithm run."""
    edge = edge or {}
    record = {
        'status': edge.get('status', 'unavailable'),
        'method': edge.get('method'),
        'reason': reason if reason is not None else edge.get('reason', ''),
        'count': int(edge.get('count', 0)),
        'circular_arc_count': int(circular_arc_count(edge)),
        'coverage': edge.get('coverage'),
        'pitch_px': edge.get('pitch_px'),
        'spacing_rms_px': edge.get('spacing_rms_px'),
        'depth_rms_px': edge.get('depth_rms_px'),
        'autocorrelation': edge.get('autocorrelation'),
        'geometry_source': edge.get('geometry_source'),
        'line': edge.get('line'),
        'elapsed_ms': float(elapsed_seconds * 1000.0),
    }
    # Shadow recovery results have not yet been mapped back to image geometry,
    # so they carry slope/intercept but not the public ``line`` field. Reliability
    # here is the same arc evidence test without requiring that later mapping.
    record['reliable'] = bool(
        edge.get('geometry_source') == 'circle_arcs'
        and circular_arc_count(edge) >= 5
        and ('line' in edge or (
            'slope' in edge and 'intercept' in edge
        ))
    )
    return record


def boundary_profile(binary, n0, n1, start, end):
    """First sustained run of paper pixels; binary is normal x along-edge."""
    strip = binary[n0:n1+3, start:end]
    sustained = strip[:-3] & strip[1:-2] & strip[2:-1] & strip[3:]
    depth = sustained.argmax(axis=0).astype(float)+n0
    depth[~sustained.any(axis=0)] = np.nan
    return depth


def adaptive_color_edge(lab, gray, threshold, positions, n0, n1, short_size):
    """Recover a paper/hinge boundary using locally selected Lab thresholds.

    Restrict the search to the outer edge band, require periodic valleys and
    agreement between neighbouring thresholds. Do not replace a usable existing
    brightness profile. Colour alone never establishes a perforated edge.
    """
    start, end = int(positions[0]), int(positions[-1])+1
    region = lab[n0:n1+3, start:end]
    light = gray[n0:n1+3, start:end] > threshold
    candidates = []
    for channel, name in ((1, 'a'), (2, 'b')):
        values = region[:, :, channel][light]
        if len(values) < len(positions)*4:
            continue
        low, high = np.percentile(values, [5, 95])
        if high-low < 6:
            continue
        levels = np.unique(np.rint(np.linspace(low, high, 21))).astype(int)
        for polarity in (1, -1):
            for level in levels:
                selected = ((lab[:, :, channel] > level) if polarity == 1
                            else (lab[:, :, channel] < level))
                binary = (selected & (gray > threshold)).astype(np.uint8)
                depth = boundary_profile(binary, n0, n1, start, end)
                edge = measure_profile(positions, depth, short_size)
                if (edge.get('count', 0) < 6 or edge.get('coverage', 0) < 0.6 or
                        edge.get('autocorrelation', 0) < 0.45 or
                        edge['spacing_rms_px']/edge['pitch_px'] > 0.08):
                    continue
                edge.update(method=f'adaptive_lab_{name}', color_threshold=int(level),
                            color_polarity=polarity, status='review',
                            reason='adaptive color boundary; verify against photograph')
                candidates.append(edge)
    stable = []
    center = float(np.mean(positions))
    for edge in candidates:
        peers = [other for other in candidates if other is not edge and
                 other['method'] == edge['method'] and
                 other['color_polarity'] == edge['color_polarity'] and
                 0 < abs(other['color_threshold']-edge['color_threshold']) <= 3 and
                 abs(other['pitch_px']/edge['pitch_px']-1) < 0.04 and
                 abs((other['slope']-edge['slope'])*center +
                     other['intercept']-edge['intercept']) < max(3, short_size*0.006)]
        if peers:
            # Coverage and count outrank very regular but sparse accidental peaks.
            score = edge['count']*edge['coverage']*edge['autocorrelation']
            stable.append((score, edge))
    if not stable:
        return None
    edge = max(stable, key=lambda item: item[0])[1]
    channel = 1 if edge['method'].endswith('_a') else 2
    selected = ((region[:, :, channel] > edge['color_threshold']) if edge['color_polarity'] == 1
                else (region[:, :, channel] < edge['color_threshold']))
    edge['debug_mask'] = ((selected & light)*255).astype(np.uint8)
    return edge


def robust_line(x, y, tolerance=2.0):
    keep = np.ones(len(x), dtype=bool)
    for _ in range(6):
        if keep.sum() < 3:
            break
        a, b = np.polyfit(x[keep], y[keep], 1)
        residual = y - (a*x + b)
        center = np.median(residual[keep])
        mad = np.median(np.abs(residual[keep] - center))
        updated = np.abs(residual-center) <= max(tolerance, 3*1.4826*mad)
        if np.array_equal(updated, keep):
            break
        keep = updated
    if keep.sum() < 3:
        return None
    a, b = np.polyfit(x[keep], y[keep], 1)
    return float(a), float(b), keep


def measure_profile(position, depth, short_size):
    """Positive depth points inward. Return a line through periodic valley bases."""
    valid = np.isfinite(depth)
    unavailable = {'status': 'unavailable', 'reason': 'insufficient edge profile'}
    if valid.sum() < 0.7*len(depth) or len(depth) < 40:
        return unavailable
    baseline = robust_line(position[valid], depth[valid], tolerance=3)
    if baseline is None:
        return unavailable
    a, b, _ = baseline
    filled = np.interp(position, position[valid], depth[valid])
    smooth = cv2.GaussianBlur(filled.reshape(1, -1), (0, 0), 1.0).ravel()
    signal = smooth - (a*position+b)
    low, high = np.percentile(signal[valid], [10, 90])
    amplitude = high-low
    if amplitude < max(2.5, short_size*0.003):
        return {'status': 'unavailable', 'reason': 'no distinct valleys'}
    signal = np.clip(signal, low-amplitude*0.5, high+amplitude*0.5)
    signal -= np.mean(signal)
    min_lag = max(8, int(short_size/45))
    max_lag = min(len(signal)//5, int(short_size/7))
    if max_lag <= min_lag:
        return unavailable
    correlations = []
    for lag in range(min_lag, max_lag+1):
        lhs, rhs = signal[:-lag], signal[lag:]
        weight = valid[:-lag] & valid[lag:]
        lhs, rhs = lhs[weight], rhs[weight]
        correlations.append(float(np.dot(lhs, rhs) /
                                  max(1e-9, np.linalg.norm(lhs)*np.linalg.norm(rhs))))
    correlations = np.array(correlations)
    maxima = [i for i in range(1, len(correlations)-1)
              if correlations[i] >= correlations[i-1] and correlations[i] > correlations[i+1]]
    if not maxima:
        return {'status': 'unavailable', 'reason': 'no periodic edge signal'}
    best = max(correlations[i] for i in maxima)
    lag_index = next(i for i in maxima if correlations[i] >= max(0.25, best*0.60)) if best >= 0.25 else None
    if lag_index is None:
        return {'status': 'unavailable', 'reason': 'weak periodicity'}
    period = float(min_lag+lag_index)
    # Find individual peaks in the detrended profile, then reject shallow texture.
    detrended = smooth-(a*position+b)
    radius = max(3, int(period*0.45))
    candidates = []
    for i in range(radius, len(smooth)-radius):
        if not valid[max(0, i-2):i+3].all():
            continue
        if detrended[i] >= detrended[i-1] and detrended[i] > detrended[i+1]:
            prominence = detrended[i] - max(np.min(detrended[i-radius:i]),
                                            np.min(detrended[i+1:i+radius+1]))
            if prominence >= max(2, amplitude*0.22):
                candidates.append((i, prominence))
    selected = []
    for i, prominence in sorted(candidates, key=lambda item: -item[1]):
        if all(abs(i-j) >= period*0.55 for j in selected):
            selected.append(i)
    selected.sort()
    if len(selected) < 5:
        return {'status': 'unavailable', 'reason': 'fewer than five valleys'}
    indices = np.array(selected)
    x = position[indices].astype(float)
    y = smooth[indices].astype(float)
    # Subpixel quadratic peak interpolation over three profile samples.
    for j, i in enumerate(indices):
        denom = detrended[i-1]-2*detrended[i]+detrended[i+1]
        shift = np.clip(0.5*(detrended[i-1]-detrended[i+1])/denom, -0.5, 0.5) if denom < -1e-9 else 0
        x[j] += shift
        y[j] = np.interp(x[j], position, smooth)
    phase = max(x, key=lambda origin: np.sum(np.abs((x-origin+period/2) % period-period/2) < period*0.18))
    lattice = np.rint((x-phase)/period)
    periodic = np.abs(x-(phase+lattice*period)) < period*0.18
    if periodic.sum() < 5:
        return {'status': 'unavailable', 'reason': 'irregular valley spacing'}
    for _ in range(3):
        if periodic.sum() < 5 or len(np.unique(lattice[periodic])) < 5:
            return {'status': 'unavailable', 'reason': 'unstable valley spacing'}
        period_fit, phase_fit = np.polyfit(lattice[periodic], x[periodic], 1)
        residual = x-(phase_fit+lattice*period_fit)
        periodic = np.abs(residual) < period_fit*0.15
    if periodic.sum() < 5:
        return {'status': 'unavailable', 'reason': 'unstable valley spacing'}
    fit = robust_line(x[periodic], y[periodic], tolerance=max(2, amplitude*0.15))
    if fit is None:
        return unavailable
    slope, intercept, inliers = fit
    accepted = np.zeros(len(x), dtype=bool)
    accepted[np.flatnonzero(periodic)[inliers]] = True
    count = int(accepted.sum())
    if count < 5:
        return {'status': 'unavailable', 'reason': 'too few consistent valley bases'}
    pitch_axis, phase_fit = np.polyfit(lattice[accepted], x[accepted], 1)
    pitch = float(pitch_axis*np.sqrt(1+slope*slope))
    spacing_rms = float(np.sqrt(np.mean((x[accepted] - (phase_fit+lattice[accepted]*pitch_axis))**2)))
    depth_rms = float(np.sqrt(np.mean((y[accepted]-(slope*x[accepted]+intercept))**2)))
    coverage = float((x[accepted].max()-x[accepted].min())/(position[-1]-position[0]))
    expected = int(lattice[accepted].max()-lattice[accepted].min()+1)
    status = 'ok' if (count >= 6 and coverage >= 0.65 and
                       count/expected >= 0.75 and correlations[lag_index] >= 0.45 and
                       spacing_rms/pitch < 0.08 and depth_rms < max(3, amplitude*0.2)) else 'review'
    return {'status': status, 'reason': '', 'slope': slope, 'intercept': intercept,
            'pitch_px': pitch, 'valleys': np.column_stack([x, y]), 'accepted': accepted,
            'count': count, 'coverage': coverage, 'autocorrelation': float(correlations[lag_index]),
            'spacing_rms_px': spacing_rms, 'depth_rms_px': depth_rms,
            'amplitude_px': float(amplitude)}


def refine_edge(edge, signal, positions, depth):
    actual = edge['valleys'].copy()
    actual_accepted = edge['accepted'].copy()
    pitch_axis = edge['pitch_px'] / np.sqrt(1 + edge['slope']**2)
    origin = actual[actual_accepted][0, 0]
    actual_lattice = np.rint((actual[:, 0] - origin) / pitch_axis).astype(int)
    accepted_lattice = actual_lattice[actual_accepted]
    pitch_fit, phase_fit = np.polyfit(
        accepted_lattice, actual[actual_accepted, 0], 1
    )
    first_index = int(np.ceil((positions[0] - phase_fit) / pitch_fit))
    last_index = int(np.floor((positions[-1] - phase_fit) / pitch_fit))
    occupied = {
        int(index) for point, index in zip(actual, actual_lattice)
        if first_index <= index <= last_index
        and abs(point[0] - (phase_fit + index * pitch_fit)) < abs(pitch_fit) * 0.35
    }
    hypothetical = np.array([
        [phase_fit + index * pitch_fit,
         edge['slope'] * (phase_fit + index * pitch_fit) + edge['intercept']]
        for index in range(first_index, last_index + 1)
        if index not in occupied
    ], dtype=float).reshape(-1, 2)
    before = np.vstack([actual, hypothetical]) if len(hypothetical) else actual
    seed_accepted = np.concatenate([
        actual_accepted, np.zeros(len(hypothetical), dtype=bool)
    ])
    fits = refine_valleys(signal, positions, depth, before, edge['pitch_px'])
    after = before.copy()
    for i, fit in enumerate(fits):
        if fit:
            after[i] = fit['point']
    edge['initial_valleys'] = actual
    edge['refinement_fits'] = [None]*len(actual)
    edge['refined_count'] = 0
    if not any(fit is not None for fit in fits):
        return
    if sum(fit is not None and fit.get('model') == 'circle' for fit in fits) >= 5:
        if refine_edge_from_arc_lattice(edge, signal, positions, depth, before, fits):
            return
    # Refit spacing and the base line, allowing refined formerly rejected points
    # back in only if both their phase and depth agree with the other holes.
    pitch_axis = edge['pitch_px']/np.sqrt(1+edge['slope']**2)
    origin = before[seed_accepted][0, 0]
    lattice = np.rint((before[:, 0]-origin)/pitch_axis)
    spacing = np.polyfit(lattice[seed_accepted], after[seed_accepted, 0], 1)
    eligible = seed_accepted | np.array([fit is not None for fit in fits])
    circle_indices = np.array([
        i for i, fit in enumerate(fits)
        if fit is not None and fit.get('model') == 'circle'
    ], dtype=int)
    if len(circle_indices) >= 3:
        rejected = circle_indices[~arc_line_inliers(after[circle_indices], edge['pitch_px'])]
        eligible[rejected] = False
    eligible &= np.abs(after[:, 0]-np.polyval(spacing, lattice)) < pitch_axis*0.15
    if eligible.sum() < 5:
        return
    line = robust_line(after[eligible, 0], after[eligible, 1], max(2, edge['amplitude_px']*0.15))
    if line is None:
        return
    slope, intercept, keep = line
    accepted = np.zeros(len(after), bool)
    accepted[np.flatnonzero(eligible)[keep]] = True
    if accepted.sum() < 5:
        return
    spacing = np.polyfit(lattice[accepted], after[accepted, 0], 1)
    edge.update(valleys=after, initial_valleys=before, accepted=accepted,
                slope=slope, intercept=intercept,
                refinement_fits=fits, refined_count=sum(fit is not None for fit in fits),
                coverage=float(np.ptp(after[accepted, 0])/(positions[-1]-positions[0])),
                count=int(accepted.sum()), pitch_px=float(spacing[0]*np.sqrt(1+slope*slope)),
                spacing_rms_px=float(np.sqrt(np.mean((after[accepted, 0]-np.polyval(spacing, lattice[accepted]))**2))),
                depth_rms_px=float(np.sqrt(np.mean((after[accepted, 1]-slope*after[accepted, 0]-intercept)**2))))
    expected = int(np.ptp(lattice[accepted])+1)
    if (edge['count'] < 6 or edge['coverage'] < 0.65 or edge['count']/expected < 0.75 or
            edge['spacing_rms_px']/edge['pitch_px'] >= 0.08 or
            edge['depth_rms_px'] >= max(3, edge['amplitude_px']*0.2)):
        edge['status'] = 'review'


def fit_arc_lattice(values, positions, reference_pitch=None):
    """Fit an integer lattice used while constructing the edge geometry."""
    values = np.asarray(values, dtype=float)
    if len(values) < 5:
        return None
    span = float(positions[-1] - positions[0])
    min_pitch = max(8.0, span / 45.0)
    max_pitch = span / 5.0
    differences = np.abs(values[:, None] - values[None, :])
    differences = differences[np.triu_indices(len(values), 1)]
    candidates = {
        round((difference / multiple) * 2) / 2
        for difference in differences
        for multiple in range(1, min(12, int(difference // min_pitch)) + 1)
        if min_pitch <= difference / multiple <= max_pitch
        and (reference_pitch is None or
             abs(difference / multiple / reference_pitch - 1) <= 0.05)
    }

    best = None
    for candidate in candidates:
        for origin in values:
            lattice = np.rint((values - origin) / candidate).astype(int)
            residual = np.abs(values - (origin + lattice * candidate))
            chosen = []
            for index in np.unique(lattice):
                members = np.flatnonzero(lattice == index)
                chosen.append(members[np.argmin(residual[members])])
            chosen = np.array(chosen, dtype=int)
            if len(chosen) < 5:
                continue
            for _ in range(3):
                spacing = np.polyfit(lattice[chosen], values[chosen], 1)
                if not np.isfinite(spacing).all() or spacing[0] <= 0:
                    chosen = np.empty(0, dtype=int)
                    break
                lattice = np.rint((values - spacing[1]) / spacing[0]).astype(int)
                residual = np.abs(values - np.polyval(spacing, lattice))
                eligible = np.flatnonzero(residual < abs(spacing[0]) * 0.15)
                chosen = np.array([
                    members[np.argmin(residual[members])]
                    for index in np.unique(lattice[eligible])
                    for members in [eligible[lattice[eligible] == index]]
                ], dtype=int)
                if len(chosen) < 5:
                    break
            if len(chosen) < 5:
                continue
            spacing = np.polyfit(lattice[chosen], values[chosen], 1)
            if (reference_pitch is not None and
                    abs(spacing[0] / reference_pitch - 1) > 0.05):
                continue
            indices = lattice[chosen]
            expected = int(np.ptp(indices) + 1)
            occupancy = len(chosen) / expected
            rms_ratio = float(np.sqrt(np.mean(
                (values[chosen] - np.polyval(spacing, indices)) ** 2
            )) / abs(spacing[0]))
            direct_gaps = np.diff(np.sort(values[chosen]))
            direct_support = int(np.sum(
                np.abs(direct_gaps / spacing[0] - 1) < 0.12
            ))
            score = (direct_support, len(chosen), -int(rms_ratio / 0.01),
                     -(abs(spacing[0] / reference_pitch - 1)
                       if reference_pitch is not None else 0),
                     occupancy, -rms_ratio)
            if best is None or score > best[0]:
                best = (score, float(spacing[0]), float(spacing[1]), chosen, indices)
    if best is None:
        return None
    _, pitch, phase, chosen, indices = best
    return pitch, phase, chosen, indices


def fit_extreme_arc_lattice(values, positions, reference_pitch=None):
    """Fit the coarsest integer lattice supported by all circular-arc points.

    The extreme points define an integer number of periods.  Every interior
    point must then lie close to the same lattice.  A point is omitted only
    when no all-point lattice exists and leave-one-out identifies one unique
    incompatible point.
    """
    values = np.asarray(values, dtype=float)
    if len(values) < 5:
        return None
    order = np.argsort(values)
    values = values[order]
    span = float(positions[-1] - positions[0])
    min_pitch = max(8.0, span / 45.0)
    max_pitch = span / 5.0

    def coarsest_fit(sample):
        sample = np.asarray(sample, dtype=float)
        distance = float(sample[-1] - sample[0])
        first = max(len(sample) - 1, int(np.ceil(distance / max_pitch)))
        last = int(np.floor(distance / min_pitch))
        candidates = []
        for intervals in range(first, last + 1):
            initial_pitch = distance / intervals
            lattice = np.rint(
                (sample - sample[0]) / initial_pitch
            ).astype(int)
            if (lattice[0] != 0 or lattice[-1] != intervals
                    or np.any(np.diff(lattice) <= 0)):
                continue
            pitch, phase = np.polyfit(lattice, sample, 1)
            if (not np.isfinite(pitch) or pitch <= 0
                    or (reference_pitch is not None
                        and abs(pitch / reference_pitch - 1) > 0.05)):
                continue
            residual = np.abs(sample - np.polyval((pitch, phase), lattice))
            max_ratio = float(np.max(residual) / pitch)
            rms_ratio = float(np.sqrt(np.mean(residual ** 2)) / pitch)
            if max_ratio <= 0.12 and rms_ratio <= 0.06:
                candidates.append((float(pitch), float(phase), lattice,
                                   max_ratio, rms_ratio))
        if not candidates:
            return None
        return max(candidates, key=lambda fit: (fit[0], -fit[3], -fit[4]))

    fit = coarsest_fit(values)
    chosen_sorted = np.arange(len(values), dtype=int)
    if fit is None:
        omitted = []
        for excluded in range(len(values)):
            candidate = coarsest_fit(np.delete(values, excluded))
            if candidate is not None:
                omitted.append((excluded, candidate))
        if not omitted:
            return None
        omitted.sort(key=lambda item: (
            item[1][3], item[1][4], -item[1][0],
        ))
        # Do not invent an outlier when several omissions explain the points
        # equally well.  Only one clearly incompatible point may be discarded.
        if (len(omitted) > 1
                and omitted[0][1][3] + 0.025 >= omitted[1][1][3]):
            return None
        excluded, fit = omitted[0]
        chosen_sorted = np.delete(chosen_sorted, excluded)

    pitch, phase, lattice, _, _ = fit
    chosen = order[chosen_sorted]
    return pitch, phase, chosen, lattice


def arc_line_inliers(points, pitch):
    """Find the largest straight-edge consensus without letting one arc tilt it."""
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return np.ones(len(points), dtype=bool)
    tolerance = max(2.5, pitch * 0.07)
    best = None
    for first in range(len(points)):
        for second in range(first + 1, len(points)):
            dx = points[second, 0] - points[first, 0]
            if abs(dx) < pitch * 0.8:
                continue
            slope = (points[second, 1] - points[first, 1]) / dx
            intercept = points[first, 1] - slope * points[first, 0]
            keep = np.abs(points[:, 1] - slope * points[:, 0] - intercept) <= tolerance
            if keep.sum() < 3:
                continue
            # Refit the consensus, then test against the final line as well.
            for _ in range(2):
                slope, intercept = np.polyfit(points[keep, 0], points[keep, 1], 1)
                updated = np.abs(points[:, 1] - slope * points[:, 0] - intercept) <= tolerance
                if np.array_equal(updated, keep) or updated.sum() < 3:
                    break
                keep = updated
            residual = np.abs(points[:, 1] - slope * points[:, 0] - intercept)
            keep &= residual <= tolerance
            if keep.sum() < 3:
                continue
            score = (int(keep.sum()), -float(np.mean(residual[keep] ** 2)),
                     float(np.ptp(points[keep, 0])))
            if best is None or score > best[0]:
                best = (score, keep)
    return best[1] if best is not None else np.ones(len(points), dtype=bool)


def refine_edge_from_arc_lattice(edge, signal, positions, depth, before, fits):
    """Complete a perforation lattice and fit its edge using circular arcs only."""
    circle_indices = [
        i for i, fit in enumerate(fits)
        if fit is not None and fit.get('model') == 'circle'
    ]
    circle_points = np.array([fits[i]['point'] for i in circle_indices])
    circle_indices = np.array(circle_indices, dtype=int)[
        arc_line_inliers(circle_points, edge['pitch_px'])
    ]
    if len(circle_indices) < 5:
        return False
    circle_points = np.array([fits[i]['point'] for i in circle_indices])
    lattice_fit = fit_arc_lattice(circle_points[:, 0], positions,
                                  edge.get('reference_pitch_px'))
    if lattice_fit is None:
        return False
    pitch_fit, phase_fit, chosen, selected_lattice = lattice_fit
    selected = circle_indices[chosen]
    selected_points = np.array([fits[i]['point'] for i in selected])
    if not np.isfinite(pitch_fit) or pitch_fit <= 0:
        return False
    line = robust_line(selected_points[:, 0], selected_points[:, 1],
                       max(2, edge['amplitude_px'] * 0.15))
    if line is None:
        return False
    slope, intercept, _ = line

    first_index = int(np.ceil((positions[0] - phase_fit) / pitch_fit))
    last_index = int(np.floor((positions[-1] - phase_fit) / pitch_fit))
    occupied = set(int(index) for index in selected_lattice)
    missing = [i for i in range(first_index, last_index + 1) if i not in occupied]
    predicted = np.array([
        [phase_fit + i * pitch_fit,
         slope * (phase_fit + i * pitch_fit) + intercept]
        for i in missing
    ], dtype=float).reshape(-1, 2)

    added_seeds = []
    added_fits = []
    if len(predicted):
        hypothetical_fits = refine_valleys(
            signal, positions, depth, predicted,
            float(abs(pitch_fit) * np.sqrt(1 + slope*slope)),
        )
        for seed, fit in zip(predicted, hypothetical_fits):
            if fit is not None and fit.get('model') == 'circle':
                added_seeds.append(seed)
                added_fits.append(fit)

    combined_before = np.vstack([before, added_seeds]) if added_seeds else before
    combined_fits = list(fits) + added_fits
    combined_after = combined_before.copy()
    for i, fit in enumerate(combined_fits):
        if fit is not None:
            combined_after[i] = fit['point']

    arc_indices = np.array([
        i for i, fit in enumerate(combined_fits)
        if fit is not None and fit.get('model') == 'circle'
    ], dtype=int)
    arc_indices = arc_indices[arc_line_inliers(combined_after[arc_indices],
                                               edge['pitch_px'])]
    if len(arc_indices) < 5:
        return False
    arc_points = combined_after[arc_indices]
    final_lattice = fit_arc_lattice(arc_points[:, 0], positions,
                                    edge.get('reference_pitch_px'))
    if final_lattice is None:
        return False
    final_pitch, final_phase, chosen, arc_lattice = final_lattice
    arc_indices = arc_indices[chosen]
    arc_points = combined_after[arc_indices]
    line = robust_line(arc_points[:, 0], arc_points[:, 1],
                       max(2, edge['amplitude_px'] * 0.15))
    if line is None:
        return False
    slope, intercept, keep = line
    accepted = np.zeros(len(combined_after), bool)
    accepted[arc_indices[keep]] = True
    if accepted.sum() < 5:
        return False

    accepted_points = combined_after[accepted]
    accepted_lattice = np.rint(
        (accepted_points[:, 0] - final_phase) / final_pitch
    ).astype(int)
    spacing = np.polyfit(accepted_lattice, accepted_points[:, 0], 1)
    edge.update(
        valleys=combined_after,
        initial_valleys=combined_before,
        accepted=accepted,
        slope=slope,
        intercept=intercept,
        refinement_fits=combined_fits,
        refined_count=len(arc_indices),
        coverage=float(np.ptp(accepted_points[:, 0]) / (positions[-1] - positions[0])),
        count=int(accepted.sum()),
        pitch_px=float(spacing[0] * np.sqrt(1 + slope*slope)),
        spacing_rms_px=float(np.sqrt(np.mean(
            (accepted_points[:, 0] - np.polyval(spacing, accepted_lattice))**2
        ))),
        depth_rms_px=float(np.sqrt(np.mean(
            (accepted_points[:, 1] - slope*accepted_points[:, 0] - intercept)**2
        ))),
        geometry_source='circle_arcs',
    )
    expected = int(np.ptp(accepted_lattice) + 1)
    quality_ok = (
        edge['count'] >= 6 and edge['coverage'] >= 0.65
        and edge['count'] / expected >= 0.75
        and edge['spacing_rms_px'] / edge['pitch_px'] < 0.08
        and edge['depth_rms_px'] < max(3, edge['amplitude_px'] * 0.2)
    )
    # Arc completion may lower confidence, but it must not promote an edge
    # already marked for review (for example an adaptive colour boundary).
    if not quality_ok:
        edge['status'] = 'review'
    return True


def side_to_patch(points, side, work_normal_size):
    """Convert side coordinates (along, inward) to rectified patch x/y."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    along, normal = points[:, 0], points[:, 1]
    if side in ('bottom', 'right'):
        normal = work_normal_size - 1 - normal
    if side in ('left', 'right'):
        return np.column_stack([normal, along])
    return np.column_stack([along, normal])


def patch_to_side(points, side, work_normal_size):
    """Convert rectified patch x/y to side coordinates (along, inward)."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if side in ('left', 'right'):
        along, normal = points[:, 1], points[:, 0]
    else:
        along, normal = points[:, 0], points[:, 1]
    if side in ('bottom', 'right'):
        normal = work_normal_size - 1 - normal
    return np.column_stack([along, normal])


def side_line_to_patch(slope, intercept, side, work_normal_size):
    """Convert a side-local line to its common rectified-patch equation."""
    if side in ('bottom', 'right'):
        return -float(slope), float(work_normal_size - 1 - intercept)
    return float(slope), float(intercept)


def patch_line(edge):
    """Return the common-coordinate line, with compatibility for old fixtures."""
    return edge.get('line_patch', edge.get('line'))


def add_edge_image_geometry(edge, side, context, basis, origin):
    """Attach common patch/image coordinates after a side measurement succeeds."""
    work_normal_size = context['work_shape'][0]
    points = side_to_patch(edge['valleys'], side, work_normal_size)
    edge['points_image'] = points @ basis.T + origin
    initial = side_to_patch(edge['initial_valleys'], side, work_normal_size)
    edge['initial_points_image'] = initial @ basis.T + origin
    for fit in edge['refinement_fits']:
        if fit:
            curve = side_to_patch(fit['curve'], side, work_normal_size)
            fit['curve_image'] = curve @ basis.T + origin
    a, b = edge['slope'], edge['intercept']
    edge['line_side'] = (float(a), float(b))
    edge['line_patch'] = side_line_to_patch(
        a, b, side, work_normal_size,
    )
    # ``line`` remains a compatibility alias, but is always patch-coordinate.
    edge['line'] = edge['line_patch']
    positions = context['positions']
    endpoints = np.array([[positions[0], a*positions[0]+b],
                          [positions[-1], a*positions[-1]+b]])
    line = side_to_patch(endpoints, side, work_normal_size)
    edge['line_image'] = line @ basis.T + origin


def _horizontal_vertical_intersection(horizontal, vertical):
    """Intersect y = ah*x+bh with x = av*y+bv in patch coordinates."""
    ah, bh = horizontal
    av, bv = vertical
    denominator = 1 - ah * av
    if abs(denominator) < 1e-6:
        return None
    x = (av * bh + bv) / denominator
    return np.array([x, ah * x + bh], dtype=float)


def exclude_points_outside_corners(sides, contexts):
    """Reject side holes whose apices lie beyond adjacent edge intersections.

    The profile search intentionally includes almost the full oriented bounding
    box. That is useful for damaged edges, but it also lets a corner hole be
    assigned to both adjoining sides. Once all four edge lines are known, use
    their actual intersections as the admissible interval for each side and
    refit the surviving arc lattice.
    """
    adjacent = {
        'top': ('left', 'right'), 'bottom': ('left', 'right'),
        'left': ('top', 'bottom'), 'right': ('top', 'bottom'),
    }
    # Compute every admissible interval from one immutable geometry snapshot.
    # Re-fitting one side must not change the coordinate system seen by sides
    # processed later in this pass.
    lines = {
        side: patch_line(sides.get(side, {}))
        for side in SIDES
    }
    bounds = {}
    for side in SIDES:
        edge = sides.get(side, {})
        if lines[side] is None or 'valleys' not in edge:
            continue
        first_name, second_name = adjacent[side]
        if lines[first_name] is None or lines[second_name] is None:
            continue
        if side in ('top', 'bottom'):
            corners = [
                _horizontal_vertical_intersection(lines[side], lines[first_name]),
                _horizontal_vertical_intersection(lines[side], lines[second_name]),
            ]
            axis = 0
        else:
            corners = [
                _horizontal_vertical_intersection(lines[first_name], lines[side]),
                _horizontal_vertical_intersection(lines[second_name], lines[side]),
            ]
            axis = 1
        if any(corner is None or not np.all(np.isfinite(corner))
               for corner in corners):
            continue
        bounds[side] = tuple(sorted(corner[axis] for corner in corners))

    for side in SIDES:
        edge = sides.get(side, {})
        accepted = np.asarray(edge.get('accepted', []), dtype=bool)
        if not len(accepted) or side not in bounds or 'valleys' not in edge:
            continue
        lower, upper = bounds[side]
        along = edge['valleys'][:, 0]
        inside = (along >= lower) & (along <= upper)
        edge['inside_corner_bounds'] = inside
        trimmed = accepted & inside
        if np.array_equal(trimmed, accepted):
            continue
        edge['accepted'] = trimmed
        if trimmed.sum() < 5:
            edge['count'] = int(trimmed.sum())
            edge['status'] = 'review'
            edge['reason'] = 'fewer than five arcs inside corner boundaries'
            edge.pop('pitch_px', None)
            edge.pop('geometry_source', None)
            continue

        points = edge['valleys'][trimmed]
        reference_axis_pitch = edge['pitch_px'] / np.sqrt(1 + edge['slope']**2)
        lattice = np.rint((points[:, 0] - points[0, 0]) / reference_axis_pitch).astype(int)
        if len(np.unique(lattice)) != len(lattice):
            edge['status'] = 'review'
            edge['reason'] = 'duplicate lattice nodes after corner filtering'
            edge.pop('pitch_px', None)
            edge.pop('geometry_source', None)
            continue
        spacing = np.polyfit(lattice, points[:, 0], 1)
        line = robust_line(points[:, 0], points[:, 1],
                           max(2, edge['amplitude_px'] * 0.15))
        if line is None:
            edge['status'] = 'review'
            edge['reason'] = 'unstable edge after corner filtering'
            edge.pop('pitch_px', None)
            edge.pop('geometry_source', None)
            continue
        slope, intercept, keep = line
        kept_indices = np.flatnonzero(trimmed)
        if not np.all(keep):
            trimmed[kept_indices[~keep]] = False
            edge['accepted'] = trimmed
            points = edge['valleys'][trimmed]
            if len(points) < 5:
                edge['count'] = len(points)
                edge['status'] = 'review'
                edge['reason'] = 'fewer than five arcs inside corner boundaries'
                edge.pop('pitch_px', None)
                edge.pop('geometry_source', None)
                continue
            lattice = np.rint(
                (points[:, 0] - points[0, 0]) / reference_axis_pitch
            ).astype(int)
            spacing = np.polyfit(lattice, points[:, 0], 1)
            slope, intercept = np.polyfit(points[:, 0], points[:, 1], 1)
        pitch = float(spacing[0] * np.sqrt(1 + slope*slope))
        edge.update(
            slope=float(slope), intercept=float(intercept),
            line_side=(float(slope), float(intercept)), count=int(trimmed.sum()),
            pitch_px=pitch,
            coverage=float(np.ptp(points[:, 0]) /
                           np.ptp(contexts[side]['positions'])),
            spacing_rms_px=float(np.sqrt(np.mean(
                (points[:, 0] - np.polyval(spacing, lattice))**2
            ))),
            depth_rms_px=float(np.sqrt(np.mean(
                (points[:, 1] - slope*points[:, 0] - intercept)**2
            ))),
        )
        work_shape = contexts[side].get('work_shape')
        if work_shape is not None:
            edge['line_patch'] = side_line_to_patch(
                slope, intercept, side, work_shape[0],
            )
        elif side in ('top', 'left'):
            edge['line_patch'] = (float(slope), float(intercept))
        else:
            edge['line_patch'] = patch_line(edge)
        edge['line'] = edge['line_patch']


def extended_adjacent_anchors(edge, side, context, direction):
    """Return the extreme known arc plus confirmed one/two-pitch extensions."""
    fits = edge.get('refinement_fits', [])
    accepted = edge.get('accepted', [])
    arc_indices = [i for i, fit in enumerate(fits)
                   if fit is not None and fit.get('model') == 'circle'
                   and (i >= len(accepted) or accepted[i])]
    # Only circle-refined (magenta) points are reliable enough to propagate
    # geometry into another side.
    usable = arc_indices
    if not usable:
        return []
    along = edge['valleys'][usable, 0]
    extreme_index = usable[int(np.argmin(along) if direction < 0 else np.argmax(along))]
    extreme = edge['valleys'][extreme_index, 0]
    anchors = [side_to_patch(edge['valleys'][extreme_index], side,
                             context['work_shape'][0])[0]]
    pitch_axis = edge['pitch_px'] / np.sqrt(1 + edge['slope']**2)
    seeds = np.array([
        [extreme + direction * step * pitch_axis,
         edge['slope'] * (extreme + direction * step * pitch_axis) + edge['intercept']]
        for step in (1, 2)
    ], dtype=float).reshape(-1, 2)
    # Even when the corner shape prevents an arc fit on the adjacent edge, its
    # extrapolated lattice node remains a useful hypothesis. The missing edge
    # itself must still confirm that hypothesis with at least five circle fits.
    anchors.extend(
        side_to_patch(seed, side, context['work_shape'][0])[0]
        for seed in seeds
    )
    fit_seeds = np.array([
        seed for seed in seeds
        if context['positions'][0] <= seed[0] <= context['positions'][-1]
    ], dtype=float).reshape(-1, 2)
    if not len(fit_seeds):
        return anchors
    predicted_depth = edge['slope'] * context['positions'] + edge['intercept']
    attempts = refine_valleys(context['signal'], context['positions'],
                              predicted_depth, fit_seeds, edge['pitch_px'])
    anchors.extend(
        side_to_patch(fit['point'], side, context['work_shape'][0])[0]
        for fit in attempts
        if fit is not None and fit.get('model') == 'circle'
    )
    return anchors


def circular_arc_count(edge):
    """Number of accepted circular arcs supporting an edge."""
    accepted = edge.get('accepted', [])
    return sum(
        fit is not None and fit.get('model') == 'circle'
        and (i >= len(accepted) or accepted[i])
        for i, fit in enumerate(edge.get('refinement_fits', []))
    )


def reliable_side(edge):
    """A line is geometrically reliable only with five circular arc fits."""
    return (
        'line' in edge
        and edge.get('geometry_source') == 'circle_arcs'
        and circular_arc_count(edge) >= 5
    )


def _side_confidence(edge):
    accepted = edge.get('accepted', [])
    circles = [
        fit for i, fit in enumerate(edge.get('refinement_fits', []))
        if fit is not None and fit.get('model') == 'circle'
        and (i >= len(accepted) or accepted[i])
    ]
    rms = np.median([fit.get('rms_px', np.inf) for fit in circles]) if circles else np.inf
    return (
        len(circles),
        float(edge.get('coverage', 0)),
        edge.get('status') == 'ok',
        -float(rms),
    )


def recovery_targets(sides, parallel_tolerance_deg=1.5):
    """Return absent, arc-poor, or geometrically inconsistent sides."""
    targets = {
        side for side in SIDES
        if not reliable_side(sides.get(side, {}))
    }
    for first_name, second_name in (('top', 'bottom'), ('left', 'right')):
        first, second = sides.get(first_name, {}), sides.get(second_name, {})
        if 'line' not in first or 'line' not in second:
            continue
        difference = abs(np.rad2deg(
            np.arctan(patch_line(first)[0]) - np.arctan(patch_line(second)[0])
        ))
        if difference <= parallel_tolerance_deg:
            continue
        # An already arc-poor side is the natural repair target. If both have
        # five magenta points, retain the stronger fit and rebuild the weaker.
        weak = [name for name in (first_name, second_name) if name in targets]
        if not weak:
            weak = [min((first_name, second_name),
                        key=lambda name: _side_confidence(sides[name]))]
        targets.update(weak)
    return targets


def recover_missing_side(side, sides, contexts):
    """Recover an edge from two adjacent sides, or one plus its opposite."""
    opposite = {'top': 'bottom', 'bottom': 'top',
                'left': 'right', 'right': 'left'}[side]
    adjacent = {'top': ('left', 'right'), 'bottom': ('left', 'right'),
                'left': ('top', 'bottom'), 'right': ('top', 'bottom')}[side]
    direction = -1 if side in ('top', 'left') else 1
    opposite_reliable = reliable_side(sides.get(opposite, {}))
    reliable_adjacent = [
        name for name in adjacent if reliable_side(sides.get(name, {}))
    ]
    # Two adjacent sides determine the missing line on their own. With only
    # one adjacent side, a reliable opposite edge must supply the parallel
    # direction while the adjacent edge supplies one corner anchor.
    if len(reliable_adjacent) < 2 and not (
            len(reliable_adjacent) == 1 and opposite_reliable):
        return None

    anchor_options = [
        extended_adjacent_anchors(sides[name], name, contexts[name], direction)
        for name in reliable_adjacent
    ]
    if any(not options for options in anchor_options):
        return None

    # Work in common patch coordinates. Prefer a reliable opposite edge. With
    # only two adjacent reliable sides, their corner extensions define the
    # missing line directly; an existing weak target is merely a pairing hint.
    if opposite_reliable:
        reference_slope = patch_line(sides[opposite])[0]
    elif patch_line(sides.get(side, {})) is not None:
        reference_slope = patch_line(sides[side])[0]
    else:
        reference_slope = 0.0

    def coordinates(anchor):
        return (anchor[1], anchor[0]) if side in ('left', 'right') else (anchor[0], anchor[1])

    target = sides.get(side, {})
    target_line = patch_line(target)
    desired_slope = reference_slope
    if (target_line is not None and np.isfinite(target_line[0])
            and abs(np.rad2deg(np.arctan(target_line[0])
                               - np.arctan(reference_slope))) <= 3.0):
        desired_slope = float(target_line[0])

    def implied_slope(pair):
        first_along, first_normal = coordinates(pair[0])
        second_along, second_normal = coordinates(pair[1])
        delta = second_along - first_along
        return ((second_normal - first_normal) / delta
                if abs(delta) >= 1 else np.inf)

    if len(anchor_options) == 2:
        # Two adjacent sides can reveal a genuinely non-parallel damaged edge.
        # Use the weak target only as a bounded slope hint; circle fits below
        # still decide whether the resulting line exists.
        anchors = min(
            ((first, second) for first in anchor_options[0]
             for second in anchor_options[1]),
            key=lambda pair: abs(implied_slope(pair) - desired_slope),
        )
    else:
        # The normal-offset search below spans +/-1.5 periods, so the extreme
        # confirmed arc is the least speculative single-corner anchor.
        anchors = (anchor_options[0][0],)
    anchors = np.array(anchors)
    if side in ('left', 'right'):
        anchor_along, anchor_normal = anchors[:, 1], anchors[:, 0]
    else:
        anchor_along, anchor_normal = anchors[:, 0], anchors[:, 1]
    if len(anchor_options) == 2:
        delta_along = anchor_along[1] - anchor_along[0]
        if abs(delta_along) < 1:
            return None
        common_slope = float(
            (anchor_normal[1] - anchor_normal[0]) / delta_along
        )
    else:
        common_slope = reference_slope
    common_intercept = float(np.mean(anchor_normal - common_slope * anchor_along))

    context = contexts[side]
    normal_size = context['work_shape'][0]
    if side in ('bottom', 'right'):
        slope = -common_slope
        intercept = normal_size - 1 - common_intercept
    else:
        slope = common_slope
        intercept = common_intercept

    pitch_sources = (side, opposite, *adjacent)
    pitch_candidates = []
    for name in pitch_sources:
        pitch = sides.get(name, {}).get('pitch_px')
        if pitch and np.isfinite(pitch) and pitch > 0 and all(
                abs(pitch / existing - 1) > 0.04 for existing in pitch_candidates):
            pitch_candidates.append(float(pitch))
    if not pitch_candidates:
        return None

    # A corner extension fixes the phase well, but its normal coordinate is not
    # the valley-base coordinate of the perpendicular edge. Search a bounded
    # neighbourhood while keeping every hypothesis parallel to the opposite.
    phase_fractions = (0, -0.25, 0.25, -0.5, 0.5)
    best = None
    target_arc_indices = [
        index for index, fit in enumerate(target.get('refinement_fits', []))
        if fit is not None and fit.get('model') == 'circle'
        and index < len(target.get('accepted', []))
        and target['accepted'][index]
    ]
    target_accepted = list(np.flatnonzero(target.get('accepted', [])))
    target_indices = target_arc_indices or target_accepted
    has_target_geometry = (
        'line' in target and 'intercept' in target
        and 'valleys' in target and bool(target_indices)
    )
    for hypothesis_pitch in pitch_candidates:
        pitch_axis = hypothesis_pitch / np.sqrt(1 + common_slope**2)
        phase = (float(target['valleys'][target_indices[0], 0])
                 if has_target_geometry else float(min(anchor_along)))
        first = int(np.ceil((context['positions'][0] - phase) / pitch_axis))
        last = int(np.floor((context['positions'][-1] - phase) / pitch_axis))
        seeds = np.array([
            [phase + index*pitch_axis,
             slope*(phase + index*pitch_axis) + intercept]
            for index in range(first, last + 1)
        ], dtype=float).reshape(-1, 2)
        if len(seeds) < 5:
            continue
        enough = max(5, int(np.ceil(len(seeds) * 0.65)))
        if has_target_geometry:
            normal_center = (target['intercept'] - intercept) / hypothesis_pitch
            normal_fractions = tuple(
                normal_center + value
                for step in range(3)
                for value in ((0,) if step == 0 else (-step*0.125, step*0.125))
            )
            candidate_phase_fractions = (0, -0.15, 0.15)
        else:
            normal_fractions = tuple(
                value
                for step in range(13)
                for value in ((0,) if step == 0 else (-step*0.125, step*0.125))
            )
            candidate_phase_fractions = phase_fractions
        for phase_fraction in candidate_phase_fractions:
            for normal_fraction in normal_fractions:
                phase_shift = phase_fraction * pitch_axis
                normal_shift = normal_fraction * hypothesis_pitch
                candidate_seeds = seeds + [phase_shift, slope*phase_shift + normal_shift]
                candidate_depth = (slope * context['positions'] + intercept + normal_shift)
                candidate_fits = refine_valleys(
                    context.get('recovery_signal', context['signal']),
                    context['positions'], candidate_depth,
                    candidate_seeds, hypothesis_pitch,
                )
                circles = [fit for fit in candidate_fits
                           if fit is not None and fit.get('model') == 'circle']
                score = (len(circles),
                         -sum(fit.get('rms_px', 0) for fit in circles),
                         -abs(phase_fraction), -abs(normal_fraction))
                if best is None or score > best[0]:
                    best = (score, candidate_seeds, candidate_depth,
                            candidate_fits, intercept + normal_shift,
                            hypothesis_pitch)
                if len(circles) >= enough:
                    break
            if best is not None and best[0][0] >= enough:
                break
        if best is not None and best[0][0] >= enough:
            break
    if best is None or best[0][0] < 5:
        return None
    _, seeds, predicted_depth, fits, intercept, hypothesis_pitch = best

    edge = {
        'status': 'review',
        'reason': (
            'recovered from two adjacent arc-supported edges'
            if len(reliable_adjacent) == 2
            else 'recovered from opposite and one adjacent arc-supported edge'
        ),
        'method': 'parallel_edge_arc_recovery',
        'slope': slope,
        'intercept': intercept,
        'pitch_px': hypothesis_pitch,
        'reference_pitch_px': sides[opposite]['pitch_px'] if opposite_reliable else None,
        'amplitude_px': sides.get(side, {}).get(
            'amplitude_px', sides.get(opposite, {}).get(
                'amplitude_px', max(3, hypothesis_pitch*0.1))),
        'valleys': seeds,
        'accepted': np.zeros(len(seeds), dtype=bool),
    }
    if not refine_edge_from_arc_lattice(
            edge, context.get('recovery_signal', context['signal']),
            context['positions'], predicted_depth, seeds, fits):
        return None
    # A sparse reconstructed line must at least span the side and follow one
    # coherent period; otherwise a local run of false arcs can invent a side.
    if (edge['coverage'] < 0.65 or
            edge['spacing_rms_px'] / edge['pitch_px'] >= 0.06):
        return None
    if opposite_reliable:
        opposite_final_pitch = sides[opposite]['pitch_px']
        if abs(edge['pitch_px'] / opposite_final_pitch - 1) > 0.15:
            return None
    edge['status'] = 'review'
    return edge


def recover_side_by_parallel_scan(side, sides, contexts):
    """Search inward lines when the first-light boundary is false.

    A bright scanner/background strip can appear before the actual stamp edge.
    In that case boundary_profile() is flat and the normal recovery remains
    anchored to the wrong boundary.  Use the reliable opposite-side pitch and
    scan normal offsets inward from the oriented envelope.  Prefer the weak
    target's measured slope when it is plausible, then fall back to a line
    parallel to the opposite side.  The strict circle fitter remains the final
    gate, so a noisy preliminary slope cannot establish an edge by itself.
    """
    opposite = {'top': 'bottom', 'bottom': 'top',
                'left': 'right', 'right': 'left'}[side]
    reference = sides.get(opposite, {})
    if not reliable_side(reference):
        return None
    pitch = reference.get('pitch_px')
    if not pitch or not np.isfinite(pitch) or pitch <= 0:
        return None
    context = contexts[side]
    bounds = context.get('normal_search')
    expected_normal = context.get('expected_normal')
    if bounds is None or expected_normal is None:
        return None
    n0, n1 = bounds
    start_normal = max(float(n0), float(expected_normal) - 0.5 * pitch)
    normal_values = np.arange(start_normal, float(n1) + 1, pitch * 0.125)
    positions = context['positions']
    center_along = float(np.mean(positions))
    reference_patch_slope = float(patch_line(reference)[0])
    reference_side_slope = (-reference_patch_slope
                            if side in ('bottom', 'right')
                            else reference_patch_slope)
    slope_candidates = []
    target = sides.get(side, {})
    target_slope = context.get('recovery_slope', target.get('slope'))
    if (target_slope is not None and np.isfinite(target_slope)
            and abs(np.rad2deg(np.arctan(target_slope))) <= 3.0):
        slope_candidates.append(float(target_slope))
    if all(abs(reference_side_slope - value) > 1e-4
           for value in slope_candidates):
        slope_candidates.append(reference_side_slope)
    phase_fractions = (0.0, 0.25, 0.5, 0.75)
    best = None
    for slope_index, slope in enumerate(slope_candidates):
        for center_normal in normal_values:
            intercept = float(center_normal - slope * center_along)
            predicted_depth = slope * positions + intercept
            for phase_fraction in phase_fractions:
                phase = float(positions[0] + phase_fraction * pitch)
                first = int(np.ceil((positions[0] - phase) / pitch))
                last = int(np.floor((positions[-1] - phase) / pitch))
                seeds = np.array([
                    [phase + index * pitch,
                     slope * (phase + index * pitch) + intercept]
                    for index in range(first, last + 1)
                ], dtype=float).reshape(-1, 2)
                if len(seeds) < 5:
                    continue
                fits = refine_valleys(
                    context.get('recovery_signal', context['signal']),
                    positions, predicted_depth, seeds, pitch,
                )
                circles = [
                    fit for fit in fits
                    if fit is not None and fit.get('model') == 'circle'
                ]
                score = (
                    len(circles),
                    -sum(fit.get('rms_px', 0) for fit in circles),
                    -slope_index,
                    -abs(center_normal - expected_normal),
                )
                if best is None or score > best[0]:
                    best = (score, seeds, predicted_depth, fits,
                            slope, intercept)
    if best is None or best[0][0] < 5:
        return None
    _, seeds, predicted_depth, fits, slope, intercept = best
    edge = {
        'status': 'review',
        'reason': 'recovered by parallel inward scan using opposite-side pitch',
        'method': 'parallel_normal_scan_recovery',
        'slope': float(slope),
        'intercept': float(intercept),
        'pitch_px': float(pitch),
        'reference_pitch_px': float(pitch),
        'amplitude_px': reference.get('amplitude_px', max(3, pitch * 0.1)),
        'valleys': seeds,
        'accepted': np.zeros(len(seeds), dtype=bool),
    }
    if not refine_edge_from_arc_lattice(
            edge, context.get('recovery_signal', context['signal']),
            positions, predicted_depth, seeds, fits):
        return None
    if (edge['coverage'] < 0.55 or
            edge['spacing_rms_px'] / edge['pitch_px'] >= 0.08):
        return None
    return edge


def _fit_kmeans_sine(context):
    """Fit one k=3 colour-boundary sinusoid without external pitch input."""
    positions = context.get('positions')
    lab_signal = context.get('lab_signal')
    bounds = context.get('normal_search')
    expected_normal = context.get('expected_normal')
    if (positions is None or lab_signal is None or bounds is None
            or expected_normal is None or len(positions) < 30):
        return None
    n0, n1 = bounds
    columns = np.rint(positions).astype(int)
    if (columns.min() < 0 or columns.max() >= lab_signal.shape[1]
            or n1 <= n0 + 3):
        return None
    colour_band = lab_signal[n0:n1 + 1, columns]
    samples = colour_band.reshape(-1, 3).astype(np.float32)
    cv2.setRNGSeed(31003)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                60, 0.25)
    _, labels, _ = cv2.kmeans(
        samples, 3, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(colour_band.shape[:2])

    span = float(positions[-1] - positions[0])
    min_pitch = max(8.0, span / 45.0)
    max_pitch = min(span / 5.0, span / 7.0)
    if max_pitch <= min_pitch:
        return None

    def evaluate(x, y, start, stop, period, inliers=None):
        omega = 2 * np.pi / period
        matrix = np.column_stack((
            np.ones_like(x), np.sin(omega * x), np.cos(omega * x)))
        if inliers is None:
            inliers = np.ones(len(x), dtype=bool)
            for _ in range(4):
                if inliers.sum() < 20:
                    return None
                coefficients = np.linalg.lstsq(
                    matrix[inliers], y[inliers], rcond=None)[0]
                residual = y - matrix @ coefficients
                median = np.median(residual[inliers])
                mad = np.median(np.abs(residual[inliers] - median))
                cutoff = max(3.0, 2.5 * 1.4826 * mad)
                updated = np.abs(residual - median) <= cutoff
                if np.array_equal(updated, inliers):
                    break
                inliers = updated
        if inliers.sum() < 20:
            return None
        coefficients = np.linalg.lstsq(
            matrix[inliers], y[inliers], rcond=None)[0]
        fitted = matrix @ coefficients
        residual = y - fitted
        rmse = float(np.sqrt(np.mean(residual[inliers] ** 2)))
        baseline = np.ones((len(x), 1), dtype=float)
        baseline_coefficients = np.linalg.lstsq(
            baseline[inliers], y[inliers], rcond=None)[0]
        baseline_residual = (y[inliers]
                             - baseline[inliers] @ baseline_coefficients)
        baseline_sse = float(np.sum(baseline_residual ** 2))
        sine_sse = float(np.sum(residual[inliers] ** 2))
        improvement = max(0.0, 1.0 - sine_sse / max(baseline_sse, 1e-6))
        amplitude = float(np.hypot(coefficients[1], coefficients[2]))
        slope = 0.0
        mean_normal = float(coefficients[0])
        if (not period * 0.05 <= amplitude <= period * 0.35
                or abs(mean_normal - expected_normal) > period * 0.65):
            return None
        coverage = float(inliers.sum() / (stop - start))
        center_factor = np.exp(
            -abs(mean_normal - expected_normal) / (period * 0.5))
        amplitude_factor = min(amplitude / max(period * 0.12, 1), 1.5)
        score = (improvement * coverage * center_factor * amplitude_factor
                 / (1.0 + rmse / 6.0))
        phase = float(np.arctan2(coefficients[1], coefficients[2]) / omega)
        intercept = float(coefficients[0])
        return {
            'score': float(score), 'period': float(period),
            'amplitude': amplitude, 'slope': slope,
            'intercept': intercept, 'phase': phase,
            'rmse': rmse, 'improvement': improvement,
            'x': x, 'y': y, 'inliers': inliers,
            'start': start, 'stop': stop,
        }

    best = None
    widths = sorted(set(
        max(90, min(len(positions), int(round(len(positions) * fraction))))
        for fraction in (0.22, 0.30, 0.38, 0.46)
    ))
    period_grid = np.arange(np.ceil(min_pitch), np.floor(max_pitch) + 1, 1.0)
    for selected_label in range(3):
        curve = np.full(len(positions), np.nan, dtype=float)
        selected = labels == selected_label
        runs = selected[:-2] & selected[1:-1] & selected[2:]
        for column_index in range(runs.shape[1]):
            starts = np.flatnonzero(runs[:, column_index])
            if starts.size:
                curve[column_index] = float(n0 + starts[0])
        for width in widths:
            step = max(10, width // 6)
            starts = list(range(0, len(curve) - width + 1, step))
            final_start = len(curve) - width
            if final_start not in starts:
                starts.append(final_start)
            for start in starts:
                stop = start + width
                valid = np.isfinite(curve[start:stop])
                if valid.sum() < int(width * 0.70):
                    continue
                x = positions[start:stop][valid].astype(float)
                y = curve[start:stop][valid]
                for period in period_grid:
                    if width / period < 3.0:
                        continue
                    candidate = evaluate(
                        x, y, start, stop, float(period))
                    if (candidate is not None
                            and (best is None
                                 or candidate['score'] > best['score'])):
                        best = candidate
    # With no linear trend term, residual edge tilt lowers the sinusoid's
    # global score.  Keep this gate permissive and let the subsequent circle
    # fits, coverage and spacing checks decide whether the hypothesis is real.
    if best is None or best['score'] < 0.06 or best['improvement'] < 0.10:
        return None

    refined = best
    for period in np.round(np.arange(best['period'] - 1.0,
                                     best['period'] + 1.01, 0.1), 1):
        if not min_pitch <= period <= max_pitch:
            continue
        candidate = evaluate(
            best['x'], best['y'], best['start'], best['stop'],
            float(period), best['inliers'])
        if candidate is not None and candidate['score'] > refined['score']:
            refined = candidate
    return refined


def recover_side_by_periodic_attenuation(side, sides, contexts):
    """Recover circle arcs after a self-fitted k=3 sinusoidal attenuation."""
    context = contexts.get(side, {})
    positions = context.get('positions')
    signal = context.get('recovery_signal', context.get('signal'))
    bounds = context.get('normal_search')
    expected_normal = context.get('expected_normal')
    if (positions is None or signal is None or bounds is None
            or expected_normal is None or len(positions) < 5):
        return None

    sine_fit = _fit_kmeans_sine(context)
    if sine_fit is None:
        return None
    pitch = sine_fit['period']

    n0, n1 = bounds
    pad = int(np.ceil(pitch * 0.75))
    band0 = max(0, int(np.floor(n0)) - pad)
    band1 = min(signal.shape[0], int(np.ceil(n1)) + pad + 1)
    if band1 - band0 < 9:
        return None
    base_signal = signal[band0:band1].astype(np.float32)

    slope_candidates = [sine_fit['slope']]

    center_along = float(np.mean(positions))
    fitted_center = float(sine_fit['slope'] * center_along
                          + sine_fit['intercept'])
    normal_centers = [
        fitted_center + offset * pitch for offset in (-0.10, 0.0, 0.10)
    ]
    if not normal_centers:
        return None

    phase_values = [
        float(sine_fit['phase']) + offset * pitch
        for offset in (-0.10, 0.0, 0.10)
    ]
    amplitudes = [float(sine_fit['amplitude'])]

    columns = np.rint(positions).astype(int)
    valid_columns = ((columns >= 0) & (columns < signal.shape[1]))
    if not valid_columns.all():
        return None
    local_rows = np.arange(band1 - band0, dtype=np.float32)[:, None]
    best = None
    for slope_index, slope in enumerate(slope_candidates):
        pitch_axis = pitch / np.sqrt(1 + slope*slope)
        for center_normal in normal_centers:
            intercept = float(center_normal - slope * center_along)
            line_depth = slope * positions + intercept
            for amplitude in amplitudes:
                for phase in phase_values:
                    angle = 2 * np.pi * (positions - phase) / pitch_axis
                    sine_depth = line_depth + amplitude * np.cos(angle)
                    local_depth = sine_depth - band0
                    distance_above = local_depth[None, :] - local_rows
                    fade = np.clip(
                        1.0 - distance_above / (0.5 * pitch), 0.0, 1.0)
                    attenuation = np.where(distance_above > 0,
                                           fade * fade, 1.0)
                    attenuated = base_signal.copy()
                    attenuated[:, columns] *= attenuation

                    first = int(np.ceil((positions[0] - phase) / pitch_axis))
                    last = int(np.floor((positions[-1] - phase) / pitch_axis))
                    seed_x = phase + np.arange(first, last + 1) * pitch_axis
                    seed_y = (slope * seed_x + intercept + amplitude - band0)
                    seeds = np.column_stack((seed_x, seed_y))
                    if len(seeds) < 5:
                        continue
                    fits = refine_valleys(
                        attenuated, positions, local_depth, seeds, pitch)
                    circles = [
                        fit for fit in fits
                        if fit is not None and fit.get('model') == 'circle'
                    ]
                    score = (
                        len(circles),
                        -sum(fit.get('rms_px', 0) for fit in circles),
                        -slope_index,
                        -abs(center_normal - expected_normal),
                    )
                    if best is None or score > best[0]:
                        best = (score, seeds, local_depth, fits, attenuated,
                                slope, intercept - band0, amplitude)

    if best is None or best[0][0] < 5:
        return None
    _, seeds, local_depth, fits, attenuated, slope, local_intercept, amplitude = best
    edge = {
        'status': 'review',
        'reason': 'recovered by self-fitted k=3 sinusoidal attenuation',
        'method': 'sinusoidal_attenuation_recovery',
        'slope': float(slope),
        'intercept': float(local_intercept),
        'pitch_px': float(pitch),
        'amplitude_px': float(amplitude),
        'valleys': seeds,
        'accepted': np.zeros(len(seeds), dtype=bool),
    }
    if not refine_edge_from_arc_lattice(
            edge, attenuated, positions, local_depth, seeds, fits):
        return None
    if (edge['coverage'] < 0.50
            or edge['spacing_rms_px'] / edge['pitch_px'] >= 0.06
            or abs(edge['pitch_px'] / pitch - 1) > 0.08
            or circular_arc_count(edge) < 5):
        return None

    # Convert the normal coordinate back from the cropped search band into the
    # complete oriented side image used by add_edge_image_geometry().
    edge['intercept'] += band0
    for key in ('valleys', 'initial_valleys'):
        if key in edge:
            edge[key] = np.asarray(edge[key], dtype=float).copy()
            edge[key][:, 1] += band0
    for arc_fit in edge.get('refinement_fits', []):
        if arc_fit is None:
            continue
        if 'point' in arc_fit:
            arc_fit['point'] = np.asarray(arc_fit['point'], dtype=float).copy()
            arc_fit['point'][1] += band0
        if 'curve' in arc_fit:
            arc_fit['curve'] = np.asarray(arc_fit['curve'], dtype=float).copy()
            arc_fit['curve'][:, 1] += band0
    return edge


def measure_stamp(image, orientation, threshold, collect_trials=False):
    if orientation is None:
        return {'status': 'unavailable', 'sides': {}}
    angle = np.deg2rad(orientation['angle_deg'])
    basis = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
    width, height = orientation['width_px'], orientation['height_px']
    margin = int(np.ceil(min(width, height)*0.12))
    shape = (int(np.ceil(width))+2*margin, int(np.ceil(height))+2*margin)
    origin = np.array([orientation['center_x'], orientation['center_y']]) - basis @ np.array([shape[0]/2, shape[1]/2])
    transform = np.column_stack([basis, origin])
    patch = cv2.warpAffine(image, transform, shape,
                           flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0.6)
    binary = (gray > threshold).astype(np.uint8)
    # Chromatic discrimination needs slightly stronger noise suppression than
    # brightness, but no morphological closing that would fill the holes.
    lab = cv2.cvtColor(cv2.GaussianBlur(patch, (5, 5), 1.0), cv2.COLOR_BGR2LAB)
    sides = {}
    contexts = {}
    algorithm_trials = {side: {} for side in SIDES} if collect_trials else None
    for side in SIDES:
        brightness_started = time.perf_counter()
        vertical = side in ('left', 'right')
        flipped = side in ('bottom', 'right')
        work = binary.T if vertical else binary
        if flipped:
            work = work[::-1]
        along_length = height if vertical else width
        normal_length = width if vertical else height
        # Keep almost the entire edge; only the outermost 1% is excluded.
        start = int(margin+along_length*0.01)
        end = int(margin+along_length*0.99)
        n0 = max(0, int(margin-normal_length*0.055))
        n1 = min(work.shape[0]-3, int(margin+normal_length*0.13))
        depth = boundary_profile(work, n0, n1, start, end)
        positions = np.arange(start, end, dtype=float)
        gray_work = gray.T if vertical else gray
        lab_work = lab.transpose(1, 0, 2) if vertical else lab
        if flipped:
            gray_work = gray_work[::-1]
            lab_work = lab_work[::-1]
        contexts[side] = {
            'positions': positions,
            'signal': gray_work,
            'lab_signal': lab_work,
            'depth': depth,
            'work_shape': work.shape,
            'normal_search': (n0, n1),
            'expected_normal': margin,
        }
        edge = measure_profile(positions, depth, min(width, height))
        edge['method'] = 'brightness'
        if 'valleys' in edge:
            refine_edge(edge, gray_work, positions, depth)
        if collect_trials:
            algorithm_trials[side]['brightness'] = _algorithm_trial(
                copy.deepcopy(edge), time.perf_counter() - brightness_started,
            )

        # A brightness profile may look periodic along a translucent hinge or
        # other obstruction while none of its candidates fits a true hole arc.
        # In that case the colour boundary deserves the same fallback search
        # as a completely missing brightness profile.
        if collect_trials or circular_arc_count(edge) < 5:
            color_started = time.perf_counter()
            color_work = lab_work
            # Stay outside the printed frame: only the outer 7% is eligible.
            color_n1 = min(n1, int(margin+normal_length*0.07))
            adaptive = adaptive_color_edge(color_work, gray_work, threshold,
                                           positions, n0, color_n1, min(width, height))
            if adaptive is not None:
                channel = 1 if adaptive['method'].endswith('_a') else 2
                color_signal = color_work[:, :, channel].astype(float) * adaptive['color_polarity']
                selected = ((color_signal > adaptive['color_threshold'] * adaptive['color_polarity'])
                            & (gray_work > threshold))
                color_depth = boundary_profile(selected.astype(np.uint8), n0, color_n1,
                                               start, end)
                refine_edge(adaptive, color_signal, positions, color_depth)
                if circular_arc_count(adaptive) > circular_arc_count(edge):
                    # A sub-threshold colour result cannot establish the side,
                    # but it can supply a cleaner signal and slope hypothesis
                    # to the later recovery, which still requires five circles.
                    contexts[side]['recovery_signal'] = color_signal
                    contexts[side]['recovery_depth'] = color_depth
                    contexts[side]['recovery_slope'] = adaptive.get('slope')
                if collect_trials:
                    algorithm_trials[side]['adaptive_color'] = _algorithm_trial(
                        copy.deepcopy(adaptive), time.perf_counter() - color_started,
                    )
                if circular_arc_count(adaptive) >= 5 and adaptive.get('geometry_source') == 'circle_arcs':
                    # A forced diagnostic trial must not replace a successful
                    # brightness result merely because it was also evaluated.
                    if circular_arc_count(edge) < 5:
                        edge = adaptive
                        contexts[side]['signal'] = color_signal
                        contexts[side]['depth'] = color_depth
            elif collect_trials:
                algorithm_trials[side]['adaptive_color'] = _algorithm_trial(
                    None, time.perf_counter() - color_started,
                    'no stable adaptive Lab boundary',
                )
        if 'valleys' in edge:
            add_edge_image_geometry(edge, side, contexts[side], basis, origin)
        sides[side] = edge

    # A corner hole can be found by both adjoining profiles. It is not a
    # trustworthy sample for either side when its apex falls beyond the actual
    # intersection of the fitted edge lines.
    exclude_points_outside_corners(sides, contexts)
    for side, edge in sides.items():
        if 'valleys' in edge and 'line' in edge:
            add_edge_image_geometry(edge, side, contexts[side], basis, origin)

    # Run both recovery families for every side against one identical baseline.
    # These shadow trials are diagnostics only and do not alter the chosen edge.
    if collect_trials:
        baseline_sides = copy.deepcopy(sides)
        for side in SIDES:
            recovery_started = time.perf_counter()
            recovered = recover_missing_side(side, baseline_sides, contexts)
            algorithm_trials[side]['parallel_edge_recovery'] = _algorithm_trial(
                recovered, time.perf_counter() - recovery_started,
                None if recovered is not None
                else 'prerequisites or quality gates failed',
            )
            scan_started = time.perf_counter()
            scanned = recover_side_by_parallel_scan(
                side, baseline_sides, contexts,
            )
            algorithm_trials[side]['parallel_normal_scan'] = _algorithm_trial(
                scanned, time.perf_counter() - scan_started,
                None if scanned is not None else (
                    'reliable opposite side unavailable or quality gates failed'
                ),
            )
            attenuation_started = time.perf_counter()
            if circular_arc_count(baseline_sides.get(side, {})) < 5:
                attenuated = recover_side_by_periodic_attenuation(
                    side, baseline_sides, contexts,
                )
                attenuation_reason = None if attenuated is not None else (
                    'k=3 sinusoid unavailable or circle quality gates failed'
                )
            else:
                attenuated = None
                attenuation_reason = 'existing side already has five circular arcs'
            algorithm_trials[side]['sinusoidal_attenuation'] = _algorithm_trial(
                attenuated, time.perf_counter() - attenuation_started,
                attenuation_reason,
            )

    # Rebuild not only absent sides, but every side unsupported by five
    # magenta arc points and the weaker member of a non-parallel opposite pair.
    # Recovery requires two reliable adjacent sides; a reliable opposite side
    # improves the hypothesis but is no longer mandatory.
    pending = recovery_targets(sides)
    for _ in range(2):
        progress = False
        for side in SIDES:
            if side not in pending:
                continue
            recovered = recover_missing_side(side, sides, contexts)
            if recovered is None:
                recovered = recover_side_by_parallel_scan(side, sides, contexts)
            attenuated = recover_side_by_periodic_attenuation(
                side, sides, contexts,
            )
            if attenuated is not None:
                recovered_arcs = circular_arc_count(recovered or {})
                attenuated_arcs = circular_arc_count(attenuated)
                recovered_pitch = (recovered or {}).get('pitch_px')
                attenuated_pitch = attenuated.get('pitch_px')
                doubled_period = (
                    recovered_pitch is not None
                    and attenuated_pitch is not None
                    and 1.75 <= recovered_pitch / attenuated_pitch <= 2.25
                    and attenuated_arcs >= recovered_arcs
                )
                if (recovered is None or doubled_period
                        or _side_confidence(attenuated)
                        > _side_confidence(recovered)):
                    recovered = attenuated
            if recovered is None:
                continue
            add_edge_image_geometry(recovered, side, contexts[side], basis, origin)
            sides[side] = recovered
            pending.remove(side)
            progress = True
        if not progress or not pending:
            break
    for side in pending:
        edge = sides[side]
        if 'line' not in edge:
            continue
        edge['status'] = 'review'
        if not edge.get('reason'):
            edge['reason'] = (
                'fewer than five circular arc fits'
                if circular_arc_count(edge) < 5
                else 'opposite edges are not sufficiently parallel'
            )
    result = {'sides': sides, 'status': 'unavailable'}
    for dimension, near, far, center, span in (
            ('width', 'left', 'right', shape[1]/2, height),
            ('height', 'top', 'bottom', shape[0]/2, width)):
        first, second = sides[near], sides[far]
        if 'line' not in first or 'line' not in second:
            continue
        a1, b1 = patch_line(first); a2, b2 = patch_line(second)
        # Intersect both fitted edges with the normal to their mean direction,
        # passing through the midpoint of the two lines at the stamp center.
        slope = np.tan((np.arctan(a1)+np.arctan(a2))/2)
        norm = np.sqrt(1+slope*slope)
        def distance_at(along):
            middle = ((a1+a2)*along+b1+b2)/2
            t1 = (a1*along+b1-middle)*norm/(1+a1*slope)
            t2 = (a2*along+b2-middle)*norm/(1+a2*slope)
            return abs(t2-t1)
        result[f'valley_{dimension}_px'] = float(distance_at(center))
        result[f'{dimension}_variation_px'] = float(abs(distance_at(center-span*0.35)-distance_at(center+span*0.35)))
        result[f'{dimension}_edge_angle_difference_deg'] = float(abs(np.rad2deg(np.arctan(a1)-np.arctan(a2))))
    for near, far in (('top', 'bottom'), ('left', 'right')):
        first, second = sides[near], sides[far]
        # The preliminary autocorrelation pitch is only a search hypothesis.
        # Expose a final pitch solely for an arc-only fitted lattice.
        for edge in (first, second):
            if not reliable_side(edge):
                edge.pop('pitch_px', None)
        if 'pitch_px' in first and 'pitch_px' in second:
            ratio = max(first['pitch_px'], second['pitch_px'])/min(first['pitch_px'], second['pitch_px'])
            if 1.8 < ratio < 2.2:
                for edge in (first, second):
                    edge['status'] = 'review'
                    edge['reason'] = 'possible doubled/halved period; opposite edges disagree'
    complete = all('line' in side for side in sides.values())
    result['status'] = ('ok' if complete and all(side['status'] == 'ok' for side in sides.values())
                        and max(result['width_edge_angle_difference_deg'], result['height_edge_angle_difference_deg']) < 1.5
                        else 'review' if any('line' in side for side in sides.values()) else 'unavailable')
    if collect_trials:
        for side in SIDES:
            algorithm_trials[side]['selected'] = _algorithm_trial(
                sides.get(side), 0.0,
            )
        result['algorithm_trials'] = algorithm_trials
    return result


def draw_measurement(image, result, offset=(0, 0), thickness=2):
    for edge in result['sides'].values():
        if 'points_image' not in edge:
            continue
        fits = edge.get('refinement_fits', [None] * len(edge['points_image']))
        inside = edge.get(
            'inside_corner_bounds', np.ones(len(edge['points_image']), dtype=bool)
        )
        for point, accepted, fit, inside_corners in zip(
                edge['points_image']+offset, edge['accepted'], fits, inside):
            if not inside_corners:
                continue
            center = tuple(np.rint(point).astype(int))
            if fit is not None and fit.get('model') == 'circle':
                color = (255, 0, 255)  # Magenta: apex refined from a circular arc.
                if not accepted:
                    cv2.circle(image, center, max(3, thickness*2),
                               color, max(1, thickness), cv2.LINE_AA)
                    continue
            else:
                # Orange is an unrefined working approximation.  Only a filled
                # magenta circular-arc point participates in the final result.
                color = (0, 120, 255)
            cv2.circle(image, center, max(3, thickness*2), color, -1, cv2.LINE_AA)
        line = np.rint(edge['line_image']+offset).astype(int)
        cv2.line(image, tuple(line[0]), tuple(line[1]), (255, 220, 0), thickness, cv2.LINE_AA)


def save_measurements(path, measurements, offset=(0, 0), dpi=None):
    fields = ['stamp', 'valley_width_px', 'valley_height_px',
              'width_variation_px', 'height_variation_px',
              'width_edge_angle_difference_deg', 'height_edge_angle_difference_deg']
    if dpi:
        fields += ['valley_width_mm', 'valley_height_mm']
    for side in SIDES:
        fields += [f'{side}_{key}' for key in ('method', 'color_threshold', 'color_polarity', 'count', 'refined_count', 'pitch_px',
                   'coverage', 'autocorrelation', 'spacing_rms_px', 'depth_rms_px')]
        if dpi:
            fields += [f'{side}_per_20mm']
    details = []
    with open(path, 'w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for number, result in enumerate(measurements, 1):
            row = {'stamp': number, **{
                key: value for key, value in result.items()
                if key != 'sides' and key in fields
            }}
            if dpi:
                for dim in ('width', 'height'):
                    if f'valley_{dim}_px' in result:
                        row[f'valley_{dim}_mm'] = result[f'valley_{dim}_px']*25.4/dpi
            detail = {'stamp': number, 'sides': {}}
            for name in SIDES:
                edge = result['sides'].get(name, {'status': 'unavailable', 'reason': 'no orientation'})
                for key, value in edge.items():
                    if f'{name}_{key}' in fields:
                        row[f'{name}_{key}'] = value
                if dpi and 'pitch_px' in edge:
                    row[f'{name}_per_20mm'] = 20*dpi/(25.4*edge['pitch_px'])
                detail['sides'][name] = {
                    'method': edge.get('method', 'unavailable')
                }
                if 'debug_mask' in edge:
                    directory = Path(path).with_suffix('')
                    directory = directory.with_name(directory.name + '_edge_masks')
                    directory.mkdir(parents=True, exist_ok=True)
                    mask_path = directory/f'stamp_{number:02}_{name}.png'
                    if not cv2.imwrite(str(mask_path), edge['debug_mask']):
                        raise RuntimeError(f'Cannot write {mask_path}')
                    detail['sides'][name].update(color_threshold=edge['color_threshold'],
                                               color_polarity=edge['color_polarity'],
                                               mask=str(mask_path.resolve()),
                                               mask_axes='x along edge; y inward toward stamp')
                if 'points_image' in edge:
                    detail['sides'][name].update(
                        points=(edge['points_image']+offset).tolist(),
                        initial_points=(edge['initial_points_image']+offset).tolist(),
                        refinement=[None if fit is None else {
                            'model': fit['model'], 'rms_px': fit['rms_px'],
                            'curve': (fit['curve_image']+offset).tolist()
                        } for fit in edge['refinement_fits']],
                        accepted=edge['accepted'].tolist(),
                        line=(edge['line_image']+offset).tolist())
            writer.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
            details.append(detail)
    Path(path).with_suffix('.json').write_text(json.dumps(
        {'coordinate_system': 'original image pixels; x right, y down',
         'dpi': dpi, 'stamps': details}, ensure_ascii=False, indent=2), encoding='utf-8')
