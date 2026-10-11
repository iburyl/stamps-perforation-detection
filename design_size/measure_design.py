"""Measure the printed design rectangle of every stamp in one scan.

The input is a scan plus the ``data_*_perf.json`` written by ``segment_stamps.py``.
Every stamp is placed in a common frame by rotation and translation only, so
true scale is preserved, and the set is then jointly registered against its own
running average (congealing). The converged average is the golden design; its
frame rectangle is measured once, and every stamp's rectangle is that one
measurement scaled by the stamp's own fitted scale.

Registering against the whole design rather than fitting an edge per stamp is
what makes this work at all: the perforation routinely cuts the design away at
a corner, so the rectangle being reported is often not wholly present on the
stamp it is reported for.

The measurand is the centreline of a printed frame rule, not its outer edge.
A rule has width and ink spreads with inking pressure, so its edges move with
the printing and its middle does not. This mirrors the choice of valley bases
over tooth tips in the perforation measurement.

Scales are fitted independently in x and y. Rotation is fitted as well, and
shear is fitted and then discarded; both are diagnostics rather than parts of
the answer, because the perforation geometry already removed the rotation and
nothing shears a printed design.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

SIDES = ('top', 'bottom', 'left', 'right')

# Congealing schedule: (downscale factor, registration rounds at that factor).
# The coarse levels carry the de-centring, which is several tens of pixels at
# 1200 dpi and larger than the frame line is wide.
DEFAULT_SCHEDULE = ((8, 3), (4, 3), (2, 2), (1, 2))

# The perforation rectangle has already removed the gross rotation and scale.
# These limits therefore apply only to one ECC refinement in that rectified
# coordinate system.  They are deliberately much wider than the observed
# printing variation; their purpose is to stop a cancellation from becoming a
# geometrically convincing local optimum.
MAX_RESIDUAL_ROTATION_DEG = 3.0
MAX_FREE_SHEAR = 0.05
MIN_ABSOLUTE_SCALE = 0.80
MAX_ABSOLUTE_SCALE = 1.25
MIN_SCALE_COHORT_FRACTION = 0.04


def valley_quad(stamp):
    """Corners of the perforation valley-base rectangle, in image pixels.

    Returns ``None`` unless all four sides were published, because three lines
    cannot bound the stamp and a reconstructed fourth would import the
    perforation measurement's own error into this one.
    """
    sides = stamp.get('measurement', {}).get('sides', {})
    lines = {}
    for name in SIDES:
        entry = sides.get(name) or {}
        points = entry.get('line_image')
        if not points:
            return None
        (x0, y0), (x1, y1) = points
        origin = np.array([x0, y0], float)
        direction = np.array([x1 - x0, y1 - y0], float)
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return None
        lines[name] = (origin, direction / norm)

    def cross(a, b):
        (p1, d1), (p2, d2) = lines[a], lines[b]
        matrix = np.array([d1, -d2]).T
        if abs(np.linalg.det(matrix)) < 1e-9:
            return None
        t = np.linalg.solve(matrix, p2 - p1)
        return p1 + t[0] * d1

    corners = [cross(*pair) for pair in
               (('top', 'left'), ('top', 'right'),
                ('bottom', 'right'), ('bottom', 'left'))]
    if any(corner is None for corner in corners):
        return None
    return np.array(corners, float)


def quad_angle_deg(quad):
    """Angle of the valley rectangle, from its two horizontal edges."""
    horizontal = (quad[1] - quad[0]) + (quad[2] - quad[3])
    return float(np.degrees(np.arctan2(horizontal[1], horizontal[0])))


def quad_size(quad):
    width = (np.linalg.norm(quad[1] - quad[0]) +
             np.linalg.norm(quad[2] - quad[3])) / 2.0
    height = (np.linalg.norm(quad[3] - quad[0]) +
              np.linalg.norm(quad[2] - quad[1])) / 2.0
    return float(width), float(height)


def stamp_patch(image, quad, canvas, erode_px):
    """Rotate and translate one stamp into the common canvas.

    No scaling is applied, so the patch is still in scan pixels. The mask is
    the valley quadrilateral: every pixel inside it is paper by construction,
    since a valley base is the innermost point of a hole.

    The scan-to-canvas affine is returned alongside, so that a result measured
    on the golden can be carried back to where it belongs on the scan.
    """
    height, width = canvas
    centre = quad.mean(axis=0)
    angle = quad_angle_deg(quad)
    matrix = cv2.getRotationMatrix2D((float(centre[0]), float(centre[1])),
                                     angle, 1.0)
    matrix[0, 2] += width / 2.0 - centre[0]
    matrix[1, 2] += height / 2.0 - centre[1]

    patch = cv2.warpAffine(image, matrix, (width, height),
                           flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    grey = lab[:, :, 0].astype(np.float32)

    corners = cv2.transform(quad.reshape(1, 4, 2).astype(np.float32), matrix)
    mask = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(mask, np.round(corners[0]).astype(np.int32), 255)
    if erode_px > 0:
        kernel = np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8)
        mask = cv2.erode(mask, kernel)

    valid = mask > 0
    if valid.sum() < 1000:
        return None, None, None
    # Per-stamp photometric normalisation: ink density and paper tone vary
    # between stamps, and only the geometry is being compared.
    grey = (grey - grey[valid].mean()) / max(grey[valid].std(), 1e-6)
    grey[~valid] = 0.0
    return grey, mask, matrix


def apply_affine(matrix, points):
    """Map points through a 2x3 affine."""
    return np.asarray(points, float) @ np.asarray(matrix)[:, :2].T \
        + np.asarray(matrix)[:, 2]


def rectangle_on_scan(rectangle, warp, placement):
    """Corners of a golden rectangle, in original scan pixels.

    The golden is in the reference frame, ``warp`` carries the reference to the
    stamp's canvas, and ``placement`` put that canvas there from the scan. Since
    neither step shears and the second is rigid, the drawn quadrilateral's side
    lengths are exactly the reported width and height: the annotation is the
    measurement rather than an illustration of it.
    """
    corners = np.array([
        [rectangle['left_px'], rectangle['top_px']],
        [rectangle['right_px'], rectangle['top_px']],
        [rectangle['right_px'], rectangle['bottom_px']],
        [rectangle['left_px'], rectangle['bottom_px']],
    ], float)
    return apply_affine(cv2.invertAffineTransform(placement),
                        apply_affine(warp, corners))


def pyramid(array, factor, interpolation):
    if factor == 1:
        return array
    height, width = array.shape[:2]
    return cv2.resize(array, (width // factor, height // factor),
                      interpolation=interpolation)


def rescale_warp(warp, from_factor, to_factor):
    """Move a reference-to-patch affine between pyramid levels."""
    scaled = warp.copy()
    scaled[:, 2] *= from_factor / to_factor
    return scaled


def build_golden(patches, masks, warps, weights, size):
    """Masked, weighted average of the stamps mapped into the reference frame.

    ``coverage`` is returned in units of stamps — one fully contributing stamp
    adds one — so that downstream code can demand a quorum.
    """
    height, width = size
    total = np.zeros((height, width), np.float32)
    coverage = np.zeros((height, width), np.float32)
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    for patch, mask, warp, weight in zip(patches, masks, warps, weights):
        aligned = cv2.warpAffine(patch, warp, (width, height), flags=flags,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        share = np.minimum(mask, 1).astype(np.float32) * weight
        valid = cv2.warpAffine(share, warp, (width, height), flags=flags,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        total += aligned * valid
        coverage += valid
    golden = np.where(coverage > 1e-6, total / np.maximum(coverage, 1e-6), 0.0)
    return golden.astype(np.float32), coverage


def align_to_golden(golden, patch, mask, warp, iterations, eps, motion):
    """One ECC refinement of a stamp against the golden image.

    ``cv2.findTransformECC`` maximises the correlation coefficient, so it is
    invariant to the linear brightness and contrast differences between stamps,
    and it returns a continuous sub-pixel transform. A discrete size sweep would
    be both slower and less precise.
    """
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, eps)
    current = warp.astype(np.float32).copy()
    try:
        _, updated = cv2.findTransformECC(
            golden, patch, current, motion, criteria, mask, 5)
    except cv2.error:
        return warp, False
    if not np.all(np.isfinite(updated)):
        return warp, False
    return updated, True


def registration_patch(golden, patch, mask, warp, weight, threshold=0.5):
    """Replace cancellation-like outliers before the next ECC refinement.

    OpenCV's binary ECC mask becomes numerically fragile when it contains many
    small residual holes.  Replacing those pixels with the current prediction
    has the same zero-residual effect while preserving one stable, contiguous
    valley mask for the optimiser.
    """
    height, width = patch.shape
    rendered = cv2.warpAffine(golden, warp, (width, height),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    replace = (mask > 0) & (weight < threshold)
    cleaned = patch.copy()
    cleaned[replace] = rendered[replace]
    return cleaned


def decompose(warp):
    """Split a reference-to-patch affine into scale, rotation, shear, shift.

    QR of the linear part gives ``A = R(theta) @ [[sx, shear*sy], [0, sy]]``,
    so ``sx`` and ``sy`` are the scales along the stamp's own axes.
    """
    linear = warp[:, :2]
    q, r = np.linalg.qr(linear)
    sign = np.sign(np.diag(r))
    sign[sign == 0] = 1.0
    q = q * sign
    r = r * sign[:, None]
    sx, sy = float(r[0, 0]), float(r[1, 1])
    return {
        'scale_x': sx,
        'scale_y': sy,
        'rotation_deg': float(np.degrees(np.arctan2(q[1, 0], q[0, 0]))),
        'shear': float(r[0, 1] / sy) if abs(sy) > 1e-9 else float('nan'),
        'shift_x': float(warp[0, 2]),
        'shift_y': float(warp[1, 2]),
    }


def project_no_shear(warp):
    """Drop the shear term from an affine, keeping scale, rotation and shift.

    Paper shrinkage and plate differences scale the design along its own axes
    and the platen is rigid, so there is no mechanism that shears a design.
    Fitting the sixth degree of freedom anyway only lets registration noise
    leak into the two scales that are being measured.
    """
    parts = decompose(warp)
    angle = np.radians(parts['rotation_deg'])
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    out = warp.copy()
    out[:, :2] = (rotation @ np.diag([parts['scale_x'], parts['scale_y']])
                  ).astype(warp.dtype)
    return out


def _robust_log_scale_limits(parts):
    """Per-axis cohort limits for simultaneous affine ECC proposals."""
    limits = []
    for key in ('scale_x', 'scale_y'):
        values = np.log([entry[key] for entry in parts])
        centre = float(np.median(values))
        mad_sigma = float(1.4826 * np.median(np.abs(values - centre)))
        radius = max(np.log1p(MIN_SCALE_COHORT_FRACTION), 6.0 * mad_sigma)
        limits.append((centre - radius, centre + radius))
    return limits


def plausible_affine_updates(updates, ok):
    """Judge one simultaneous set of affine ECC proposals robustly.

    A stamp is compared with the rest of the collection, not with a presumed
    design size.  Real shared changes therefore remain measurable, while the
    isolated 8--25 percent collapses caused by cancellations are rejected.
    """
    accepted = [False] * len(updates)
    candidate_indices = []
    candidate_parts = []
    for index, (warp, converged) in enumerate(zip(updates, ok)):
        if not converged:
            continue
        parts = decompose(warp)
        if not all(np.isfinite(parts[key]) for key in
                   ('scale_x', 'scale_y', 'rotation_deg', 'shear')):
            continue
        if not (MIN_ABSOLUTE_SCALE <= parts['scale_x'] <= MAX_ABSOLUTE_SCALE and
                MIN_ABSOLUTE_SCALE <= parts['scale_y'] <= MAX_ABSOLUTE_SCALE):
            continue
        if abs(parts['rotation_deg']) > MAX_RESIDUAL_ROTATION_DEG:
            continue
        if abs(parts['shear']) > MAX_FREE_SHEAR:
            continue
        candidate_indices.append(index)
        candidate_parts.append(parts)

    if len(candidate_parts) < 3:
        for index in candidate_indices:
            accepted[index] = True
        return accepted

    x_limits, y_limits = _robust_log_scale_limits(candidate_parts)
    for index, parts in zip(candidate_indices, candidate_parts):
        log_x = np.log(parts['scale_x'])
        log_y = np.log(parts['scale_y'])
        accepted[index] = (x_limits[0] <= log_x <= x_limits[1] and
                           y_limits[0] <= log_y <= y_limits[1])
    return accepted


def normalise_gauge(warps):
    """Fix the free gauge of the reference frame.

    Congealing determines the stamps only up to one transform shared by all of
    them, so scale, rotation and position of the reference frame have to be
    pinned or the whole set drifts. Pinning the geometric mean scale to 1 also
    makes the golden's own rectangle the collection mean, so it can be read
    directly, and pinning mean rotation to 0 keeps the per-stamp angle usable
    as the independent check that it is meant to be.
    """
    parts = [decompose(warp) for warp in warps]
    gx = float(np.exp(np.mean(np.log([abs(p['scale_x']) for p in parts]))))
    gy = float(np.exp(np.mean(np.log([abs(p['scale_y']) for p in parts]))))
    angle = np.radians(np.mean([p['rotation_deg'] for p in parts]))
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    linear = rotation @ np.diag([gx, gy])
    gauge = np.eye(3)
    gauge[:2, :2] = linear
    gauge[0, 2] = float(np.mean([p['shift_x'] for p in parts]))
    gauge[1, 2] = float(np.mean([p['shift_y'] for p in parts]))
    inverse = np.linalg.inv(gauge)
    out = []
    for warp in warps:
        full = np.vstack([warp, [0.0, 0.0, 1.0]])
        out.append((full @ inverse)[:2].astype(np.float32))
    return out


def outlier_weights(patches, masks, warps, golden, size, cutoff):
    """Per-stamp weight maps that drop pixels disagreeing with the golden.

    Cancellation ink, hinge remnants and tears are uncorrelated between stamps,
    so they appear as large local residuals against the average and are removed
    here rather than being modelled.
    """
    height, width = size
    forward = cv2.INTER_LINEAR
    weights = []
    for patch, mask, warp in zip(patches, masks, warps):
        rendered = cv2.warpAffine(golden, warp, (width, height),
                                  flags=forward,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        valid = mask > 0
        residual = np.abs(patch - rendered)
        scale = np.median(residual[valid]) * 1.4826
        if scale < 1e-6:
            weights.append(np.ones((height, width), np.float32))
            continue
        keep = (residual < cutoff * scale).astype(np.float32)
        keep[~valid] = 0.0
        weights.append(keep)
    return weights


def congeal(patches, masks, size, schedule, iterations, eps, cutoff, verbose):
    count = len(patches)
    warps = [np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], np.float32)
             for _ in range(count)]
    weights = [np.ones(size, np.float32) for _ in range(count)]
    failures = [0] * count
    rejections = [0] * count
    affine_accepts = [0] * count
    # The shear the unconstrained fit asked for before it was projected away.
    # It should be near zero; a large value means that stamp did not register.
    free_shear = [0.0] * count
    previous = schedule[0][0]
    golden = None
    coverage = None

    for factor, rounds in schedule:
        level_size = (size[0] // factor, size[1] // factor)
        level_patches = [pyramid(p, factor, cv2.INTER_AREA) for p in patches]
        level_masks = [pyramid(m, factor, cv2.INTER_NEAREST) for m in masks]
        warps = [rescale_warp(w, previous, factor) for w in warps]
        previous = factor

        for step in range(rounds):
            level_weights = [pyramid(w, factor, cv2.INTER_AREA) for w in weights]
            golden, coverage = build_golden(level_patches, level_masks, warps,
                                            level_weights, level_size)
            motion = cv2.MOTION_AFFINE
            updates = []
            converged = []
            for index in range(count):
                updated, ok = align_to_golden(
                    golden, level_patches[index], level_masks[index],
                    warps[index], iterations, eps, motion)
                updates.append(updated)
                converged.append(ok)

            accepted = plausible_affine_updates(updates, converged)
            # A rejected ordinary fit gets one robust retry.  Normal stamps
            # retain the unmodified intensity field and therefore the original
            # sub-pixel scale precision; only a suspect stamp has its previous
            # residual outliers replaced by the current prediction.
            if any(not value for value in accepted):
                retried_updates = list(updates)
                retried_converged = list(converged)
                for index, accept in enumerate(accepted):
                    if accept:
                        continue
                    cleaned = registration_patch(
                        golden, level_patches[index], level_masks[index],
                        warps[index], level_weights[index])
                    retried_updates[index], retried_converged[index] = \
                        align_to_golden(
                            golden, cleaned, level_masks[index], warps[index],
                            iterations, eps, motion)
                updates = retried_updates
                converged = retried_converged
                accepted = plausible_affine_updates(updates, converged)

            for index, (updated, ok, accept) in enumerate(
                    zip(updates, converged, accepted)):
                if ok:
                    free_shear[index] = decompose(updated)['shear']
                if accept:
                    warps[index] = project_no_shear(updated)
                    affine_accepts[index] += 1
                elif not ok:
                    failures[index] += 1
                else:
                    rejections[index] += 1
            warps = normalise_gauge(warps)
            golden, coverage = build_golden(level_patches, level_masks, warps,
                                            level_weights, level_size)
            level_weight_maps = outlier_weights(level_patches, level_masks,
                                                warps, golden, level_size, cutoff)
            weights = [cv2.resize(w, (size[1], size[0]),
                                  interpolation=cv2.INTER_LINEAR)
                       for w in level_weight_maps]
            if verbose:
                spread = np.std([decompose(w)['scale_x'] for w in warps])
                print(f"  level 1/{factor} round {step + 1}: "
                      f"scale_x sd {spread * 100:.3f}%")

    warps = [rescale_warp(w, previous, 1) for w in warps]
    golden, coverage = build_golden(patches, masks, warps, weights, size)
    return (golden, coverage, warps, weights, failures, rejections,
            affine_accepts, free_shear)


def robust_outliers(values, minimum_radius, sigma_multiplier=6.0,
                    logarithmic=False, upper_only=False):
    """Boolean robust outlier mask with a practical minimum tolerance."""
    values = np.asarray(values, float)
    transformed = np.log(values) if logarithmic else values
    centre = float(np.median(transformed))
    mad_sigma = float(1.4826 * np.median(np.abs(transformed - centre)))
    radius = max(minimum_radius, sigma_multiplier * mad_sigma)
    difference = transformed - centre
    return ((difference > radius) if upper_only
            else (np.abs(difference) > radius))


def assess_registrations(results):
    """Attach conservative publication status to registration results.

    Scale outliers and stamps for which no affine refinement was ever accepted
    are unsafe measurements.  Rotation, shear, residual, or a recovered ECC
    failure are review signals only: they may describe a real printing offset
    and must not by themselves erase the measurement.
    """
    if not results:
        return
    scale_x_outlier = robust_outliers(
        [row['scale_x'] for row in results],
        np.log1p(MIN_SCALE_COHORT_FRACTION), logarithmic=True)
    scale_y_outlier = robust_outliers(
        [row['scale_y'] for row in results],
        np.log1p(MIN_SCALE_COHORT_FRACTION), logarithmic=True)
    rotation_outlier = robust_outliers(
        [row['residual_rotation_deg'] for row in results], 2.0)
    shear_outlier = robust_outliers(
        [row['unconstrained_shear'] for row in results], 0.02)
    residual_outlier = robust_outliers(
        [row['residual_rms'] for row in results], 0.15, upper_only=True)

    for index, row in enumerate(results):
        rejected = []
        review = []
        if scale_x_outlier[index]:
            rejected.append('scale_x cohort outlier')
        if scale_y_outlier[index]:
            rejected.append('scale_y cohort outlier')
        if row['accepted_affine_updates'] == 0:
            rejected.append('no accepted affine refinement')
        if row['alignment_failures']:
            review.append(f"{row['alignment_failures']} ECC failures")
        if row['alignment_rejections']:
            review.append(f"{row['alignment_rejections']} implausible ECC updates")
        if rotation_outlier[index]:
            review.append('residual rotation outlier')
        if shear_outlier[index]:
            review.append('free shear outlier')
        if residual_outlier[index]:
            review.append('residual RMS outlier')
        reasons = rejected + review
        row['registration_status'] = (
            'rejected' if rejected else 'review' if review else 'ok')
        row['registration_reasons'] = reasons

        if rejected:
            row['candidate_design_width_px'] = row.pop('design_width_px')
            row['candidate_design_height_px'] = row.pop('design_height_px')
            row['candidate_design_corners_image'] = row.pop(
                'design_corners_image')


def smooth_profile(profile, sigma):
    radius = max(1, int(round(3 * sigma)))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(np.pad(profile, radius, mode='edge'), kernel, 'valid')


def profile_noise(profile, sigma=1.5):
    """Robust scatter of a profile against a lightly smoothed copy of itself.

    This is the yardstick everything else is measured in. Expressing a rule's
    strength as a multiple of the profile's own noise makes the number
    dimensionless and independent of exposure, ink colour and how dark the
    design interior happens to be.
    """
    residual = profile - smooth_profile(profile, sigma)
    return float(1.4826 * np.median(np.abs(residual - np.median(residual))))


def peak_prominence(signal, index):
    """Topographic prominence of a local maximum, and its key saddle.

    From the summit, how far must one descend before being able to climb to
    anything higher? Unlike height above a fixed level, this is unaffected by
    adding a constant, and it judges the peak against its own neighbourhood
    rather than against the full range of the profile.
    """
    peak = signal[index]
    saddles = []
    for step in (-1, 1):
        position = index
        lowest = peak
        while 0 <= position + step < len(signal):
            position += step
            if signal[position] > peak:
                break
            lowest = min(lowest, signal[position])
        saddles.append(lowest)
    saddle = max(saddles)
    return float(peak - saddle), float(saddle)


def _crossing(signal, inside, outside, level):
    """Sub-pixel index where ``signal`` crosses ``level`` between two samples."""
    span = signal[inside] - signal[outside]
    if abs(span) < 1e-12:
        return float(inside)
    return float(inside + (signal[inside] - level) / span * (outside - inside))


def peak_candidates(profile, search_span, min_significance, max_candidates=12):
    """Every plausible rule on one side, described but not yet judged.

    The profile runs outside-in, so index 0 is the edge of the canvas. Each
    candidate carries its sub-pixel centre, its width at half prominence, and
    its prominence in units of the profile's noise. Nothing is accepted or
    rejected here: which candidates are real is decided later, by whether the
    four sides agree, not by any per-side threshold.

    Width is taken at half prominence rather than at a fixed darkness, because
    the width of a line at a fixed level depends on how strongly that line
    printed. Half prominence is relative to the peak's own height, so the same
    rule measures the same width whether it printed heavily or faintly.
    """
    noise = profile_noise(profile)
    if noise < 1e-12:
        return []
    smooth = smooth_profile(profile, 1.0)
    limit = max(3, min(len(smooth) - 1, int(search_span * len(smooth))))
    found = []
    for index in range(1, limit):
        if smooth[index] <= smooth[index - 1] or smooth[index] < smooth[index + 1]:
            continue
        prominence, saddle = peak_prominence(smooth, index)
        if prominence < min_significance * noise:
            continue
        level = saddle + prominence / 2.0
        low = index
        while low > 0 and smooth[low - 1] > level:
            low -= 1
        high = index
        while high < len(smooth) - 1 and smooth[high + 1] > level:
            high += 1
        start = 0.0 if low == 0 else _crossing(smooth, low, low - 1, level)
        end = (float(high) if high == len(smooth) - 1
               else _crossing(smooth, high, high + 1, level))
        # Centre from the unsmoothed profile, so smoothing sets the support
        # but does not move the answer.
        weight = np.clip(profile[low:high + 1] - level, 0.0, None)
        if weight.sum() <= 0:
            continue
        found.append({
            'centre_px': float(np.sum(weight * np.arange(low, high + 1))
                               / weight.sum()),
            'width_px': float(end - start),
            'prominence': prominence,
            'significance': prominence / noise,
        })
    found.sort(key=lambda item: -item['significance'])
    found = found[:max_candidates]
    found.sort(key=lambda item: item['centre_px'])
    return found


def masked_profile(darkness, coverage, axis, span, min_stamps,
                   min_line_fraction=0.5):
    """Median darkness along ``axis``, over the central ``span`` of the other.

    Ornaments and value text interrupt the frame on any single row, so a median
    is taken rather than a mean.

    ``min_stamps`` is the decisive guard. Near the canvas margin only one or two
    stamps reach the pixel, so whatever they happen to carry there — a tooth,
    backing paper, a torn edge — survives averaging intact and looks exactly
    like a rule. Requiring agreement from a quorum of stamps is what makes the
    average, rather than any individual stamp, the thing being measured.

    The validity flags are returned alongside, because filling the rejected
    lines with a constant puts a step into the profile that is an artefact of
    this function rather than anything on the stamps. Callers that look for
    peaks must search inside the valid span, not fill and then threshold.
    """
    size = darkness.shape[1 - axis]
    margin = int((1.0 - span) / 2.0 * size)
    if axis == 0:
        window = (slice(margin, size - margin), slice(None))
    else:
        window = (slice(None), slice(margin, size - margin))
    values = np.ma.masked_array(darkness[window],
                                mask=coverage[window] < min_stamps)
    profile = np.ma.median(values, axis=axis)
    enough = values.count(axis=axis) >= min_line_fraction * values.shape[axis]
    filled = np.ma.filled(profile, np.nan)
    valid = enough & np.isfinite(filled)
    usable = filled[valid]
    return np.where(valid, filled, usable.min() if usable.size else 0.0), valid


def _index_tuples(depth, count):
    """All 4-tuples of candidate offsets summing to ``depth``."""
    for a in range(min(depth, count - 1) + 1):
        for b in range(min(depth - a, count - 1) + 1):
            for c in range(min(depth - a - b, count - 1) + 1):
                d = depth - a - b - c
                if d < count:
                    yield (a, b, c, d)


def choose_consistent_rule(candidates, start, width_tolerance, max_depth=10):
    """The outermost choice of one candidate per side on which all four agree.

    This is where the decision is made, and it is deliberately not made per
    side. The four sides are not four independent detections; they are one
    printed rectangle observed four times. So rather than asking each side
    whether its peak clears some bar, the sides are asked whether they agree
    on the *shape* of the feature they found — its width at half prominence.

    Picking the thin outer rule on two sides and the thick frame band on the
    other two, which is exactly how 3K and 70K failed, is precisely what this
    test rejects: those widths differ threefold. Searching outermost-first then
    makes "the outer rule" mean what it says.
    """
    sides = ('left', 'right', 'top', 'bottom')
    available = [candidates[side][start[side]:] for side in sides]
    if any(not entries for entries in available):
        return None
    count = max(len(entries) for entries in available)
    for depth in range(max_depth + 1):
        for offsets in _index_tuples(depth, count):
            if any(offset >= len(entries)
                   for offset, entries in zip(offsets, available)):
                continue
            chosen = {side: entries[offset] for side, offset, entries
                      in zip(sides, offsets, available)}
            widths = [entry['width_px'] for entry in chosen.values()]
            if min(widths) <= 0:
                continue
            if max(widths) / min(widths) > width_tolerance:
                continue
            return {side: (start[side] + offset, chosen[side])
                    for side, offset in zip(sides, offsets)}
    return None


def frame_rectangle(golden, coverage, span, min_stamps, search_span,
                    min_significance, width_tolerance, band_count=2,
                    max_candidates=12):
    """Measure the design's frame rules on the golden image.

    The returned ``rules`` list runs from the outermost rule inwards; the first
    entry is used as the published rectangle.
    """
    height, width = golden.shape
    darkness = -golden.astype(np.float32)
    columns, column_valid = masked_profile(darkness, coverage, 0, span,
                                           min_stamps)
    rows, row_valid = masked_profile(darkness, coverage, 1, span, min_stamps)
    # Each side reads outside-in, so a candidate's position means the same
    # thing on all four and the agreement test compares like with like.
    profiles = {'left': (columns, column_valid),
                'right': (columns[::-1], column_valid[::-1]),
                'top': (rows, row_valid),
                'bottom': (rows[::-1], row_valid[::-1])}
    origin = {'left': 0.0, 'right': width - 1.0,
              'top': 0.0, 'bottom': height - 1.0}
    direction = {'left': 1.0, 'right': -1.0, 'top': 1.0, 'bottom': -1.0}

    candidates = {}
    for side, (profile, valid) in profiles.items():
        present = np.flatnonzero(valid)
        if present.size < 8:
            candidates[side] = []
            continue
        first, last = int(present[0]), int(present[-1])
        found = peak_candidates(profile[first:last + 1], search_span,
                                min_significance, max_candidates)
        for entry in found:
            entry['centre_px'] += first
        candidates[side] = found

    rules = []
    start = {side: 0 for side in profiles}
    for number in range(1, band_count + 1):
        choice = choose_consistent_rule(candidates, start, width_tolerance)
        if choice is None:
            break
        rule = {'rule': number}
        for side, (index, entry) in choice.items():
            rule[f'{side}_px'] = origin[side] + direction[side] * entry['centre_px']
            start[side] = index + 1
        rule['width_px'] = float(rule['right_px'] - rule['left_px'])
        rule['height_px'] = float(rule['bottom_px'] - rule['top_px'])
        rule['band_px'] = {side: entry['width_px']
                           for side, (_, entry) in choice.items()}
        rule['significance'] = {side: entry['significance']
                                for side, (_, entry) in choice.items()}
        rules.append(rule)
    return rules, candidates


RULE_COLORS = ((0, 0, 255), (0, 200, 255), (0, 255, 0))


def golden_preview(golden, coverage, rules, selected, min_stamps,
                   max_width=1600):
    """The golden image with every detected rule drawn; the published one red.

    Pixels below the stamp quorum are dimmed, so the region the measurement
    refused to read is visible rather than implied.
    """
    image = np.zeros(golden.shape, np.uint8)
    valid = coverage >= min_stamps
    if valid.any():
        values = golden[valid]
        low, high = np.percentile(values, 0.5), np.percentile(values, 99.5)
        scaled = np.clip((golden - low) / max(high - low, 1e-6), 0, 1) * 255
        image = scaled.astype(np.uint8)
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    canvas[~valid] = (canvas[~valid] * 0.35 + 150 * 0.65).astype(np.uint8)
    for index, rule in enumerate(rules):
        color = RULE_COLORS[0] if index == selected else \
            RULE_COLORS[min(index + 1, len(RULE_COLORS) - 1)]
        cv2.rectangle(canvas,
                      (int(round(rule['left_px'])), int(round(rule['top_px']))),
                      (int(round(rule['right_px'])), int(round(rule['bottom_px']))),
                      color, 2)
    if canvas.shape[1] > max_width:
        factor = max_width / canvas.shape[1]
        canvas = cv2.resize(canvas, None, fx=factor, fy=factor,
                            interpolation=cv2.INTER_AREA)
    return canvas


DESIGN_COLOR = (0, 0, 255)
CORNER_COLOR = (255, 255, 0)
REVIEW_COLOR = (0, 165, 255)
SKIPPED_COLOR = (180, 180, 180)


def draw_design(image, results, skipped, unit):
    """Annotate the scan with each stamp's design rectangle and size.

    Line and font sizes follow ``segment_stamps.py`` so that this image and
    ``*_perf.jpg`` can be read side by side at the same zoom.
    """
    image = image.copy()
    height, width = image.shape[:2]
    thickness = max(2, round(max(height, width) / 2500))
    font_scale = max(0.7, max(height, width) / 5000)

    def put_label(text, x0, x1, y0, color):
        span = max(1, x1 - x0)
        text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                     font_scale, thickness)[0][0]
        scale = font_scale * min(1.0, span / max(1, text_width))
        cv2.putText(image, text, (int(x0), int(max(thickness * 4 + 5,
                                                   y0 - thickness * 3))),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness,
                    cv2.LINE_AA)

    for row in results:
        box = row.get('bbox')
        status = row.get('registration_status', 'ok')
        if status == 'rejected':
            if box:
                cv2.rectangle(image, (box['x0'], box['y0']),
                              (box['x1'], box['y1']), SKIPPED_COLOR,
                              thickness, cv2.LINE_AA)
                put_label(f"{row['stamp']}: registration rejected",
                          box['x0'], box['x1'], box['y0'], SKIPPED_COLOR)
            continue

        corners = np.rint(row['design_corners_image']).astype(np.int32)
        color = REVIEW_COLOR if status == 'review' else DESIGN_COLOR
        cv2.polylines(image, [corners], True, color, thickness,
                      cv2.LINE_AA)
        for point in corners:
            cv2.circle(image, tuple(int(value) for value in point),
                       max(4, thickness * 3), CORNER_COLOR, -1, cv2.LINE_AA)
        # Label above the detection box, where ``*_perf.jpg`` puts it, so
        # the two images can be compared without hunting for the caption.
        anchor = ((box['x0'], box['x1'], box['y0']) if box else
                  (corners[:, 0].min(), corners[:, 0].max(),
                    corners[:, 1].min()))
        suffix = ' [review]' if status == 'review' else ''
        put_label(f"{row['stamp']}: {row[f'design_width_{unit}']:.2f} x "
                  f"{row[f'design_height_{unit}']:.2f} {unit}{suffix}",
                  *anchor, color)

    # A stamp the perforation stage could not close has no design rectangle.
    # Labelling it keeps the omission visible instead of silent.
    for number, box in skipped:
        cv2.rectangle(image, (box['x0'], box['y0']), (box['x1'], box['y1']),
                      SKIPPED_COLOR, thickness, cv2.LINE_AA)
        put_label(f'{number}: not measured', box['x0'], box['x1'], box['y0'],
                  SKIPPED_COLOR)
    return image


def photometric_match(predicted, actual, valid, weight):
    """Carry the golden onto one stamp's own ink with a gain and an offset.

    The per-stamp normalisation in ``stamp_patch`` equalises mean and spread
    but not ink density, so the design does not cancel without this. The fit
    uses inliers only, which is what keeps the cancellation out of it: a heavy
    postmark would otherwise drag the gain and leave the design behind.
    """
    use = valid & (weight > 0.5)
    if use.sum() < 100:
        use = valid
    gain, offset = np.polyfit(predicted[use].astype(np.float64),
                              actual[use].astype(np.float64), 1)
    return (gain * predicted + offset).astype(np.float32)


def deghost(residual, design, valid, window):
    """Remove the part of a residual that a local misregistration explains.

    A shift of d turns the design into design + d.grad(design), so the design's
    own edges dominate the residual wherever registration is imperfect by a
    fraction of a pixel. Fitting that two-parameter model over a sliding window
    and subtracting it leaves only what no shift could account for. A postmark
    is not a shifted copy of the design, so it survives.
    """
    keep = valid.astype(np.float32)
    gx = cv2.Sobel(design, cv2.CV_32F, 1, 0, ksize=3) * keep
    gy = cv2.Sobel(design, cv2.CV_32F, 0, 1, ksize=3) * keep
    masked = residual * keep

    def box(array):
        return cv2.boxFilter(array, cv2.CV_32F, (window, window))

    xx, yy, xy = box(gx * gx), box(gy * gy), box(gx * gy)
    xr, yr = box(gx * masked), box(gy * masked)
    determinant = xx * yy - xy * xy
    safe = np.abs(determinant) > 1e-8
    divisor = np.where(safe, determinant, 1.0)
    along_x = np.where(safe, (yy * xr - xy * yr) / divisor, 0.0)
    along_y = np.where(safe, (xx * yr - xy * xr) / divisor, 0.0)
    return residual - (along_x * gx + along_y * gy)


def reveal_stamp(patch, predicted, mask, weight, fade, window):
    """Darkness on one stamp that the shared design does not account for.

    In the patch's own photometric units. Where the design prints solid the
    postmark would be ink on ink, so there is nothing to recover and the result
    is blank: absence of residual is not evidence of absence of a cancel.
    """
    valid = mask > 0
    design = photometric_match(predicted, patch, valid, weight)
    paper = np.percentile(patch[valid], 90.0)
    design_paper = np.percentile(design[valid], 90.0)
    revealed = (paper - patch) - fade * (design_paper - design)
    revealed = deghost(revealed, design, valid, window)
    revealed[~valid] = 0.0
    return revealed


def draw_reveal(image, patches, masks, placements, warps, weights, golden,
                canvas, fade, window):
    """The scan with the shared design removed from every stamp.

    The geometry is the original scan's, so this can be laid beside
    ``_perf.jpg``. Outside the measured stamps the scan is left as a pale
    ghost, which keeps the layout and the perforation visible without
    competing with what has been revealed.
    """
    height, width = image.shape[:2]
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    canvas_shape = (canvas[1], canvas[0])
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP

    revealed = np.zeros((height, width), np.float32)
    painted = np.zeros((height, width), np.float32)
    for patch, mask, placement, warp, weight in zip(
            patches, masks, placements, warps, weights):
        predicted = cv2.warpAffine(golden, warp, canvas_shape,
                                   flags=cv2.INTER_LINEAR)
        extra = reveal_stamp(patch, predicted, mask, weight, fade, window)
        for source, target in ((extra, revealed),
                               (np.minimum(mask, 1).astype(np.float32),
                                painted)):
            target += cv2.warpAffine(source, placement, (width, height),
                                     flags=flags,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0)

    inside = painted > 0.5
    if inside.any():
        # One scale for the whole scan, so that stamps stay comparable.
        span = np.percentile(revealed[inside], 99.5)
        strength = np.clip(revealed / max(span, 1e-6), 0.0, 1.0)
    else:
        strength = np.zeros_like(revealed)

    paper = np.clip(1.0 - 0.18 * (1.0 - grey), 0.0, 1.0)
    out = np.where(inside, 1.0 - strength, paper)
    return np.clip(cv2.cvtColor(out, cv2.COLOR_GRAY2BGR) * 255.0,
                   0, 255).astype(np.uint8)


CANCEL_SEED, CANCEL_EDGE, CANCEL_SMOOTH = 4.0, 1.5, 3.0
CANCEL_CORE_FRACTION = 5e-5


def cancel_strength(revealed, valid):
    """How much of each pixel the cancellation, rather than the stamp, owns.

    The residual is never zero — fine design detail does not survive averaging
    and comes back as texture — so the stamp's own residual noise is the scale.
    A single threshold on it cannot work, because a stroke's soft edge is as
    faint as the texture and only its shape tells them apart. Two things are
    true of a postmark and of nothing else here: it is connected, and it is
    large. So each mark is followed out from a core of four robust sigma down
    to one and a half, and a mark is only believed if its core covers enough
    of the stamp to have been meant. Speckle, however deep a single excursion
    happens to be, is neither.
    """
    sample = revealed[valid]
    noise = 1.4826 * np.median(np.abs(sample - np.median(sample)))
    core = (revealed > CANCEL_SEED * noise) & valid
    edge = (revealed > CANCEL_EDGE * noise) & valid
    count, labels = cv2.connectedComponents(edge.astype(np.uint8),
                                            connectivity=8)

    enough = max(16, int(CANCEL_CORE_FRACTION * valid.sum()))
    core_size = np.bincount(labels[core], minlength=count)
    seeded = core_size >= enough
    seeded[0] = False

    span = max((CANCEL_SEED - CANCEL_EDGE) * noise, 1e-6)
    alpha = np.clip((revealed - CANCEL_EDGE * noise) / span, 0.0, 1.0)
    alpha = np.where(seeded[labels], alpha, 0.0).astype(np.float32)
    alpha = cv2.GaussianBlur(alpha, (0, 0), CANCEL_SMOOTH)
    alpha[~valid] = 0.0
    return alpha


def restore_stamp(colour, design, mask, weight, alpha):
    """One stamp with the cancellation replaced by the shared design.

    Returned as the change to apply, not as a new patch, so that everything the
    cancellation does not touch keeps the stamp's own pixels untouched rather
    than being resampled.

    Lightness comes from the golden, carried into this stamp's own exposure.
    Colour cannot: under a black mark the stamp's own hue is destroyed too. It
    is rebuilt from the restored lightness instead, on the relation between the
    two that holds over the uncancelled part of this stamp — a one-colour print
    on tinted paper has only the two hues, and lightness says which.
    """
    valid = mask > 0
    lab = cv2.cvtColor(colour, cv2.COLOR_BGR2LAB).astype(np.float32)
    clear = valid & (alpha < 0.1)
    if clear.sum() < 100:
        clear = valid

    blended = lab.copy()
    lightness = photometric_match(design, lab[:, :, 0], clear, weight)
    blended[:, :, 0] = lightness
    for channel in (1, 2):
        blended[:, :, channel] = photometric_match(
            lightness, lab[:, :, channel], clear, weight)
    mix = (alpha * (mask > 0))[..., None]
    merged = np.clip(lab * (1.0 - mix) + blended * mix, 0, 255)
    restored = cv2.cvtColor(merged.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return restored.astype(np.float32) - colour.astype(np.float32)


def scan_window(mask, placement, shape):
    """Where on the scan one canvas patch lands, and the affine that puts it
    there, so that only that part of a large scan has to be touched."""
    x, y, w, h = cv2.boundingRect(mask)
    corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], float)
    on_scan = apply_affine(cv2.invertAffineTransform(placement), corners)
    x0 = max(int(np.floor(on_scan[:, 0].min())) - 2, 0)
    y0 = max(int(np.floor(on_scan[:, 1].min())) - 2, 0)
    x1 = min(int(np.ceil(on_scan[:, 0].max())) + 2, shape[1])
    y1 = min(int(np.ceil(on_scan[:, 1].max())) + 2, shape[0])
    shifted = placement.copy()
    shifted[:, 2] += shifted[:, :2] @ np.array([x0, y0], float)
    return (slice(y0, y1), slice(x0, x1)), shifted, (x1 - x0, y1 - y0)


def draw_uncancelled(image, patches, masks, placements, warps, weights,
                     golden, canvas, fade, window):
    """The scan with the cancellations taken off and the design put back.

    The complement of ``draw_reveal`` from the same decomposition: that one
    keeps what the stamps do not share, this one restores it to what they do.
    Only the marked pixels change, so away from a postmark the scan is the
    original, down to the bit.
    """
    out = image.copy()
    canvas_shape = (canvas[1], canvas[0])
    for patch, mask, placement, warp, weight in zip(
            patches, masks, placements, warps, weights):
        design = cv2.warpAffine(golden, warp, canvas_shape,
                                flags=cv2.INTER_LINEAR)
        revealed = reveal_stamp(patch, design, mask, weight, 1.0, window)
        alpha = cancel_strength(revealed, mask > 0) * fade
        colour = cv2.warpAffine(image, placement, canvas_shape,
                                flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        change = restore_stamp(colour, design, mask, weight, alpha)

        where, shifted, size = scan_window(mask, placement, image.shape[:2])
        back = cv2.warpAffine(change, shifted, size,
                              flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        out[where] = np.clip(out[where].astype(np.float32) + back,
                             0, 255).astype(np.uint8)
    return out


def measure_scan(image, document, schedule, iterations, eps, cutoff,
                 erode_px, pad, search_span, min_significance,
                 width_tolerance, span, rule_index, min_coverage, verbose):
    usable = []
    for stamp in document['stamps']:
        quad = valley_quad(stamp)
        if quad is not None:
            usable.append((stamp, quad))
    if len(usable) < 3:
        raise RuntimeError(
            f'need at least 3 stamps with four published sides, found {len(usable)}')

    sizes = np.array([quad_size(quad) for _, quad in usable])
    canvas = (int(sizes[:, 1].max()) + 2 * pad, int(sizes[:, 0].max()) + 2 * pad)
    factor = max(f for f, _ in schedule)
    canvas = (canvas[0] - canvas[0] % factor, canvas[1] - canvas[1] % factor)

    patches, masks, placements, kept = [], [], [], []
    for stamp, quad in usable:
        patch, mask, placement = stamp_patch(image, quad, canvas, erode_px)
        if patch is None:
            continue
        patches.append(patch)
        masks.append(mask)
        placements.append(placement)
        kept.append((stamp, quad))
    measured = {stamp['stamp'] for stamp, _ in kept}
    skipped = [(stamp['stamp'], stamp['bbox'])
               for stamp in document['stamps']
               if stamp['stamp'] not in measured and 'bbox' in stamp]
    if verbose:
        print(f"Congealing {len(patches)} stamps on a "
              f"{canvas[1]}x{canvas[0]} canvas")

    golden, coverage, warps, weights, failures, rejections, affine_accepts, \
        free_shear = congeal(
            patches, masks, canvas, schedule, iterations, eps, cutoff, verbose)
    min_stamps = max(2.0, min_coverage * len(patches))
    rules, candidates = frame_rectangle(
        golden, coverage, span, min_stamps, search_span, min_significance,
        width_tolerance, band_count=max(2, rule_index + 1))
    if len(rules) <= rule_index:
        raise RuntimeError(
            f'golden image has {len(rules)} frame rules all four sides agree '
            f'on, so rule {rule_index + 1} cannot be measured')
    rectangle = rules[rule_index]

    results = []
    for index, (stamp, quad) in enumerate(kept):
        parts = decompose(warps[index])
        width_px = rectangle['width_px'] * parts['scale_x']
        height_px = rectangle['height_px'] * parts['scale_y']
        valid = masks[index] > 0
        rendered = cv2.warpAffine(golden, warps[index],
                                  (canvas[1], canvas[0]),
                                  flags=cv2.INTER_LINEAR)
        residual = float(np.sqrt(np.mean(
            (patches[index][valid] - rendered[valid]) ** 2)))
        results.append({
            'stamp': stamp['stamp'],
            'bbox': stamp.get('bbox'),
            'design_width_px': float(width_px),
            'design_height_px': float(height_px),
            'scale_x': parts['scale_x'],
            'scale_y': parts['scale_y'],
            'residual_rotation_deg': parts['rotation_deg'],
            'unconstrained_shear': free_shear[index],
            'offset_x_px': parts['shift_x'],
            'offset_y_px': parts['shift_y'],
            'residual_rms': residual,
            'kept_fraction': float(weights[index][valid].mean()),
            'alignment_failures': failures[index],
            'alignment_rejections': rejections[index],
            'accepted_affine_updates': affine_accepts[index],
            'design_corners_image': rectangle_on_scan(
                rectangle, warps[index], placements[index]).tolist(),
        })
    assess_registrations(results)
    registration = {'patches': patches, 'masks': masks,
                    'placements': placements, 'warps': warps,
                    'weights': weights}
    return golden, coverage, rules, rule_index, results, canvas, min_stamps, \
        skipped, candidates, registration


def main():
    parser = argparse.ArgumentParser(
        description=('Measure the printed design rectangle of every stamp in a '
                     'scan, by congealing the stamps into one golden image and '
                     'fitting each stamp back to it.'))
    parser.add_argument('input', help='Input scan, as given to segment_stamps.py')
    parser.add_argument(
        '--perf', help='Perforation JSON (default: data_<input stem>_perf.json)')
    parser.add_argument(
        '--dpi', type=float,
        help='Scan DPI for mm output (default: the value in the perforation JSON)')
    parser.add_argument(
        '--iterations', type=int, default=120,
        help='Maximum ECC iterations per stamp per round (default: 120)')
    parser.add_argument(
        '--eps', type=float, default=1e-6,
        help='ECC convergence threshold (default: 1e-6)')
    parser.add_argument(
        '--outlier-cutoff', type=float, default=3.0,
        help=('Residual cutoff in robust sigma above which a pixel stops '
              'contributing, which is how cancellations are removed '
              '(default: 3.0)'))
    parser.add_argument(
        '--erode', type=int, default=3,
        help='Erosion of the valley-rectangle mask, in pixels (default: 3)')
    parser.add_argument(
        '--pad', type=int, default=64,
        help='Canvas margin around the largest valley rectangle (default: 64)')
    parser.add_argument(
        '--rule-search-span', type=float, default=0.35,
        help=('Fraction of each profile, measured inwards from the edge, that '
              'is searched for frame rules (default: 0.35)'))
    parser.add_argument(
        '--rule-significance', type=float, default=6.0,
        help=('Smallest peak prominence, in multiples of the profile noise, '
              'that is enumerated as a candidate rule. This limits the '
              'candidate list; which candidate wins is decided by four-side '
              'agreement (default: 6.0)'))
    parser.add_argument(
        '--rule-width-tolerance', type=float, default=1.5,
        help=('Largest ratio between the widest and narrowest of the four '
              'sides of one rule, at half prominence, for the four to count '
              'as the same printed feature (default: 1.5)'))
    parser.add_argument(
        '--profile-span', type=float, default=0.5,
        help='Central fraction of the opposite axis used per profile (default: 0.5)')
    parser.add_argument(
        '--reveal-fade', type=float, default=1.0,
        help=('How much of the design to remove, from 0 for none to 1 for all '
              '(default: 1.0)'))
    parser.add_argument(
        '--uncancel-fade', type=float, default=1.0,
        help=('How much of the cancellation to take back off the stamps, from '
              '0 for none to 1 for all (default: 1.0)'))
    parser.add_argument(
        '--reveal-window', type=int, default=129,
        help=('Side of the window, in pixels, over which the local '
              'misregistration that would otherwise leave the design edges '
              'ghosting is fitted and removed (default: 129)'))
    parser.add_argument(
        '--min-coverage', type=float, default=0.3,
        help=('Fraction of the stamps that must reach a pixel before the '
              'golden is read there, which keeps the thinly covered canvas '
              'margin out of the rule search (default: 0.3)'))
    parser.add_argument(
        '--rule', type=int, default=1,
        help=('Which frame rule to publish, counted inwards from the outside. '
              'All detected rules are recorded either way (default: 1)'))
    parser.add_argument('--quiet', action='store_true', help='Suppress progress')
    args = parser.parse_args()

    if args.dpi is not None and (not np.isfinite(args.dpi) or args.dpi <= 0):
        parser.error('--dpi must be finite and positive')
    if not 0.0 < args.profile_span <= 1.0:
        parser.error('--profile-span must be in (0, 1]')
    if args.rule < 1:
        parser.error('--rule must be a positive rule number')
    if not 0.0 < args.min_coverage <= 1.0:
        parser.error('--min-coverage must be in (0, 1]')
    if not 0.0 <= args.reveal_fade <= 1.0:
        parser.error('--reveal-fade must be in [0, 1]')
    if not 0.0 <= args.uncancel_fade <= 1.0:
        parser.error('--uncancel-fade must be in [0, 1]')
    if args.reveal_window < 3 or args.reveal_window % 2 == 0:
        parser.error('--reveal-window must be an odd number of pixels, >= 3')

    input_path = Path(args.input)
    perf_path = Path(args.perf) if args.perf else \
        input_path.with_name('data_' + input_path.stem + '_perf.json')
    document = json.loads(perf_path.read_text(encoding='utf-8'))
    dpi = args.dpi if args.dpi is not None else document.get('dpi')

    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f'Cannot read image: {input_path}')
    verbose = not args.quiet
    if verbose:
        print(f'Image: {input_path}')

    golden, coverage, rules, rule_index, results, canvas, min_stamps, \
        skipped, candidates, registration = measure_scan(
            image, document, DEFAULT_SCHEDULE, args.iterations, args.eps,
            args.outlier_cutoff, args.erode, args.pad, args.rule_search_span,
            args.rule_significance, args.rule_width_tolerance,
            args.profile_span, args.rule - 1, args.min_coverage, verbose)
    rectangle = rules[rule_index]

    unit = 'mm' if dpi else 'px'
    if dpi:
        factor = 25.4 / dpi
        for row in results:
            if 'design_width_px' in row:
                row['design_width_mm'] = row['design_width_px'] * factor
                row['design_height_mm'] = row['design_height_px'] * factor
            else:
                row['candidate_design_width_mm'] = (
                    row['candidate_design_width_px'] * factor)
                row['candidate_design_height_mm'] = (
                    row['candidate_design_height_px'] * factor)

    golden_path = input_path.with_name(input_path.stem + '_golden.png')
    cv2.imwrite(str(golden_path),
                golden_preview(golden, coverage, rules, rule_index, min_stamps))

    drawn_path = input_path.with_name(input_path.stem + '_design_size.jpg')
    if not cv2.imwrite(str(drawn_path),
                       draw_design(image, results, skipped, unit)):
        raise RuntimeError(f'Cannot write image: {drawn_path}')

    parts = (registration['patches'], registration['masks'],
             registration['placements'], registration['warps'],
             registration['weights'], golden, canvas)
    reveal_path = input_path.with_name(input_path.stem + '_design_cancel.jpg')
    if not cv2.imwrite(str(reveal_path), draw_reveal(
            image, *parts, args.reveal_fade, args.reveal_window)):
        raise RuntimeError(f'Cannot write image: {reveal_path}')

    clean_path = input_path.with_name(input_path.stem + '_design_clean.jpg')
    if not cv2.imwrite(str(clean_path), draw_uncancelled(
            image, *parts, args.uncancel_fade, args.reveal_window)):
        raise RuntimeError(f'Cannot write image: {clean_path}')

    payload = {
        'format': 'stamp-design-size',
        'version': 1,
        'source': input_path.name,
        'source_path': str(input_path.resolve()),
        'perforation_source': str(perf_path.resolve()),
        'dpi': dpi,
        'measurand': (f'centreline of design frame rule {args.rule}, '
                      'counted inwards from the outside'),
        'golden': {
            'canvas_height_px': canvas[0],
            'canvas_width_px': canvas[1],
            'published_rule': args.rule,
            'min_stamps_per_pixel': min_stamps,
            'rectangle': rectangle,
            'rules': rules,
            'candidates': candidates,
            'preview': golden_path.name,
        },
        'annotated_scan': drawn_path.name,
        'cancellation_scan': reveal_path.name,
        'uncancelled_scan': clean_path.name,
        'parameters': {
            'schedule': [list(level) for level in DEFAULT_SCHEDULE],
            'iterations': args.iterations,
            'eps': args.eps,
            'outlier_cutoff': args.outlier_cutoff,
            'erode': args.erode,
            'pad': args.pad,
            'rule_search_span': args.rule_search_span,
            'rule_significance': args.rule_significance,
            'rule_width_tolerance': args.rule_width_tolerance,
            'profile_span': args.profile_span,
            'min_coverage': args.min_coverage,
            'rule': args.rule,
            'reveal_fade': args.reveal_fade,
            'uncancel_fade': args.uncancel_fade,
            'reveal_window': args.reveal_window,
        },
        'stamps': results,
    }
    out_path = input_path.with_name('data_' + input_path.stem + '_design.json')
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding='utf-8')

    if verbose:
        print('\nCandidate rules per side, by prominence over profile noise:')
        for side in ('left', 'right', 'top', 'bottom'):
            summary = ', '.join(
                f"{entry['centre_px']:.1f}px w{entry['width_px']:.1f} "
                f"({entry['significance']:.0f}s)"
                for entry in candidates[side][:4])
            print(f'  {side:<6} {summary}')
        print('\nGolden frame rules, outermost first:')
        for rule in rules:
            mark = ' <- published' if rule['rule'] == args.rule else ''
            widths = tuple(round(value, 1) for value in rule['band_px'].values())
            print(f"  rule {rule['rule']}: {rule['width_px']:8.2f} x "
                  f"{rule['height_px']:8.2f} px, half-prominence widths "
                  f"{widths} px, weakest "
                  f"{min(rule['significance'].values()):.0f}s{mark}")
        print(f"{'stamp':>5} {'width':>9} {'height':>9} {'sx':>8} {'sy':>8} "
              f"{'rot':>7} {'shear':>8} {'kept':>6} {'rms':>6} "
              f"{'status':>8}   ({unit})")
        for row in results:
            width = row.get(f'design_width_{unit}')
            height = row.get(f'design_height_{unit}')
            if width is None:
                width_text = height_text = '       --'
            else:
                width_text = f'{width:9.4f}'
                height_text = f'{height:9.4f}'
            print(f"{row['stamp']:5d} {width_text} {height_text} "
                  f"{row['scale_x']:8.5f} {row['scale_y']:8.5f} "
                  f"{row['residual_rotation_deg']:7.3f} "
                  f"{row['unconstrained_shear']:8.5f} "
                  f"{row['kept_fraction']:6.3f} {row['residual_rms']:6.3f} "
                  f"{row['registration_status']:>8}")
        published = [row for row in results
                     if f'design_width_{unit}' in row]
        widths = np.array([row[f'design_width_{unit}'] for row in published])
        heights = np.array([row[f'design_height_{unit}'] for row in published])
        if len(published):
            ddof = 1 if len(published) > 1 else 0
            print(f"\nwidth  mean {widths.mean():.4f} sd "
                  f"{widths.std(ddof=ddof):.4f} range "
                  f"{widths.min():.4f}..{widths.max():.4f} {unit}")
            print(f"height mean {heights.mean():.4f} sd "
                  f"{heights.std(ddof=ddof):.4f} range "
                  f"{heights.min():.4f}..{heights.max():.4f} {unit}")
        rejected = [row for row in results
                    if row['registration_status'] == 'rejected']
        if rejected:
            print('\nRejected registrations:')
            for row in rejected:
                print(f"  {row['stamp']}: "
                      + '; '.join(row['registration_reasons']))
        if skipped:
            print('\nNot measured, no four published perforation sides: '
                  + ', '.join(str(number) for number, _ in skipped))
        print(f"\nWrote {out_path}\nWrote {golden_path}\nWrote {drawn_path}\n"
              f"Wrote {reveal_path}\nWrote {clean_path}")


if __name__ == '__main__':
    main()
