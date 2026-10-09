"""Measure the printed design rectangle of every stamp in one scan.

The input is a scan plus the ``*_perf.json`` written by ``segment_stamps.py``.
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


def align_to_golden(golden, patch, mask, warp, iterations, eps):
    """One ECC affine refinement of a stamp against the golden image.

    ``cv2.findTransformECC`` maximises the correlation coefficient, so it is
    invariant to the linear brightness and contrast differences between stamps,
    and it returns a continuous sub-pixel transform. A discrete size sweep would
    be both slower and less precise.
    """
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, eps)
    current = warp.astype(np.float32).copy()
    try:
        _, updated = cv2.findTransformECC(
            golden, patch, current, cv2.MOTION_AFFINE, criteria, mask, 5)
    except cv2.error:
        return warp, False
    if not np.all(np.isfinite(updated)):
        return warp, False
    return updated, True


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
            for index in range(count):
                updated, ok = align_to_golden(
                    golden, level_patches[index], level_masks[index],
                    warps[index], iterations, eps)
                free_shear[index] = decompose(updated)['shear']
                warps[index] = project_no_shear(updated)
                if not ok:
                    failures[index] += 1
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
    return golden, coverage, warps, weights, failures, free_shear


def edge_bands(profile, min_run, level, count):
    """The outermost ``count`` sustained dark runs at each end of a profile.

    The Arms design carries more than one rule per side — a thin outer rule and
    a thick main frame band — and either is a defensible definition of "the
    design rectangle". Both are returned so the choice is recorded rather than
    made silently, and so the separation between them can be checked.

    Each run is reported as its sub-pixel darkness centroid and its width. A
    centroid is used rather than an edge because a rule's width grows with
    inking pressure, roughly symmetrically, which moves its edges but not its
    middle.
    """
    low, high = np.percentile(profile, 5), np.percentile(profile, 98)
    if high - low < 1e-6:
        return [], []
    norm = (profile - low) / (high - low)
    dark = norm > level
    length = len(norm)
    results = []
    for flip in (False, True):
        sequence = dark[::-1] if flip else dark
        values = norm[::-1] if flip else norm
        found = []
        index = 0
        while index < length and len(found) < count:
            if not sequence[index]:
                index += 1
                continue
            end = index
            while end < length and sequence[end]:
                end += 1
            if end - index >= min_run:
                weight = values[index:end] - level
                offset = float(np.sum(weight * np.arange(index, end)) /
                               max(np.sum(weight), 1e-6))
                found.append({
                    'centre_px': length - 1 - offset if flip else offset,
                    'width_px': int(end - index),
                })
            index = end
        results.append(found)
    return results[0], results[1]


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
    usable = filled[enough & np.isfinite(filled)]
    return np.where(enough & np.isfinite(filled), filled,
                    usable.min() if usable.size else 0.0)


def frame_rectangle(golden, coverage, min_run, level, span, min_stamps,
                    band_count=2):
    """Measure the design's frame rules on the golden image.

    The returned ``rules`` list runs from the outermost rule inwards; the first
    entry is used as the published rectangle.
    """
    darkness = -golden.astype(np.float32)
    columns = masked_profile(darkness, coverage, 0, span, min_stamps)
    rows = masked_profile(darkness, coverage, 1, span, min_stamps)
    left, right = edge_bands(columns, min_run, level, band_count)
    top, bottom = edge_bands(rows, min_run, level, band_count)

    rules = []
    for index in range(min(len(left), len(right), len(top), len(bottom))):
        found = {'left': left[index], 'right': right[index],
                 'top': top[index], 'bottom': bottom[index]}
        rule = {'rule': index + 1}
        rule.update({f'{side}_px': entry['centre_px']
                     for side, entry in found.items()})
        rule['width_px'] = float(found['right']['centre_px'] -
                                 found['left']['centre_px'])
        rule['height_px'] = float(found['bottom']['centre_px'] -
                                  found['top']['centre_px'])
        rule['band_px'] = {side: entry['width_px']
                           for side, entry in found.items()}
        rules.append(rule)
    return rules


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
SKIPPED_COLOR = (180, 180, 180)


def draw_design(image, results, skipped, unit):
    """Annotate the scan with each stamp's design rectangle and size.

    Line and font sizes follow ``segment_stamps.py`` so that this image and
    ``*_detected.jpg`` can be read side by side at the same zoom.
    """
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
        corners = np.rint(row['design_corners_image']).astype(np.int32)
        cv2.polylines(image, [corners], True, DESIGN_COLOR, thickness,
                      cv2.LINE_AA)
        for point in corners:
            cv2.circle(image, tuple(int(value) for value in point),
                       max(4, thickness * 3), CORNER_COLOR, -1, cv2.LINE_AA)
        # Label above the detection box, where ``*_detected.jpg`` puts it, so
        # the two images can be compared without hunting for the caption.
        box = row.get('bbox')
        anchor = ((box['x0'], box['x1'], box['y0']) if box else
                  (corners[:, 0].min(), corners[:, 0].max(),
                   corners[:, 1].min()))
        put_label(f"{row['stamp']}: {row[f'design_width_{unit}']:.2f} x "
                  f"{row[f'design_height_{unit}']:.2f} {unit}",
                  *anchor, DESIGN_COLOR)

    # A stamp the perforation stage could not close has no design rectangle.
    # Labelling it keeps the omission visible instead of silent.
    for number, box in skipped:
        cv2.rectangle(image, (box['x0'], box['y0']), (box['x1'], box['y1']),
                      SKIPPED_COLOR, thickness, cv2.LINE_AA)
        put_label(f'{number}: not measured', box['x0'], box['x1'], box['y0'],
                  SKIPPED_COLOR)
    return image


def measure_scan(image, document, schedule, iterations, eps, cutoff,
                 erode_px, pad, min_run, level, span, rule_index,
                 min_coverage, verbose):
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

    golden, coverage, warps, weights, failures, free_shear = congeal(
        patches, masks, canvas, schedule, iterations, eps, cutoff, verbose)
    min_stamps = max(2.0, min_coverage * len(patches))
    rules = frame_rectangle(golden, coverage, min_run, level, span, min_stamps)
    if len(rules) <= rule_index:
        raise RuntimeError(
            f'golden image has {len(rules)} frame rules on all four sides, '
            f'so rule {rule_index + 1} cannot be measured')
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
            'design_corners_image': rectangle_on_scan(
                rectangle, warps[index], placements[index]).tolist(),
        })
    return golden, coverage, rules, rule_index, results, canvas, min_stamps, \
        skipped


def main():
    parser = argparse.ArgumentParser(
        description=('Measure the printed design rectangle of every stamp in a '
                     'scan, by congealing the stamps into one golden image and '
                     'fitting each stamp back to it.'))
    parser.add_argument('input', help='Input scan, as given to segment_stamps.py')
    parser.add_argument(
        '--perf', help='Perforation JSON (default: <input stem>_perf.json)')
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
        '--band-min-run', type=int, default=5,
        help='Shortest run of dark pixels accepted as the frame band (default: 5)')
    parser.add_argument(
        '--band-level', type=float, default=0.5,
        help='Normalised darkness defining the frame band (default: 0.5)')
    parser.add_argument(
        '--profile-span', type=float, default=0.5,
        help='Central fraction of the opposite axis used per profile (default: 0.5)')
    parser.add_argument(
        '--min-coverage', type=float, default=0.6,
        help=('Fraction of the stamps that must reach a pixel before the '
              'golden is read there, which keeps the thinly covered canvas '
              'margin out of the rule search (default: 0.6)'))
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

    input_path = Path(args.input)
    perf_path = Path(args.perf) if args.perf else \
        input_path.with_name(input_path.stem + '_perf.json')
    document = json.loads(perf_path.read_text(encoding='utf-8'))
    dpi = args.dpi if args.dpi is not None else document.get('dpi')

    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f'Cannot read image: {input_path}')
    verbose = not args.quiet
    if verbose:
        print(f'Image: {input_path}')

    golden, coverage, rules, rule_index, results, canvas, min_stamps, \
        skipped = measure_scan(
            image, document, DEFAULT_SCHEDULE, args.iterations, args.eps,
            args.outlier_cutoff, args.erode, args.pad, args.band_min_run,
            args.band_level, args.profile_span, args.rule - 1,
            args.min_coverage, verbose)
    rectangle = rules[rule_index]

    unit = 'mm' if dpi else 'px'
    if dpi:
        factor = 25.4 / dpi
        for row in results:
            row['design_width_mm'] = row['design_width_px'] * factor
            row['design_height_mm'] = row['design_height_px'] * factor

    golden_path = input_path.with_name(input_path.stem + '_golden.png')
    cv2.imwrite(str(golden_path),
                golden_preview(golden, coverage, rules, rule_index, min_stamps))

    drawn_path = input_path.with_name(input_path.stem + '_design_size.jpg')
    if not cv2.imwrite(str(drawn_path),
                       draw_design(image, results, skipped, unit)):
        raise RuntimeError(f'Cannot write image: {drawn_path}')

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
            'preview': golden_path.name,
        },
        'annotated_scan': drawn_path.name,
        'parameters': {
            'schedule': [list(level) for level in DEFAULT_SCHEDULE],
            'iterations': args.iterations,
            'eps': args.eps,
            'outlier_cutoff': args.outlier_cutoff,
            'erode': args.erode,
            'pad': args.pad,
            'band_min_run': args.band_min_run,
            'band_level': args.band_level,
            'profile_span': args.profile_span,
            'min_coverage': args.min_coverage,
            'rule': args.rule,
        },
        'stamps': results,
    }
    out_path = input_path.with_name(input_path.stem + '_design.json')
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding='utf-8')

    if verbose:
        print('\nGolden frame rules, outermost first:')
        for rule in rules:
            mark = ' <- published' if rule['rule'] == args.rule else ''
            print(f"  rule {rule['rule']}: {rule['width_px']:8.2f} x "
                  f"{rule['height_px']:8.2f} px, band widths "
                  f"{tuple(rule['band_px'].values())} px{mark}")
        print(f"{'stamp':>5} {'width':>9} {'height':>9} {'sx':>8} {'sy':>8} "
              f"{'rot':>7} {'shear':>8} {'kept':>6} {'rms':>6}   ({unit})")
        for row in results:
            width = row.get(f'design_width_{unit}', row['design_width_px'])
            height = row.get(f'design_height_{unit}', row['design_height_px'])
            print(f"{row['stamp']:5d} {width:9.4f} {height:9.4f} "
                  f"{row['scale_x']:8.5f} {row['scale_y']:8.5f} "
                  f"{row['residual_rotation_deg']:7.3f} "
                  f"{row['unconstrained_shear']:8.5f} "
                  f"{row['kept_fraction']:6.3f} {row['residual_rms']:6.3f}")
        widths = np.array([row.get(f'design_width_{unit}', row['design_width_px'])
                           for row in results])
        heights = np.array([row.get(f'design_height_{unit}', row['design_height_px'])
                            for row in results])
        print(f"\nwidth  mean {widths.mean():.4f} sd {widths.std(ddof=1):.4f} "
              f"range {widths.min():.4f}..{widths.max():.4f} {unit}")
        print(f"height mean {heights.mean():.4f} sd {heights.std(ddof=1):.4f} "
              f"range {heights.min():.4f}..{heights.max():.4f} {unit}")
        if skipped:
            print('\nNot measured, no four published perforation sides: '
                  + ', '.join(str(number) for number, _ in skipped))
        print(f"\nWrote {out_path}\nWrote {golden_path}\nWrote {drawn_path}")


if __name__ == '__main__':
    main()
