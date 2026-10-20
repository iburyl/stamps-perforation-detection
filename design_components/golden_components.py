"""Rebuild the golden design one printing pass at a time, then recombine.

``design_size\\measure_design.py`` averages whole impressions into one golden.
That is the right thing for measuring the design rectangle, but it blurs a
two-pass stamp: the passes register independently, so whichever pass dominates
the alignment is sharpened and the other is smeared, and every postmark stroke
is averaged in along with the ink.

This separates each impression into its printing passes first, registers each
pass on its own evidence, averages each into its own golden, and only then puts
them back together.  Postmarks are identified as a component and excluded
rather than averaged down.

Which passes a stamp has is read off the scan rather than assumed, because
these denominations were not all printed the same way.  ``ink_palette`` names
the major inks, and there turn out to be two cases.

A single major ink means one press run, which laid the design and the ruled
ground behind it down together.  Those two are registered by different means,
because they offer different evidence.  The design is a unique aperiodic
pattern, so it is registered by correlation against the running template; a
sharp unambiguous peak exists and the half-set check confirms it.  The ground
is periodic, so no template-based match is well posed on it; instead its phase
is read against an analytic wave at its own measured pitch, which no template
bootstrap can bias.  Only the component across the ruling is observable that
way, the component along it being a symmetry of the screen, so only that
component is corrected.  The two plates still sit rotated with respect to each
other, by about a tenth of a degree in these sheets -- enough to displace a
corner by more than a pixel -- so each impression's screen bearing is measured
and brought to a common angle before the phase is matched.

Several major inks mean several press runs.  The bi-coloured denominations
printed the frame from one plate and the centre from another, each a separate
trip through the press, so each colour landed with its own small shift *and*
its own small turn and nothing ties them together.  Each colour is registered
as a rigid motion against its own running template.  Measured that way the
centre plate moves three to five times as far as the frame, and turns three to
five times as much, which is why averaging whole impressions smears it.

Relative *scale* is left alone throughout, deliberately: the inks lie on the
same fibres, so every later movement of the paper carries them equally, and
measurement puts any residual below 0.04%.

Driven by ``measure_design.py``, which calls :func:`build` once it has written
the design JSON this reads.  Writes ``<stem>_golden_components.png`` next to
``<stem>_golden.png``.

Experimental.  Reads only the JSON written by the production pipeline.
"""

from __future__ import annotations

import json
import math
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

import background_ink
import ink_palette
from ink_palette import unmix
from background_ink import (
    optical_density,
    render_ink,
    screen_window,
    separate,
    to_linear,
    to_srgb,
)
from pass_geometry import fit_wave
from three_colour_refinement import add_label, bgr_hex, estimate_papers, paper_ring, shift_image
from two_pass_decomposition import aligned_collection, masked_colour_median


MIN_COLOUR_AGREEMENT = 0.10
BACKGROUND_DIRECTION_RELATIVE_TOLERANCE = 0.10
AUTO_COLOUR_SCREEN_MIN_STRENGTH = 4.5
AUTO_COLOUR_SCREEN_STRONG_STRENGTH = 10.0
AUTO_COLOUR_SCREEN_MIN_HELD_OUT = 0.01
DUPLICATE_INK_ANGLE_DEG = 18.0
DUPLICATE_ACHROMATIC_CORRELATION = 0.45
DUPLICATE_BACKGROUND_CORRELATION = 0.35


def collection_quorum(count: int) -> int:
    """Useful majority coverage, including collections smaller than three."""
    return max(1, math.ceil(0.6 * count))


def background_direction_resolved(tilt: float, scatter: float) -> bool:
    """Whether a proposed second-ink direction clears measurement scatter.

    The two estimates move by a few tenths of a degree when the design masks are
    rebuilt.  Treating their exact crossing as a physical boundary made 7K flip
    from a strong, repeatable background to a single-ink fallback at 4.141 vs
    4.152 degrees.  A narrow relative uncertainty band keeps that near-tie in
    the evidence-gathering branch, while 5K's 3.524 vs 4.132 degree failure
    remains outside it and is still read as one ink.
    """
    if not np.isfinite(tilt) or not np.isfinite(scatter):
        return False
    if scatter <= 1e-9:
        return tilt > 0.0
    return tilt >= (1.0 - BACKGROUND_DIRECTION_RELATIVE_TOLERANCE) * scatter


def move_image(image: np.ndarray, motion: np.ndarray, interpolation: int) -> np.ndarray:
    """Apply a motion, which is either a translation or a whole affine.

    Both are applied as inverse maps, so a motion always means the same thing:
    where in this impression to read the value that belongs at ``x``.
    """
    motion = np.asarray(motion, np.float32)
    if motion.shape == (2,):
        return shift_image(image, motion, interpolation)
    height, width = image.shape[:2]
    return cv2.warpAffine(
        image,
        motion,
        (width, height),
        flags=interpolation | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def recentre(warps: np.ndarray, centre: tuple[float, float]) -> np.ndarray:
    """Pin the median rotation, and the median displacement at the centre, to
    zero.

    Only differences between impressions are measured; where the collection as
    a whole sits is not, so a gauge has to be chosen rather than discovered.
    """
    origin = np.array([centre[0], centre[1], 1.0])
    angles = np.arctan2(warps[:, 1, 0], warps[:, 0, 0])
    offsets = warps @ origin - np.asarray(centre)
    angles = angles - np.median(angles)
    offsets = offsets - np.median(offsets, axis=0)
    rebuilt = np.zeros_like(warps)
    for index, (angle, offset) in enumerate(zip(angles, offsets)):
        turn = np.array(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
        )
        rebuilt[index, :, :2] = turn
        rebuilt[index, :, 2] = np.asarray(centre) + offset - turn @ np.asarray(centre)
    return rebuilt


def robust_golden(
    layers: np.ndarray, masks: np.ndarray, motions: np.ndarray, quorum: int, floor: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Move every impression onto the common frame and take the median.

    ``quorum`` is the coverage the returned mask reports as trustworthy;
    ``floor`` is the coverage below which no value is emitted at all.
    """
    stack, coverage = [], []
    for layer, mask, motion in zip(layers, masks, motions):
        stack.append(move_image(layer, motion, cv2.INTER_CUBIC))
        coverage.append(move_image(mask.astype(np.uint8), motion, cv2.INTER_NEAREST) > 0)
    values = np.stack(stack)
    seen = np.stack(coverage)
    values[~seen] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        golden = np.nanmedian(values, axis=0)
    tally = seen.sum(axis=0)
    enough = tally >= quorum
    # Keep the thin-coverage median rather than punching a hole in the design.
    # Under a postmark that lands on most impressions the few readable samples
    # are still paper and ink, just noisier.
    golden[~np.isfinite(golden) | (tally < floor)] = 0.0
    return golden.astype(np.float32), enough


def best_preserving_colour_merge(
    entries: list[dict],
    shifts: np.ndarray,
    preferred_sources: np.ndarray,
    preservation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Apply screen geometry without allowing averaging to erase the best plate.

    The clean periodic field chooses the transformations, not the density of the
    finished colour plate.  The strongest preservation-ranked impression remains
    authoritative wherever it is readable; aligned peers only fill pixels hidden
    from it by a postmark or an invalid mask.
    """
    wanted = set(int(index) for index in np.asarray(preferred_sources).ravel())
    positions = [
        position
        for position, entry in enumerate(entries)
        if int(entry["source_index"]) in wanted
    ]
    if not positions:
        raise RuntimeError("No preservation-selected colour impression fits the screen")

    sources = np.array([int(entries[position]["source_index"]) for position in positions])
    layers = np.stack([entries[position]["coverage"] for position in positions])
    masks = np.stack([entries[position]["readable"] for position in positions])
    motions = np.asarray(shifts, np.float32)[positions]
    merged, merged_seen = robust_golden(layers, masks, motions, quorum=1, floor=1)

    best_local = int(np.argmax(preservation[sources]))
    best_source = int(sources[best_local])
    best_layer = move_image(layers[best_local], motions[best_local], cv2.INTER_CUBIC)
    best_seen = move_image(
        masks[best_local].astype(np.uint8), motions[best_local], cv2.INTER_NEAREST
    ) > 0
    merged[best_seen] = best_layer[best_seen]
    return merged.astype(np.float32), best_seen | merged_seen, best_source


def colour_preservation_selection(
    layers: np.ndarray, masks: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Select a well-preserved colour group without prescribing its size.

    Water loss removes both area and density from an ink.  Their product is a
    useful one-dimensional preservation measure.  Work in log space and cut
    only across a conspicuous gap; a smooth quality gradient contains no
    defensible boundary, so all readable impressions remain in play.  There
    is deliberately no minimum group size: one surviving impression is still
    better evidence than abandoning the colour pass altogether.

    Returns the selection, scores relative to the strongest impression, and
    the relative-score boundary when a distinct group was found.
    """
    raw = np.zeros(len(layers), np.float64)
    for index, (layer, mask) in enumerate(zip(layers, masks)):
        values = np.clip(layer[mask], 0.0, None)
        positive = values[values > 0.0]
        if not len(positive):
            continue
        # Mean density retains completeness; the upper quantile represents
        # how much ink survived where the plate is still legible.
        raw[index] = float(values.mean()) * float(np.percentile(positive, 90))

    available = raw > 0.0
    selected = available.copy()
    relative = raw / max(float(raw.max()), 1e-12)
    live = np.flatnonzero(available)
    if len(live) < 2:
        return selected, relative, None

    ordered = live[np.argsort(np.log(raw[live]))]
    levels = np.log(raw[ordered])
    gaps = np.diff(levels)
    split = int(np.argmax(gaps))
    gap = float(gaps[split])

    # A pair supplies no distribution from which to estimate a typical gap,
    # but a twofold preservation difference is strong enough to prefer the
    # better observation.  With three or more, require both a material ratio
    # and an outlying gap relative to the other spacings.
    if len(live) == 2:
        distinct = gap >= math.log(2.0)
    else:
        other = np.delete(gaps, split)
        typical = float(np.median(other)) if len(other) else 0.0
        distinct = gap >= math.log(1.5) and gap >= 1.5 * max(typical, 1e-9)
    if not distinct:
        return selected, relative, None

    selected[:] = False
    selected[ordered[split + 1 :]] = True
    boundary = math.sqrt(
        float(relative[ordered[split]]) * float(relative[ordered[split + 1]])
    )
    return selected, relative, boundary


def register_pattern(
    layers: np.ndarray,
    masks: np.ndarray,
    quorum: int,
    iterations: int,
    limit: float,
    spin: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Motion per impression, by correlation against the running template.

    Sound for any plate that printed a unique aperiodic pattern, whose
    correlation peak is sharp and single.  It is deliberately not used for the
    periodic background screen, where no such peak exists.

    With ``spin`` set to a bearing in degrees the plate is also allowed to have
    landed turned, and the returned motions are whole rigid transforms rather
    than translations.  That is what a colour struck in its own press run
    needs; a plate that printed in the same run as the template does not.
    """
    warps = np.repeat(np.eye(2, 3, dtype=np.float32)[None], len(layers), axis=0)
    scores = np.full(len(layers), np.nan, np.float32)
    mode = cv2.MOTION_EUCLIDEAN if spin > 0.0 else cv2.MOTION_TRANSLATION
    centre = (layers.shape[2] / 2.0, layers.shape[1] / 2.0)
    template, _ = robust_golden(layers, masks, warps, quorum, floor=quorum)
    for _ in range(iterations):
        positive = template[template > 0]
        if len(positive) < 500:
            break
        support = cv2.dilate(
            (template >= np.percentile(positive, 20)).astype(np.uint8),
            np.ones((5, 5), np.uint8),
        )
        target = np.clip(template / max(float(np.percentile(positive, 96)), 1e-6), 0, 1)
        for index, layer in enumerate(layers):
            inside = layer[layer > 0]
            if len(inside) < 300:
                continue
            moving = np.clip(layer / max(float(np.percentile(inside, 96)), 1e-6), 0, 1)
            warp = warps[index].copy()
            try:
                score, warp = cv2.findTransformECC(
                    target,
                    moving,
                    warp,
                    mode,
                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 120, 1e-7),
                    inputMask=(support & masks[index].astype(np.uint8)) * 255,
                    gaussFiltSize=5,
                )
            except cv2.error:
                continue
            if not np.all(np.isfinite(warp)):
                continue
            # Judge the fit by how far it moves the middle of the design, which
            # a rotation about the top-left corner wildly overstates.
            travel = warp @ np.array([centre[0], centre[1], 1.0]) - np.asarray(centre)
            turned = abs(math.degrees(math.atan2(warp[1, 0], warp[0, 0])))
            if np.max(np.abs(travel)) <= limit and turned <= max(spin, 1e-9):
                warps[index] = warp
                scores[index] = float(score)
        good = np.isfinite(scores)
        if good.any():
            warps[good] = recentre(warps[good], centre)
        template, _ = robust_golden(layers, masks, warps, quorum, floor=quorum)
    if mode == cv2.MOTION_TRANSLATION:
        return warps[:, :, 2].copy(), scores
    return warps, scores


def shared_wave(patches: list[np.ndarray], start: np.ndarray) -> np.ndarray:
    """The one screen frequency that best fits every impression at once.

    Each impression's own FFT peak can sit a bin away from its neighbour's, and
    a bin here is a 1.7% pitch difference, which drifts a full period across the
    window.  Searching off the grid around the anchor's peak for the frequency
    with the most total amplitude gives all impressions a common ruler.
    """
    best, best_power = start, -1.0
    step = 1.0 / background_ink.SCREEN_WINDOW
    for dy in np.arange(-1.0, 1.01, 0.2):
        for dx in np.arange(-1.0, 1.01, 0.2):
            wave = start + np.array([dx, dy]) * step
            power = sum(
                abs(background_ink.screen_coefficient(patch, wave).sum()) for patch in patches
            )
            if power > best_power:
                best, best_power = wave, power
    return best


def phase_shifts(results: list[dict], corner: tuple[int, int], anchor: int) -> tuple:
    """Across-ruling shift of each screen, from its phase against an analytic wave.

    Every phase is read in the same window at the same frequency, so the
    differences are comparable.  Only the component along the wave vector is
    recoverable; sliding along the ruling leaves the screen unchanged.

    The phase is read in the extracted screen itself, not in the density it came
    out of.  Those are different projections of the same colour -- the screen is
    the sliver of density standing clear of the main ink's plane, while a
    reading off the raw density is dominated by what the channels have in common
    -- and they only agree when the two inks are well separated.  On 5K, where
    the sliver is six hundredths of the density, they disagreed by 59 degrees of
    phase, so the impressions were aligned to the wrong thing and merging them
    cancelled the ruling instead of building it.  Read in the layer that is
    actually being merged, the same impressions agree to 1.3 degrees.
    """
    patches = [
        background_ink.screen_patch(r["coverage"][:, :, None], r["field"], corner)
        for r in results
    ]
    wave = shared_wave(patches, results[anchor]["wave"])
    shifts = np.zeros((len(results), 2), np.float32)
    squared = float(wave @ wave)
    if squared < 1e-12:
        return shifts, wave, 0.0
    phases, weights = [], []
    for patch in patches:
        coefficient = background_ink.screen_coefficient(patch, wave)[0]
        phases.append(np.angle(coefficient))
        weights.append(abs(coefficient))
    reference = phases[anchor]
    for index, phase in enumerate(phases):
        turns = (phase - reference) / (2.0 * math.pi)
        turns -= round(turns)  # nearest repeat; the screen is ambiguous beyond it
        # shift_image resamples at x + shift, so the shift opposes the phase lead.
        shifts[index] = (-turns / squared) * wave
    shifts -= np.median(shifts, axis=0)
    return shifts, wave, float(np.median(weights))


def screen_cell_feature(layer: np.ndarray, mask: np.ndarray, pitch: float) -> np.ndarray:
    """High-pass screen evidence without broad design or paper shading.

    Cell selection needs the small two-dimensional departures from a perfect
    sinusoid.  The fundamental alone cannot distinguish neighbouring repeats,
    while broad leakage from the main plate can.  A mask-normalised blur
    removes that broad component without manufacturing a dark rim around held
    out design and postmark pixels.
    """
    weight = mask.astype(np.float32)
    sigma = max(1.5, 0.55 * pitch)
    numerator = cv2.GaussianBlur(
        np.where(mask, layer, 0.0).astype(np.float32), (0, 0), sigma
    )
    denominator = cv2.GaussianBlur(weight, (0, 0), sigma)
    broad = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0.15,
    )
    feature = np.where(mask, layer - broad, 0.0).astype(np.float32)
    values = np.abs(feature[mask])
    if len(values):
        cap = float(np.percentile(values, 99.5))
        if cap > 1e-9:
            feature = np.clip(feature, -cap, cap)
    return feature


def held_out_cell_score(
    layers: np.ndarray,
    masks: np.ndarray,
    shifts: np.ndarray,
    corner: tuple[int, int],
    pitch: float,
) -> float:
    """Pair agreement outside the window used to resolve the screen cell.

    Fourier strength cannot distinguish neighbouring periods.  The resolver's
    central window can also reward a local coincidence, so candidate choice is
    made on the rest of the stamp.  High-pass features retain the cell's
    aperiodic detail, and every pair gets one vote so no candidate is selected
    merely because a single strong impression served as its anchor.
    """
    features = np.stack([
        screen_cell_feature(layer, mask, pitch)
        for layer, mask in zip(layers, masks)
    ])
    aligned, covered = [], []
    for feature, mask, shift in zip(features, masks, shifts):
        aligned.append(shift_image(feature, shift, cv2.INTER_CUBIC))
        covered.append(
            shift_image(mask.astype(np.uint8), shift, cv2.INTER_NEAREST) > 0
        )
    aligned, covered = np.stack(aligned), np.stack(covered)

    held_out = np.ones(layers.shape[1:], bool)
    y0, x0 = corner
    size = background_ink.SCREEN_WINDOW
    margin = 16
    held_out[
        max(0, y0 - margin):min(held_out.shape[0], y0 + size + margin),
        max(0, x0 - margin):min(held_out.shape[1], x0 + size + margin),
    ] = False
    correlations = []
    for first in range(len(layers)):
        for second in range(first):
            common = covered[first] & covered[second] & held_out
            if int(common.sum()) < 3000:
                continue
            one = aligned[first][common].astype(np.float64)
            two = aligned[second][common].astype(np.float64)
            one -= one.mean()
            two -= two.mean()
            denominator = math.sqrt(float(one @ one) * float(two @ two))
            if denominator > 1e-12:
                correlations.append(float((one @ two) / denominator))
    return float(np.mean(correlations)) if correlations else float("nan")


def masked_translation_surface(
    moving: np.ndarray,
    moving_mask: np.ndarray,
    reference: np.ndarray,
    reference_mask: np.ndarray,
    corner: tuple[int, int],
    limit: int,
) -> np.ndarray:
    """Masked zero-mean correlation for every integer shift in ``limit``.

    Six OpenCV correlations evaluate the whole grid at once.  This is much
    cheaper than warping a full stamp for every candidate and, unlike filling
    masked pixels with zero, does not reward candidates merely for changing
    the overlap.
    """
    height, width = reference.shape
    y0, x0 = corner
    size = background_ink.SCREEN_WINDOW
    box_height = min(size, height - y0)
    box_width = min(size, width - x0)
    if box_height < 32 or box_width < 32:
        return np.full((2 * limit + 1, 2 * limit + 1), np.nan, np.float32)

    ref = reference[y0:y0 + box_height, x0:x0 + box_width].astype(np.float32)
    ref_mask = reference_mask[
        y0:y0 + box_height, x0:x0 + box_width
    ].astype(np.float32)
    padded_moving = cv2.copyMakeBorder(
        moving.astype(np.float32), limit, limit, limit, limit,
        cv2.BORDER_CONSTANT, value=0,
    )
    padded_mask = cv2.copyMakeBorder(
        moving_mask.astype(np.float32), limit, limit, limit, limit,
        cv2.BORDER_CONSTANT, value=0,
    )
    search = padded_moving[
        y0:y0 + box_height + 2 * limit,
        x0:x0 + box_width + 2 * limit,
    ]
    search_mask = padded_mask[
        y0:y0 + box_height + 2 * limit,
        x0:x0 + box_width + 2 * limit,
    ]

    def correlate(image: np.ndarray, template: np.ndarray) -> np.ndarray:
        return cv2.matchTemplate(image, template, cv2.TM_CCORR).astype(np.float64)

    ref_values = ref * ref_mask
    moving_values = search * search_mask
    overlap = correlate(search_mask, ref_mask)
    sum_reference = correlate(search_mask, ref_values)
    square_reference = correlate(search_mask, ref_values * ref)
    sum_moving = correlate(moving_values, ref_mask)
    square_moving = correlate(moving_values * search, ref_mask)
    product = correlate(moving_values, ref_values)

    safe = np.maximum(overlap, 1.0)
    covariance = product - sum_reference * sum_moving / safe
    variance_reference = square_reference - sum_reference * sum_reference / safe
    variance_moving = square_moving - sum_moving * sum_moving / safe
    denominator = np.sqrt(
        np.maximum(variance_reference, 0.0) * np.maximum(variance_moving, 0.0)
    )
    minimum_overlap = max(1000.0, 0.30 * float(ref_mask.sum()))
    valid = (overlap >= minimum_overlap) & (denominator > 1e-12)
    surface = np.full(overlap.shape, np.nan, np.float32)
    surface[valid] = (covariance[valid] / denominator[valid]).astype(np.float32)
    return surface


def best_screen_cell(surface: np.ndarray, wave: np.ndarray) -> dict:
    """Best correlation peak whose normal displacement is a whole period."""
    limit = surface.shape[0] // 2
    yy, xx = np.mgrid[-limit:limit + 1, -limit:limit + 1]
    turns = xx * wave[0] + yy * wave[1]
    cells = np.rint(turns).astype(np.int32)
    tolerance = max(0.12, 0.75 * float(np.hypot(*wave)))
    candidates = np.isfinite(surface) & (np.abs(turns - cells) <= tolerance)
    candidates[limit, limit] = np.isfinite(surface[limit, limit])
    if not candidates.any():
        return {
            "shift": np.zeros(2, np.float32),
            "score": float("nan"),
            "current": float("nan"),
            "cell_gap": 0.0,
        }
    ranked = np.where(candidates, surface, -np.inf)
    flat = int(np.argmax(ranked))
    row, column = np.unravel_index(flat, ranked.shape)
    chosen_cell = int(cells[row, column])
    cell_scores = [
        float(np.max(surface[candidates & (cells == cell)]))
        for cell in np.unique(cells[candidates])
    ]
    cell_scores.sort(reverse=True)
    gap = cell_scores[0] - cell_scores[1] if len(cell_scores) > 1 else 0.0
    return {
        "shift": np.array([column - limit, row - limit], np.float32),
        "score": float(surface[row, column]),
        "current": float(surface[limit, limit]),
        "cell_gap": float(gap),
        "cell": chosen_cell,
    }


def recentre_screen_cells(residuals: np.ndarray, wave: np.ndarray) -> np.ndarray:
    """Choose a zero gauge without averaging two neighbouring screen cells."""
    length = float(np.hypot(*wave))
    if length < 1e-12:
        return residuals - np.median(residuals, axis=0)
    normal = wave / length
    tangent = np.array([-normal[1], normal[0]])
    pitch = 1.0 / length
    across = residuals @ normal
    along = residuals @ tangent
    # Every residual must remain on a whole-period branch after the graph fit.
    across = np.rint(across / pitch) * pitch
    across -= round(float(np.median(across)) / pitch) * pitch
    along -= np.median(along)
    return (across[:, None] * normal + along[:, None] * tangent).astype(np.float32)


def solve_screen_cell_graph(
    count: int, edges: list[dict], wave: np.ndarray, limit: float
) -> np.ndarray:
    """Robustly synchronize pairwise cell offsets over the whole collection."""
    usable = [edge for edge in edges if np.isfinite(edge["score"]) and edge["score"] > 0.05]
    if count < 2 or not usable:
        return np.zeros((count, 2), np.float32)
    matrix = np.zeros((len(usable), max(count - 1, 1)), np.float64)
    length = float(np.hypot(*wave))
    if length < 1e-12:
        return np.zeros((count, 2), np.float32)
    normal = wave / length
    tangent = np.array([-normal[1], normal[0]])
    pitch = 1.0 / length
    target = np.zeros((len(usable), 2), np.float64)
    base_weights = np.zeros(len(usable), np.float64)
    for row, edge in enumerate(usable):
        first, second = edge["first"], edge["second"]
        if first:
            matrix[row, first - 1] -= 1.0
        if second:
            matrix[row, second - 1] += 1.0
        # The first coordinate is discrete by construction.  Solving it in
        # integer turns prevents an inconsistent graph from averaging two
        # neighbouring repeats into a half-period blur.
        target[row, 0] = edge["cell"]
        target[row, 1] = float(edge["shift"] @ tangent) / pitch
        evidence = max(
            edge["cell_gap"], edge["score"] - edge["current"], 0.01
        )
        base_weights[row] = edge["score"] ** 2 * min(evidence, 0.25)

    weights = base_weights.copy()
    solution = np.zeros((count, 2), np.float64)
    for _ in range(4):
        root = np.sqrt(np.maximum(weights, 1e-9))[:, None]
        fitted, *_ = np.linalg.lstsq(matrix * root, target * root, rcond=None)
        solution[1:] = fitted
        error = np.linalg.norm(matrix @ fitted - target, axis=1)
        scale = max(float(np.median(error)), 0.5)
        weights = base_weights * np.minimum(1.0, 1.5 * scale / np.maximum(error, 1e-9))
    turns = np.rint(solution[:, 0])
    turns -= round(float(np.median(turns)))
    most_turns = max(1, int(math.floor(limit / pitch + 1e-6)))
    turns = np.clip(turns, -most_turns, most_turns)
    along = solution[:, 1] * pitch
    along -= np.median(along)
    along = np.clip(along, -limit, limit)
    return (
        turns[:, None] * pitch * normal + along[:, None] * tangent
    ).astype(np.float32)


def resolve_screen_cells(
    layers: np.ndarray,
    masks: np.ndarray,
    phase: np.ndarray,
    corner: tuple[int, int],
    wave: np.ndarray,
    limit: float,
    strongest: int,
) -> dict:
    """Resolve the integer-period and along-ruling ambiguity of screen phase.

    Two candidates are returned for side-by-side evaluation.  The anchor form
    aligns every impression directly to the strongest single observation.  The
    consensus form synchronizes all pairwise observations and is the intended
    production solution; neither lets the chosen anchor set the absolute gauge.
    """
    started = time.perf_counter()
    count = len(layers)
    search = max(1, int(math.ceil(limit)))
    pitch = 1.0 / max(float(np.hypot(*wave)), 1e-9)
    aligned_layers = np.stack([
        shift_image(layer, shift, cv2.INTER_CUBIC)
        for layer, shift in zip(layers, phase)
    ])
    aligned_masks = np.stack([
        shift_image(mask.astype(np.uint8), shift, cv2.INTER_NEAREST) > 0
        for mask, shift in zip(masks, phase)
    ])
    features = np.stack([
        screen_cell_feature(layer, mask, pitch)
        for layer, mask in zip(aligned_layers, aligned_masks)
    ])

    edges = []
    for first in range(count):
        for second in range(first + 1, count):
            surface = masked_translation_surface(
                features[second], aligned_masks[second],
                features[first], aligned_masks[first], corner, search,
            )
            match = best_screen_cell(surface, wave)
            edges.append({"first": first, "second": second, **match})

    anchor_residuals = np.zeros((count, 2), np.float32)
    for edge in edges:
        if edge["first"] == strongest:
            anchor_residuals[edge["second"]] = edge["shift"]
        elif edge["second"] == strongest:
            anchor_residuals[edge["first"]] = -edge["shift"]
    anchor_residuals = recentre_screen_cells(anchor_residuals, wave)
    consensus_residuals = solve_screen_cell_graph(count, edges, wave, limit)

    incident = [[] for _ in range(count)]
    for edge in edges:
        incident[edge["first"]].append(edge)
        incident[edge["second"]].append(edge)
    confidence = []
    for matches in incident:
        confidence.append({
            "median_correlation": float(np.median([m["score"] for m in matches]))
            if matches else float("nan"),
            "median_cell_gap": float(np.median([m["cell_gap"] for m in matches]))
            if matches else 0.0,
        })
    return {
        "anchor_shifts": phase + anchor_residuals,
        "consensus_shifts": phase + consensus_residuals,
        "anchor_residuals": anchor_residuals,
        "consensus_residuals": consensus_residuals,
        "confidence": confidence,
        "pair_count": len(edges),
        "elapsed_seconds": time.perf_counter() - started,
    }


def screen_peak_floor(layer: np.ndarray, mask: np.ndarray, corner, wave: np.ndarray) -> tuple:
    """The ruling's Fourier amplitude, and the broadband floor beside it."""
    size = background_ink.SCREEN_WINDOW
    y0, x0 = corner
    box = (slice(y0, y0 + size), slice(x0, x0 + size))
    weight = mask[box].astype(np.float64)
    if weight.sum() < 15000:
        return 0.0, 0.0
    patch = layer[box].astype(np.float64)
    patch = patch - patch[weight > 0].mean()
    patch = patch * np.outer(np.hanning(size), np.hanning(size)) * weight

    peak = abs(background_ink.screen_coefficient(patch[:, :, None], wave)[0])
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(patch)))
    centre = size // 2
    yy, xx = np.mgrid[:size, :size]
    radius = np.hypot(xx - centre, yy - centre)
    target = np.hypot(*wave) * size
    ring = np.abs(radius - target) < 4.0
    for sign in (1.0, -1.0):  # the peak and its conjugate are not the floor
        ring &= (
            np.hypot(xx - centre - sign * wave[0] * size, yy - centre - sign * wave[1] * size)
            > 6.0
        )
    return float(peak), float(np.median(spectrum[ring])) if ring.any() else 0.0


def place_in_impression(
    golden: np.ndarray, shift: np.ndarray, rotation: float, shape: tuple[int, int]
) -> np.ndarray:
    """Put a merged golden back where it sits in one impression.

    The merge brought every impression onto a common frame by turning it to the
    collection's bearing and then sliding it; this undoes both, in reverse, so
    that a quantity measured once from the whole collection can be used against
    a single impression's own measurements.
    """
    local = shift_image(golden, -np.asarray(shift, np.float32), cv2.INTER_CUBIC)
    if abs(rotation) > 1e-6:
        height, width = shape
        matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), -rotation, 1.0)
        local = cv2.warpAffine(
            local, matrix, (width, height),
            flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
    return local


def repeatability_gain(
    layers: np.ndarray, fields: np.ndarray, shifts: np.ndarray, common: np.ndarray
) -> np.ndarray:
    """Per-frequency gain from how well the impressions agree about the plate.

    The background is one plate, so whatever it really printed repeats in every
    impression once they are registered, while what the 1/separability division
    amplified does not.  Comparing the coherent sum against the mean power, bin
    by bin, separates the two without having to assume which frequencies a
    background is allowed to occupy -- the assumption band-limiting got wrong,
    by discarding the background's own low-frequency design along with the noise.

    It is measured over the whole shared field rather than the screen window.
    The window is 256 px and the background's design varies over the whole
    stamp, so a window cannot see the low frequencies at all: measured there,
    5K kept a thousandth of its layer and the design envelope went with it.

    One over the impression count is the coherence independent noise shows by
    chance, so it is subtracted before the gain is read.
    """
    aperture = common.astype(np.float64)
    height, width = aperture.shape
    aperture = aperture * np.outer(np.hanning(height), np.hanning(width))
    total = np.zeros((height, width), np.complex128)
    power = np.zeros((height, width), np.float64)
    count = 0
    for layer, mask, shift in zip(layers, fields, shifts):
        moved = shift_image(layer, shift, cv2.INTER_CUBIC).astype(np.float64)
        weight = shift_image(mask.astype(np.uint8), shift, cv2.INTER_NEAREST) > 0
        inside = weight & common
        if inside.sum() < 15000:
            continue
        spectrum = np.fft.fft2((moved - moved[inside].mean()) * aperture * weight)
        total += spectrum
        power += np.abs(spectrum) ** 2
        count += 1
    if count < 3:
        return np.ones((height, width), np.float64)
    coherence = np.abs(total / count) ** 2 / np.maximum(power / count, 1e-20)
    gain = (coherence - 1.0 / count) / (1.0 - 1.0 / count)
    # A per-bin coherence is itself a noisy statistic; neighbouring bins belong
    # to the same structure, so the gain is smoothed before it is believed.
    return np.fft.ifftshift(
        np.clip(cv2.GaussianBlur(np.fft.fftshift(gain), (0, 0), 2.0), 0.0, 1.0)
    )


def apply_gain(layer: np.ndarray, field: np.ndarray, gain: np.ndarray) -> np.ndarray:
    """Filter a layer by a gain measured across the collection."""
    keep = field.astype(np.float32)
    spectrum = np.fft.fft2(layer.astype(np.float64) * keep)
    return (np.real(np.fft.ifft2(spectrum * gain)) * keep).astype(np.float32)


def screen_strength(layer: np.ndarray, mask: np.ndarray, corner, wave: np.ndarray) -> float:
    """How far the ruling stands above the broadband floor in one layer.

    This is the crispness measure for the background, and it is comparable
    between a single impression and an average of several, which mean gradient
    magnitude is not: noise raises a gradient but not a Fourier peak.  A merge
    that reinforces the screen scores above its parts, a merge that smears it
    scores below them.
    """
    peak, floor = screen_peak_floor(layer, mask, corner, wave)
    return float(peak / max(floor, 1e-9))


def blend(values: np.ndarray, seen: np.ndarray, weights: np.ndarray, floor: int = 1):
    """Weighted mean of the impressions that cover each pixel.

    Impressions differ several-fold in how much background ink they took, and
    they differ again in how noisy that ink reads.  A median throws both facts
    away and returns the middling impression; a weighted mean keeps the
    well-inked, quiet ones in proportion.  This is safe here only because the
    layers have been put in one colour basis, so they share a sign and a scale,
    and because the field already excludes postmarks and heavy ink, so there
    are no outliers left for a median to guard against.
    """
    spread = weights[:, None, None] * seen
    total = spread.sum(axis=0)
    golden = (values * spread).sum(axis=0) / np.maximum(total, 1e-9)
    golden[seen.sum(axis=0) < floor] = 0.0
    return golden


def matched_weights(cropped: np.ndarray, covered: np.ndarray, wave: np.ndarray) -> np.ndarray:
    """Weight each impression by the ruling it brings per unit of its own noise.

    Maximising the merged screen against the merged noise wants amplitude over
    variance, so an impression that carries twice the ruling counts twice, and
    one that carries twice the noise counts a quarter.
    """
    weights = []
    for layer, mask in zip(cropped, covered):
        peak, floor = screen_peak_floor(layer, mask, (0, 0), wave)
        weights.append(peak / max(floor, 1e-9) ** 2)
    weights = np.array(weights, np.float64)
    return weights / max(weights.max(), 1e-12)


def window_scorer(layers: np.ndarray, masks: np.ndarray, shifts: np.ndarray, corner, wave):
    """Score any subset of impressions by the crispness of their merge.

    The shifted layers are cropped to the shared screen window once, so trying
    a subset costs only a weighted mean over it.
    """
    size = background_ink.SCREEN_WINDOW
    y0, x0 = corner
    window = (slice(y0, y0 + size), slice(x0, x0 + size))
    cropped, covered = [], []
    for layer, mask, shift in zip(layers, masks, shifts):
        cropped.append(shift_image(layer, shift, cv2.INTER_CUBIC)[window])
        covered.append(shift_image(mask.astype(np.uint8), shift, cv2.INTER_NEAREST)[window] > 0)
    cropped, covered = np.stack(cropped), np.stack(covered)
    weights = matched_weights(cropped, covered, wave)

    def score(members) -> float:
        members = list(members)
        golden = blend(cropped[members].astype(np.float64), covered[members], weights[members])
        enough = covered[members].sum(axis=0) >= max(1, len(members) // 2)
        golden[~enough] = 0.0
        return screen_strength(golden, enough, (0, 0), wave)

    return score, weights


def choose_impressions(score, count: int) -> tuple[list[int], list[float], list[float]]:
    """Drop only the impressions whose presence blunts the merge.

    Averaging every impression is not automatically right: the screen is
    quasi-periodic, so an impression whose ruling does not phase up with the
    rest subtracts modulation instead of adding it.  But the search has to run
    backwards, from the whole set, because growing one forwards walks into a
    dead end.  On 5K the strongest impression scores 44 alone, every one of its
    fourteen pairings scores less, and all fifteen together score 57: the
    impressions only pay off in bulk, each one adding a little ruling and a
    little less noise than it adds ruling, and no pair shows that.
    """
    alone = [score([index]) for index in range(count)]
    chosen = list(range(count))
    trail = [score(chosen)]
    while len(chosen) > 2:
        best, drop = max(
            (score([i for i in chosen if i != candidate]), candidate) for candidate in chosen
        )
        if best <= trail[-1]:
            break
        chosen.remove(drop)
        trail.append(best)
    # Backward elimination can be trapped by a set where removing any one
    # member is insufficient, even though the complete merge is worse than a
    # clean singleton.  Never publish a merge weaker than the evidence already
    # available in its best constituent.
    best_single = int(np.argmax(alone))
    if alone[best_single] > trail[-1]:
        chosen = [best_single]
        trail.append(alone[best_single])
    return chosen, trail, alone


def screen_golden(
    layers: np.ndarray, masks: np.ndarray, shifts: np.ndarray, weights: np.ndarray, quorum: int
) -> tuple[np.ndarray, np.ndarray]:
    """Shift the screen layers onto the common frame and take the weighted mean."""
    stack, coverage = [], []
    for layer, mask, shift in zip(layers, masks, shifts):
        stack.append(shift_image(layer, shift, cv2.INTER_CUBIC))
        coverage.append(shift_image(mask.astype(np.uint8), shift, cv2.INTER_NEAREST) > 0)
    values, seen = np.stack(stack).astype(np.float64), np.stack(coverage)
    golden = blend(values, seen, weights)
    return golden.astype(np.float32), seen.sum(axis=0) >= quorum


def select_and_merge(
    layers: np.ndarray, field: np.ndarray, shifts: np.ndarray, corner, wave: np.ndarray
) -> dict:
    """Keep only the impressions whose merge sharpens the screen, then merge them."""
    score, weights = window_scorer(layers, field, shifts, corner, wave)
    chosen, trail, alone = choose_impressions(score, len(layers))
    quorum = min(len(chosen), max(2, len(chosen) // 2))
    golden, seen = screen_golden(
        layers[chosen], field[chosen], shifts[chosen], weights[chosen], quorum
    )
    return {
        "golden": golden,
        "seen": seen,
        "layers": layers,
        "field": field,
        "shifts": shifts,
        "wave": wave,
        "chosen": chosen,
        "quorum": quorum,
        "trail": trail,
        "alone": alone,
        "all_merged": score(range(len(layers))),
        "weights": weights,
    }


def merge_gain(merged: dict, corner, wave: np.ndarray) -> float:
    """Screen amplitude of the merge against the average of one impression.

    Above one the impressions reinforced; below one they cancelled, which is
    the thing to watch, since a merge can come out smoother than any single
    impression while holding less ruling than any of them.
    """
    chosen = merged["chosen"]
    singles = [
        abs(background_ink.screen_coefficient(
            background_ink.screen_patch(
                merged["layers"][index][:, :, None], merged["field"][index], corner
            ),
            wave,
        )[0])
        for index in chosen
    ]
    together = abs(background_ink.screen_coefficient(
        background_ink.screen_patch(
            merged["golden"][:, :, None], merged["seen"], corner
        ),
        wave,
    )[0])
    return float(together / max(np.mean(singles), 1e-9))


def background_golden(
    results: list[dict], corner, anchor: int, downsample: int, shift_limit: float = 12.0,
    cell_min_strength: float | None = None,
) -> dict:
    """Build the legacy phase merge and candidate cell-resolved merges."""
    field = np.stack([r["field"] for r in results])
    layers = np.stack(
        [np.where(r["field"], r["coverage"], 0.0).astype(np.float32) for r in results]
    )
    shifts, wave, _ = phase_shifts(results, corner, anchor)
    merged = select_and_merge(layers, field, shifts, corner, wave)
    merged["gain"] = merge_gain(merged, corner, wave)
    merged["pitch"] = 1.0 / max(float(np.hypot(*wave)), 1e-9) * downsample
    merged["used"] = [results[index]["number"] for index in merged["chosen"]]
    merged["skipped"] = [
        r["number"] for index, r in enumerate(results) if index not in merged["chosen"]
    ]

    ranked = np.argsort(-np.asarray(merged["alone"]))
    comparisons = []

    def add_variant(kind: str, members: np.ndarray, variant_shifts: np.ndarray, note: str):
        candidate = select_and_merge(
            layers[members], field[members], variant_shifts[members], corner, wave
        )
        chosen_local = np.asarray(candidate["chosen"], np.int32)
        result = {
            "kind": kind,
            "golden": candidate["golden"],
            "seen": candidate["seen"],
            "score": candidate["trail"][-1],
            "note": note,
            "chosen": np.asarray(members, np.int32)[chosen_local],
            "chosen_shifts": candidate["shifts"][chosen_local],
            "chosen_weights": candidate["weights"][chosen_local],
            "quorum": candidate["quorum"],
            # Every member's registration, not only the kept ones, so the merged
            # result can be placed back into any impression's own frame.
            "members": np.asarray(members, np.int32),
            "shifts": candidate["shifts"],
        }
        comparisons.append(result)
        return result

    if len(ranked):
        best = int(ranked[0])
        add_variant(
            "best", np.array([best]), shifts,
            f"background from best single stamp #{results[best]['number']} only",
        )
    if len(ranked) >= 2:
        second = int(ranked[1])
        add_variant(
            "second", np.array([second]), shifts,
            f"background from second-best single stamp #{results[second]['number']} only",
        )
        pair = ranked[:2]
        add_variant(
            "pair", pair, shifts,
            f"background phase-merged from best two stamps "
            f"#{results[int(pair[0])]['number']} + #{results[int(pair[1])]['number']}",
        )

    strongest_screen = max(merged["alone"], default=float("nan"))
    resolve_cells = (
        len(results) >= 2
        and (
            cell_min_strength is None
            or (
                np.isfinite(strongest_screen)
                and strongest_screen >= cell_min_strength
            )
        )
    )
    if resolve_cells:
        strongest = int(ranked[0])
        cells = resolve_screen_cells(
            layers, field, shifts, corner, wave, shift_limit / max(downsample, 1), strongest
        )
        all_members = np.arange(len(results))
        anchor_candidate = add_variant(
            "anchor", all_members, cells["anchor_shifts"],
            f"background 2D cells against strongest stamp #{results[strongest]['number']}",
        )
        consensus_candidate = add_variant(
            "consensus", all_members, cells["consensus_shifts"],
            "background global pairwise 2D-cell consensus",
        )
        internal_pitch = 1.0 / max(float(np.hypot(*wave)), 1e-9)
        validation = {
            "anchor": held_out_cell_score(
                layers, field, cells["anchor_shifts"], corner, internal_pitch
            ),
            "consensus": held_out_cell_score(
                layers, field, cells["consensus_shifts"], corner, internal_pitch
            ),
        }
        finite = {key: value for key, value in validation.items() if np.isfinite(value)}
        ordered_validation = sorted(finite, key=finite.get, reverse=True)
        preferred = ordered_validation[0] if ordered_validation else None
        # Near ties are not evidence for choosing one registration branch.
        if (
            len(ordered_validation) > 1
            and finite[ordered_validation[0]] - finite[ordered_validation[1]] < 0.01
        ):
            preferred = None
        if preferred is not None:
            selected = anchor_candidate if preferred == "anchor" else consensus_candidate
            selected["note"] += (
                f"; automatic preferred ({validation[preferred]:+.3f} held-out cell score)"
            )
        merged["cell_validation"] = validation
        merged["preferred_cell_kind"] = preferred
        merged["cell_resolution"] = cells
    else:
        merged["cell_resolution"] = None
        merged["cell_validation"] = {}
        merged["preferred_cell_kind"] = None
    merged["comparisons"] = comparisons
    return merged


def derotate(results: list[dict], corner, start: np.ndarray, window: int, step: int) -> tuple:
    """Rotate each impression so its screen ruling lies at a common bearing.

    The two passes come off separate plates in separate press runs, so they can
    sit rotated with respect to each other.  They cannot sit scaled with respect
    to each other, both inks being on the same fibres, which is why only the
    bearing is corrected here and the pitch is left alone.

    Each impression's own wave vector is fitted by a plane through the screen
    phase read in a grid of windows; a single FFT bin is far too coarse for the
    tenth of a degree that matters.
    """
    fitted = []
    for result in results:
        fitted.append(fit_wave(result["density"], result["field"], start, window, step))
    bearings = np.array([entry["bearing"] for entry in fitted])
    reference = float(np.median(bearings))

    height, width = results[0]["field"].shape
    centre = (width / 2.0, height / 2.0)
    turned = []
    for result, entry in zip(results, fitted):
        angle = reference - entry["bearing"]
        matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)
        spun = dict(result)
        spun["density"] = cv2.warpAffine(
            result["density"], matrix, (width, height),
            flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderValue=0,
        )
        spun["coverage"] = cv2.warpAffine(
            result["coverage"], matrix, (width, height),
            flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderValue=0,
        )
        spun["field"] = cv2.warpAffine(
            result["field"].astype(np.uint8), matrix, (width, height),
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP, borderValue=0,
        ) > 0
        if "readable" in result:
            spun["readable"] = cv2.warpAffine(
                result["readable"].astype(np.uint8), matrix, (width, height),
                flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP, borderValue=0,
            ) > 0
        spun["rotation"] = angle
        turned.append(spun)
    return turned, bearings, reference


def edge_band(layer: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """A narrow band along the design's own edges.

    Sharpness has to be judged here rather than over the whole stamp.  A
    postmark carries far more edge energy than the design does, so a golden
    that still contains postmarks scores higher everywhere else simply for
    being dirtier.
    """
    gradient = cv2.magnitude(
        cv2.Sobel(layer, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(layer, cv2.CV_32F, 0, 1, ksize=3)
    )
    inside = gradient[mask]
    if not len(inside):
        return np.zeros(layer.shape, bool)
    strong = (gradient >= np.percentile(inside, 92)).astype(np.uint8)
    return (cv2.dilate(strong, np.ones((3, 3), np.uint8)) > 0) & mask


def sharpness(image: np.ndarray, mask: np.ndarray) -> float:
    """Mean gradient magnitude of the luminance over the given band."""
    grey = cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    gradient = cv2.magnitude(
        cv2.Sobel(grey.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(grey.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3),
    )
    return float(np.mean(gradient[mask])) if mask.any() else float("nan")


def layer_sharpness(layer: np.ndarray, mask: np.ndarray) -> float:
    """Mean gradient magnitude of a scalar density layer."""
    gradient = cv2.magnitude(
        cv2.Sobel(layer.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(layer.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3),
    )
    return float(np.mean(gradient[mask])) if mask.any() else float("nan")


def layer_structure_correlation(
    first: np.ndarray,
    second: np.ndarray,
    mask: np.ndarray,
    sigma: float,
) -> float:
    """Correlation of plate geometry while ignoring density and fine texture.

    Two colour rays can describe one physical ink when fading or the scanner
    changes its saturation.  Their scalar layers then reproduce the same
    outlines even though the pixels assigned to the two rays need not overlap.
    Comparing blurred gradient magnitudes catches that shared geometry.  A
    genuinely separate plate may occupy the same broad part of the stamp, but
    its individual strokes and therefore its gradient field remain distinct.
    """
    gradients = []
    for layer in (first, second):
        layer = np.clip(np.asarray(layer, np.float32), 0.0, None)
        gradient = cv2.magnitude(
            cv2.Sobel(layer, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(layer, cv2.CV_32F, 0, 1, ksize=3),
        )
        gradients.append(cv2.GaussianBlur(gradient, (0, 0), max(sigma, 0.5)))
    valid = mask & np.isfinite(gradients[0]) & np.isfinite(gradients[1])
    if int(valid.sum()) < 1000:
        return float("nan")
    one = gradients[0][valid].astype(np.float64)
    two = gradients[1][valid].astype(np.float64)
    one -= one.mean()
    two -= two.mean()
    denominator = math.sqrt(float(one @ one) * float(two @ two))
    return float((one @ two) / denominator) if denominator > 1e-12 else float("nan")


def achromatic_layer(
    image: np.ndarray,
    paper: np.ndarray,
    valid: np.ndarray,
    ray: np.ndarray,
) -> np.ndarray:
    """Extract one solid near-neutral printing ink as a discrete mask.

    A plate does not intentionally print a family of grey inks.  Intermediate
    values in a scan are predominantly paper, wear, focus and resampling, and
    retaining them is precisely how a faint coloured ground leaks into the
    black component.  Identify the dense class with Otsu's split, require its
    optical-density direction to remain close to the black ink ray, and emit
    one representative density for every accepted pixel.
    """
    density, magnitude = optical_density(image, paper)
    ray = np.asarray(ray, np.float64)
    ray /= max(float(np.linalg.norm(ray)), 1e-12)
    amount = np.maximum(density @ ray, 0.0)
    cosine = np.divide(
        density @ ray,
        magnitude,
        out=np.zeros_like(amount),
        where=magnitude > 1e-6,
    )

    values = amount[valid & np.isfinite(amount)]
    if len(values) < 100:
        return np.zeros(amount.shape, np.float32)
    cap = max(float(np.percentile(values, 99.5)), 1e-6)
    scaled = np.clip(np.rint(amount * (255.0 / cap)), 0, 255).astype(np.uint8)
    threshold_u8, _ = cv2.threshold(
        scaled[valid].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    # A weak near-neutral wash is still background, even if its hue happens to
    # sit close to black.  The floor deliberately sacrifices worn grey pixels:
    # this branch models the one solid printing ink, not scan-derived tones.
    threshold = max(0.60, float(threshold_u8) * cap / 255.0)
    belongs = (
        valid
        & (amount >= threshold)
        & (cosine >= math.cos(math.radians(8.0)))
    )
    if int(belongs.sum()) < 100:
        return np.zeros(amount.shape, np.float32)
    strength = float(np.median(amount[belongs]))
    return np.where(belongs, strength, 0.0).astype(np.float32)


def select_achromatic_golden(
    refined: np.ndarray,
    baseline: np.ndarray,
    seen: np.ndarray,
    covered: np.ndarray,
    design_support: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, bool]:
    """Keep refinement only when it beats the ordinary averaged black layer.

    The comparison is deliberately made in the same filtered density domain,
    over repeatable design structure when that support is available.  Besides
    edge energy, require nearly all of the baseline's ink mass; otherwise a
    sparse but contrasty fragment could win by looking artificially sharp.
    """
    # Do not intersect this with ``seen``.  Missing candidate coverage is a
    # defect to score, not an excuse to remove those pixels from the test.
    scoring = covered.copy()
    if design_support is not None:
        scoring &= design_support
    band = edge_band(baseline, scoring)
    before = layer_sharpness(baseline, band)
    after = layer_sharpness(refined, band)
    baseline_mass = float(np.sum(baseline[scoring]))
    refined_mass = float(np.sum(refined[scoring]))
    keeps_coverage = float(np.mean(seen[scoring])) >= 0.97 if scoring.any() else False
    keeps_ink = refined_mass >= 0.97 * baseline_mass and keeps_coverage
    improved = bool(
        np.isfinite(after)
        and np.isfinite(before)
        and after >= before
        and keeps_ink
    )
    if improved:
        return refined, seen, band, before, after, True
    return baseline, covered.copy(), band, before, after, False


def half_set_agreement(
    layers: np.ndarray,
    masks: np.ndarray,
    shifts: np.ndarray,
    quorum: int,
    mask: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """Correlation of two goldens built from disjoint halves of the collection.

    Structure the two halves do not agree on was produced by the averaging, not
    recovered by it.  Each half is built the way the pass itself is built, so
    the check answers a question about the pipeline rather than about a merge
    rule the pipeline does not use.
    """
    order = np.arange(len(layers))
    first, second = order[0::2], order[1::2]
    if len(first) < 2 or len(second) < 2:
        return float("nan")
    share = max(2, quorum // 2)
    if weights is None:
        one, seen_one = robust_golden(layers[first], masks[first], shifts[first], share)
        two, seen_two = robust_golden(layers[second], masks[second], shifts[second], share)
    else:
        one, seen_one = screen_golden(
            layers[first], masks[first], shifts[first], weights[first], share
        )
        two, seen_two = screen_golden(
            layers[second], masks[second], shifts[second], weights[second], share
        )
    common = mask & seen_one & seen_two
    if common.sum() < 1000:
        return float("nan")
    a = one[common] - one[common].mean()
    b = two[common] - two[common].mean()
    denominator = math.sqrt(float(a @ a) * float(b @ b))
    return float((a @ b) / denominator) if denominator > 1e-12 else float("nan")


def contrast(values: np.ndarray, mask: np.ndarray, gamma: float = 0.9) -> np.ndarray:
    inside = values[mask & (values > 0)]
    if not len(inside):
        return np.full(values.shape, 255, np.uint8)
    low, high = np.percentile(inside, [5, 99.0])
    if high - low < 1e-6:
        return np.where(mask & (values > 0), 0, 255).astype(np.uint8)
    scaled = np.clip((values - low) / max(high - low, 1e-6), 0, 1) ** gamma
    return (255.0 * (1.0 - scaled)).astype(np.uint8)


def reconstruct(
    layers: list[np.ndarray], rays: list[np.ndarray], paper: np.ndarray
) -> np.ndarray:
    density = np.zeros((*layers[0].shape, 3), np.float64)
    for layer, ray in zip(layers, rays):
        density += np.clip(layer, 0.0, None)[:, :, None] * ray[None, None, :]
    linear = to_linear(np.asarray(paper, np.float64))[None, None, :] * 10.0 ** (-density)
    return to_srgb(linear).astype(np.uint8)


def remove_background_projection(
    along_main: np.ndarray,
    background_coverage: np.ndarray,
    main_ray: np.ndarray,
    background_ray: np.ndarray,
) -> np.ndarray:
    """Remove the recovered background contribution already inside main."""
    projection = float(np.asarray(background_ray) @ np.asarray(main_ray))
    return (
        np.asarray(along_main, np.float32)
        - np.asarray(background_coverage, np.float32) * projection
    ).astype(np.float32)


@dataclass
class Settings:
    """How the rebuild is steered.  The paths default to the names the
    production pipeline writes beside the scan."""

    perf: Path | None = None
    design: Path | None = None
    output: Path | None = None
    downsample: int = 1
    iterations: int = 3
    shift_limit: float = 12.0
    screen_tilt: float = 12.0
    no_background: bool = False  # for a single-pass denomination
    validate: bool = False  # build each pass from two halves and compare them
    window: int = 256  # phase window, px
    step: int = 128  # phase window stride, px
    panel_width: int = 760
    inks: int = 4  # most colour plates to look for
    # When supplied, bypass the staged achromatic/chromatic detector and find
    # exactly this many primary inks together in full optical-density space.
    # A value of one still enters the existing main-plus-background route.
    ink_count: int | None = None
    separation: float = ink_palette.SEPARATION_DEG
    spin_limit: float = 1.0  # most a colour plate may be turned, deg
    # Zero probes every coloured plate automatically; a one-based value forces
    # that plate through the screen route for diagnostics.
    screen_ink: int = 0


@dataclass
class Collection:
    """One scan's impressions, already brought onto a common canvas."""

    scan: Path
    images: np.ndarray
    valid: np.ndarray
    numbers: list[int]
    papers: np.ndarray
    paper: np.ndarray
    baseline: np.ndarray
    covered: np.ndarray
    corner: tuple[int, int]
    # The design sidecar, kept for where each stamp sits in the original scan.
    design: dict = field(default_factory=dict)


def unified_palette(collection: Collection, count: int) -> dict:
    """Find a fixed number of primary inks without a hue/neutral split.

    Optical density is the useful full colour space here: pigment absorption
    adds linearly, while normalising its three-channel vector separates ink
    direction from how heavily it printed.  Initial modes come from the
    registered median, where moving cancellations have mostly disappeared.
    Per-impression rays are then re-estimated inside each mode's spatial
    support, preserving batch, water-damage and scanner colour differences.
    """
    density, magnitude = optical_density(collection.baseline, collection.paper)
    values = magnitude[collection.covered]
    if len(values) < 500:
        raise RuntimeError("Too little registered coverage to name printing inks")
    # Do not use a top-density percentile here.  On the V issue the neutral
    # engraved plate is much denser than the coloured ground, so such a cut
    # removes the second requested ink before clustering even starts.
    density_floor = 0.06
    selected = collection.covered & (magnitude >= density_floor)
    if int(selected.sum()) < max(500, 100 * count):
        raise RuntimeError("Too few dense median pixels to name printing inks")
    directions = density[selected] / np.maximum(magnitude[selected][:, None], 1e-12)
    rays, shares = ink_palette.fixed_modes(directions, count)

    assignment = np.argmax(directions @ rays.T, axis=1)
    cores: list[np.ndarray] = []
    supports: list[np.ndarray] = []
    for ink in range(count):
        assigned = np.zeros(collection.covered.shape, bool)
        assigned[selected] = assignment == ink
        amount = np.maximum(density @ rays[ink], 0.0)
        values = amount[assigned]
        if len(values) >= 100:
            cap = max(float(np.percentile(values, 99.5)), 1e-6)
            scaled = np.clip(
                np.rint(values * (255.0 / cap)), 0, 255
            ).astype(np.uint8)
            threshold_u8, _ = cv2.threshold(
                scaled.reshape(-1, 1),
                0,
                255,
                cv2.THRESH_BINARY | cv2.THRESH_OTSU,
            )
            threshold = max(0.05, float(threshold_u8) * cap / 255.0)
            core = assigned & (amount >= threshold)
        else:
            core = assigned
        cores.append(core)
        # A near-neutral ink is colour-indistinguishable from cancellation and
        # therefore needs a tight repeatable spatial support.  This is a common
        # classifier with a nuisance ambiguity, not a separate achromatic
        # reconstruction: its density, local colour and registration continue
        # through exactly the same path as every other ink.
        neutral_angle = math.degrees(math.acos(float(np.clip(
            rays[ink] @ background_ink.NEUTRAL, -1.0, 1.0
        ))))
        width = 5 if neutral_angle <= 12.0 else 25
        support = cv2.dilate(
            core.astype(np.uint8), np.ones((width, width), np.uint8)
        ) > 0
        supports.append(support)

    local_rays = np.full(
        (len(collection.images), count, 3), np.nan, np.float64
    )
    cosine = math.cos(math.radians(60.0))
    for stamp, (image, valid, paper) in enumerate(
        zip(collection.images, collection.valid, collection.papers)
    ):
        stamp_density, stamp_magnitude = optical_density(image, paper)
        stamp_directions = np.divide(
            stamp_density,
            stamp_magnitude[:, :, None],
            out=np.zeros_like(stamp_density),
            where=stamp_magnitude[:, :, None] > 1e-8,
        )
        for ink, core in enumerate(cores):
            # Estimate colour only where the registered median assigned this
            # ink itself.  The dilated support is useful for geometry, but its
            # surrounding paper lets a dark cancellation drag a weak ink back
            # toward grey (most visibly the green oval on 35K).
            area = valid & core
            if int(area.sum()) < 100:
                continue
            local_floor = max(float(np.percentile(stamp_magnitude[area], 45)), 0.05)
            evidence = area & (stamp_magnitude >= local_floor)
            if int(evidence.sum()) < 50:
                continue
            candidates, _ = ink_palette.modes(
                stamp_directions[evidence],
                separation=8.0,
                limit=4,
                share=0.02,
            )
            allowed = (candidates @ rays[ink]) >= cosine
            if not allowed.any():
                continue
            # A postmark or paper cast may be the largest local mode.  The
            # registered median already supplied the plate's direction, so use
            # it as a weak identity prior and take the nearby mode that best
            # continues that plate, without forcing every stamp to one hue.
            ray = candidates[allowed][
                int(np.argmax(candidates[allowed] @ rays[ink]))
            ]
            local_rays[stamp, ink] = ray

    if count > 1:
        gram = np.clip(rays @ rays.T, -1.0, 1.0)
        angles = np.degrees(np.arccos(gram))
        apart = float(np.min(angles[~np.eye(count, dtype=bool)]))
    else:
        apart = float("inf")
    observed = np.isfinite(local_rays).all(axis=2).sum(axis=0)
    distances = np.stack([
        cv2.distanceTransform((~core).astype(np.uint8), cv2.DIST_L2, 5)
        for core in cores
    ])
    ownership = np.argmin(distances, axis=0).astype(np.int16)
    print(
        f"  unified fixed-count palette in optical-density space: {count} ink(s); "
        f"local estimates per ink {observed.tolist()}"
    )
    return {
        "rays": rays,
        "shares": shares / max(float(shares.sum()), 1e-12),
        "count": count,
        "separation_deg": apart,
        "achromatic": np.zeros(count, bool),
        "achromatic_added": np.zeros(count, bool),
        "achromatic_angle_deg": ink_palette.ACHROMATIC_DEG,
        "achromatic_supports": [None] * count,
        "achromatic_thresholds": [None] * count,
        "ink_supports": supports,
        "ink_cores": cores,
        "ink_distances": distances,
        "ink_ownership": ownership,
        "local_ink_rays": local_rays,
        "unified": True,
    }


def averaged_achromatic_pass(collection: Collection, palette: dict) -> list[Pass]:
    """Safe result for a lone neutral ink with no separable coloured screen.

    Without a colour distinction a black cancellation cannot yet be assigned
    per impression reliably.  Returning the colour-isolated ordinary
    golden is conservative, complete, and gives later work a well-defined
    baseline instead of applying the sparse repeatability mask as artwork.
    """
    ray = np.asarray(palette["rays"][0], np.float64)
    golden = achromatic_layer(
        collection.baseline, collection.paper, collection.covered, ray
    )
    support = palette.get("achromatic_supports", [None])[0]
    scoring = collection.covered if support is None else collection.covered & support
    return [
        Pass(
            name="achromatic pass",
            golden=golden,
            seen=collection.covered.copy(),
            ray=ray,
            density_note="achromatic pass, density; averaged baseline",
            ink_note=f"achromatic pass in its own ink; averaged baseline; "
                     f"{bgr_hex(to_srgb(to_linear(collection.paper) * 10.0 ** (-ray)))}",
            band=edge_band(golden, scoring),
        )
    ]


def achromatic_support(
    collection: Collection, rays: np.ndarray, index: int
) -> tuple[np.ndarray, float] | None:
    """Repeatable dark structure belonging to one near-neutral ink ray.

    Colour alone cannot distinguish a black printing plate from a black or grey
    cancellation.  The plate repeats in the registered median while the
    cancellation does not.  Build the support by applying the same solid-ink
    classifier to that median.  The small dilation admits the residual
    per-impression plate movement that this module subsequently measures.
    """
    layer = achromatic_layer(
        collection.baseline, collection.paper, collection.covered, rays[index]
    )
    threshold = 0.60
    core = collection.covered & (layer > 0)
    minimum = max(500, int(round(0.001 * collection.covered.sum())))
    if int(core.sum()) < minimum:
        return None

    # A repeatable plate has strokes, not merely enough isolated dark samples
    # to cross the area threshold.  On badly cancelled scans the registered
    # median can contain hundreds of one-pixel neutral remnants spread over
    # the stamp; counting their total promoted that noise to a printing ink.
    # Close only one-pixel gaps, then require at least one coherent piece of
    # design.  The threshold scales with the usable canvas but retains small
    # lettering on the synthetic and low-resolution cases.
    coherent = cv2.morphologyEx(
        core.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    )
    component_count, _, statistics, _ = cv2.connectedComponentsWithStats(
        coherent, connectivity=8
    )
    largest = (
        int(statistics[1:, cv2.CC_STAT_AREA].max())
        if component_count > 1
        else 0
    )
    coherent_minimum = max(64, int(round(0.0001 * collection.covered.sum())))
    if largest < coherent_minimum:
        print(
            "  rejected achromatic candidate: "
            f"{int(core.sum())} dark pixels form {component_count - 1} fragments; "
            f"largest {largest}, need {coherent_minimum}"
        )
        return None
    support = cv2.dilate(core.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    return support, threshold


def validate_achromatic_palette(collection: Collection, palette: dict) -> dict:
    """Attach spatial supports and discard unsupported added neutral modes.

    Existing colour modes are never removed.  A mode that the ordinary palette
    already found may merely lose its achromatic label; only a separately added
    neutral candidate is removed when no repeatable dark design supports it.
    """
    rays = palette["rays"]
    flags = np.asarray(
        palette.get("achromatic", np.zeros(len(rays), bool)), bool
    ).copy()
    added = np.asarray(
        palette.get("achromatic_added", np.zeros(len(rays), bool)), bool
    )
    supports: list[np.ndarray | None] = [None] * len(rays)
    thresholds: list[float | None] = [None] * len(rays)
    keep = np.ones(len(rays), bool)
    for index in np.flatnonzero(flags):
        found = achromatic_support(collection, rays, int(index))
        if found is None:
            if added[index]:
                keep[index] = False
            else:
                flags[index] = False
            continue
        supports[index], thresholds[index] = found

    if not keep.all():
        rays = rays[keep]
        shares = palette["shares"][keep]
        flags = flags[keep]
        added = added[keep]
        supports = [value for value, retain in zip(supports, keep) if retain]
        thresholds = [value for value, retain in zip(thresholds, keep) if retain]
        palette = dict(palette)
        palette.update(
            rays=rays,
            shares=shares,
            count=len(rays),
            achromatic=flags,
            achromatic_added=added,
        )
    else:
        palette = dict(palette)
        palette["achromatic"] = flags
    palette["achromatic_supports"] = supports
    palette["achromatic_thresholds"] = thresholds
    return palette


def local_colour_estimates(
    collection: Collection, remaining: np.ndarray, separation: float
) -> tuple[np.ndarray, np.ndarray]:
    """Dominant residual ink direction and evidence count per impression.

    Ink batches and scan colour balance can move the observed ray substantially
    between stamps.  Estimate it locally; the spatial layer, not RGB distance,
    is what will later decide whether these observations are the same plate.
    Missing estimates are represented by NaNs.
    """
    rays = np.full((len(collection.images), 3), np.nan, np.float64)
    evidence = np.zeros(len(collection.images), np.float64)
    for index, (image, valid, paper) in enumerate(
        zip(collection.images, remaining, collection.papers)
    ):
        if valid.sum() < 500:
            continue
        density, magnitude = optical_density(image, paper)
        chroma = magnitude - 0.999 * (density @ background_ink.NEUTRAL)
        density_cut = max(float(np.percentile(magnitude[valid], 70)), 0.08)
        selected = (
            valid
            & (magnitude > density_cut)
            & (chroma > np.percentile(chroma[valid], 70))
        )
        if selected.sum() < 200:
            continue
        directions = density[selected] / magnitude[selected][:, None]
        local, _ = ink_palette.modes(directions, separation, 1, ink_palette.MINOR_SHARE)
        rays[index] = local[0]
        evidence[index] = float(selected.sum())
    return rays, evidence


def achromatic_exclusion(
    collection: Collection,
    rays: np.ndarray,
    supports: list[np.ndarray | None],
) -> np.ndarray:
    """Pixels safely attributable to validated achromatic printing plates.

    The raw colour classifier also responds to neutral cancellation, dirt and
    scan defects.  A validated support is the spatial evidence that separates
    those from a printing plate, so it must be applied *before* the exclusion
    is dilated.  Otherwise a tiny valid plate support can still cause the whole
    raw neutral mask to erase genuine coloured ink from the second stage.
    """
    excluded = np.zeros_like(collection.valid)
    kernel = np.ones((7, 7), np.uint8)
    for ray, support in zip(rays, supports):
        for index, (image, paper, valid) in enumerate(
            zip(collection.images, collection.papers, collection.valid)
        ):
            mask = valid & (achromatic_layer(image, paper, valid, ray) > 0)
            if support is not None:
                mask &= support
            excluded[index] |= cv2.dilate(mask.astype(np.uint8), kernel) > 0
    return excluded


def persistent_colour_modes(
    collection: Collection,
    remaining: np.ndarray,
    rays: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Keep colour modes that recur as an ink within individual stamps.

    A pooled colour cloud can turn a batch or fading difference into another
    mode: a few stamps then carry almost all observations of that "ink".  A
    separate press run instead contributes a measurable cluster within most
    usable impressions, even when it covers only a small part of each stamp.
    Requiring a strict majority is proportional rather than a fixed stamp
    count, so one- and two-stamp collections still receive a best effort.
    """
    rays = np.asarray(rays, np.float64)
    shares = np.zeros((len(collection.images), len(rays)), np.float64)
    usable = np.zeros(len(collection.images), bool)
    for index, (image, valid, paper) in enumerate(
        zip(collection.images, remaining, collection.papers)
    ):
        try:
            directions = ink_palette.ink_directions(
                image[None, ...], valid[None, ...], paper[None, ...]
            )
        except RuntimeError:
            continue
        assigned = np.argmax(directions @ rays.T, axis=1)
        shares[index] = np.bincount(
            assigned, minlength=len(rays)
        ) / len(assigned)
        usable[index] = True

    usable_count = int(usable.sum())
    required = max(1, usable_count // 2 + 1)
    presence = np.sum(
        shares[usable] >= ink_palette.MINOR_SHARE, axis=0
    ) if usable_count else np.zeros(len(rays), int)
    keep = presence >= required
    if not keep.any() and len(rays):
        keep[int(np.argmax(presence))] = True
    return keep, presence, required


def staged_palette(collection: Collection, args: Settings) -> dict:
    """Find neutral inks first, then name colours on what they did not print."""
    first = None
    try:
        neutral_directions = ink_palette.achromatic_directions(
            collection.images, collection.valid, collection.papers
        )
        neutral_ray = np.median(neutral_directions, axis=0)
        neutral_ray /= max(float(np.linalg.norm(neutral_ray)), 1e-12)
        first = validate_achromatic_palette(
            collection,
            {
                "rays": neutral_ray[None, :],
                "shares": np.ones(1),
                "count": 1,
                "separation_deg": float("inf"),
                "achromatic": np.ones(1, bool),
                "achromatic_added": np.ones(1, bool),
                "achromatic_angle_deg": ink_palette.ACHROMATIC_DEG,
            },
        )
    except RuntimeError:
        pass

    if first is None or not np.asarray(first["achromatic"], bool).any():
        directions = ink_palette.ink_directions(
            collection.images, collection.valid, collection.papers
        )
        result = ink_palette.find_coloured_palette(
            directions, args.inks, args.separation
        )
        result["achromatic_supports"] = [None] * result["count"]
        result["achromatic_thresholds"] = [None] * result["count"]
        return result

    neutral = np.asarray(first["achromatic"], bool)

    neutral_rays = first["rays"][neutral]
    neutral_shares = first["shares"][neutral]
    neutral_supports = [
        support for support, keep in zip(first["achromatic_supports"], neutral) if keep
    ]
    neutral_thresholds = [
        threshold for threshold, keep in zip(first["achromatic_thresholds"], neutral) if keep
    ]

    excluded = achromatic_exclusion(
        collection, neutral_rays, neutral_supports
    )
    remaining = collection.valid & ~excluded

    try:
        directions = ink_palette.ink_directions(
            collection.images, remaining, collection.papers
        )
        colours = ink_palette.find_coloured_palette(
            directions,
            max(1, args.inks - len(neutral_rays)),
            args.separation,
        )
    except RuntimeError:
        print("  colour search after achromatic extraction: no usable colour")
        result = dict(first)
        result.update(
            rays=neutral_rays,
            shares=neutral_shares,
            count=len(neutral_rays),
            achromatic=np.ones(len(neutral_rays), bool),
            achromatic_added=np.zeros(len(neutral_rays), bool),
            achromatic_supports=neutral_supports,
            achromatic_thresholds=neutral_thresholds,
        )
        return result

    if colours["count"] > 1:
        persistent, presence, required = persistent_colour_modes(
            collection, remaining, colours["rays"]
        )
        if not persistent.all():
            descriptions = ", ".join(
                f"{bgr_hex(ink_palette.ink_colour(collection.paper, ray))} "
                f"in {int(found)} usable stamps"
                for ray, found in zip(colours["rays"][~persistent], presence[~persistent])
            )
            print(
                "  rejected pooled colour variant(s) without per-stamp recurrence: "
                f"{descriptions}; need {required} stamps"
            )
            colours = dict(colours)
            colours["rays"] = colours["rays"][persistent]
            colours["shares"] = colours["shares"][persistent]
            colours["count"] = int(persistent.sum())

    colour_rays = colours["rays"]
    local_rays = None
    variants = colour_rays
    variant_shares = colours["shares"]
    if colours["count"] == 1:
        # Preserve the per-impression hue adaptation used for inks whose batch
        # or water damage varies from stamp to stamp.  It is valid only when
        # the residual has one spatial colour component; with two actual inks
        # a single local median would merge them again.
        estimates, colour_evidence = local_colour_estimates(
            collection, remaining, args.separation
        )
        has_colour = np.isfinite(estimates).all(axis=1)
        if has_colour.any():
            observed = estimates[has_colour]
            colour_ray = np.median(observed, axis=0)
            colour_ray /= max(float(np.linalg.norm(colour_ray)), 1e-12)
            colour_rays = colour_ray[None, :]
            local_rays = estimates
            variants, variant_shares = ink_palette.modes(
                observed, separation=12.0, limit=args.inks, share=0.12
            )
            colour_weight = float(colour_evidence.sum())
        else:
            colour_weight = float(len(directions) * 9)
    else:
        has_colour = np.ones(len(collection.images), bool)
        colour_weight = float(len(directions) * 9)

    rays = np.vstack([neutral_rays, colour_rays])
    colour_weights = np.asarray(colours["shares"], np.float64)
    colour_weights /= max(float(colour_weights.sum()), 1e-12)
    shares = np.concatenate(
        [np.full(len(neutral_rays), float(excluded.sum()) / len(neutral_rays)),
         colour_weight * colour_weights]
    )
    shares /= max(float(shares.sum()), 1e-12)
    flags = np.concatenate(
        [np.ones(len(neutral_rays), bool), np.zeros(len(colour_rays), bool)]
    )
    gram = np.clip(rays @ rays.T, -1.0, 1.0)
    angles = np.degrees(np.arccos(gram))
    apart = float(np.min(angles[~np.eye(len(rays), dtype=bool)]))
    print(
        f"  staged palette: excluded {excluded.sum() / max(collection.valid.sum(), 1):.1%} "
        f"as achromatic before colour search"
    )
    colour_text = ", ".join(
        f"{bgr_hex(ink_palette.ink_colour(collection.paper, ray))} {share:.0%}"
        for ray, share in zip(variants, variant_shares)
    )
    if colours["count"] == 1 and local_rays is not None:
        print(
            f"    one spatial colour component, {int(has_colour.sum())} local estimates; "
            f"observed variants: {colour_text}"
        )
    else:
        print(
            f"    {colours['count']} spatial colour components after neutral extraction: "
            f"{colour_text}"
        )
    result = {
        "rays": rays,
        "shares": shares,
        "count": len(rays),
        "separation_deg": apart,
        "achromatic": flags,
        "achromatic_added": np.zeros(len(rays), bool),
        "achromatic_angle_deg": ink_palette.ACHROMATIC_DEG,
        "achromatic_supports": neutral_supports + [None] * len(colour_rays),
        "achromatic_thresholds": neutral_thresholds + [None] * len(colour_rays),
        "colour_variants": variants,
        "colour_variant_shares": variant_shares,
    }
    if local_rays is not None:
        result["local_colour_rays"] = local_rays
    return reject_duplicate_achromatic_plate(collection, result, args)


def reject_duplicate_achromatic_plate(
    collection: Collection, palette: dict, args: Settings
) -> dict:
    """Collapse a false grey plate that is the coloured plate in disguise.

    The achromatic-first stage is intentionally permissive so that a real
    black plate is not missed.  On a worn one-colour issue, however, the dense
    desaturated centres of strokes can pass that test while their coloured
    fringes become a second palette entry.  Before accepting two press runs,
    build conservative unregistered layers and ask whether their *geometry*
    is the same.  The colour angle is only a gate; the spatial correlation is
    the deciding evidence.
    """
    neutral = np.asarray(
        palette.get("achromatic", np.zeros(palette["count"], bool)), bool
    )
    coloured = np.flatnonzero(~neutral)
    achromatic = np.flatnonzero(neutral)
    if len(achromatic) != 1 or len(coloured) != 1:
        return palette

    grey_index, colour_index = int(achromatic[0]), int(coloured[0])
    cosine = float(np.clip(
        palette["rays"][grey_index] @ palette["rays"][colour_index], -1.0, 1.0
    ))
    angle = math.degrees(math.acos(cosine))
    if angle > DUPLICATE_INK_ANGLE_DEG:
        return palette

    layers, readable = colour_layers(collection, palette)
    grey = achromatic_layer(
        collection.baseline,
        collection.paper,
        collection.covered,
        palette["rays"][grey_index],
    )
    colour, _ = robust_golden(
        layers[colour_index],
        readable[colour_index],
        np.zeros((len(collection.images), 2), np.float32),
        collection_quorum(len(collection.images)),
        floor=1,
    )
    correlation = layer_structure_correlation(
        grey,
        colour,
        collection.covered,
        sigma=8.0 / max(args.downsample, 1),
    )
    print(
        f"    achromatic/colour structure check: {angle:.1f} deg apart, "
        f"gradient correlation {correlation:+.3f}"
    )
    if not np.isfinite(correlation) or correlation < DUPLICATE_ACHROMATIC_CORRELATION:
        return palette

    # Re-estimate the one physical ink on all chromatic observations.  This
    # restores the colour suppressed by the false grey split and sends the
    # resulting single plate through the ordinary residual-background path.
    directions = ink_palette.ink_directions(
        collection.images, collection.valid, collection.papers
    )
    collapsed = ink_palette.find_coloured_palette(
        directions, args.inks, args.separation
    )
    collapsed["achromatic_supports"] = [None] * collapsed["count"]
    collapsed["achromatic_thresholds"] = [None] * collapsed["count"]
    collapsed["duplicate_achromatic_rejected"] = {
        "angle_deg": angle,
        "structure_correlation": correlation,
    }
    print(
        "    rejecting the achromatic split: both rays reproduce one plate; "
        f"re-estimated {collapsed['count']} coloured ink(s) from the full residual"
    )
    return collapsed


@dataclass
class Pass:
    """What one plate contributed, and how well it was recovered."""

    name: str
    golden: np.ndarray
    seen: np.ndarray
    ray: np.ndarray
    density_note: str
    ink_note: str
    # Short, stable label for the accepted reconstruction column.
    primary_note: str = ""
    # Edges the rebuild is scored over.  Carried rather than derived from the
    # golden above, so that an option which changes how many impressions merge
    # is still measured over the same pixels as the run it is compared with.
    band: np.ndarray | None = None
    # Optional diagnostics built from alternative impression selections.  Each
    # position is one panel column; plates without an alternative are left at
    # their normal golden in that comparison.
    comparison_goldens: list[np.ndarray] = field(default_factory=list)
    comparison_seen: list[np.ndarray] = field(default_factory=list)
    comparison_rays: list[np.ndarray] = field(default_factory=list)
    comparison_notes: list[str] = field(default_factory=list)


def load_collection(scan: Path, args: Settings) -> Collection:
    """Read the scan and its sidecars, and bring every impression onto one canvas."""
    perf_path = args.perf or scan.with_name(f"data_{scan.stem}_perf.json")
    design_path = args.design or scan.with_name(f"data_{scan.stem}_design.json")
    perf = json.loads(perf_path.read_text(encoding="utf-8"))
    design = json.loads(design_path.read_text(encoding="utf-8"))
    images, valid, numbers = aligned_collection(
        scan, perf, design, args.downsample, minimum=1
    )
    print(f"Printing passes, from {len(numbers)} stamps on a "
          f"{images.shape[2]}x{images.shape[1]} canvas:")

    papers = estimate_papers(
        images, valid, paper_ring(design, images.shape[1:3], args.downsample)
    )
    # One screen window for the whole collection, so phases are comparable.
    _, magnitude = optical_density(images[0], papers[0])
    corner = screen_window(
        valid[0] & (magnitude < np.percentile(magnitude[valid[0]], 60)), images.shape[1:3]
    )
    return Collection(
        scan=scan,
        images=images,
        valid=valid,
        numbers=list(numbers),
        papers=papers,
        paper=np.median(papers, axis=0),
        baseline=np.clip(masked_colour_median(images, valid), 0, 255).astype(np.uint8),
        covered=valid.sum(axis=0) >= collection_quorum(len(images)),
        corner=corner,
        design=design,
    )


def screen_passes(collection: Collection, palette: dict, args: Settings) -> list[Pass]:
    """The main design, and the ruled screen printed behind it.

    One press run laid both down, so the design is registered by correlation
    and the screen by its own phase, against the same impressions.
    """
    scan, paper, covered, corner = (
        collection.scan, collection.paper, collection.covered, collection.corner
    )
    main_is_achromatic = bool(palette.get("achromatic", [False])[0])
    main_name = "achromatic pass" if main_is_achromatic else "main pass"
    results, kept = [], []
    for number, image, mask, stamp_paper in zip(
        collection.numbers, collection.images, collection.valid, collection.papers
    ):
        try:
            # The impression's own ink ray is allowed to be useless here; the
            # coverage is re-read below against the one the collection agrees on.
            result = separate(image, stamp_paper, mask, args.screen_tilt, corner,
                              min_separability=0.0)
        except RuntimeError as error:
            print(f"  stamp {number:2d}: {error}")
            continue
        result["number"] = number
        result["valid"] = mask
        result["image"] = image
        result["paper"] = stamp_paper
        results.append(result)
        kept.append(number)
    if not results:
        raise RuntimeError("No impression carries separable screen evidence")
    print(f"  separated {len(results)} impressions: {kept}")

    main_ray = np.median(np.stack([r["main_ray"] for r in results]), axis=0)
    main_ray /= np.linalg.norm(main_ray)
    back_ray = np.median(np.stack([r["ink_ray"] for r in results]), axis=0)
    back_ray /= np.linalg.norm(back_ray)

    # One colour basis for the whole collection.  Both inks belong to the issue,
    # so they are estimated once from every impression and each impression is
    # then projected through the same frame, which puts every background layer
    # on one scale and one sign.
    own = np.array([r["separability"] for r in results])
    basis = background_ink.orthonormal_frame(main_ray)
    for result in results:
        result["coverage"], separability = background_ink.coverage_against(
            basis, result["density"], back_ray
        )
        result["separability"] = separability
    # The overall sign is a gauge, not a fact; what matters is that no
    # impression disagrees with the rest about which way the screen runs.
    against = int((np.sign(own) != np.sign(separability)).sum())
    print(
        f"  background ink read in one shared basis: separability "
        f"[{own.min():+.3f}..{own.max():+.3f}] per impression -> {separability:+.3f} "
        f"for all, reversing {against} of {len(own)} that read inverted on their own"
    )
    # Is there a second ink here at all?  Everything downstream divides by the
    # separability, so what has to be settled first is whether the background's
    # direction has been resolved.  Its tilt out of the main ink's plane is the
    # signal; the scatter between the impressions' own estimates of that
    # direction is the error on it.  Where the tilt is the smaller of the two,
    # the collection has not measured a second ink, and dividing by it
    # manufactures one out of noise: on 5K a 3.5 degree tilt against 4.1 of
    # scatter produced a background that claimed 65 per cent of the main plate's
    # density, which no background tint can be.  Hue separation is no guide --
    # 7K is the next closest pair at 7.8 degrees and resolves cleanly, because
    # its difference from the main ink lies out of the plane rather than toward
    # neutral, where the main ray and the postmark grey already account for it.
    tilt = math.degrees(math.asin(min(1.0, abs(separability))))
    scatter = float(np.mean(np.degrees(np.arccos(np.clip(
        np.stack([r["ink_ray"] for r in results]) @ back_ray, -1.0, 1.0
    )))))
    separable = background_direction_resolved(tilt, scatter)
    ratio = tilt / max(scatter, 1e-9)
    print(
        f"  background ink stands {tilt:.2f} deg out of the main ink's plane "
        f"against {scatter:.2f} deg of scatter in that direction "
        f"(ratio {ratio:.3f}, required "
        f"{1.0 - BACKGROUND_DIRECTION_RELATIVE_TOLERANCE:.3f}): "
        + ("a resolved second ink" if separable
           else "not resolved, reading the sheet as a single ink")
    )
    publishing = separable and not args.no_background
    pitch = float(np.median([r["pitch"] for r in results]))
    hue = float(np.degrees(np.arccos(np.clip(main_ray @ back_ray, -1, 1))))
    print(
        f"  main ink {bgr_hex(to_srgb(to_linear(paper) * 10.0 ** (-main_ray * 1.0)))}, "
        f"background ink {bgr_hex(to_srgb(to_linear(paper) * 10.0 ** (-back_ray * 0.35)))}, "
        f"{hue:.1f} deg apart; screen pitch {pitch * args.downsample:.2f} px"
    )

    quorum = collection_quorum(len(results))
    registration_readable = np.stack([r["valid"] & ~r["postmark"] for r in results])
    design_support = palette.get("achromatic_supports", [None])[0] \
        if main_is_achromatic else None
    if main_is_achromatic:
        main_layers = np.stack(
            [achromatic_layer(r["image"], r["paper"], mask, main_ray)
             for r, mask in zip(results, registration_readable)]
        )
        registration_masks = registration_readable
        registration_layers = main_layers
        # An achromatic plate is read as a solid mask, not as a projection of the
        # whole density, so there is no background component inside it to remove.
        conserved_layers = main_layers
        if design_support is not None:
            # The support is only an alignment focus.  It is intentionally not
            # applied to the recovered layer, where it used to erase legitimate
            # grey tones and small details.
            focus = cv2.dilate(
                design_support.astype(np.uint8), np.ones((9, 9), np.uint8)
            ) > 0
            registration_masks = registration_readable & focus[None, :, :]
            registration_layers = np.where(registration_masks, main_layers, 0.0)
    else:
        # Two versions of the same plate, for two different jobs.
        #
        # ``along_main`` is the whole density along the main ray, so it still
        # carries the background ray's projection onto that ray.  Rebuilding the
        # recovered background on top of it would count that component twice, so
        # conservation requires taking it out -- but each impression's coverage
        # is its out-of-plane residual divided by the separability, and where the
        # two inks are close that divisor amplifies the coverage's own noise.  On
        # 5K, separability 0.061 with the rays 7.2 degrees apart, subtracting it
        # put sixteen times the background's noise into the layer, which is
        # ruinous for registration: matching on it dropped the plate's ECC from
        # 0.910 to 0.822 and spread an essentially rigid plate's vertical travel
        # from 0.33 to 2.35 px.
        #
        # So the plate is matched on the uncorrected layer, where the evidence is
        # cleanest, and the golden is then merged from the corrected ones using
        # those same offsets.  The correction stays per impression, in the frame
        # where the double counting actually happened, and the merge averages
        # down the noise it brings.  Correcting the merged goldens instead is
        # quieter still but wrong in geometry, the two plates being registered in
        # different frames: it leaves an inverted imprint of the background in
        # the main golden, which showed up as the duplicate-structure correlation
        # rising above even doing nothing at all.
        main_layers = np.stack(
            [np.where(mask, np.maximum(r["along_main"], 0.0), 0.0).astype(np.float32)
             for r, mask in zip(results, registration_readable)]
        )
        conserved_layers = np.stack(
            [np.where(
                mask,
                np.maximum(
                    remove_background_projection(
                        r["along_main"], r["coverage"], main_ray, back_ray
                    ),
                    0.0,
                ),
                0.0,
             ).astype(np.float32)
             for r, mask in zip(results, registration_readable)]
        )
        registration_layers = main_layers
        registration_masks = registration_readable
    main_shifts, main_scores = register_pattern(
        registration_layers,
        registration_masks,
        quorum,
        args.iterations,
        args.shift_limit,
    )
    readable = registration_readable
    comparison_band = None
    selection_note = ""
    candidate_score = baseline_score = float("nan")
    # Conserve density only against a background that is actually going to be
    # rebuilt.  With no background pass to lay that ink back down, subtracting
    # it would simply lose it.
    golden_main, seen_main = robust_golden(
        conserved_layers if publishing else main_layers, readable, main_shifts, quorum
    )
    if main_is_achromatic:
        baseline_main = achromatic_layer(
            collection.baseline, collection.paper, covered, main_ray
        )
        (
            golden_main,
            seen_main,
            comparison_band,
            baseline_score,
            candidate_score,
            improved,
        ) = select_achromatic_golden(
            golden_main, baseline_main, seen_main, covered, design_support
        )
        selection_note = "independently registered" if improved else "averaged fallback"
        print(
            f"  {main_name}: filtered edge energy {baseline_score:.4f} -> "
            f"{candidate_score:.4f}; using {selection_note}"
        )
    thin = covered & ~seen_main
    print(
        f"  {main_name}: shift span {np.ptp(main_shifts[:, 0]) * args.downsample:.2f}"
        f"x{np.ptp(main_shifts[:, 1]) * args.downsample:.2f} px, "
        f"median ECC {np.nanmedian(main_scores):.3f}, "
        f"{thin.sum() / max(covered.sum(), 1):.1%} on fewer than {quorum} impressions"
    )

    if not separable and not selection_note:
        selection_note = "one ink only, no separable background"
    passes = [
        Pass(
            name=main_name,
            golden=golden_main,
            seen=seen_main,
            ray=main_ray,
            primary_note="single ink" if not separable else "",
            density_note=f"{main_name}, density"
                         f"{'; ' + selection_note if selection_note else ''}; "
                         f"ECC {np.nanmedian(main_scores):.2f}",
            ink_note=f"{main_name} in its own ink"
                     f"{'; ' + selection_note if selection_note else ''}; "
                     f"{bgr_hex(to_srgb(to_linear(paper) * 10.0 ** (-main_ray)))}",
            band=comparison_band if comparison_band is not None else
                 edge_band(golden_main, covered & seen_main),
        )
    ]
    if not publishing:
        return passes

    anchor = int(np.argmax([r["snr"] * r["coherence"] for r in results]))
    # Bring every screen to a common bearing before matching phase.  The passes
    # come off separate plates, so their relative rotation is a free parameter;
    # left uncorrected it stops impressions from phasing up.
    turned, bearings, reference = derotate(
        results, corner, results[anchor]["wave"], args.window, args.step
    )
    back = background_golden(
        turned, corner, anchor, args.downsample, args.shift_limit
    )
    consensus_candidate = next(
        (
            candidate for candidate in back["comparisons"]
            if candidate["kind"] == "consensus"
        ),
        None,
    )
    accepted_back = consensus_candidate if consensus_candidate is not None else back
    # Refit the bearings on the rotated layers.  If the sign of the correction
    # were wrong this spread would double rather than close.
    settled = np.array(
        [fit_wave(r["density"], r["field"], back["wave"], args.window, args.step)["bearing"]
         for r in turned]
    )
    trail, used, skipped = back["trail"], back["used"], back["skipped"]
    print(
        f"  background pass: anchored on stamp {results[anchor]['number']}, "
        f"shared screen {back['pitch']:.3f} px, merge gain {back['gain']:.2f}x, "
        f"shift span {np.ptp(back['shifts'][:, 0]) * args.downsample:.2f}"
        f"x{np.ptp(back['shifts'][:, 1]) * args.downsample:.2f} px"
    )
    print(
        f"    bearing brought to {reference:+.3f} deg: spread "
        f"{np.ptp(bearings):.3f} -> {np.ptp(settled):.3f} deg "
        f"(sd {np.std(bearings):.3f} -> {np.std(settled):.3f})"
    )
    report_selection(back, results, len(results))
    report_cell_variants(back)

    duplicate_correlation = layer_structure_correlation(
        golden_main,
        accepted_back["golden"],
        covered & seen_main,
        sigma=8.0 / max(args.downsample, 1),
    )
    duplicate_background = bool(
        hue <= DUPLICATE_INK_ANGLE_DEG
        and np.isfinite(duplicate_correlation)
        and duplicate_correlation >= DUPLICATE_BACKGROUND_CORRELATION
    )
    print(
        f"    main/background structure check: {hue:.1f} deg apart, "
        f"gradient correlation {duplicate_correlation:+.3f}; "
        + ("possible duplicate flagged" if duplicate_background else "distinct enough")
    )

    selection = {
        "scan": scan.name,
        "screen_pitch_px": back["pitch"],
        "reference_bearing_deg": reference,
        "bearing_deg": {str(r["number"]): float(b) for r, b in zip(results, bearings)},
        "best_single": trail[0],
        "merged": trail[-1],
        "used": used,
        "skipped": skipped,
        "duplicate_layer_check": {
            "hue_angle_deg": hue,
            "structure_correlation": duplicate_correlation,
            "flagged": duplicate_background,
            "rejected": False,
        },
        "strength_alone": {
            str(result["number"]): back["alone"][index]
            for index, result in enumerate(results)
        },
        "variants": {
            entry["kind"]: float(entry["score"])
            for entry in back["comparisons"]
        },
        "accepted_variant": "consensus" if consensus_candidate is not None
                            else "legacy_phase_only",
    }
    cells = back.get("cell_resolution")
    if cells is not None:
        selection["cell_resolution"] = {
            "pair_count": cells["pair_count"],
            "elapsed_seconds": cells["elapsed_seconds"],
            "held_out_scores": back.get("cell_validation", {}),
            "preferred": back.get("preferred_cell_kind"),
            "per_stamp": {
                str(result["number"]): {
                    "anchor_residual": cells["anchor_residuals"][index].tolist(),
                    "consensus_residual": cells["consensus_residuals"][index].tolist(),
                    **cells["confidence"][index],
                }
                for index, result in enumerate(results)
            },
        }
    sidecar = scan.with_name(f"data_{scan.stem}_background_selection.json")
    sidecar.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(f"  Wrote {sidecar}")

    if args.validate:
        print("  half-set agreement (structure the halves independently agree on):")
        print(f"    {main_name:<15} "
              f"{half_set_agreement(conserved_layers, readable, main_shifts, quorum, covered):+.3f}")
        report_halves(back, covered)

    by_kind = {candidate["kind"]: candidate for candidate in back["comparisons"]}
    main_variants: list[tuple[np.ndarray, str]] = []
    # Fixed, comparable columns, in a stable order: the single best impression
    # first as the honest "one good stamp" reference, then one column per fix
    # under test.  Both plates contribute to every column, so each one is a
    # complete rebuild rather than a change to the background alone.
    diagnostic_candidates = []
    if "best" in by_kind:
        diagnostic_candidates.append(by_kind["best"])
        main_variants.append((passes[0].golden, f"{main_name} as accepted"))

    shape = registration_readable.shape[1:]
    chosen_back = np.asarray(accepted_back["chosen"], np.int32)
    chosen_field = np.stack([turned[index]["field"] for index in chosen_back])

    def conserved_against(backgrounds: list[np.ndarray]) -> np.ndarray:
        return np.stack(
            [np.where(
                mask,
                np.maximum(
                    remove_background_projection(
                        r["along_main"], estimate, main_ray, back_ray
                    ),
                    0.0,
                ),
                0.0,
             ).astype(np.float32)
             for r, mask, estimate in zip(results, registration_readable, backgrounds)]
        )

    # Fix 1.  Subtract the background the collection agreed on, placed back into
    # each impression's own frame, rather than that impression's own coverage.
    # The two are the same quantity, but the merged one has been through ten to
    # twenty impressions, so it carries a fraction of the noise that dividing by
    # the separability put into any single one of them.
    placed = np.array(back["shifts"], np.float32).copy()
    placed[accepted_back["members"]] = accepted_back["shifts"]
    registered = [
        place_in_impression(accepted_back["golden"], placed[index], spun["rotation"], shape)
        for index, spun in enumerate(turned)
    ]
    registered_main, _ = robust_golden(
        conserved_against(registered), readable, main_shifts, quorum
    )
    diagnostic_candidates.append({
        "golden": accepted_back["golden"],
        "seen": accepted_back["seen"],
        "note": "background as accepted",
    })
    main_variants.append(
        (registered_main, f"{main_name} conserved against the registered background")
    )

    # Fix 2.  Keep each impression's own coverage, but first discount the
    # frequencies the impressions do not agree on, which is where the division
    # by the separability left noise rather than ink.
    gain = repeatability_gain(
        np.stack([turned[index]["coverage"] for index in chosen_back]),
        chosen_field,
        accepted_back["chosen_shifts"],
        accepted_back["seen"],
    )
    agreed = [
        apply_gain(r["coverage"], mask, gain)
        for r, mask in zip(results, registration_readable)
    ]
    agreed_golden, agreed_seen = screen_golden(
        np.stack([apply_gain(turned[index]["coverage"], turned[index]["field"], gain)
                  for index in chosen_back]),
        chosen_field,
        accepted_back["chosen_shifts"],
        accepted_back["chosen_weights"],
        accepted_back["quorum"],
    )
    agreed_main, _ = robust_golden(
        conserved_against(agreed), readable, main_shifts, quorum
    )
    diagnostic_candidates.append({
        "golden": agreed_golden,
        "seen": agreed_seen,
        "note": "background kept only where impressions agree",
    })
    main_variants.append(
        (agreed_main, f"{main_name} conserved against the agreed background")
    )

    # What each option actually charges the main plate, so the panels can be
    # read against a number.  Sampled on a quarter grid; this is a diagnostic.
    def debit(corrected: np.ndarray) -> float:
        sparse, whole = corrected[:, ::4, ::4], main_layers[:, ::4, ::4]
        seen = registration_readable[:, ::4, ::4]
        if not seen.any():
            return float("nan")
        return float(
            np.percentile(np.abs(whole - sparse)[seen], 90)
            / max(np.percentile(np.abs(whole)[seen], 90), 1e-9)
        )

    print(
        f"    conservation debits the main plate "
        f"{debit(conserved_layers):.1%} of its density as accepted, "
        f"{debit(conserved_against(registered)):.1%} against the registered background, "
        f"{debit(conserved_against(agreed)):.1%} against the agreed one"
    )

    passes[0].comparison_goldens = [golden for golden, _ in main_variants]
    passes[0].comparison_seen = [seen_main for _ in main_variants]
    passes[0].comparison_rays = [main_ray.copy() for _ in main_variants]
    passes[0].comparison_notes = [note for _, note in main_variants]
    passes.append(
        Pass(
            name="background pass",
            golden=accepted_back["golden"],
            seen=accepted_back["seen"],
            ray=back_ray,
            primary_note="consensus" if consensus_candidate is not None else "legacy",
            density_note=f"background pass, density; screen {pitch * args.downsample:.2f} px "
                         f"at {reference:+.2f} deg; "
                         + ("consensus registration" if consensus_candidate is not None
                            else f"legacy merge {','.join(str(n) for n in used)} "
                                 f"of {len(results)}"),
            ink_note=f"background in its own ink, {hue:.0f} deg from main; "
                     + ("global pairwise cell consensus"
                        if consensus_candidate is not None
                        else f"strength {trail[-1]:.1f} vs {trail[0]:.1f} single"),
            comparison_goldens=[entry["golden"] for entry in diagnostic_candidates],
            comparison_seen=[entry["seen"] for entry in diagnostic_candidates],
            comparison_rays=[back_ray.copy() for _ in diagnostic_candidates],
            comparison_notes=[entry["note"] for entry in diagnostic_candidates],
        )
    )
    return passes


def report_selection(back: dict, results: list[dict], total: int) -> None:
    trail = back["trail"]
    print(
        f"    screen strength: best single {max(back['alone']):.2f}, "
        f"all {total} merged {back['all_merged']:.2f}, "
        f"merge of the {len(back['chosen'])} kept {trail[-1]:.2f}"
    )
    for index, result in enumerate(results):
        print(
            f"    stamp {result['number']:2d}: alone {back['alone'][index]:6.2f}  "
            + ("kept" if index in back["chosen"] else "dropped")
        )
    print(f"    used     {back['used']}")
    print(f"    skipped  {back['skipped']}")


def report_cell_variants(back: dict) -> None:
    """Report cell-sensitive candidates beside the phase-only score."""
    cells = back.get("cell_resolution")
    if cells is None:
        return
    scores = {
        candidate["kind"]: candidate["score"]
        for candidate in back.get("comparisons", [])
        if candidate["kind"] in ("anchor", "consensus")
    }
    print(
        f"    2D cell resolver: {cells['pair_count']} pairs in "
        f"{cells['elapsed_seconds']:.2f} s; "
        f"strongest-anchor {scores.get('anchor', float('nan')):.2f}, "
        f"global consensus {scores.get('consensus', float('nan')):.2f}"
    )
    validation = back.get("cell_validation", {})
    if validation:
        preferred = back.get("preferred_cell_kind") or "none (near tie)"
        print(
            "      held-out cell agreement: "
            f"anchor {validation.get('anchor', float('nan')):+.3f}, "
            f"consensus {validation.get('consensus', float('nan')):+.3f}; "
            f"preferred {preferred}"
        )


def report_halves(back: dict, covered: np.ndarray) -> None:
    """Whether two disjoint halves of the kept impressions agree on the screen."""
    chosen = back["chosen"]
    layers, field, shifts = back["layers"][chosen], back["field"][chosen], back["shifts"][chosen]
    agreement = half_set_agreement(layers, field, shifts, back["quorum"], covered)
    control = half_set_agreement(
        layers, field, np.zeros_like(shifts), back["quorum"], covered
    )
    if math.isnan(agreement):
        print(f"    background pass: {len(chosen)} impressions kept, too few to split")
        return
    print(f"    background pass {agreement:+.3f} (unaligned control {control:+.3f})")
    if agreement < 0.2:
        print(
            "    The background golden is not reproduced by independent halves.\n"
            "    Read that pass off single impressions with background_ink.py."
        )


def unified_colour_layers(
    collection: Collection, palette: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Split fixed-count inks without treating neutral colour as another pass.

    Every requested ink competes in the same three-dimensional density space.
    Neutral is offered only as a nuisance cancellation direction when none of
    the requested inks is itself near-neutral; otherwise moving cancellations
    are left for the collection-wide robust merge to remove.
    """
    rays = np.asarray(palette["rays"], np.float64)
    local = palette.get("local_ink_rays")
    supports = palette.get("ink_supports", [None] * len(rays))
    ownership = palette.get("ink_ownership")
    count, shape = len(rays), collection.images.shape[1:3]
    layers = np.zeros((count, len(collection.images), *shape), np.float32)
    readable = np.zeros((count, len(collection.images), *shape), bool)
    has_neutral_ink = bool(
        np.any(rays @ background_ink.NEUTRAL >= math.cos(math.radians(12.0)))
    )
    for index, (image, valid, paper) in enumerate(
        zip(collection.images, collection.valid, collection.papers)
    ):
        stamp_rays = rays.copy()
        if local is not None:
            for ink in range(count):
                if np.isfinite(local[index, ink]).all():
                    stamp_rays[ink] = local[index, ink]
        # Keep plate identity tied to the collection-wide centres.  A badly
        # faded coloured ink can move close to the neutral ink on one stamp;
        # letting that local estimate redefine the decision boundary makes the
        # two plates exchange pixels and turns cancellations into artwork.  The
        # local ray still measures density and renders that stamp's own colour.
        columns = (
            rays
            if has_neutral_ink
            else np.vstack([rays, background_ink.NEUTRAL])
        )
        density, magnitude = optical_density(image, paper)
        directions = np.divide(
            density,
            magnitude[:, :, None],
            out=np.zeros_like(density),
            where=magnitude[:, :, None] > 1e-8,
        )
        flat = density.reshape(-1, 3)
        amounts = flat @ columns.T
        residual = np.linalg.norm(
            flat[:, None, :] - amounts[:, :, None] * columns[None, :, :],
            axis=2,
        )
        residual[amounts < 0.0] = np.inf
        assigned = np.argmin(residual, axis=1).reshape(shape)
        nuisance = (
            np.zeros(shape, bool)
            if has_neutral_ink
            else assigned == count
        )
        base = valid & ~nuisance
        for ink in range(count):
            similarity = directions @ stamp_rays[ink]
            if ownership is None:
                identity = assigned == ink
            else:
                identity = ownership == ink
            ink_mask = (
                base
                & identity
                & (magnitude >= 0.05)
                & (similarity >= math.cos(math.radians(50.0)))
            )
            if supports[ink] is not None:
                ink_mask &= supports[ink]
            projection = np.maximum(density @ stamp_rays[ink], 0.0)
            layers[ink, index] = np.where(ink_mask, projection, 0.0)
            readable[ink, index] = base
    return layers, readable


def colour_layers(
    collection: Collection, palette: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Split every impression between the inks, and say where it is readable.

    Achromatic inks are extracted as solid masks first.  Their pixels are then
    removed before the remaining density is assigned between coloured inks;
    the colour model therefore cannot leak back into the black component.
    """
    if palette.get("unified"):
        return unified_colour_layers(collection, palette)

    rays = palette["rays"]
    count, shape = len(rays), collection.images.shape[1:3]
    layers = np.zeros((count, len(collection.images), *shape), np.float32)
    readable = np.zeros((count, len(collection.images), *shape), bool)
    neutral = np.asarray(
        palette.get("achromatic", np.zeros(count, bool)), bool
    )
    supports = palette.get("achromatic_supports", [None] * count)
    coloured_indices = np.flatnonzero(~neutral)
    local_colour_rays = palette.get("local_colour_rays")
    for index, (image, mask, paper) in enumerate(
        zip(collection.images, collection.valid, collection.papers)
    ):
        density, _ = optical_density(image, paper)
        excluded = np.zeros(shape, bool)
        for ink in np.flatnonzero(neutral):
            layer = achromatic_layer(image, paper, mask, rays[ink])
            ink_mask = mask & (layer > 0)
            if supports[ink] is not None:
                ink_mask &= supports[ink]
            excluded |= cv2.dilate(
                ink_mask.astype(np.uint8), np.ones((7, 7), np.uint8)
            ) > 0
            readable[ink, index] = ink_mask
            layers[ink, index] = np.where(ink_mask, layer, 0.0)

        remaining = mask & ~excluded
        if not len(coloured_indices) or remaining.sum() < 200:
            continue
        colour_rays = rays[coloured_indices]
        if local_colour_rays is not None:
            local_ray = local_colour_rays[index]
            if not np.isfinite(local_ray).all():
                continue
            # The staged local-colour path currently represents one spatial
            # residual component whose observed ink ray may vary by stamp.
            colour_rays = local_ray[None, :]
        amounts = unmix(density, colour_rays)
        base = remaining & ~ink_palette.postmark_mask(amounts, remaining)
        for local, ink in enumerate(coloured_indices):
            readable[ink, index] = base
            layers[ink, index] = np.where(base, amounts[local], 0.0)
    return layers, readable


def shared_screen_corner(fields: list[np.ndarray]) -> tuple[int, int]:
    """Place the common phase window where the collection has most evidence."""
    size = background_ink.SCREEN_WINDOW
    support = np.stack(fields).sum(axis=0).astype(np.float32)
    height, width = support.shape
    if height < size or width < size:
        raise RuntimeError("The aligned canvas is smaller than the screen window")
    integral = cv2.integral(support, sdepth=cv2.CV_64F)
    scores = (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )
    y0, x0 = np.unravel_index(int(np.argmax(scores)), scores.shape)
    return int(y0), int(x0)


def screen_from_colour(
    collection: Collection,
    layers: np.ndarray,
    readable: np.ndarray,
    ray: np.ndarray,
    args: Settings,
    indices: np.ndarray | None = None,
    cell_min_strength: float | None = None,
) -> dict:
    """Read one colour plate as a ruled screen and merge it on its phase.

    The engraved ground of a plate is periodic where its design is not, so the
    phase of that ruling locates the impression far more finely than its
    outline does.  This hands the colour to the machinery the screened
    single-colour stamps already use, unchanged.
    """
    entries = []
    if indices is None:
        indices = np.arange(len(layers))
    indices = np.asarray(indices, np.int32)
    if len(indices) != len(layers):
        raise ValueError("screen colour indices must match the selected layers")
    for index, source_index in enumerate(indices):
        number = collection.numbers[int(source_index)]
        coverage = layers[index]
        # The solid lines of the design are held out along with the postmark,
        # leaving the engraved ground.  Where the plate laid down nothing at all
        # the coverage is a measured zero, not a gap, so those pixels stay in:
        # holding them out as well would starve the windows the phase is read
        # over, since one plate covers only part of the design.
        present = readable[index] & (coverage > 0.0)
        if present.sum() < 15000:
            continue
        heavy = cv2.dilate(
            (coverage > np.percentile(coverage[present], 85)).astype(np.uint8),
            np.ones((5, 5), np.uint8),
        ) > 0
        field = readable[index] & ~heavy
        density = coverage[:, :, None] * ray[None, None, :]
        try:
            screen = background_ink.screen_ray(
                density, field, (7.5, 10.5), args.screen_tilt
            )
        except RuntimeError as error:
            print(f"    stamp {number:2d}: {error}")
            continue
        entries.append(
            {**screen, "number": number, "source_index": int(source_index),
             "density": density, "coverage": coverage,
             "field": field, "readable": readable[index],
             "valid": collection.valid[int(source_index)]}
        )
    if not entries:
        raise RuntimeError("No impression carries a readable ruling in that ink")

    # The collection corner is chosen for the full design/background route.
    # A partial colour plate can occupy a very different region, so locate one
    # shared phase window from this plate's own clear-field masks.
    screen_corner = shared_screen_corner([entry["field"] for entry in entries])

    anchor = int(np.argmax([e["snr"] * e["coherence"] for e in entries]))
    start = entries[anchor]["wave"]

    # A sparse colour plate may not fill the 256 px windows that work well for
    # an all-over background.  Use the largest geometry window that leaves at
    # least two measurable impressions, retaining the best single-impression
    # attempt as a last-resort result for genuinely tiny collections.
    geometry_options: list[tuple[int, int]] = []
    smallest = min(96, args.window)
    for factor in (1.0, 0.75, 0.5, 0.375):
        size = max(smallest, int(round(args.window * factor / 16.0)) * 16)
        size = min(size, min(collection.images.shape[1:3]))
        step = min(args.step, max(32, size // 2))
        option = (size, step)
        if option not in geometry_options:
            geometry_options.append(option)

    best_entries: list[dict] = []
    best_errors: dict[int, RuntimeError] = {}
    geometry_window, geometry_step = geometry_options[-1]
    best_geometry_score = -1.0
    for size, step in geometry_options:
        fitted_entries, errors = [], {}
        for entry in entries:
            try:
                fit_wave(entry["density"], entry["field"], start, size, step)
            except RuntimeError as error:
                errors[entry["number"]] = error
                continue
            fitted_entries.append(entry)
        # Angular precision rises steeply with the window baseline; balance it
        # against the number of surviving impressions instead of blindly
        # choosing either the largest window or the largest collection.
        geometry_score = len(fitted_entries) * size**3
        if geometry_score > best_geometry_score:
            best_entries, best_errors = fitted_entries, errors
            geometry_window, geometry_step = size, step
            best_geometry_score = geometry_score

    entries = best_entries
    for number, error in best_errors.items():
        print(f"    stamp {number:2d}: cannot fit screen bearing: {error}")
    if not entries:
        raise RuntimeError("No impression carries enough field to fit the screen bearing")
    anchor = int(np.argmax([e["snr"] * e["coherence"] for e in entries]))
    turned, bearings, reference = derotate(
        entries,
        screen_corner,
        entries[anchor]["wave"],
        geometry_window,
        geometry_step,
    )
    merged = background_golden(
        turned,
        screen_corner,
        anchor,
        args.downsample,
        args.shift_limit,
        cell_min_strength=cell_min_strength,
    )
    # The screen field is only a registration target.  Once its transforms are
    # known, apply them to the complete colour plate so strong ornament strokes
    # excluded from phase estimation are not also erased from the result.
    for candidate in merged["comparisons"]:
        chosen = np.asarray(candidate["chosen"], np.int32)
        full_layers = np.stack([turned[index]["coverage"] for index in chosen])
        full_readable = np.stack([turned[index]["readable"] for index in chosen])
        full_golden, full_seen = screen_golden(
            full_layers,
            full_readable,
            candidate["chosen_shifts"],
            candidate["chosen_weights"],
            candidate["quorum"],
        )
        candidate["golden"] = full_golden
        candidate["seen"] = full_seen
    merged["bearings"] = bearings
    merged["reference"] = reference
    merged["entries"] = entries
    merged["turned_entries"] = turned
    merged["anchor"] = anchor
    merged["corner"] = screen_corner
    merged["geometry_window"] = geometry_window
    merged["geometry_step"] = geometry_step
    return merged


def automatic_colour_screen_decision(screen: dict) -> tuple[bool, str]:
    """Decide whether a colour plate is genuinely registered by a ruling.

    A periodic peak by itself is insufficient: a compact solid motif can put a
    weak accidental peak in the allowed pitch band.  Strong periodic evidence
    is useful as a small-collection best effort; otherwise require the resolved
    two-dimensional cells to agree on held-out image structure.
    """
    alone = np.asarray(screen.get("alone", []), np.float64)
    finite = alone[np.isfinite(alone)]
    strength = float(finite.max()) if len(finite) else float("nan")
    if not np.isfinite(strength):
        return False, "no finite screen-strength measurement"
    if strength >= AUTO_COLOUR_SCREEN_STRONG_STRENGTH:
        return True, (
            f"strong periodic evidence {strength:.2f} >= "
            f"{AUTO_COLOUR_SCREEN_STRONG_STRENGTH:.2f}"
        )
    if strength < AUTO_COLOUR_SCREEN_MIN_STRENGTH:
        return False, (
            f"periodic evidence {strength:.2f} < "
            f"{AUTO_COLOUR_SCREEN_MIN_STRENGTH:.2f}"
        )

    held_out = float(
        screen.get("cell_validation", {}).get("consensus", float("nan"))
    )
    if not np.isfinite(held_out):
        return False, "moderate periodic evidence has no held-out cell validation"
    if held_out < AUTO_COLOUR_SCREEN_MIN_HELD_OUT:
        return False, (
            f"held-out cell agreement {held_out:+.3f} < "
            f"{AUTO_COLOUR_SCREEN_MIN_HELD_OUT:+.3f}"
        )
    return True, (
        f"periodic evidence {strength:.2f} with held-out cell agreement "
        f"{held_out:+.3f}"
    )


def colour_passes(collection: Collection, palette: dict, args: Settings) -> list[Pass]:
    """One pass per colour plate, each registered on its own evidence.

    These plates went through the press separately, so each landed with its own
    small shift *and* its own small turn, and nothing ties them together.  Each
    colour is therefore registered against its own running template as a rigid
    motion, not as a shift.
    """
    rays, shares = palette["rays"], palette["shares"]
    neutral = np.asarray(
        palette.get("achromatic", np.zeros(len(rays), bool)), bool
    )
    layers, readable = colour_layers(collection, palette)
    passes = []
    for ink, (ray, share) in enumerate(zip(rays, shares)):
        colour = bgr_hex(ink_palette.ink_colour(collection.paper, ray))
        role = "achromatic ink" if neutral[ink] else "ink"
        name = f"{role} {ink + 1} {colour}"
        if neutral[ink]:
            golden = achromatic_layer(
                collection.baseline, collection.paper, collection.covered, ray
            )
            support = palette.get("achromatic_supports", [None] * len(rays))[ink]
            scoring = collection.covered if support is None else collection.covered & support
            band = edge_band(golden, scoring) if not passes else None
            print(f"  {name}, {share:.0%} of the ink: solid averaged mask")
            passes.append(
                Pass(
                    name=name,
                    golden=golden,
                    seen=collection.covered.copy(),
                    ray=ray,
                    density_note=f"{name}, density; solid averaged mask",
                    ink_note=f"{name} in its own ink, {share:.0%} of the inked area",
                    band=band,
                )
            )
            continue
        selected, preservation, boundary = colour_preservation_selection(
            layers[ink], readable[ink]
        )
        chosen = np.flatnonzero(selected)
        if not len(chosen):
            print(f"  {name}, {share:.0%} of the ink: no readable impression")
            continue
        chosen_numbers = [collection.numbers[index] for index in chosen]
        if boundary is None:
            print(
                f"    colour preservation: no distinct quality break; using "
                f"{len(chosen)} impression(s) {chosen_numbers}"
            )
        else:
            dropped_numbers = [
                collection.numbers[index] for index in np.flatnonzero(~selected)
            ]
            print(
                f"    colour preservation: adaptive boundary {boundary:.2f} of best; "
                f"using {len(chosen)} {chosen_numbers}, excluding {dropped_numbers}"
            )
        print(
            "      relative evidence "
            + ", ".join(
                f"{number}:{preservation[index]:.2f}"
                for index, number in enumerate(collection.numbers)
                if preservation[index] > 0.0
            )
        )

        selected_layers = layers[ink, selected]
        selected_readable = readable[ink, selected]
        quorum = collection_quorum(len(chosen))
        warps, scores = register_pattern(
            selected_layers,
            selected_readable,
            quorum,
            args.iterations,
            args.shift_limit,
            args.spin_limit,
        )
        turns = np.degrees(np.arctan2(warps[:, 1, 0], warps[:, 0, 0]))
        centre = np.array([collection.images.shape[2] / 2.0, collection.images.shape[1] / 2.0])
        travel = warps @ np.array([centre[0], centre[1], 1.0]) - centre
        golden, seen = robust_golden(
            selected_layers,
            selected_readable,
            warps,
            quorum,
            floor=min(2, len(chosen)),
        )
        agreement = half_set_agreement(
            selected_layers, selected_readable, warps, quorum, collection.covered
        )
        if np.isfinite(agreement) and agreement < MIN_COLOUR_AGREEMENT:
            print(
                f"  {name}, {share:.0%} of the ink: rejected; "
                f"half-set agreement {agreement:+.3f}"
            )
            continue
        validation = (
            f"half-set agreement {agreement:+.3f}"
            if np.isfinite(agreement)
            else "best effort; too few selected impressions for a half-set check"
        )

        local_rays = palette.get("local_ink_rays")
        if local_rays is None:
            local_rays = palette.get("local_colour_rays")

        def local_ray(index: int) -> np.ndarray | None:
            if local_rays is None:
                return None
            candidate = (
                local_rays[index, ink]
                if np.asarray(local_rays).ndim == 3
                else local_rays[index]
            )
            if not np.isfinite(candidate).all():
                return None
            result = np.asarray(candidate, np.float64).copy()
            result /= max(float(np.linalg.norm(result)), 1e-12)
            return result

        if local_rays is not None:
            usable_rays = [local_ray(int(index)) for index in chosen]
            usable_rays = [value for value in usable_rays if value is not None]
            if len(usable_rays):
                ray = np.median(np.stack(usable_rays), axis=0)
                ray /= max(float(np.linalg.norm(ray)), 1e-12)
                colour = bgr_hex(ink_palette.ink_colour(collection.paper, ray))
                name = f"{role} {ink + 1} {colour}"

        ranked = np.flatnonzero(preservation > 0.0)
        ranked = ranked[np.argsort(-preservation[ranked])]
        comparison_goldens: list[np.ndarray] = []
        comparison_seen: list[np.ndarray] = []
        comparison_rays: list[np.ndarray] = []
        comparison_notes: list[str] = []
        primary_note = ""

        def observed_ray(index: int) -> np.ndarray:
            result = ray.copy()
            observed = local_ray(index)
            if observed is not None:
                result = observed
            return result

        # The first two columns preserve the observations exactly as extracted:
        # they deliberately use no other colour stamp for density or hue.
        for rank, index in enumerate(ranked[:2], start=1):
            number = collection.numbers[index]
            ordinal = "best" if rank == 1 else "second-best"
            comparison_goldens.append(layers[ink, index].copy())
            comparison_seen.append(readable[ink, index].copy())
            comparison_rays.append(observed_ray(index))
            comparison_notes.append(
                f"colour from {ordinal} single stamp #{number} only"
            )

        # Then show what the same merger does to only those two observations.
        if len(ranked) >= 2:
            pair = ranked[:2]
            pair_layers = layers[ink, pair]
            pair_readable = readable[ink, pair]
            pair_quorum = collection_quorum(len(pair))
            pair_warps, _ = register_pattern(
                pair_layers,
                pair_readable,
                pair_quorum,
                args.iterations,
                args.shift_limit,
                args.spin_limit,
            )
            pair_golden, pair_seen = robust_golden(
                pair_layers, pair_readable, pair_warps, pair_quorum, floor=1
            )
            pair_ray = np.median(
                np.stack([observed_ray(int(index)) for index in pair]), axis=0
            )
            pair_ray /= max(float(np.linalg.norm(pair_ray)), 1e-12)
            pair_numbers = [collection.numbers[int(index)] for index in pair]
            comparison_goldens.append(pair_golden)
            comparison_seen.append(pair_seen)
            comparison_rays.append(pair_ray)
            comparison_notes.append(
                f"colour merged from best two stamps #{pair_numbers[0]} + #{pair_numbers[1]}"
            )
        band = edge_band(golden, collection.covered & seen) if ink == 0 else None
        median_score = (
            float(np.nanmedian(scores)) if np.isfinite(scores).any() else float("nan")
        )
        print(
            f"  {name}, {share:.0%} of the ink: turn sd {np.std(turns):.3f} deg "
            f"(spread {np.ptp(turns):.3f}), centre travel "
            f"{np.ptp(travel[:, 0]) * args.downsample:.2f}"
            f"x{np.ptp(travel[:, 1]) * args.downsample:.2f} px, "
            f"median ECC {median_score:.3f}, {validation}"
        )
        note = f"turn sd {np.std(turns):.3f} deg, ECC {median_score:.2f}"

        forced_screen = args.screen_ink == ink + 1
        automatic_screen = args.screen_ink == 0
        if forced_screen or automatic_screen:
            mode = "forced" if forced_screen else "automatic probe"
            print(f"    reading {name} as a ruled screen as well ({mode}):")
            # Preservation ranks the whole coloured motif for the ordinary
            # ECC merge.  It is intentionally not an admission test for the
            # screen resolver: a worn impression can be poor overall while
            # still exposing a large, clean patch of ruling.  Give that route
            # every measurable impression and let its own screen-strength
            # selection decide which ones contribute.
            try:
                crisped = screen_from_colour(
                    collection,
                    layers[ink],
                    readable[ink],
                    ray,
                    args,
                    np.arange(len(collection.numbers)),
                    cell_min_strength=(
                        None if forced_screen else AUTO_COLOUR_SCREEN_MIN_STRENGTH
                    ),
                )
            except RuntimeError as error:
                print(f"    screen route unavailable; keeping ordinary colour merge: {error}")
                crisped = None

            if crisped is not None:
                settled = np.array(
                    [fit_wave(
                        e["density"],
                        e["field"],
                        crisped["wave"],
                        crisped["geometry_window"],
                        crisped["geometry_step"],
                    )["bearing"] for e in crisped["entries"]]
                )
                print(
                    f"    screen {crisped['pitch']:.3f} px, "
                    f"merge gain {crisped['gain']:.2f}x, "
                    f"bearing brought to {crisped['reference']:+.3f} deg: spread "
                    f"{np.ptp(crisped['bearings']):.3f} -> {np.ptp(settled):.3f} deg; "
                    f"geometry window {crisped['geometry_window']}/"
                    f"{crisped['geometry_step']} px"
                )
                report_selection(crisped, crisped["entries"], len(crisped["entries"]))
                report_cell_variants(crisped)
                accepted, decision = (
                    (True, "explicit --screen-ink override")
                    if forced_screen
                    else automatic_colour_screen_decision(crisped)
                )
                decision_kind = "forced" if forced_screen else "automatic"
                print(
                    f"    {decision_kind} screen decision: "
                    f"{'accepted' if accepted else 'rejected'}; {decision}"
                )

                by_kind = {
                    candidate["kind"]: candidate
                    for candidate in crisped["comparisons"]
                }
                cells = crisped.get("cell_resolution")
                if accepted and cells is not None:
                    for kind in ("anchor", "consensus"):
                        candidate = by_kind.get(kind)
                        if candidate is None:
                            continue
                        try:
                            full_golden, full_seen, best_source = best_preserving_colour_merge(
                                crisped["turned_entries"],
                                cells[f"{kind}_shifts"],
                                chosen,
                                preservation,
                            )
                        except RuntimeError as error:
                            print(f"    {kind} full-colour merge unavailable: {error}")
                            continue
                        candidate["golden"] = full_golden
                        candidate["seen"] = full_seen
                        candidate["note"] += (
                            f"; full colour anchored to preservation-best stamp "
                            f"#{collection.numbers[best_source]}"
                        )
                        print(
                            f"    {kind} full-colour merge preserves stamp "
                            f"#{collection.numbers[best_source]} and fills only its unreadable pixels"
                        )
                consensus = by_kind.get("consensus")
                anchor = by_kind.get("anchor")
                screen_primary = (
                    consensus if consensus is not None else by_kind.get("best")
                )
            else:
                accepted = False
                consensus = anchor = screen_primary = None

            if accepted and screen_primary is not None:
                # Use exactly the same primary/diagnostic layout as the
                # single-colour background route.  Here "legacy" deliberately
                # means the established whole-design ECC result: it is the
                # result this optional screen interpretation is replacing.
                legacy_golden, legacy_seen = golden, seen
                single_goldens = comparison_goldens[:2]
                single_seen = comparison_seen[:2]
                single_rays = comparison_rays[:2]

                golden, seen = screen_primary["golden"], screen_primary["seen"]
                if consensus is not None:
                    primary_note = "consensus"
                    note = (
                        f"screen {crisped['pitch']:.2f} px; "
                        "global pairwise 2D-cell consensus"
                    )
                else:
                    primary_note = "screen best effort"
                    note = f"screen {crisped['pitch']:.2f} px; strongest single impression"
                comparison_goldens = [legacy_golden, *single_goldens]
                comparison_seen = [legacy_seen, *single_seen]
                comparison_rays = [ray.copy(), *single_rays]
                comparison_notes = [
                    "legacy",
                    *(["best"] if len(single_goldens) >= 1 else []),
                    *(["second"] if len(single_goldens) >= 2 else []),
                ]
                if anchor is not None:
                    comparison_goldens.append(anchor["golden"])
                    comparison_seen.append(anchor["seen"])
                    comparison_rays.append(ray.copy())
                    comparison_notes.append("anchor")

        passes.append(
            Pass(
                name=name,
                golden=golden,
                seen=seen,
                ray=ray,
                density_note=f"{name}, density; {note}",
                ink_note=f"{name} in its own ink, {share:.0%} of the inked area",
                band=band,
                primary_note=primary_note,
                comparison_goldens=comparison_goldens,
                comparison_seen=comparison_seen,
                comparison_rays=comparison_rays,
                comparison_notes=comparison_notes,
            )
        )
    return passes


def stamp_ink_colours(
    collection: Collection, passes: list[Pass]
) -> dict[int, list[tuple[float, float, float] | None]]:
    """Each stamp's own rendering of every plate's ink.

    The templates say where a plate printed; the colour is then read from the
    impressions themselves, so a stamp struck in a different shade reports a
    different colour instead of being averaged into the collection's.  Each
    plate's core is taken where it printed heavily and the other plates did not,
    so one circle stands for one ink rather than for an overprint.
    """
    levels = []
    for entry in passes:
        # Measured over the pixels the plate actually printed on, so a minor
        # ink covering a tenth of the sheet still gets a usable threshold.
        printed = entry.seen & collection.covered & (entry.golden > 0)
        if printed.sum() < 400:
            levels.append(None)
            continue
        heavy, light = np.percentile(entry.golden[printed], [80, 40])
        levels.append((printed, float(heavy), float(light)))

    cores = []
    for index, entry in enumerate(passes):
        if levels[index] is None:
            cores.append(None)
            continue
        printed, heavy, _ = levels[index]
        core = printed & (entry.golden >= heavy)
        for other, level in enumerate(levels):
            if other != index and level is not None:
                core &= passes[other].golden <= level[2]
        core = cv2.erode(core.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        cores.append(core if core.sum() >= 150 else None)

    colours: dict[int, list] = {}
    for index, number in enumerate(collection.numbers):
        image = collection.images[index]
        grey = cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        row = []
        for core in cores:
            picked = None if core is None else core & collection.valid[index]
            if picked is None or picked.sum() < 50:
                row.append(None)
                continue
            values, luminance = image[picked], grey[picked]
            # A cancellation only ever darkens, so dropping the darkest quarter
            # leaves a struck impression reporting its ink and not the postmark.
            keep = luminance >= np.percentile(luminance, 25)
            row.append(tuple(float(value) for value in np.median(values[keep], axis=0)))
        colours[number] = row
    return colours


def mark_clean_scan(collection: Collection, passes: list[Pass]) -> Path | None:
    """Write each stamp's number and its own ink colours onto the cleaned scan.

    The cleaned scan is the one image in the set that still has every stamp in
    its original place, which makes it the natural sheet to compare impressions
    on.  Each stamp is marked with its number and one filled circle per plate,
    in the order the passes were recovered.
    """
    clean = collection.scan.with_name(f"{collection.scan.stem}_design_clean.jpg")
    if not clean.exists():
        return None
    image = cv2.imread(str(clean), cv2.IMREAD_COLOR)
    if image is None:
        return None
    anchors = {
        row["stamp"]: np.asarray(row["design_corners_image"], np.float64)
        for row in collection.design.get("stamps", [])
        if row.get("design_corners_image")
    }
    colours = stamp_ink_colours(collection, passes)
    height, width = image.shape[:2]
    radius = max(6, round(max(height, width) / 220))
    thickness = max(1, round(max(height, width) / 1800))
    scale = max(0.5, max(height, width) / 2600)
    ink = (40, 40, 40)
    for number, inks in colours.items():
        corners = anchors.get(number)
        shown = [colour for colour in inks if colour is not None]
        if corners is None or not shown:
            continue
        label = str(number)
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness + 1
        )
        step = round(radius * 2.4)
        x = int(round(corners[:, 0].min())) + radius * 2
        y = int(round(corners[:, 1].min())) + radius * 2 + text_height
        span = text_width + radius * 2 + step * len(shown)
        plate = image.copy()
        cv2.rectangle(
            plate, (x - radius, y - text_height - radius), (x + span, y + radius),
            (255, 255, 255), -1,
        )
        cv2.addWeighted(plate, 0.72, image, 0.28, 0.0, image)
        cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, ink,
                    thickness + 1, cv2.LINE_AA)
        middle = y - text_height // 2
        centre = x + text_width + radius * 2
        for colour in shown:
            spot = tuple(int(round(min(255.0, max(0.0, value)))) for value in colour)
            cv2.circle(image, (centre, middle), radius, spot, -1, cv2.LINE_AA)
            cv2.circle(image, (centre, middle), radius, ink, thickness, cv2.LINE_AA)
            centre += step
    if not cv2.imwrite(str(clean), image):
        raise RuntimeError(f"Cannot write image: {clean}")
    return clean


def write_panel(
    rows: list[list[tuple[np.ndarray, str]]], output: Path, width: int
) -> Path:
    rendered = []
    for row in rows:
        cells = []
        for image, text in row:
            factor = width / image.shape[1]
            resized = cv2.resize(
                image,
                (width, max(1, int(round(image.shape[0] * factor)))),
                interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_CUBIC,
            )
            cells.append(add_label(resized, text))
        tall = max(cell.shape[0] for cell in cells)
        padded = [
            cv2.copyMakeBorder(
                cell, 0, tall - cell.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )
            for cell in cells
        ]
        rendered.append(np.hstack(padded))
    sheet = np.vstack(rendered)
    if not cv2.imwrite(str(output), sheet):
        raise RuntimeError(f"Cannot write image: {output} ({sheet.shape} {sheet.dtype})")
    print(f"Wrote {output}")
    return output


def build(scan: Path, args: Settings) -> Path:
    """Rebuild one scan's golden from its printing passes and write the panel."""
    output = args.output or scan.with_name(f"{scan.stem}_golden_components.png")
    collection = load_collection(scan, args)

    palette = (
        unified_palette(collection, args.ink_count)
        if args.ink_count is not None
        else staged_palette(collection, args)
    )
    print("  " + ink_palette.describe(palette, collection.paper))
    for index, support in enumerate(palette["achromatic_supports"]):
        if support is not None:
            print(
                f"    achromatic ink {index + 1}: {int(support.sum())} repeatable pixels "
                f"above density {palette['achromatic_thresholds'][index]:.3f}"
            )

    # Several colours mean several press runs, each free to have landed turned.
    # One colour means one run, with the ruled ground printed alongside it.
    exactly_neutral = bool(
        palette["count"] == 1
        and palette["achromatic"][0]
        and math.degrees(
            math.acos(float(np.clip(palette["rays"][0] @ background_ink.NEUTRAL, -1, 1)))
        ) < 1.0
    )
    if palette["count"] > 1:
        passes = colour_passes(collection, palette, args)
        if len(passes) == 1:
            retained = int(np.argmax(palette["rays"] @ passes[0].ray))
            single = {
                "rays": palette["rays"][[retained]],
                "shares": np.ones(1),
                "count": 1,
                "separation_deg": float("inf"),
                "achromatic": palette["achromatic"][[retained]],
                "achromatic_added": np.zeros(1, bool),
                "achromatic_angle_deg": palette["achromatic_angle_deg"],
                "achromatic_supports": [palette["achromatic_supports"][retained]],
                "achromatic_thresholds": [palette["achromatic_thresholds"][retained]],
            }
            print("  only one repeatable ink remains; trying the background model")
            if single["achromatic"][0] and math.degrees(
                math.acos(float(np.clip(single["rays"][0] @ background_ink.NEUTRAL, -1, 1)))
            ) < 1.0:
                passes = averaged_achromatic_pass(collection, single)
            else:
                passes = screen_passes(collection, single, args)
        elif not passes:
            raise RuntimeError("No repeatable printing ink survived validation")
    elif exactly_neutral:
        passes = averaged_achromatic_pass(collection, palette)
    else:
        passes = screen_passes(collection, palette, args)

    rebuilt = reconstruct(
        [p.golden for p in passes], [p.ray for p in passes], collection.paper
    )
    band = passes[0].band
    if band is None:
        band = edge_band(passes[0].golden, collection.covered & passes[0].seen)
    before = sharpness(collection.baseline, band)
    after = sharpness(rebuilt, band)
    gain = after / max(before, 1e-6) - 1
    print(
        f"  design edge energy over {band.sum()} edge pixels: "
        f"golden {before:.2f} -> rebuilt {after:.2f} ({gain:+.1%})"
    )

    marked = mark_clean_scan(collection, passes)
    if marked is not None:
        print(
            f"  Marked {len(passes)} ink(s) per stamp on {marked.name}"
        )

    comparison_count = max((len(entry.comparison_goldens) for entry in passes), default=0)
    comparison_rebuilds = []
    comparison_labels = []
    for comparison in range(comparison_count):
        selected_layers = [
            entry.comparison_goldens[comparison]
            if comparison < len(entry.comparison_goldens)
            else entry.golden
            for entry in passes
        ]
        selected_rays = [
            entry.comparison_rays[comparison]
            if comparison < len(entry.comparison_rays)
            else entry.ray
            for entry in passes
        ]
        comparison_rebuilds.append(
            reconstruct(selected_layers, selected_rays, collection.paper)
        )
        notes = [
            entry.comparison_notes[comparison]
            for entry in passes
            if comparison < len(entry.comparison_notes)
        ]
        comparison_labels.append("; ".join(notes))

    primary_notes = [entry.primary_note for entry in passes if entry.primary_note]
    primary_label = "; ".join(primary_notes) if primary_notes else "rebuilt from its passes"

    # The optional third column is a direct diagnostic against the merged
    # colour: it keeps every other pass unchanged and takes colour from only
    # the best-preserved individual impression.
    rows = [
        [
            (collection.baseline,
             f"{scan.stem}: golden as averaged now ({len(collection.numbers)} stamps), "
             f"paper {bgr_hex(collection.paper)}"),
            (rebuilt, f"{primary_label}; design edge energy {gain:+.1%}"),
            *list(zip(comparison_rebuilds, comparison_labels)),
        ]
    ]
    for entry in passes:
        row = [
            (cv2.cvtColor(contrast(entry.golden, entry.seen), cv2.COLOR_GRAY2BGR),
             entry.density_note),
            (render_ink(entry.golden, collection.paper, entry.ray), entry.ink_note),
        ]
        for comparison in range(comparison_count):
            if comparison >= len(entry.comparison_goldens):
                row.append(
                    (render_ink(entry.golden, collection.paper, entry.ray),
                     f"same {entry.name}; unchanged in colour comparison")
                )
            else:
                row.append(
                    (render_ink(entry.comparison_goldens[comparison], collection.paper,
                                entry.comparison_rays[comparison]),
                     entry.comparison_notes[comparison])
                )
        rows.append(row)
    return write_panel(rows, output, args.panel_width)
