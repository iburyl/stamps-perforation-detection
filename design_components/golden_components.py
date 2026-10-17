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
import warnings
from dataclasses import dataclass
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


def background_golden(results: list[dict], corner, anchor: int, downsample: int) -> dict:
    """Phase-register, select and merge the background pass of a collection."""
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
    separation: float = ink_palette.SEPARATION_DEG
    spin_limit: float = 1.0  # most a colour plate may be turned, deg
    screen_ink: int = 0  # which colour, if any, to also read as a screen


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


@dataclass
class Pass:
    """What one plate contributed, and how well it was recovered."""

    name: str
    golden: np.ndarray
    seen: np.ndarray
    ray: np.ndarray
    density_note: str
    ink_note: str
    # Edges the rebuild is scored over.  Carried rather than derived from the
    # golden above, so that an option which changes how many impressions merge
    # is still measured over the same pixels as the run it is compared with.
    band: np.ndarray | None = None


def load_collection(scan: Path, args: Settings) -> Collection:
    """Read the scan and its sidecars, and bring every impression onto one canvas."""
    perf_path = args.perf or scan.with_name(f"data_{scan.stem}_perf.json")
    design_path = args.design or scan.with_name(f"data_{scan.stem}_design.json")
    perf = json.loads(perf_path.read_text(encoding="utf-8"))
    design = json.loads(design_path.read_text(encoding="utf-8"))
    images, valid, numbers = aligned_collection(scan, perf, design, args.downsample)
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
        covered=valid.sum(axis=0) >= max(3, math.ceil(0.6 * len(images))),
        corner=corner,
    )


def screen_passes(collection: Collection, args: Settings) -> list[Pass]:
    """The main design, and the ruled screen printed behind it.

    One press run laid both down, so the design is registered by correlation
    and the screen by its own phase, against the same impressions.
    """
    scan, paper, covered, corner = (
        collection.scan, collection.paper, collection.covered, collection.corner
    )
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
        results.append(result)
        kept.append(number)
    if len(results) < 5:
        raise RuntimeError("Need at least five separable impressions")
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
    pitch = float(np.median([r["pitch"] for r in results]))
    hue = float(np.degrees(np.arccos(np.clip(main_ray @ back_ray, -1, 1))))
    print(
        f"  main ink {bgr_hex(to_srgb(to_linear(paper) * 10.0 ** (-main_ray * 1.0)))}, "
        f"background ink {bgr_hex(to_srgb(to_linear(paper) * 10.0 ** (-back_ray * 0.35)))}, "
        f"{hue:.1f} deg apart; screen pitch {pitch * args.downsample:.2f} px"
    )

    quorum = max(3, math.ceil(0.6 * len(results)))
    readable = np.stack([r["valid"] & ~r["postmark"] for r in results])
    main_layers = np.stack(
        [np.where(r["valid"] & ~r["postmark"], r["along_main"], 0.0).astype(np.float32)
         for r in results]
    )
    main_shifts, main_scores = register_pattern(
        main_layers, readable, quorum, args.iterations, args.shift_limit
    )
    golden_main, seen_main = robust_golden(main_layers, readable, main_shifts, quorum)
    thin = covered & ~seen_main
    print(
        f"  main pass: shift span {np.ptp(main_shifts[:, 0]) * args.downsample:.2f}"
        f"x{np.ptp(main_shifts[:, 1]) * args.downsample:.2f} px, "
        f"median ECC {np.nanmedian(main_scores):.3f}, "
        f"{thin.sum() / max(covered.sum(), 1):.1%} on fewer than {quorum} impressions"
    )

    passes = [
        Pass(
            name="main pass",
            golden=golden_main,
            seen=seen_main,
            ray=main_ray,
            density_note=f"main pass, density; ECC {np.nanmedian(main_scores):.2f}",
            ink_note="main pass in its own ink "
                     f"{bgr_hex(to_srgb(to_linear(paper) * 10.0 ** (-main_ray)))}",
            band=edge_band(golden_main, covered & seen_main),
        )
    ]
    if args.no_background:
        return passes

    anchor = int(np.argmax([r["snr"] * r["coherence"] for r in results]))
    # Bring every screen to a common bearing before matching phase.  The passes
    # come off separate plates, so their relative rotation is a free parameter;
    # left uncorrected it stops impressions from phasing up.
    turned, bearings, reference = derotate(
        results, corner, results[anchor]["wave"], args.window, args.step
    )
    back = background_golden(turned, corner, anchor, args.downsample)
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

    selection = {
        "scan": scan.name,
        "screen_pitch_px": back["pitch"],
        "reference_bearing_deg": reference,
        "bearing_deg": {str(r["number"]): float(b) for r, b in zip(results, bearings)},
        "best_single": trail[0],
        "merged": trail[-1],
        "used": used,
        "skipped": skipped,
        "strength_alone": {
            str(result["number"]): back["alone"][index]
            for index, result in enumerate(results)
        },
    }
    sidecar = scan.with_name(f"data_{scan.stem}_background_selection.json")
    sidecar.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(f"  Wrote {sidecar}")

    if args.validate:
        print("  half-set agreement (structure the halves independently agree on):")
        print(f"    main pass       "
              f"{half_set_agreement(main_layers, readable, main_shifts, quorum, covered):+.3f}")
        report_halves(back, covered)

    passes.append(
        Pass(
            name="background pass",
            golden=back["golden"],
            seen=back["seen"],
            ray=back_ray,
            density_note=f"background pass, density; screen {pitch * args.downsample:.2f} px "
                         f"at {reference:+.2f} deg; "
                         f"merged {','.join(str(n) for n in used)} of {len(results)}",
            ink_note=f"background in its own ink, {hue:.0f} deg from main; "
                     f"strength {trail[-1]:.1f} vs {trail[0]:.1f} single",
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


def colour_layers(
    collection: Collection, rays: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Split every impression between the inks, and say where it is readable.

    The postmark is measured separately from the split rather than solved for
    alongside the inks, which is what keeps a nearly achromatic plate intact.
    """
    count, shape = len(rays), collection.images.shape[1:3]
    layers = np.zeros((count, len(collection.images), *shape), np.float32)
    readable = np.zeros((len(collection.images), *shape), bool)
    for index, (image, mask, paper) in enumerate(
        zip(collection.images, collection.valid, collection.papers)
    ):
        density, _ = optical_density(image, paper)
        amounts = unmix(density, rays)
        readable[index] = mask & ~ink_palette.postmark_mask(amounts, mask)
        for ink in range(count):
            layers[ink, index] = np.where(readable[index], amounts[ink], 0.0)
    return layers, readable


def screen_from_colour(
    collection: Collection, layers: np.ndarray, readable: np.ndarray, ray: np.ndarray, args: Settings
) -> dict:
    """Read one colour plate as a ruled screen and merge it on its phase.

    The engraved ground of a plate is periodic where its design is not, so the
    phase of that ruling locates the impression far more finely than its
    outline does.  This hands the colour to the machinery the screened
    single-colour stamps already use, unchanged.
    """
    entries = []
    for index, number in enumerate(collection.numbers):
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
                density, field, (7.5, 10.5), args.screen_tilt, collection.corner
            )
        except RuntimeError as error:
            print(f"    stamp {number:2d}: {error}")
            continue
        entries.append(
            {**screen, "number": number, "density": density, "coverage": coverage,
             "field": field, "valid": collection.valid[index]}
        )
    if len(entries) < 5:
        raise RuntimeError("Too few impressions carry a readable ruling in that ink")

    anchor = int(np.argmax([e["snr"] * e["coherence"] for e in entries]))
    turned, bearings, reference = derotate(
        entries, collection.corner, entries[anchor]["wave"], args.window, args.step
    )
    merged = background_golden(turned, collection.corner, anchor, args.downsample)
    merged["bearings"] = bearings
    merged["reference"] = reference
    merged["entries"] = entries
    merged["anchor"] = anchor
    return merged


def colour_passes(collection: Collection, palette: dict, args: Settings) -> list[Pass]:
    """One pass per colour plate, each registered on its own evidence.

    These plates went through the press separately, so each landed with its own
    small shift *and* its own small turn, and nothing ties them together.  Each
    colour is therefore registered against its own running template as a rigid
    motion, not as a shift.
    """
    rays, shares = palette["rays"], palette["shares"]
    layers, readable = colour_layers(collection, rays)
    quorum = max(3, math.ceil(0.6 * len(collection.images)))
    passes = []
    for ink, (ray, share) in enumerate(zip(rays, shares)):
        colour = bgr_hex(ink_palette.ink_colour(collection.paper, ray))
        name = f"ink {ink + 1} {colour}"
        warps, scores = register_pattern(
            layers[ink], readable, quorum, args.iterations, args.shift_limit, args.spin_limit
        )
        turns = np.degrees(np.arctan2(warps[:, 1, 0], warps[:, 0, 0]))
        centre = np.array([collection.images.shape[2] / 2.0, collection.images.shape[1] / 2.0])
        travel = warps @ np.array([centre[0], centre[1], 1.0]) - centre
        golden, seen = robust_golden(layers[ink], readable, warps, quorum)
        band = edge_band(golden, collection.covered & seen) if ink == 0 else None
        print(
            f"  {name}, {share:.0%} of the ink: turn sd {np.std(turns):.3f} deg "
            f"(spread {np.ptp(turns):.3f}), centre travel "
            f"{np.ptp(travel[:, 0]) * args.downsample:.2f}"
            f"x{np.ptp(travel[:, 1]) * args.downsample:.2f} px, "
            f"median ECC {np.nanmedian(scores):.3f}"
        )
        note = f"turn sd {np.std(turns):.3f} deg, ECC {np.nanmedian(scores):.2f}"

        if args.screen_ink == ink + 1:
            print(f"    reading {name} as a ruled screen as well:")
            crisped = screen_from_colour(collection, layers[ink], readable, ray, args)
            settled = np.array(
                [fit_wave(e["density"], e["field"], crisped["wave"], args.window, args.step)
                 ["bearing"] for e in crisped["entries"]]
            )
            print(
                f"    screen {crisped['pitch']:.3f} px, merge gain {crisped['gain']:.2f}x, "
                f"bearing brought to {crisped['reference']:+.3f} deg: spread "
                f"{np.ptp(crisped['bearings']):.3f} -> {np.ptp(settled):.3f} deg"
            )
            report_selection(crisped, crisped["entries"], len(crisped["entries"]))
            golden, seen = crisped["golden"], crisped["seen"]
            note = (f"screen {crisped['pitch'] * args.downsample:.2f} px at "
                    f"{crisped['reference']:+.2f} deg, merged "
                    f"{len(crisped['chosen'])} of {len(crisped['entries'])}")

        if args.validate:
            agreement = half_set_agreement(layers[ink], readable, warps, quorum, collection.covered)
            print(f"    half-set agreement {agreement:+.3f}")

        passes.append(
            Pass(
                name=name,
                golden=golden,
                seen=seen,
                ray=ray,
                density_note=f"{name}, density; {note}",
                ink_note=f"{name} in its own ink, {share:.0%} of the inked area",
                band=band,
            )
        )
    return passes


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

    palette = ink_palette.find_palette(
        ink_palette.ink_directions(collection.images, collection.valid, collection.papers),
        args.inks,
        args.separation,
    )
    print("  " + ink_palette.describe(palette, collection.paper))

    # Several colours mean several press runs, each free to have landed turned.
    # One colour means one run, with the ruled ground printed alongside it.
    if palette["count"] > 1:
        passes = colour_passes(collection, palette, args)
    else:
        passes = screen_passes(collection, args)

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

    # Two columns: each pass sits beside what it contributes to, density on the
    # left and the same layer in its own ink on the right.
    rows = [
        [
            (collection.baseline,
             f"{scan.stem}: golden as averaged now ({len(collection.numbers)} stamps), "
             f"paper {bgr_hex(collection.paper)}"),
            (rebuilt, f"rebuilt from its passes; design edge energy {gain:+.1%}"),
        ]
    ]
    for entry in passes:
        rows.append(
            [
                (cv2.cvtColor(contrast(entry.golden, entry.seen), cv2.COLOR_GRAY2BGR),
                 entry.density_note),
                (render_ink(entry.golden, collection.paper, entry.ray), entry.ink_note),
            ]
        )
    return write_panel(rows, output, args.panel_width)
