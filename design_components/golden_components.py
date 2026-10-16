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

The two passes are registered by different means, because they offer different
evidence.  The main design is a unique aperiodic pattern, so it is registered
by correlation against the running template; a sharp unambiguous peak exists
and the half-set check confirms it.  The background screen is periodic, so no
template-based match is well posed on it; instead its phase is read against an
analytic wave at its own measured pitch, which no template bootstrap can bias.
Only the component across the ruling is observable that way, the component
along it being a symmetry of the screen, so only that component is corrected.

The two passes come off separate plates in separate press runs, so they also
sit rotated with respect to each other, by about a tenth of a degree in these
sheets -- enough to displace a corner by more than a pixel.  Each impression's
screen bearing is therefore measured and brought to a common angle before the
phase is matched.  Their relative *scale* is left alone deliberately: both inks
lie on the same fibres, so every later movement of the paper carries them
equally, and measurement puts any residual below 0.04%.

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


def robust_golden(
    layers: np.ndarray, masks: np.ndarray, shifts: np.ndarray, quorum: int, floor: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Shift every impression onto the common frame and take the median.

    ``quorum`` is the coverage the returned mask reports as trustworthy;
    ``floor`` is the coverage below which no value is emitted at all.
    """
    stack, coverage = [], []
    for layer, mask, shift in zip(layers, masks, shifts):
        stack.append(shift_image(layer, shift, cv2.INTER_CUBIC))
        coverage.append(shift_image(mask.astype(np.uint8), shift, cv2.INTER_NEAREST) > 0)
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
    layers: np.ndarray, masks: np.ndarray, quorum: int, iterations: int, limit: float
) -> tuple[np.ndarray, np.ndarray]:
    """Translation per impression, by correlation against the running template.

    Sound for the main design, whose correlation peak is sharp and unique.  It
    is deliberately not used for the periodic background screen.
    """
    shifts = np.zeros((len(layers), 2), np.float32)
    scores = np.full(len(layers), np.nan, np.float32)
    template, _ = robust_golden(layers, masks, shifts, quorum, floor=quorum)
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
            warp = np.array(
                [[1.0, 0.0, shifts[index, 0]], [0.0, 1.0, shifts[index, 1]]], np.float32
            )
            try:
                score, warp = cv2.findTransformECC(
                    target,
                    moving,
                    warp,
                    cv2.MOTION_TRANSLATION,
                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 120, 1e-7),
                    inputMask=(support & masks[index].astype(np.uint8)) * 255,
                    gaussFiltSize=5,
                )
            except cv2.error:
                continue
            candidate = warp[:, 2]
            if np.all(np.isfinite(candidate)) and np.max(np.abs(candidate)) <= limit:
                shifts[index] = candidate
                scores[index] = float(score)
        good = np.isfinite(scores)
        if good.any():
            shifts[good] -= np.median(shifts[good], axis=0)
        template, _ = robust_golden(layers, masks, shifts, quorum, floor=quorum)
    return shifts, scores


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
    """
    patches = [
        background_ink.screen_patch(r["density"], r["field"], corner) for r in results
    ]
    wave = shared_wave(patches, results[anchor]["wave"])
    shifts = np.zeros((len(results), 2), np.float32)
    squared = float(wave @ wave)
    if squared < 1e-12:
        return shifts, wave, 0.0
    phases, weights = [], []
    for patch in patches:
        coefficients = background_ink.screen_coefficient(patch, wave)
        total = coefficients.sum()
        phases.append(np.angle(total))
        weights.append(abs(total) / max(float(np.sum(np.abs(coefficients))), 1e-12))
    reference = phases[anchor]
    for index, phase in enumerate(phases):
        turns = (phase - reference) / (2.0 * math.pi)
        turns -= round(turns)  # nearest repeat; the screen is ambiguous beyond it
        # shift_image resamples at x + shift, so the shift opposes the phase lead.
        shifts[index] = (-turns / squared) * wave
    shifts -= np.median(shifts, axis=0)
    return shifts, wave, float(np.median(weights))


def screen_strength(layer: np.ndarray, mask: np.ndarray, corner, wave: np.ndarray) -> float:
    """How far the ruling stands above the broadband floor in one layer.

    This is the crispness measure for the background, and it is comparable
    between a single impression and an average of several, which mean gradient
    magnitude is not: noise raises a gradient but not a Fourier peak.  A merge
    that reinforces the screen scores above its parts, a merge that smears it
    scores below them.
    """
    size = background_ink.SCREEN_WINDOW
    y0, x0 = corner
    box = (slice(y0, y0 + size), slice(x0, x0 + size))
    weight = mask[box].astype(np.float64)
    if weight.sum() < 15000:
        return 0.0
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
    floor = float(np.median(spectrum[ring])) if ring.any() else 0.0
    return float(peak / max(floor, 1e-9))


def window_scorer(layers: np.ndarray, masks: np.ndarray, shifts: np.ndarray, corner, wave):
    """Score any subset of impressions by the crispness of their merge.

    The shifted layers are cropped to the shared screen window once, so trying
    a subset costs only a median over it.
    """
    size = background_ink.SCREEN_WINDOW
    y0, x0 = corner
    window = (slice(y0, y0 + size), slice(x0, x0 + size))
    cropped, covered = [], []
    for layer, mask, shift in zip(layers, masks, shifts):
        cropped.append(shift_image(layer, shift, cv2.INTER_CUBIC)[window])
        covered.append(shift_image(mask.astype(np.uint8), shift, cv2.INTER_NEAREST)[window] > 0)
    cropped, covered = np.stack(cropped), np.stack(covered)

    def score(members) -> float:
        members = list(members)
        values = cropped[members].astype(np.float64)
        seen = covered[members]
        values[~seen] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            golden = np.nanmedian(values, axis=0)
        enough = seen.sum(axis=0) >= max(1, len(members) // 2)
        golden[~np.isfinite(golden) | ~enough] = 0.0
        return screen_strength(golden, enough, (0, 0), wave)

    return score


def choose_impressions(score, count: int) -> tuple[list[int], list[float], list[float]]:
    """Greedily keep only the impressions whose merge sharpens the screen.

    Averaging every impression is wrong here.  The screen is quasi-periodic, so
    impressions whose ruling does not actually phase up with the rest subtract
    modulation instead of adding it, and the merge ends up flatter than the
    single impressions it came from.  Each candidate is admitted only if the
    golden is crisper with it than without.
    """
    alone = [score([index]) for index in range(count)]
    chosen = [int(np.argmax(alone))]
    trail = [alone[chosen[0]]]
    remaining = [index for index in range(count) if index not in chosen]
    while remaining:
        best, index = max((score(chosen + [candidate]), candidate) for candidate in remaining)
        if best <= trail[-1]:
            break
        chosen.append(index)
        trail.append(best)
        remaining.remove(index)
    return chosen, trail, alone


def select_and_merge(
    layers: np.ndarray, field: np.ndarray, shifts: np.ndarray, corner, wave: np.ndarray
) -> dict:
    """Keep only the impressions whose merge sharpens the screen, then merge them."""
    score = window_scorer(layers, field, shifts, corner, wave)
    chosen, trail, alone = choose_impressions(score, len(layers))
    quorum = max(2, len(chosen) // 2)
    golden, seen = robust_golden(layers[chosen], field[chosen], shifts[chosen], quorum)
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


def background_golden(results: list[dict], corner, anchor: int, downsample: int) -> dict:
    """Phase-register, select and merge the background pass of a collection."""
    field = np.stack([r["field"] for r in results])
    layers = np.stack(
        [np.where(r["field"], r["coverage"], 0.0).astype(np.float32) for r in results]
    )
    shifts, wave, lock = phase_shifts(results, corner, anchor)
    merged = select_and_merge(layers, field, shifts, corner, wave)
    merged["lock"] = lock
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
    layers: np.ndarray, masks: np.ndarray, shifts: np.ndarray, quorum: int, mask: np.ndarray
) -> float:
    """Correlation of two goldens built from disjoint halves of the collection.

    Structure the two halves do not agree on was produced by the averaging, not
    recovered by it.
    """
    order = np.arange(len(layers))
    first, second = order[0::2], order[1::2]
    if len(first) < 2 or len(second) < 2:
        return float("nan")
    one, seen_one = robust_golden(layers[first], masks[first], shifts[first], max(2, quorum // 2))
    two, seen_two = robust_golden(layers[second], masks[second], shifts[second], max(2, quorum // 2))
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


def build(scan: Path, args: Settings) -> Path:
    """Rebuild one scan's golden from its printing passes and write the panel."""
    perf_path = args.perf or scan.with_name(f"data_{scan.stem}_perf.json")
    design_path = args.design or scan.with_name(f"data_{scan.stem}_design.json")
    output = args.output or scan.with_name(f"{scan.stem}_golden_components.png")

    perf = json.loads(perf_path.read_text(encoding="utf-8"))
    design = json.loads(design_path.read_text(encoding="utf-8"))
    images, valid, numbers = aligned_collection(scan, perf, design, args.downsample)
    print(f"Printing passes, from {len(numbers)} stamps on a "
          f"{images.shape[2]}x{images.shape[1]} canvas:")

    ring = paper_ring(design, images.shape[1:3], args.downsample)
    papers = estimate_papers(images, valid, ring)
    paper = np.median(papers, axis=0)
    baseline = np.clip(masked_colour_median(images, valid), 0, 255).astype(np.uint8)
    covered = valid.sum(axis=0) >= max(3, math.ceil(0.6 * len(images)))

    # One screen window for the whole collection, so phases are comparable.
    density, magnitude = optical_density(images[0], papers[0])
    corner = screen_window(valid[0] & (magnitude < np.percentile(magnitude[valid[0]], 60)),
                           images.shape[1:3])

    results, kept = [], []
    for number, image, mask, stamp_paper in zip(numbers, images, valid, papers):
        try:
            result = separate(image, stamp_paper, mask, args.screen_tilt, corner)
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
        [np.where(r["valid"] & ~r["postmark"], r["along_main"], 0.0).astype(np.float32) for r in results]
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

    layers = [golden_main]
    rays = [main_ray]
    if not args.no_background:
        anchor = int(np.argmax([r["snr"] * r["coherence"] for r in results]))

        # Bring every screen to a common bearing before matching phase.  The
        # passes come off separate plates, so their relative rotation is a free
        # parameter; left uncorrected it stops impressions from phasing up.
        turned, bearings, reference = derotate(
            results, corner, results[anchor]["wave"], args.window, args.step
        )
        back = background_golden(turned, corner, anchor, args.downsample)
        # Refit the bearings on the rotated layers.  If the sign of the
        # correction were wrong this spread would double rather than close.
        settled = np.array(
            [
                fit_wave(r["density"], r["field"], back["wave"], args.window, args.step)[
                    "bearing"
                ]
                for r in turned
            ]
        )
        golden_back, seen_back = back["golden"], back["seen"]
        used, skipped, trail = back["used"], back["skipped"], back["trail"]
        print(
            f"  background pass: anchored on stamp {results[anchor]['number']}, "
            f"shared screen {back['pitch']:.3f} px, phase lock {back['lock']:.2f}, "
            f"shift span {np.ptp(back['shifts'][:, 0]) * args.downsample:.2f}"
            f"x{np.ptp(back['shifts'][:, 1]) * args.downsample:.2f} px"
        )
        print(
            f"    bearing brought to {reference:+.3f} deg: spread "
            f"{np.ptp(bearings):.3f} -> {np.ptp(settled):.3f} deg "
            f"(sd {np.std(bearings):.3f} -> {np.std(settled):.3f})"
        )
        print(
            f"    screen strength: best single {trail[0]:.2f}, "
            f"merge of {len(back['chosen'])} {trail[-1]:.2f}, "
            f"all {len(results)} merged {back['all_merged']:.2f}"
        )
        for index, result in enumerate(results):
            place = back["chosen"].index(index) + 1 if index in back["chosen"] else 0
            print(
                f"    stamp {result['number']:2d}: alone {back['alone'][index]:6.2f}  "
                + (f"used {place}; merge {trail[place - 1]:6.2f}" if place else "skipped")
            )
        print(f"    used     {used}")
        print(f"    skipped  {skipped}")
        back_quorum = back["quorum"]
        back_layers, field = back["layers"], back["field"]
        back_shifts, chosen = back["shifts"], back["chosen"]

        selection = {
            "scan": scan.name,
            "screen_pitch_px": back["pitch"],
            "reference_bearing_deg": reference,
            "bearing_deg": {
                str(r["number"]): float(b) for r, b in zip(results, bearings)
            },
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

        layers.append(golden_back)
        rays.append(back_ray)

    rebuilt = reconstruct(layers, rays, paper)
    band = edge_band(golden_main, covered & seen_main)
    before = sharpness(baseline, band)
    after = sharpness(rebuilt, band)
    gain = after / max(before, 1e-6) - 1
    print(
        f"  design edge energy over {band.sum()} edge pixels: "
        f"golden {before:.2f} -> rebuilt {after:.2f} ({gain:+.1%})"
    )

    if args.validate:
        print("  half-set agreement (structure the halves independently agree on):")
        one = half_set_agreement(main_layers, readable, main_shifts, quorum, covered)
        print(f"    main pass       {one:+.3f}")
        if not args.no_background:
            two = half_set_agreement(
                back_layers[chosen], field[chosen], back_shifts[chosen], back_quorum, covered
            )
            unaligned = half_set_agreement(
                back_layers[chosen],
                field[chosen],
                np.zeros_like(back_shifts[chosen]),
                back_quorum,
                covered,
            )
            if math.isnan(two):
                print(
                    f"    background pass: {len(chosen)} impressions kept, too few to "
                    "split into halves"
                )
            else:
                print(f"    background pass {two:+.3f} (unaligned control {unaligned:+.3f})")
            if not math.isnan(two) and two < 0.2:
                print(
                    "    The background golden is not reproduced by independent halves.\n"
                    "    Read that pass off single impressions with background_ink.py."
                )

    # Two columns: each pass sits beside what it contributes to, density on the
    # left and the same layer in its own ink on the right.
    rows: list[list[tuple[np.ndarray, str]]] = [
        [
            (
                baseline,
                f"{scan.stem}: golden as averaged now ({len(results)} stamps), "
                f"paper {bgr_hex(paper)}",
            ),
            (rebuilt, f"rebuilt from its passes; design edge energy {gain:+.1%}"),
        ],
        [
            (
                cv2.cvtColor(contrast(golden_main, seen_main), cv2.COLOR_GRAY2BGR),
                f"main pass, density; ECC {np.nanmedian(main_scores):.2f}",
            ),
            (
                render_ink(golden_main, paper, main_ray),
                "main pass in its own ink "
                f"{bgr_hex(to_srgb(to_linear(paper) * 10.0 ** (-main_ray)))}",
            ),
        ],
    ]
    if not args.no_background:
        rows.append(
            [
                (
                    cv2.cvtColor(contrast(golden_back, seen_back), cv2.COLOR_GRAY2BGR),
                    f"background pass, density; screen {pitch * args.downsample:.2f} px "
                    f"at {reference:+.2f} deg; "
                    f"merged {','.join(str(n) for n in used)} of {len(results)}",
                ),
                (
                    render_ink(golden_back, paper, back_ray),
                    f"background in its own ink, {hue:.0f} deg from main; "
                    f"strength {trail[-1]:.1f} vs {trail[0]:.1f} single",
                ),
            ]
        )

    rendered = []
    for row in rows:
        cells = []
        for image, text in row:
            factor = args.panel_width / image.shape[1]
            resized = cv2.resize(
                image,
                (args.panel_width, max(1, int(round(image.shape[0] * factor)))),
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
        raise RuntimeError(
            f"Cannot write image: {output} ({sheet.shape} {sheet.dtype})"
        )
    print(f"Wrote {output}")
    return output
