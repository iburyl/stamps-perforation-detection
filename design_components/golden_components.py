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
and ``--validate`` confirms it.  The background screen is periodic, so no
template-based match is well posed on it; instead its phase is read against an
analytic wave at its own measured pitch, which no template bootstrap can bias.
Only the component across the ruling is observable that way, the component
along it being a symmetry of the screen, so only that component is corrected.

Writes ``<stem>_golden_components.png`` next to ``<stem>_golden.png``.

Experimental.  Reads only the JSON written by the production pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild the golden one printing pass at a time, then recombine."
    )
    parser.add_argument("scan", type=Path)
    parser.add_argument("--perf", type=Path, help="Default: data_<stem>_perf.json")
    parser.add_argument("--design", type=Path, help="Default: data_<stem>_design.json")
    parser.add_argument("--output", type=Path, help="Default: <stem>_golden_components.png")
    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--shift-limit", type=float, default=12.0)
    parser.add_argument("--screen-tilt", type=float, default=12.0)
    parser.add_argument(
        "--no-background", action="store_true",
        help="Rebuild the main pass only, for a single-pass denomination",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Build each pass from two disjoint halves and report their agreement",
    )
    parser.add_argument("--panel-width", type=int, default=760)
    args = parser.parse_args()
    if args.downsample < 1:
        parser.error("--downsample must be positive")
    return args


def main() -> None:
    args = parse_args()
    scan = args.scan.resolve()
    perf_path = args.perf or scan.with_name(f"data_{scan.stem}_perf.json")
    design_path = args.design or scan.with_name(f"data_{scan.stem}_design.json")
    output = args.output or scan.with_name(f"{scan.stem}_golden_components.png")

    perf = json.loads(perf_path.read_text(encoding="utf-8"))
    design = json.loads(design_path.read_text(encoding="utf-8"))
    images, valid, numbers = aligned_collection(scan, perf, design, args.downsample)
    print(f"Scan: {scan}\n  {len(numbers)} stamps, canvas {images.shape[2]}x{images.shape[1]}")

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
        field = np.stack([r["field"] for r in results])
        back_layers = np.stack(
            [np.where(r["field"], r["coverage"], 0.0).astype(np.float32) for r in results]
        )
        anchor = int(np.argmax([r["snr"] * r["coherence"] for r in results]))
        back_shifts, wave, lock = phase_shifts(results, corner, anchor)
        golden_back, seen_back = robust_golden(back_layers, field, back_shifts, quorum)
        print(
            f"  background pass: anchored on stamp {results[anchor]['number']}, "
            f"shared screen {1.0 / max(np.hypot(*wave), 1e-9) * args.downsample:.3f} px, "
            f"phase lock {lock:.2f}, shift span "
            f"{np.ptp(back_shifts[:, 0]) * args.downsample:.2f}"
            f"x{np.ptp(back_shifts[:, 1]) * args.downsample:.2f} px"
        )
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
            two = half_set_agreement(back_layers, field, back_shifts, quorum, covered)
            flat = half_set_agreement(
                back_layers, field, np.zeros_like(back_shifts), quorum, covered
            )
            print(f"    background pass {two:+.3f} (unaligned control {flat:+.3f})")
            if not math.isnan(two) and two < 0.2:
                print(
                    "    The background golden is not reproduced by independent halves.\n"
                    "    Read that pass off single impressions with background_ink.py."
                )

    shape = baseline.shape[:2]
    swatch = np.empty((*shape, 3), np.uint8)
    swatch[:] = np.clip(np.round(paper), 0, 255).astype(np.uint8)
    panels: list[tuple[np.ndarray, str]] = [
        (baseline, f"{scan.stem}: golden as averaged now ({len(results)} stamps)"),
        (rebuilt, f"rebuilt from its passes; design edge energy {gain:+.1%}"),
        (swatch, f"paper {bgr_hex(paper)}"),
        (
            cv2.cvtColor(contrast(golden_main, seen_main), cv2.COLOR_GRAY2BGR),
            f"main pass, density; ECC {np.nanmedian(main_scores):.2f}",
        ),
        (
            render_ink(golden_main, paper, main_ray),
            f"main pass in its own ink {bgr_hex(to_srgb(to_linear(paper) * 10.0 ** (-main_ray)))}",
        ),
    ]
    if not args.no_background:
        panels.append(
            (
                cv2.cvtColor(contrast(golden_back, seen_back), cv2.COLOR_GRAY2BGR),
                f"background pass, density; screen {pitch * args.downsample:.2f} px",
            )
        )
        panels.append(
            (
                render_ink(golden_back, paper, back_ray),
                f"background pass in its own ink, {hue:.0f} deg from main",
            )
        )

    rendered = []
    for image, text in panels:
        factor = args.panel_width / image.shape[1]
        resized = cv2.resize(
            image,
            (args.panel_width, max(1, int(round(image.shape[0] * factor)))),
            interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_CUBIC,
        )
        rendered.append(add_label(resized, text))
    if not cv2.imwrite(str(output), np.vstack(rendered)):
        raise RuntimeError(f"Cannot write image: {output}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
