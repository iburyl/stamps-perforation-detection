"""Measure the background plate's geometry relative to the main plate.

The two passes are printed from separate plates in separate press runs, so they
can sit shifted and rotated with respect to each other.  They cannot sit
*scaled* with respect to each other on any single stamp, because both inks lie
on the same fibres: every later movement of the paper, which is nearly all of
the dimensional change these sheets have seen, carries both layers equally.  A
difference in the paper's moisture state between the two runs is the only leak,
and that is a property of the plate pair and the press-room routine, not of the
individual stamp.

So the relative scale between the layers is one constant for the whole
collection, while the relative rotation is a free parameter per impression.
This script tests that.

``aligned_collection`` has already applied each stamp's own design warp, fitted
by ``design_size\\measure_design.py`` from the printed ink against the golden.
That divides the main pass's scale and rotation out of the frame.  Whatever
spread is then left in the background screen is purely the inter-pass part, so
nothing here compares the perforation geometry against the image geometry --
the perforations were a later, separate operation with their own pin pitch and
their own registration error, and their scale is not the image's scale.

Precision is the whole difficulty.  One FFT bin is about 1.7% of the pitch,
which is far coarser than the effect.  Instead the screen's phase is read in a
grid of windows and a plane is fitted to it: over a 600 px baseline the screen
runs through some 68 periods, so a 0.1% pitch error shows up as roughly 24
degrees of accumulated phase.  The kernel uses absolute pixel coordinates, so
every window reports nearly the same phase and the integer-period ambiguity
that would otherwise defeat a long baseline never arises.

Reports, per impression, the across-ruling pitch and the bearing in the aligned
frame.  A constant pitch and a varying bearing is the claim above holding.

Experimental.  Reads only the JSON written by the production pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import background_ink
from background_ink import optical_density, screen_window, separate
from three_colour_refinement import estimate_papers, paper_ring
from two_pass_decomposition import aligned_collection


def window_phase(
    density: np.ndarray, field: np.ndarray, corner, size: int, wave: np.ndarray
) -> tuple[complex, float]:
    """Screen coefficient of one window, in absolute coordinates.

    Using absolute rather than window-local coordinates is what keeps every
    window's phase near a common value, so no unwrapping across periods is
    needed.
    """
    y0, x0 = corner
    box = (slice(y0, y0 + size), slice(x0, x0 + size))
    weight = field[box].astype(np.float64)
    covered = float(weight.sum()) / max(size * size, 1)
    if covered < 0.25:
        return 0j, covered
    patch = density[box].astype(np.float64)
    patch = patch - patch[weight > 0].mean(axis=0)
    patch = patch * (np.outer(np.hanning(size), np.hanning(size)) * weight)[:, :, None]
    rows = np.exp(-2j * math.pi * wave[1] * (y0 + np.arange(size)))
    columns = np.exp(-2j * math.pi * wave[0] * (x0 + np.arange(size)))
    return complex(np.einsum("y,x,yxc->c", rows, columns, patch, optimize=True).sum()), covered


def plane_fit(centres: np.ndarray, phases: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Frequency error from phase = offset + 2*pi * delta . centre."""
    anchor = int(np.argmax(weights))
    relative = phases - phases[anchor]
    relative = (relative + math.pi) % (2.0 * math.pi) - math.pi
    design = np.column_stack([np.ones(len(centres)), centres - centres[anchor]])
    root = np.sqrt(weights)[:, None]
    solution, *_ = np.linalg.lstsq(design * root, relative * np.sqrt(weights), rcond=None)
    return solution[1:] / (2.0 * math.pi)


def fit_wave(
    density: np.ndarray, field: np.ndarray, wave: np.ndarray, size: int, step: int
) -> dict:
    """Refine one impression's wave vector by a plane fit to the window phases."""
    height, width = field.shape
    centres, phases, weights, parity = [], [], [], []
    for row, y0 in enumerate(range(0, height - size + 1, step)):
        for column, x0 in enumerate(range(0, width - size + 1, step)):
            coefficient, covered = window_phase(density, field, (y0, x0), size, wave)
            if coefficient == 0j:
                continue
            centres.append((x0 + size / 2.0, y0 + size / 2.0))
            phases.append(np.angle(coefficient))
            weights.append(abs(coefficient) * covered)
            parity.append((row + column) % 2)
    if len(centres) < 12:
        raise RuntimeError(f"Only {len(centres)} usable windows")

    centres = np.array(centres, np.float64)
    phases = np.array(phases, np.float64)
    weights = np.array(weights, np.float64)
    weights /= weights.sum()
    parity = np.array(parity, bool)

    refined = wave + plane_fit(centres, phases, weights)
    magnitude = max(float(np.hypot(*refined)), 1e-12)

    # Repeatability from two interleaved halves of the windows.  The least
    # squares covariance is not usable here: the windows overlap and the phase
    # residual is systematic wander rather than independent noise, both of which
    # break the assumption it rests on.  Splitting on a checkerboard rather than
    # by position keeps each half spanning the full baseline.
    halves = []
    for side in (parity, ~parity):
        if side.sum() < 6:
            continue
        halves.append(wave + plane_fit(centres[side], phases[side], weights[side]))
    if len(halves) == 2:
        gap = halves[0] - halves[1]
        along = refined / magnitude
        across = np.array([-along[1], along[0]])
        # Half-fits differ by sqrt(2) sigma_half and sigma_full ~ sigma_half/sqrt(2).
        pitch_sigma = abs(float(along @ gap)) / 2.0 / magnitude**2
        bearing_sigma = math.degrees(abs(float(across @ gap)) / 2.0 / magnitude)
    else:
        pitch_sigma = bearing_sigma = float("nan")

    return {
        "wave": refined,
        "pitch": 1.0 / magnitude,
        "bearing": float(np.degrees(math.atan2(refined[1], refined[0]))),
        "windows": len(centres),
        "pitch_sigma": pitch_sigma,
        "bearing_sigma": bearing_sigma,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scan", type=Path)
    parser.add_argument("--perf", type=Path, help="Default: data_<stem>_perf.json")
    parser.add_argument("--design", type=Path, help="Default: data_<stem>_design.json")
    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument("--window", type=int, default=256, help="Phase window, px")
    parser.add_argument("--step", type=int, default=128, help="Window stride, px")
    parser.add_argument("--screen-tilt", type=float, default=12.0)
    args = parser.parse_args()

    scan = args.scan.resolve()
    perf_path = args.perf or scan.with_name(f"data_{scan.stem}_perf.json")
    design_path = args.design or scan.with_name(f"data_{scan.stem}_design.json")
    perf = json.loads(perf_path.read_text(encoding="utf-8"))
    design = json.loads(design_path.read_text(encoding="utf-8"))

    images, valid, numbers = aligned_collection(scan, perf, design, args.downsample)
    baseline_shape = images.shape[1:3]
    print(f"Scan: {scan}")
    print(f"  {len(numbers)} stamps, canvas {images.shape[2]}x{images.shape[1]}")

    ring = paper_ring(design, images.shape[1:3], args.downsample)
    papers = estimate_papers(images, valid, ring)
    density, magnitude = optical_density(images[0], papers[0])
    corner = screen_window(
        valid[0] & (magnitude < np.percentile(magnitude[valid[0]], 60)), images.shape[1:3]
    )

    separated, kept = [], []
    for number, image, mask, paper in zip(numbers, images, valid, papers):
        try:
            separated.append(separate(image, paper, mask, args.screen_tilt, corner))
        except RuntimeError as error:
            print(f"  stamp {number:2d}: {error}")
            continue
        kept.append(number)

    if len(separated) < 3:
        raise RuntimeError("Too few impressions carry a readable screen")

    # One refined starting frequency, then each impression is fitted freely from
    # it, so the per-impression spread is not constrained by a shared ruler.
    anchor = int(np.argmax([r["snr"] * r["coherence"] for r in separated]))
    patches = [
        background_ink.screen_patch(r["density"], r["field"], corner) for r in separated
    ]
    start = separated[anchor]["wave"]
    best, best_power = start, -1.0
    grid = 1.0 / background_ink.SCREEN_WINDOW
    for dy in np.arange(-1.0, 1.01, 0.2):
        for dx in np.arange(-1.0, 1.01, 0.2):
            trial = start + np.array([dx, dy]) * grid
            power = sum(
                abs(background_ink.screen_coefficient(patch, trial).sum()) for patch in patches
            )
            if power > best_power:
                best, best_power = trial, power
    print(f"  starting frequency: pitch {1.0 / np.hypot(*best):.4f} px")

    by_number = {int(item["stamp"]): item for item in design["stamps"]}
    rows = []
    for number, result in zip(kept, separated):
        try:
            fitted = fit_wave(result["density"], result["field"], best, args.window, args.step)
        except RuntimeError as error:
            print(f"  stamp {number:2d}: {error}")
            continue
        entry = by_number.get(number, {})
        fitted["number"] = number
        fitted["scale_y"] = float(entry.get("scale_y", float("nan")))
        fitted["design_rotation"] = float(entry.get("residual_rotation_deg", float("nan")))
        rows.append(fitted)

    print("\n  Background screen in the aligned frame (main pass already divided out)")
    print(
        "  stamp   pitch px +- sigma      bearing deg +- sigma   windows   "
        "design scale_y   design rot"
    )
    for row in rows:
        print(
            f"  {row['number']:5d}   {row['pitch'] * args.downsample:8.4f} "
            f"+-{row['pitch_sigma'] * args.downsample:.4f}   "
            f"{row['bearing']:+11.3f} +-{row['bearing_sigma']:.3f}   "
            f"{row['windows']:7d}   {row['scale_y']:14.5f}   {row['design_rotation']:+10.3f}"
        )

    pitches = np.array([row["pitch"] for row in rows])
    bearings = np.array([row["bearing"] for row in rows])
    relative = float(np.std(pitches) / max(np.mean(pitches), 1e-12))
    # The noise floor the observed spread has to beat: the per-impression
    # uncertainties combined, not the spread itself.
    pitch_floor = float(np.sqrt(np.mean(np.square([r["pitch_sigma"] for r in rows]))))
    bearing_floor = float(np.sqrt(np.mean(np.square([r["bearing_sigma"] for r in rows]))))
    print(
        f"\n  pitch   mean {np.mean(pitches) * args.downsample:.4f} px, "
        f"sd {np.std(pitches) * args.downsample:.4f} px ({relative:.3%}), "
        f"measurement floor {pitch_floor * args.downsample:.4f} px "
        f"({pitch_floor / np.mean(pitches):.3%})"
    )
    print(
        f"  bearing mean {np.mean(bearings):+.3f} deg, "
        f"sd {np.std(bearings):.3f} deg, spread {np.ptp(bearings):.3f} deg, "
        f"measurement floor {bearing_floor:.3f} deg"
    )

    # The claim: scale is common to both layers, so the aligned-frame pitch
    # should not vary; rotation is not, so the bearing may.
    scales = np.array([row["scale_y"] for row in rows])
    if np.isfinite(scales).all() and np.ptp(scales) > 0:
        print(
            f"  the design fit already removed a scale_y sd of "
            f"{np.std(np.log(scales)):.3%}; the screen still shows {relative:.3%}"
        )

    # What survives the measurement floor is the real inter-pass variation.
    pitch_real = math.sqrt(max(np.std(pitches) ** 2 - pitch_floor**2, 0.0))
    bearing_real = math.sqrt(max(np.std(bearings) ** 2 - bearing_floor**2, 0.0))
    print(
        f"\n  relative scale:    {pitch_real / np.mean(pitches):.3%} beyond the floor"
        + ("  (nothing detected)" if pitch_real <= 0.0 else "")
    )
    print(f"  relative rotation: {bearing_real:.3f} deg beyond the floor")
    reach = 0.5 * math.hypot(*baseline_shape)
    print(
        f"  at {reach:.0f} px from centre that rotation displaces "
        f"{math.radians(bearing_real) * reach * args.downsample:.2f} px, "
        f"and the full {np.ptp(bearings):.3f} deg spread displaces "
        f"{math.radians(np.ptp(bearings)) * reach * args.downsample:.2f} px"
    )

    # If the bearing spread were really the design fit's own rotation error
    # being injected into the aligned frame, the two would be anticorrelated.
    rotations = np.array([row["design_rotation"] for row in rows])
    if np.isfinite(rotations).all() and np.std(rotations) > 0:
        correlation = float(np.corrcoef(rotations, bearings)[0, 1])
        slope = correlation * np.std(bearings) / np.std(rotations)
        print(
            f"  bearing vs the design fit's own rotation: r {correlation:+.3f}, "
            f"slope {slope:+.3f} deg per deg; "
            f"{abs(slope) * np.std(rotations):.3f} deg of the bearing tracks it, "
            f"{np.std(bearings) * math.sqrt(max(1 - correlation**2, 0)):.3f} deg does not"
        )


if __name__ == "__main__":
    main()
