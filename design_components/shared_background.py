"""REJECTED EXPERIMENT: common 1--7K background from five denominations.

This dense/complement-based reconstruction blurred unrelated contour designs
into the alleged common background.  It is retained only for comparison; do
not use its output as a recovered printing plate.

The experiment uses colour only to initialise two optical-density rays.  It
then estimates a scalar background-density image for every stamp, registers
those images independently of the outline pass, takes a robust within-series
median, normalises all five series to the same frame, and finally gives every
denomination equal weight in the shared background.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import cv2
import numpy as np

from two_pass_decomposition import (
    aligned_collection,
    colour_partition,
    luminance_bgr,
    masked_colour_median,
)


SERIES = ("1K", "2K", "3K", "5K", "7K")


def quiet_nanmedian(array: np.ndarray, axis: int = 0) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        return np.nanmedian(array, axis=axis)


def optical_density(image: np.ndarray, paper_bgr: np.ndarray) -> np.ndarray:
    density = -np.log(
        np.clip(
            (image.astype(np.float64) + 2.0)
            / (paper_bgr[None, None, :].astype(np.float64) + 2.0),
            1e-3,
            1.5,
        )
    )
    return np.clip(density, 0.0, None).astype(np.float32)


def unmix_two(density: np.ndarray, bases: np.ndarray) -> np.ndarray:
    """Exact non-negative least squares for two unit colour directions."""
    shape = density.shape[:2]
    values = density.reshape(-1, 3).astype(np.float64)
    bases = bases.astype(np.float64)
    bases /= np.maximum(np.linalg.norm(bases, axis=1, keepdims=True), 1e-12)
    gram_inverse = np.linalg.pinv(bases @ bases.T)
    both = values @ bases.T @ gram_inverse
    candidates = np.zeros((len(values), 4, 2), np.float64)
    candidates[:, 0] = np.maximum(both, 0.0)
    candidates[:, 1, 0] = np.maximum(values @ bases[0], 0.0)
    candidates[:, 2, 1] = np.maximum(values @ bases[1], 0.0)
    reconstructions = np.einsum("nck,kp->ncp", candidates, bases)
    error = np.sum((reconstructions - values[:, None, :]) ** 2, axis=2)
    chosen = np.argmin(error, axis=1)
    coefficients = candidates[np.arange(len(values)), chosen]
    return coefficients.reshape(*shape, 2).astype(np.float32)


def estimate_paper(image: np.ndarray, quiet_mask: np.ndarray) -> np.ndarray:
    pixels = image[quiet_mask].astype(np.float64)
    if len(pixels) < 500:
        pixels = image.reshape(-1, 3).astype(np.float64)
    return np.percentile(pixels, 90, axis=0)


def normalise_density(density: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = density[valid & (density > 0)]
    if len(values) < 100:
        return density
    high = float(np.percentile(values, 96))
    return np.clip(density / max(high, 1e-6), 0.0, 1.5).astype(np.float32)


def align_translation(
    template: np.ndarray,
    moving: np.ndarray,
    valid: np.ndarray,
    limit: float,
) -> tuple[np.ndarray, np.ndarray, bool]:
    warp = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 150, 1e-6)
    try:
        _, fitted = cv2.findTransformECC(
            template.astype(np.float32),
            moving.astype(np.float32),
            warp,
            cv2.MOTION_TRANSLATION,
            criteria,
            valid.astype(np.uint8) * 255,
            5,
        )
        if not np.all(np.isfinite(fitted)) or np.linalg.norm(fitted[:, 2]) > limit:
            identity = np.array(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], np.float32
            )
            return moving, identity, False
        aligned = cv2.warpAffine(
            moving,
            fitted,
            (moving.shape[1], moving.shape[0]),
            flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
            borderValue=0,
        )
        return aligned, fitted, True
    except cv2.error:
        return moving, warp, False


def series_background(
    scan_path: Path,
    perf: dict,
    design: dict,
    downsample: int,
    seed: int,
) -> dict:
    images, valid, numbers = aligned_collection(scan_path, perf, design, downsample)
    reference = masked_colour_median(images, valid)
    grey = luminance_bgr(reference)
    paper_level = float(np.percentile(grey[grey > 0], 92))
    darkness = np.clip((paper_level - grey) / max(paper_level, 1.0), 0.0, 1.0)
    control = colour_partition(
        reference, darkness, np.random.default_rng(seed), max_samples=60000
    )
    bases = np.asarray(control["od_direction_centres_bgr"], np.float32)
    reference_density = optical_density(reference, control["paper_bgr"])
    reference_coefficients = unmix_two(reference_density, bases)
    group_density = []
    for group in (0, 1):
        where = control["assignment"] == group
        group_density.append(float(np.median(np.linalg.norm(reference_density[where], axis=1))))
    background_group = int(np.argmin(group_density))

    total_reference = reference_coefficients.sum(axis=2)
    quiet = total_reference <= np.percentile(total_reference[grey > 0], 28)
    per_stamp: list[np.ndarray] = []
    per_stamp_valid: list[np.ndarray] = []
    for image, stamp_valid in zip(images, valid):
        paper = estimate_paper(image, quiet & stamp_valid)
        coefficients = unmix_two(optical_density(image, paper), bases)
        background = coefficients[:, :, background_group]
        outline = coefficients[:, :, 1 - background_group]
        total = background + outline
        positive = total[stamp_valid & (total > 0)]
        floor = float(np.percentile(positive, 8)) if len(positive) else 0.0
        purity = background / np.maximum(total, 1e-6)
        background_valid = stamp_valid & (total > floor) & (purity >= 0.58)
        # Clip extreme cancellation density before the robust median. The
        # actual rejection is across stamps; clipping only prevents overflow.
        present = background[background_valid]
        if len(present) < 500:
            background_valid = stamp_valid & (background >= outline)
            present = background[background_valid]
        background = np.minimum(background, np.percentile(present, 99.5))
        background[~background_valid] = np.nan
        per_stamp.append(background)
        per_stamp_valid.append(background_valid)
    stack = np.stack(per_stamp)
    initial = quiet_nanmedian(stack, axis=0)
    initial[~np.isfinite(initial)] = 0.0
    quorum = np.sum(np.isfinite(stack), axis=0) >= max(3, int(np.ceil(0.35 * len(stack))))
    initial = normalise_density(initial, quorum)

    aligned_stamps = []
    stamp_warps = []
    stamp_ok = []
    aligned_valid = []
    for background, background_valid in zip(per_stamp, per_stamp_valid):
        moving = np.nan_to_num(background, nan=0.0)
        moving = normalise_density(moving, background_valid)
        aligned, warp, ok = align_translation(
            initial, moving, background_valid & quorum, 6.0
        )
        moved_valid = cv2.warpAffine(
            background_valid.astype(np.uint8),
            warp,
            (moving.shape[1], moving.shape[0]),
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
            borderValue=0,
        ) > 0
        aligned[~moved_valid] = np.nan
        aligned_stamps.append(aligned)
        aligned_valid.append(moved_valid)
        stamp_warps.append(warp)
        stamp_ok.append(ok)
    aligned_stack = np.stack(aligned_stamps)
    background = quiet_nanmedian(aligned_stack, axis=0)
    scatter = 1.4826 * quiet_nanmedian(
        np.abs(aligned_stack - background[None, :, :]), axis=0
    )
    background[~np.isfinite(background)] = 0.0
    scatter[~np.isfinite(scatter)] = 0.0
    final_quorum = np.sum(np.stack(aligned_valid), axis=0) >= max(
        3, int(np.ceil(0.35 * len(stack)))
    )
    background[~final_quorum] = 0.0
    background = normalise_density(background, final_quorum)
    return {
        "name": scan_path.stem,
        "background": background,
        "scatter": scatter,
        "quorum": final_quorum,
        "reference": reference,
        "rectangle": design["golden"]["rectangle"],
        "stamps": len(numbers),
        "colour_direction_separation": control["direction_separation"],
        "background_group": background_group,
        "stamp_warps": np.asarray(stamp_warps),
        "stamp_registration_successes": int(np.sum(stamp_ok)),
    }


def frame_normalise(
    image: np.ndarray,
    rectangle: dict,
    downsample: int,
    target_size: tuple[int, int],
    interpolation: int,
) -> np.ndarray:
    height, width = target_size
    pad = 18
    left = float(rectangle["left_px"]) / downsample
    right = float(rectangle["right_px"]) / downsample
    top = float(rectangle["top_px"]) / downsample
    bottom = float(rectangle["bottom_px"]) / downsample
    sx = (width - 2 * pad) / max(right - left, 1e-6)
    sy = (height - 2 * pad) / max(bottom - top, 1e-6)
    matrix = np.array(
        [[sx, 0.0, pad - sx * left], [0.0, sy, pad - sy * top]], np.float32
    )
    return cv2.warpAffine(
        image.astype(np.float32), matrix, (width, height),
        flags=interpolation, borderValue=0,
    )


def preview_density(density: np.ndarray, gamma: float = 0.75) -> np.ndarray:
    values = density[density > 0]
    scale = float(np.percentile(values, 98)) if len(values) else 1.0
    ink = np.power(np.clip(density / max(scale, 1e-6), 0, 1), gamma)
    grey = np.clip(255.0 * (1.0 - ink), 0, 255).astype(np.uint8)
    return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)


def correlation(first: np.ndarray, second: np.ndarray, mask: np.ndarray) -> float:
    a = first[mask].astype(np.float64)
    b = second[mask].astype(np.float64)
    if len(a) < 100 or a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Jointly recover the background shared by 1K--7K."
    )
    parser.add_argument("data", type=Path, help="Directory containing scans and JSON")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()
    output = args.output or args.data / "shared_background_1K_7K"
    output.mkdir(parents=True, exist_ok=True)
    per_series_dir = output / "per_series"
    per_series_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for index, name in enumerate(SERIES):
        jpg = args.data / f"{name}.jpg"
        png = args.data / f"{name}.png"
        scan = jpg if jpg.exists() else png
        perf = json.loads((args.data / f"data_{name}_perf.json").read_text(encoding="utf-8"))
        design = json.loads((args.data / f"data_{name}_design.json").read_text(encoding="utf-8"))
        print(f"{name}: separating {len(design['stamps'])} stamps ...")
        result = series_background(
            scan, perf, design, args.downsample, args.seed + index
        )
        results.append(result)

    # Common dimensions near the median physical frame size at analysis scale.
    widths = [
        (item["rectangle"]["right_px"] - item["rectangle"]["left_px"])
        / args.downsample
        for item in results
    ]
    heights = [
        (item["rectangle"]["bottom_px"] - item["rectangle"]["top_px"])
        / args.downsample
        for item in results
    ]
    target = (int(round(np.median(heights))) + 36, int(round(np.median(widths))) + 36)
    normalised = []
    normalised_masks = []
    for item in results:
        image = frame_normalise(
            item["background"], item["rectangle"], args.downsample,
            target, cv2.INTER_CUBIC,
        )
        mask = frame_normalise(
            item["quorum"].astype(np.float32), item["rectangle"], args.downsample,
            target, cv2.INTER_NEAREST,
        ) > 0.5
        image = normalise_density(image, mask)
        normalised.append(image)
        normalised_masks.append(mask)

    # Use 2K as the initial anchor because the user-identified colour contrast
    # makes its background coefficient particularly clear.
    anchor_index = SERIES.index("2K")
    anchor = normalised[anchor_index]
    registered = []
    series_warps = []
    series_ok = []
    for image, mask in zip(normalised, normalised_masks):
        aligned, warp, ok = align_translation(anchor, image, mask, 10.0)
        registered.append(aligned)
        series_warps.append(warp)
        series_ok.append(ok)
    stack = np.stack(registered)
    mask_stack = np.stack(normalised_masks)
    masked = stack.copy()
    masked[~mask_stack] = np.nan
    shared = quiet_nanmedian(masked, axis=0)
    dispersion = 1.4826 * quiet_nanmedian(
        np.abs(masked - shared[None, :, :]), axis=0
    )
    shared[~np.isfinite(shared)] = 0.0
    dispersion[~np.isfinite(dispersion)] = 0.0
    coverage = np.sum(mask_stack, axis=0).astype(np.uint8)
    persistent = coverage >= 3
    shared[~persistent] = 0.0
    shared = normalise_density(shared, persistent)

    for item, image in zip(results, registered):
        cv2.imwrite(
            str(per_series_dir / f"{item['name']}_background.png"),
            preview_density(image),
        )
    shared_path = output / "shared_background_density.png"
    shared_preview = preview_density(shared, gamma=0.68)
    cv2.imwrite(str(shared_path), shared_preview)
    evidence = cv2.cvtColor(shared_preview, cv2.COLOR_BGR2BGRA)
    evidence[:, :, 3] = np.where(
        persistent,
        np.clip(coverage / len(SERIES) * 255, 0, 255),
        0,
    ).astype(np.uint8)
    cv2.imwrite(str(output / "shared_background_evidence.png"), evidence)
    coverage_preview = cv2.applyColorMap(
        np.clip(coverage * (255 // len(SERIES)), 0, 255).astype(np.uint8),
        cv2.COLORMAP_VIRIDIS,
    )
    cv2.imwrite(str(output / "coverage.png"), coverage_preview)
    dispersion_scale = float(np.percentile(dispersion[persistent], 95))
    dispersion_preview = cv2.applyColorMap(
        np.clip(dispersion / max(dispersion_scale, 1e-6) * 255, 0, 255).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    cv2.imwrite(str(output / "dispersion.png"), dispersion_preview)

    rows = []
    for index, item in enumerate(results):
        mask = persistent & normalised_masks[index]
        rows.append({
            "series": item["name"],
            "stamps": item["stamps"],
            "colour_direction_separation": item["colour_direction_separation"],
            "within_series_background_registrations": item["stamp_registration_successes"],
            "series_shift_x_analysis_px": float(series_warps[index][0, 2]),
            "series_shift_y_analysis_px": float(series_warps[index][1, 2]),
            "series_registration_ok": bool(series_ok[index]),
            "correlation_with_shared": correlation(registered[index], shared, mask),
        })
    payload = {
        "format": "experimental-shared-background",
        "version": 1,
        "series": list(SERIES),
        "method": (
            "colour-seeded two-ray optical-density unmixing; independent "
            "background translation per stamp; equal-weight robust median "
            "across denomination medians"
        ),
        "downsample": args.downsample,
        "target_height_px": target[0],
        "target_width_px": target[1],
        "persistent_pixel_fraction": float(np.mean(persistent)),
        "median_persistent_dispersion": float(np.median(dispersion[persistent])),
        "output": shared_path.name,
        "per_series": rows,
    }
    (output / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {shared_path}")


if __name__ == "__main__":
    main()
