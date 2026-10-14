"""REJECTED EXPERIMENT: conservative shared background for the 1K--7K series.

High-confidence seeds avoided the worst complement-mask error, but pooled
cross-denomination averaging still smeared contours and did not improve the
background reconstruction.  Retained only as a negative result.

Unlike ``shared_background.py`` this version never interprets the complement
of a colour component as background.  Only high-confidence pixels belonging
to the weak colour component are observations; everything else is unknown.
Geometry is translation-only because the five detected design rectangles have
the same size to within two source pixels at 1200 dpi.
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


def optical_density(image: np.ndarray, paper: np.ndarray) -> np.ndarray:
    result = -np.log(
        np.clip(
            (image.astype(np.float64) + 2.0)
            / (paper[None, None, :].astype(np.float64) + 2.0),
            1e-3,
            1.5,
        )
    )
    return np.clip(result, 0.0, None).astype(np.float32)


def apply_affine(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:, :2].T + matrix[:, 2]


def warp_observation(
    values: np.ndarray,
    observed: np.ndarray,
    warp: np.ndarray,
    size: tuple[int, int],
    inverse: bool,
) -> tuple[np.ndarray, np.ndarray]:
    flags = cv2.INTER_CUBIC | (cv2.WARP_INVERSE_MAP if inverse else 0)
    mask_flags = cv2.INTER_NEAREST | (cv2.WARP_INVERSE_MAP if inverse else 0)
    width, height = size
    moved = cv2.warpAffine(values, warp, (width, height), flags=flags, borderValue=0)
    moved_mask = cv2.warpAffine(
        observed.astype(np.uint8), warp, (width, height),
        flags=mask_flags, borderValue=0,
    ) > 0
    return moved, moved_mask


def fit_translation(
    template: np.ndarray,
    moving: np.ndarray,
    window: np.ndarray,
    limit: float,
) -> tuple[np.ndarray, bool]:
    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], np.float32)
    if window.sum() < 300:
        return identity, False
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 180, 1e-6)
    try:
        _, fitted = cv2.findTransformECC(
            template.astype(np.float32),
            moving.astype(np.float32),
            identity.copy(),
            cv2.MOTION_TRANSLATION,
            criteria,
            window.astype(np.uint8) * 255,
            5,
        )
    except cv2.error:
        return identity, False
    if not np.all(np.isfinite(fitted)) or np.linalg.norm(fitted[:, 2]) > limit:
        return identity, False
    return fitted, True


def paper_from_quiet(image: np.ndarray, quiet: np.ndarray) -> np.ndarray:
    pixels = image[quiet].astype(np.float64)
    if len(pixels) < 500:
        pixels = image.reshape(-1, 3).astype(np.float64)
    return np.percentile(pixels, 90, axis=0)


def colour_membership(
    density: np.ndarray,
    centres: np.ndarray,
    group: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    magnitude = np.linalg.norm(density, axis=2)
    direction = density / np.maximum(magnitude[:, :, None], 1e-8)
    distances = np.stack(
        [np.sum((direction - centre[None, None, :]) ** 2, axis=2)
         for centre in centres]
    )
    other = 1 - group
    margin = (distances[other] - distances[group]) / (
        distances[other] + distances[group] + 1e-6
    )
    return magnitude, margin, distances[group] < distances[other]


def inside_rectangle(shape: tuple[int, int], rectangle: dict, downsample: int) -> np.ndarray:
    height, width = shape
    mask = np.zeros(shape, np.uint8)
    inset = max(3, int(round(8 / downsample)))
    left = int(round(rectangle["left_px"] / downsample)) + inset
    right = int(round(rectangle["right_px"] / downsample)) - inset
    top = int(round(rectangle["top_px"] / downsample)) + inset
    bottom = int(round(rectangle["bottom_px"] / downsample)) - inset
    cv2.rectangle(mask, (left, top), (right, bottom), 1, -1)
    return mask > 0


def one_series(
    scan: Path,
    perf: dict,
    design: dict,
    downsample: int,
    seed: int,
) -> dict:
    images, valid, numbers = aligned_collection(scan, perf, design, downsample)
    reference = masked_colour_median(images, valid)
    grey = luminance_bgr(reference)
    paper_level = float(np.percentile(grey[grey > 0], 92))
    darkness = np.clip((paper_level - grey) / max(paper_level, 1.0), 0.0, 1.0)
    control = colour_partition(
        reference, darkness, np.random.default_rng(seed), max_samples=80000
    )
    centres = np.asarray(control["od_direction_centres_bgr"], np.float32)
    reference_density = optical_density(reference, control["paper_bgr"])
    magnitude = np.linalg.norm(reference_density, axis=2)
    median_by_group = [
        float(np.median(magnitude[control["assignment"] == group]))
        for group in (0, 1)
    ]
    background_group = int(np.argmin(median_by_group))
    reference_magnitude, margin, member = colour_membership(
        reference_density, centres, background_group
    )
    inside = inside_rectangle(reference_magnitude.shape, design["golden"]["rectangle"], downsample)
    positive = reference_magnitude[inside & member]
    floor = float(np.percentile(positive, 12)) if len(positive) else 0.0
    # The colour margin is intentionally permissive: it is only a seed. A
    # pixel must recur across stamps and denominations before publication.
    seed_mask = inside & member & (margin > 0.05) & (reference_magnitude > floor)
    seed_template = np.where(seed_mask, reference_magnitude, 0.0)
    scale = float(np.percentile(seed_template[seed_mask], 97))
    seed_template = np.clip(seed_template / max(scale, 1e-6), 0, 1.5)
    search_window = cv2.dilate(seed_mask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    quiet = inside & (reference_magnitude <= np.percentile(reference_magnitude[inside], 25))

    observations = []
    observation_masks = []
    shifts = []
    successes = 0
    for image, image_valid in zip(images, valid):
        paper = paper_from_quiet(image, quiet & image_valid)
        density = optical_density(image, paper)
        stamp_magnitude, stamp_margin, stamp_member = colour_membership(
            density, centres, background_group
        )
        candidate = (
            inside & image_valid & stamp_member & (stamp_margin > 0.02)
            & (stamp_magnitude > floor * 0.65)
        )
        response = np.where(candidate, stamp_magnitude / max(scale, 1e-6), 0.0)
        warp, ok = fit_translation(seed_template, response, search_window & image_valid, 5.0)
        aligned, aligned_mask = warp_observation(
            response, candidate, warp,
            (response.shape[1], response.shape[0]), inverse=True,
        )
        aligned[~aligned_mask] = np.nan
        observations.append(aligned)
        observation_masks.append(aligned_mask)
        shifts.append(warp[:, 2].copy())
        successes += int(ok)

    stack = np.stack(observations)
    observed_count = np.sum(np.stack(observation_masks), axis=0)
    required = max(3, int(np.ceil(0.30 * len(images))))
    published = observed_count >= required
    median = quiet_nanmedian(stack, axis=0)
    median[~np.isfinite(median) | ~published] = 0.0
    return {
        "name": scan.stem,
        "background": median.astype(np.float32),
        "observed": published,
        "rectangle": design["golden"]["rectangle"],
        "stamps": len(numbers),
        "successes": successes,
        "shifts": np.asarray(shifts),
        "direction_separation": control["direction_separation"],
        "seed_fraction": float(np.mean(seed_mask)),
    }


def place_by_frame_centre(
    values: np.ndarray,
    observed: np.ndarray,
    rectangle: dict,
    downsample: int,
    target_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = target_shape
    source_cx = (rectangle["left_px"] + rectangle["right_px"]) / (2 * downsample)
    source_cy = (rectangle["top_px"] + rectangle["bottom_px"]) / (2 * downsample)
    warp = np.array(
        [[1.0, 0.0, width / 2.0 - source_cx],
         [0.0, 1.0, height / 2.0 - source_cy]],
        np.float32,
    )
    moved, moved_mask = warp_observation(
        values, observed, warp, (width, height), inverse=False
    )
    moved[~moved_mask] = np.nan
    return moved, moved_mask, warp


def density_preview(values: np.ndarray, observed: np.ndarray) -> np.ndarray:
    present = values[observed & (values > 0)]
    high = float(np.percentile(present, 98)) if len(present) else 1.0
    ink = np.power(np.clip(values / max(high, 1e-6), 0, 1), 0.72)
    grey = np.full(values.shape, 255, np.uint8)
    grey[observed] = np.clip(255 * (1.0 - ink[observed]), 0, 255).astype(np.uint8)
    return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Conservative shared sparse background for 1K--7K"
    )
    parser.add_argument("data", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()
    output = args.output or args.data / "shared_background_sparse"
    output.mkdir(parents=True, exist_ok=True)
    per_series = output / "per_series"
    per_series.mkdir(parents=True, exist_ok=True)

    results = []
    for index, name in enumerate(SERIES):
        jpg = args.data / f"{name}.jpg"
        png = args.data / f"{name}.png"
        scan = jpg if jpg.exists() else png
        perf = json.loads((args.data / f"data_{name}_perf.json").read_text(encoding="utf-8"))
        design = json.loads((args.data / f"data_{name}_design.json").read_text(encoding="utf-8"))
        print(f"{name}: extracting sparse background evidence ...")
        results.append(one_series(
            scan, perf, design, args.downsample, args.seed + index
        ))

    widths = [item["background"].shape[1] for item in results]
    heights = [item["background"].shape[0] for item in results]
    target_shape = (max(heights), int(np.median(widths)))
    placed = []
    placed_masks = []
    frame_warps = []
    for item in results:
        values, observed, warp = place_by_frame_centre(
            item["background"], item["observed"], item["rectangle"],
            args.downsample, target_shape,
        )
        placed.append(values)
        placed_masks.append(observed)
        frame_warps.append(warp)

    # One conservative translation refinement against the current median. It
    # is accepted only inside five analysis pixels and never changes scale.
    initial = quiet_nanmedian(np.stack(placed), axis=0)
    initial[~np.isfinite(initial)] = 0.0
    refined = []
    refined_masks = []
    refinement_warps = []
    refinement_ok = []
    for values, observed in zip(placed, placed_masks):
        moving = np.nan_to_num(values, nan=0.0)
        overlap = observed & (initial > 0)
        warp, ok = fit_translation(initial, moving, overlap, 5.0)
        moved, moved_mask = warp_observation(
            moving, observed, warp,
            (target_shape[1], target_shape[0]), inverse=True,
        )
        moved[~moved_mask] = np.nan
        refined.append(moved)
        refined_masks.append(moved_mask)
        refinement_warps.append(warp)
        refinement_ok.append(ok)

    stack = np.stack(refined)
    count = np.sum(np.stack(refined_masks), axis=0)
    shared_mask = count >= 3
    shared = quiet_nanmedian(stack, axis=0)
    shared[~np.isfinite(shared) | ~shared_mask] = 0.0

    for item, values, observed in zip(results, refined, refined_masks):
        cv2.imwrite(
            str(per_series / f"{item['name']}_sparse_background.png"),
            density_preview(np.nan_to_num(values, nan=0.0), observed),
        )
    preview = density_preview(shared, shared_mask)
    cv2.imwrite(str(output / "shared_background_sparse.png"), preview)
    evidence = cv2.cvtColor(preview, cv2.COLOR_BGR2BGRA)
    evidence[:, :, 3] = np.where(
        shared_mask, np.clip(count / len(SERIES) * 255, 0, 255), 0
    ).astype(np.uint8)
    cv2.imwrite(str(output / "shared_background_sparse_evidence.png"), evidence)
    cv2.imwrite(
        str(output / "agreement.png"),
        cv2.applyColorMap(
            np.clip(count / len(SERIES) * 255, 0, 255).astype(np.uint8),
            cv2.COLORMAP_VIRIDIS,
        ),
    )

    payload = {
        "format": "experimental-sparse-shared-background",
        "version": 1,
        "series": list(SERIES),
        "downsample": args.downsample,
        "source_stamps": int(sum(item["stamps"] for item in results)),
        "published_pixel_fraction": float(np.mean(shared_mask)),
        "rule": (
            "only recurrent pixels assigned to the weak colour component; "
            "complement remains unknown; translation-only registration"
        ),
        "per_series": [
            {
                "series": item["name"],
                "stamps": item["stamps"],
                "background_registration_successes": item["successes"],
                "colour_direction_separation": item["direction_separation"],
                "seed_fraction": item["seed_fraction"],
                "cross_series_refinement_ok": bool(refinement_ok[index]),
                "cross_series_shift_x_px": float(refinement_warps[index][0, 2]),
                "cross_series_shift_y_px": float(refinement_warps[index][1, 2]),
            }
            for index, item in enumerate(results)
        ],
        "outputs": {
            "background": "shared_background_sparse.png",
            "evidence": "shared_background_sparse_evidence.png",
            "agreement": "agreement.png",
        },
    }
    (output / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {output / 'shared_background_sparse.png'}")


if __name__ == "__main__":
    main()
