"""CURRENT EXPERIMENT: recover the faint third ink of XIX/1R before averaging.

The observable palette is modelled as three printing inks plus paper.  The two
strong inks can be initialised from the registered median.  The pale background
ink cannot: its fine lines have different sub-pixel translations and largely
cancel in that median.  For that layer this experiment forms a colour-selective
band-pass response on every individual stamp, chooses a sharp single stamp as
the initial template, estimates independent translations, and only then forms
the robust average.

This is experimental code and does not modify the production pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from pathlib import Path

import cv2
import numpy as np

from three_colour_refinement import (
    add_label,
    bgr_hex,
    estimate_papers,
    optical_density,
    paper_ring,
    register_one_colour,
    render_combined,
    render_density,
    shift_image,
    spherical_kmeans,
)
from two_pass_decomposition import aligned_collection, masked_colour_median


def learn_three_ink_directions(
    reference: np.ndarray,
    covered: np.ndarray,
    paper: np.ndarray,
    max_samples: int,
) -> dict:
    """Find the dark, red and pale ink rays; paper remains a separate colour."""
    magnitude, direction = optical_density(reference, paper)
    observed = magnitude[covered]
    cutoff = max(0.04, float(np.percentile(observed, 24)))
    usable = covered & (magnitude >= cutoff)
    vectors = direction[usable]
    strengths = magnitude[usable]

    rng = np.random.default_rng(20261012)
    if len(vectors) > max_samples:
        bins = np.quantile(strengths, [0.0, 0.2, 0.45, 0.7, 1.0])
        selected: list[np.ndarray] = []
        quota = max_samples // 4
        for low, high in zip(bins[:-1], bins[1:]):
            candidates = np.flatnonzero((strengths >= low) & (strengths <= high))
            if len(candidates) > quota:
                candidates = rng.choice(candidates, quota, replace=False)
            selected.append(candidates)
        sample = vectors[np.concatenate(selected)]
    else:
        sample = vectors

    _, centres, loss = spherical_kmeans(sample, 3)
    similarity = direction @ centres.T
    assignment = np.argmax(similarity, axis=2)
    ordered = np.sort(similarity, axis=2)
    margin = ordered[:, :, -1] - ordered[:, :, -2]
    confident = usable & (np.max(similarity, axis=2) >= 0.96) & (margin >= 0.007)

    colours: list[np.ndarray] = []
    representative_strengths: list[float] = []
    counts: list[int] = []
    for group in range(3):
        values = magnitude[confident & (assignment == group)]
        if len(values) < 500:
            raise RuntimeError(f"Ink candidate {group + 1} has too few pixels")
        strength = float(np.percentile(values, 78))
        colour = (paper + 2.0) * np.exp(-centres[group] * strength) - 2.0
        colours.append(np.clip(colour, 0.0, 255.0))
        representative_strengths.append(strength)
        counts.append(int(len(values)))

    luminance = np.asarray(
        [0.114 * colour[0] + 0.587 * colour[1] + 0.299 * colour[2] for colour in colours]
    )
    order = np.argsort(luminance)
    inverse = np.zeros(3, np.int16)
    inverse[order] = np.arange(3)
    centres = centres[order]
    colours = [colours[index] for index in order]
    representative_strengths = [representative_strengths[index] for index in order]
    counts = [counts[index] for index in order]
    assignment = inverse[assignment]

    angles = np.zeros((3, 3), np.float64)
    for first in range(3):
        for second in range(3):
            cosine = float(np.clip(centres[first] @ centres[second], -1.0, 1.0))
            angles[first, second] = math.degrees(math.acos(cosine))
    return {
        "centres": centres,
        "colours": np.asarray(colours),
        "representative_strengths": representative_strengths,
        "counts": counts,
        "cutoff": cutoff,
        "loss": loss,
        "angles": angles,
        "assignment": assignment,
        "confident": confident,
    }


def strong_ink_responses(
    image: np.ndarray,
    paper: np.ndarray,
    centres: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    magnitude, direction = optical_density(image, paper)
    similarity = direction @ centres.T
    assignment = np.argmax(similarity, axis=2)
    ordered = np.sort(similarity, axis=2)
    margin = ordered[:, :, -1] - ordered[:, :, -2]
    confident = valid & (magnitude >= 0.035) & (np.max(similarity, axis=2) >= 0.95)
    confident &= margin >= 0.005

    # Beer-Lambert mixing is additive in optical density.  A regularised
    # non-negative decomposition can therefore retain pale ink beneath/near a
    # stronger impression instead of forcing every pixel into exactly one bin.
    od = direction * magnitude[:, :, None]
    gram = centres @ centres.T
    inverse = np.linalg.inv(gram + 0.004 * np.eye(3, dtype=np.float32))
    coefficients = od @ centres.T @ inverse
    coefficients = np.clip(coefficients, 0.0, 4.0).astype(np.float32)
    coefficients[~valid] = 0.0

    responses = np.zeros((*magnitude.shape, 2), np.float32)
    for group in range(2):
        selected = confident & (assignment == group)
        responses[:, :, group][selected] = magnitude[selected]
    return responses, magnitude, similarity, assignment, coefficients


def faint_pattern_response(
    coefficients: np.ndarray,
    valid: np.ndarray,
    downsample: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Colour-selective local contrast for the pale printed lines."""
    raw = coefficients[:, :, 2]

    # Remove strong dark/red areas with a safety halo.  They must never anchor
    # the weak-layer registration through their much stronger edges.
    # Low-density paper/pattern pixels have unstable colour directions and may
    # momentarily look closer to a strong-ink ray.  Excluding them was the
    # reason the first version erased the security network.  Mask only truly
    # dense strong ink, with a small halo around its strokes.
    other = valid & (np.sum(coefficients[:, :, :2], axis=2) > 0.48)
    radius = max(2, round(4 / downsample))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    allowed = valid & ~(cv2.dilate(other.astype(np.uint8), kernel) > 0)

    sigma_small = max(0.45, 0.75 / downsample)
    sigma_large = max(2.0, 5.0 / downsample)
    fine = cv2.GaussianBlur(raw, (0, 0), sigma_small)
    local_background = cv2.GaussianBlur(raw, (0, 0), sigma_large)
    band = np.clip(fine - local_background, 0.0, None)
    band[~allowed] = 0.0

    # Map local contrast back to a plausible optical-density scale for colour
    # rendering. Registration itself is invariant to this scalar.
    positive = band[band > 0]
    if len(positive):
        scale = float(np.percentile(positive, 97))
        band = np.clip(band / max(scale, 1e-7) * 0.30, 0.0, 0.65)
    return band.astype(np.float32), allowed


def ecc_translation(
    template: np.ndarray,
    moving: np.ndarray,
    mask: np.ndarray,
    initial: np.ndarray | None = None,
    limit: float = 14.0,
) -> tuple[np.ndarray, float]:
    template_positive = template[mask & (template > 0)]
    moving_positive = moving[mask & (moving > 0)]
    if len(template_positive) < 300 or len(moving_positive) < 300:
        return np.zeros(2, np.float32), float("nan")
    target = np.clip(template / max(float(np.percentile(template_positive, 98)), 1e-7), 0, 1)
    source = np.clip(moving / max(float(np.percentile(moving_positive, 98)), 1e-7), 0, 1)
    shift = np.zeros(2, np.float32) if initial is None else initial.astype(np.float32).copy()
    warp = np.array([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]]], np.float32)
    try:
        score, warp = cv2.findTransformECC(
            target.astype(np.float32),
            source.astype(np.float32),
            warp,
            cv2.MOTION_TRANSLATION,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 160, 1e-7),
            inputMask=mask.astype(np.uint8) * 255,
            gaussFiltSize=3,
        )
    except cv2.error:
        return np.zeros(2, np.float32), float("nan")
    candidate = warp[:, 2]
    if not np.all(np.isfinite(candidate)) or np.max(np.abs(candidate)) > limit:
        return np.zeros(2, np.float32), float("nan")
    return candidate.astype(np.float32), float(score)


def register_faint_pattern(
    responses: np.ndarray,
    allowed: np.ndarray,
    valid: np.ndarray,
    ring: np.ndarray,
    anchor_index: int | None = None,
    shift_limit: float = 4.0,
) -> dict:
    """Start from the best individual stamp, never from the blurred median."""
    count = len(responses)
    quorum = valid.sum(axis=0) >= max(3, math.ceil(0.6 * count))
    base_area = ring & quorum

    candidate_scores = np.full((count, count), np.nan, np.float32)
    candidate_shifts = np.zeros((count, count, 2), np.float32)
    for reference_index in range(count):
        for moving_index in range(count):
            common = base_area & allowed[reference_index] & allowed[moving_index]
            shift, score = ecc_translation(
                responses[reference_index],
                responses[moving_index],
                common,
                limit=shift_limit,
            )
            candidate_scores[reference_index, moving_index] = score
            candidate_shifts[reference_index, moving_index] = shift

    quality = np.nanmedian(candidate_scores, axis=1)
    anchor = int(np.nanargmax(quality)) if anchor_index is None else anchor_index
    shifts = candidate_shifts[anchor].copy()
    scores = candidate_scores[anchor].copy()
    good = np.isfinite(scores)
    shifts[good] -= np.median(shifts[good], axis=0)

    def aligned_median(current_shifts: np.ndarray) -> np.ndarray:
        stack: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for response, mask, shift in zip(responses, valid, current_shifts):
            stack.append(shift_image(response, shift, cv2.INTER_CUBIC))
            masks.append(shift_image(mask.astype(np.uint8), shift, cv2.INTER_NEAREST) > 0)
        values = np.stack(stack)
        coverage = np.stack(masks)
        values[~coverage] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = np.nanmedian(values, axis=0)
        return np.nan_to_num(result).astype(np.float32)

    initial_values = responses.copy()
    initial_values[~valid] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        initial = np.nanmedian(initial_values, axis=0)
    initial = np.nan_to_num(initial).astype(np.float32)
    template = aligned_median(shifts)

    # One refinement against the now coherent multi-stamp template.
    support = cv2.dilate((template > 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    for index in range(count):
        common = base_area & allowed[index] & support
        shift, score = ecc_translation(
            template,
            responses[index],
            common,
            shifts[index],
            limit=shift_limit,
        )
        if np.isfinite(score):
            shifts[index] = shift
            scores[index] = score
    good = np.isfinite(scores)
    shifts[good] -= np.median(shifts[good], axis=0)
    refined = aligned_median(shifts)

    return {
        "initial": initial,
        "refined": refined,
        "shifts": shifts,
        "scores": scores,
        "anchor_index": anchor,
        "anchor_quality": float(quality[anchor]),
        "candidate_quality": quality,
    }


def contrast_preview(density: np.ndarray) -> np.ndarray:
    positive = density[density > 0]
    scale = float(np.percentile(positive, 98)) if len(positive) else 1.0
    grey = 255.0 * (1.0 - np.clip(density / max(scale, 1e-8), 0.0, 1.0))
    return grey.astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scan", type=Path)
    parser.add_argument("--perf", type=Path)
    parser.add_argument("--design", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument("--max-colour-samples", type=int, default=180000)
    parser.add_argument(
        "--pattern-anchor",
        type=int,
        help="Stamp number used as the sharp initial pale-pattern template",
    )
    args = parser.parse_args()

    scan = args.scan.resolve()
    perf_path = args.perf or scan.with_name(f"data_{scan.stem}_perf.json")
    design_path = args.design or scan.with_name(f"data_{scan.stem}_design.json")
    output = args.output or scan.parents[1] / "experiments" / f"{scan.stem}_background_pattern"
    output.mkdir(parents=True, exist_ok=True)

    perf = json.loads(perf_path.read_text(encoding="utf-8"))
    design = json.loads(design_path.read_text(encoding="utf-8"))
    images, valid, numbers = aligned_collection(scan, perf, design, args.downsample)
    reference = masked_colour_median(images, valid)
    covered = valid.sum(axis=0) >= max(3, math.ceil(0.6 * len(images)))
    ring = paper_ring(design, reference.shape[:2], args.downsample)
    papers = estimate_papers(images, valid, ring)
    paper = np.median(papers, axis=0)
    learned = learn_three_ink_directions(
        reference, covered, paper, args.max_colour_samples
    )

    strong_stacks = [[], []]
    pattern_stack: list[np.ndarray] = []
    allowed_stack: list[np.ndarray] = []
    for image, mask, stamp_paper in zip(images, valid, papers):
        strong, _magnitude, _similarity, _assignment, coefficients = strong_ink_responses(
            image, stamp_paper, learned["centres"], mask
        )
        for group in range(2):
            strong_stacks[group].append(strong[:, :, group])
        pattern, allowed = faint_pattern_response(
            coefficients, mask, args.downsample
        )
        pattern_stack.append(pattern)
        allowed_stack.append(allowed)

    strong_results = [
        register_one_colour(np.stack(stack), valid, iterations=2)
        for stack in strong_stacks
    ]
    anchor_index = None
    if args.pattern_anchor is not None:
        if args.pattern_anchor not in numbers:
            raise RuntimeError(f"Pattern anchor stamp {args.pattern_anchor} is unavailable")
        anchor_index = numbers.index(args.pattern_anchor)
    pattern_result = register_faint_pattern(
        np.stack(pattern_stack),
        np.stack(allowed_stack),
        valid,
        ring,
        anchor_index=anchor_index,
        shift_limit=4.0 / args.downsample,
    )
    results = strong_results + [pattern_result]

    initial_layers = [result["initial"] for result in results]
    refined_layers = [result["refined"] for result in results]
    before = render_combined(initial_layers, paper, learned["centres"])
    after = render_combined(refined_layers, paper, learned["centres"])
    cv2.imwrite(str(output / "four_colour_before.png"), before)
    cv2.imwrite(str(output / "four_colour_refined.png"), after)
    cv2.imwrite(str(output / "registered_colour_median.png"), reference.astype(np.uint8))

    diagnostic_panels = []
    for number, image, response in zip(numbers, images, pattern_stack):
        preview = image.copy()
        inset = contrast_preview(response)
        inset = cv2.cvtColor(inset, cv2.COLOR_GRAY2BGR)
        inset_width = preview.shape[1] // 3
        inset_height = preview.shape[0] // 3
        preview[:inset_height, :inset_width] = cv2.resize(
            inset, (inset_width, inset_height), interpolation=cv2.INTER_AREA
        )
        diagnostic_panels.append(add_label(preview, f"stamp {number}; inset=pale response"))
    diagnostic_rows = []
    for start in range(0, len(diagnostic_panels), 5):
        row = diagnostic_panels[start : start + 5]
        while len(row) < 5:
            row.append(np.full_like(diagnostic_panels[0], 245))
        diagnostic_rows.append(np.hstack(row))
    cv2.imwrite(str(output / "individual_pattern_inputs.png"), np.vstack(diagnostic_rows))

    panels = [
        add_label(reference.astype(np.uint8), "registered colour median"),
        add_label(before, "3 inks + paper before layer shifts"),
        add_label(after, "3 inks + paper after layer shifts"),
    ]
    ink_rows: list[dict] = []
    for group, result in enumerate(results):
        layer = render_density(result["refined"], paper, learned["centres"][group])
        cv2.imwrite(str(output / f"ink_{group + 1}_refined.png"), layer)
        cv2.imwrite(
            str(output / f"ink_{group + 1}_density.png"),
            contrast_preview(result["refined"]),
        )
        colour_hex = bgr_hex(learned["colours"][group])
        label = "pale background ink" if group == 2 else f"ink {group + 1}"
        panels.append(add_label(layer, f"{label}: {colour_hex}"))
        shifts = result["shifts"].astype(np.float64) * args.downsample
        ink_rows.append(
            {
                "ink": group + 1,
                "role": "background_pattern" if group == 2 else "strong_design",
                "hex": colour_hex,
                "representative_bgr": learned["colours"][group].tolist(),
                "od_direction_bgr": learned["centres"][group].tolist(),
                "shift_span_x_px": float(np.ptp(shifts[:, 0])),
                "shift_span_y_px": float(np.ptp(shifts[:, 1])),
                "median_ecc": float(np.nanmedian(result["scores"])),
            }
        )
    paper_panel = np.empty_like(reference, np.uint8)
    paper_panel[:] = np.clip(np.round(paper), 0, 255).astype(np.uint8)
    panels.append(add_label(paper_panel, f"paper: {bgr_hex(paper)}"))

    height, width = reference.shape[:2]
    montage_rows = []
    for start in range(0, len(panels), 3):
        row = panels[start : start + 3]
        while len(row) < 3:
            row.append(np.full((height, width, 3), 245, np.uint8))
        montage_rows.append(np.hstack(row))
    cv2.imwrite(str(output / "comparison_montage.png"), np.vstack(montage_rows))

    # High-contrast pattern-only comparison at native resolution.
    pattern_before = cv2.cvtColor(contrast_preview(pattern_result["initial"]), cv2.COLOR_GRAY2BGR)
    pattern_after = cv2.cvtColor(contrast_preview(pattern_result["refined"]), cv2.COLOR_GRAY2BGR)
    cv2.imwrite(
        str(output / "background_pattern_before_after.png"),
        np.hstack(
            [
                add_label(pattern_before, "background pattern before"),
                add_label(pattern_after, "background pattern after independent shifts"),
            ]
        ),
    )

    shift_rows = []
    for index, number in enumerate(numbers):
        row: dict[str, int | float] = {"stamp": number}
        for group, result in enumerate(results):
            row[f"ink_{group + 1}_dx_px"] = float(result["shifts"][index, 0] * args.downsample)
            row[f"ink_{group + 1}_dy_px"] = float(result["shifts"][index, 1] * args.downsample)
            row[f"ink_{group + 1}_ecc"] = float(result["scores"][index])
        shift_rows.append(row)
    with (output / "per_stamp_shifts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(shift_rows[0]))
        writer.writeheader()
        writer.writerows(shift_rows)

    report = {
        "method": "three inks plus paper; pale pattern initialised from one sharp stamp",
        "downsample": args.downsample,
        "stamps": numbers,
        "paper_hex": bgr_hex(paper),
        "paper_bgr": paper.tolist(),
        "ink_angles_deg": learned["angles"].tolist(),
        "inks": ink_rows,
        "background_anchor_stamp": numbers[pattern_result["anchor_index"]],
        "background_anchor_quality": pattern_result["anchor_quality"],
        "background_candidate_quality": pattern_result["candidate_quality"].tolist(),
        "background_input_stats": [
            {
                "stamp": number,
                "positive_pixels": int(np.sum(response > 0)),
                "p99": float(np.percentile(response[response > 0], 99))
                if np.any(response > 0)
                else 0.0,
                "ring_positive_pixels": int(np.sum((response > 0) & ring)),
            }
            for number, response in zip(numbers, pattern_stack)
        ],
        "caveat": (
            "The pale layer is a colour-selective local-contrast reconstruction. "
            "Broad low-frequency tone is assigned to paper so paper ageing does not "
            "become a fictitious printed pattern."
        ),
    }
    (output / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
