"""INTERMEDIATE EXPERIMENT: two strong inks plus paper.

This version improved the dark/red layers of XIX/1R, but it incorrectly treated
the pale security underprint as paper variation.  Keep it as a documented
baseline; use ``background_pattern_refinement.py`` for the current 1R model.

This script is intentionally isolated from the production pipeline.  It uses the
existing perforation/design JSON only to put all stamps into one coordinate
system.  It then:

1. estimates paper from the unprinted margin of every stamp;
2. finds exactly two ink directions in RGB optical-density space (the third
   known colour is paper, not another printed layer);
3. keeps only high-confidence pixels for each colour (cancellation and mixed
   pixels stay unassigned);
4. estimates an independent sub-pixel translation for both inks on every
   stamp; and
5. forms a robust median pattern for each colour separately.

No complement of a colour mask is ever treated as another printing layer.
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

from two_pass_decomposition import aligned_collection, masked_colour_median


def bgr_hex(colour: np.ndarray) -> str:
    b, g, r = np.clip(np.round(colour), 0, 255).astype(np.uint8)
    return f"#{r:02X}{g:02X}{b:02X}"


def paper_ring(design: dict, shape: tuple[int, int], downsample: int) -> np.ndarray:
    """Pixels well outside the measured outer design rule."""
    height, width = shape
    rectangle = design["golden"]["rectangle"]
    left = float(rectangle["left_px"]) / downsample
    right = float(rectangle["right_px"]) / downsample
    top = float(rectangle["top_px"]) / downsample
    bottom = float(rectangle["bottom_px"]) / downsample
    gap = max(8.0, 18.0 / downsample)
    yy, xx = np.mgrid[:height, :width]
    return (
        (xx < left - gap)
        | (xx > right + gap)
        | (yy < top - gap)
        | (yy > bottom + gap)
    )


def estimate_papers(
    images: np.ndarray,
    valid: np.ndarray,
    ring: np.ndarray,
) -> np.ndarray:
    """Per-stamp paper colour, resistant to fibres and isolated ink marks."""
    papers: list[np.ndarray] = []
    for image, mask in zip(images, valid):
        pixels = image[mask & ring].astype(np.float64)
        if len(pixels) < 1000:
            pixels = image[mask].astype(np.float64)
        # The upper-middle marginal colour rejects dark fibres/cancellations,
        # but stays below clipped highlights from the scanner.
        papers.append(np.percentile(pixels, 72, axis=0))
    return np.asarray(papers, np.float64)


def optical_density(image: np.ndarray, paper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    od = -np.log(
        np.clip(
            (image.astype(np.float32) + 2.0) / (paper[None, None, :] + 2.0),
            1e-3,
            1.35,
        )
    )
    od = np.clip(od, 0.0, 4.0)
    magnitude = np.linalg.norm(od, axis=2)
    direction = od / np.maximum(magnitude[:, :, None], 1e-7)
    return magnitude, direction


def spherical_kmeans(
    vectors: np.ndarray,
    clusters: int,
    attempts: int = 16,
    seed: int = 20261012,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Deterministic multi-start spherical k-means."""
    rng = np.random.default_rng(seed)
    best: tuple[np.ndarray, np.ndarray, float] | None = None
    count = len(vectors)
    if count < clusters:
        raise RuntimeError("Too few colour samples")

    for _ in range(attempts):
        first = int(rng.integers(count))
        chosen = [first]
        nearest = np.sum((vectors - vectors[first]) ** 2, axis=1)
        for _group in range(1, clusters):
            # k-means++ with a tiny floor so identical points are harmless.
            probability = nearest + 1e-12
            probability /= probability.sum()
            index = int(rng.choice(count, p=probability))
            chosen.append(index)
            distance = np.sum((vectors - vectors[index]) ** 2, axis=1)
            nearest = np.minimum(nearest, distance)
        centres = vectors[chosen].astype(np.float64)

        labels = np.full(count, -1, np.int16)
        for _iteration in range(80):
            similarity = vectors @ centres.T
            updated = np.argmax(similarity, axis=1).astype(np.int16)
            if np.array_equal(updated, labels):
                break
            labels = updated
            replacement = []
            collapsed = False
            for group in range(clusters):
                members = vectors[labels == group]
                if len(members) == 0:
                    collapsed = True
                    break
                centre = members.mean(axis=0)
                centre /= max(float(np.linalg.norm(centre)), 1e-12)
                replacement.append(centre)
            if collapsed:
                break
            centres = np.asarray(replacement)
        if collapsed:
            continue
        loss = float(np.mean(1.0 - np.max(vectors @ centres.T, axis=1)))
        if best is None or loss < best[2]:
            best = labels, centres.astype(np.float32), loss

    if best is None:
        raise RuntimeError("Three-colour clustering collapsed")
    return best


def learn_ink_directions(
    reference: np.ndarray,
    covered: np.ndarray,
    paper: np.ndarray,
    ring: np.ndarray,
    max_samples: int,
) -> dict:
    magnitude, direction = optical_density(reference, paper)
    observed = magnitude[covered]
    if len(observed) < 10000:
        raise RuntimeError("Too little common stamp coverage")

    # A fixed floor protects pale ink; the percentile removes paper texture.
    cutoff = max(0.045, float(np.percentile(observed, 30)))
    usable = covered & (magnitude >= cutoff)
    vectors = direction[usable]
    strengths = magnitude[usable]
    rng = np.random.default_rng(20261012)
    if len(vectors) > max_samples:
        # Give pale and dense ink comparable opportunity to seed a centre.
        bins = np.quantile(strengths, [0.0, 0.25, 0.5, 0.75, 1.0])
        selections: list[np.ndarray] = []
        quota = max_samples // 4
        for low, high in zip(bins[:-1], bins[1:]):
            candidates = np.flatnonzero((strengths >= low) & (strengths <= high))
            if len(candidates) > quota:
                candidates = rng.choice(candidates, quota, replace=False)
            selections.append(candidates)
        selected = np.concatenate(selections)
        sample = vectors[selected]
    else:
        sample = vectors

    # Three direction clusters are fitted as a diagnostic, but only two are
    # inks.  The third is ordinary paper/fibre variation and is identified by
    # its presence in the unprinted outer margin plus its low density.
    _, candidate_centres, candidate_loss = spherical_kmeans(sample, 3)
    candidate_similarity = direction @ candidate_centres.T
    candidate_assignment = np.argmax(candidate_similarity, axis=2)
    paper_scores: list[float] = []
    candidate_stats: list[dict] = []
    for group in range(3):
        members = usable & (candidate_assignment == group)
        values = magnitude[members]
        strength = float(np.percentile(values, 78)) if len(values) else 0.0
        ring_fraction = float(np.sum(members & ring) / max(1, np.sum(members)))
        score = ring_fraction / max(strength, 0.03)
        paper_scores.append(score)
        candidate_stats.append(
            {
                "group": group,
                "pixels": int(np.sum(members)),
                "strength_p78": strength,
                "ring_fraction": ring_fraction,
                "paper_score": score,
            }
        )
    paper_variation_group = int(np.argmax(paper_scores))
    ink_groups = [group for group in range(3) if group != paper_variation_group]
    centres = candidate_centres[ink_groups]
    loss = candidate_loss
    similarity = direction @ centres.T
    best = np.argmax(similarity, axis=2)
    ordered_similarity = np.sort(similarity, axis=2)
    margin = ordered_similarity[:, :, -1] - ordered_similarity[:, :, -2]
    confident = usable & (np.max(similarity, axis=2) >= 0.965) & (margin >= 0.008)

    # Estimate a representative density and colour, then sort dark-to-light.
    preview_colours: list[np.ndarray] = []
    representative_strengths: list[float] = []
    for group in range(2):
        values = magnitude[confident & (best == group)]
        if len(values) < 200:
            raise RuntimeError(f"Ink {group + 1} has too few confident pixels")
        strength = float(np.percentile(values, 78))
        representative_strengths.append(strength)
        colour = (paper + 2.0) * np.exp(-centres[group] * strength) - 2.0
        preview_colours.append(np.clip(colour, 0.0, 255.0))

    luminance = [
        float(0.114 * colour[0] + 0.587 * colour[1] + 0.299 * colour[2])
        for colour in preview_colours
    ]
    order = np.argsort(luminance)
    inverse = np.zeros(2, np.int16)
    inverse[order] = np.arange(2)
    centres = centres[order]
    preview_colours = [preview_colours[index] for index in order]
    representative_strengths = [representative_strengths[index] for index in order]
    best = inverse[best]

    pairwise_angles = np.zeros((2, 2), np.float64)
    for first in range(2):
        for second in range(2):
            cosine = float(np.clip(centres[first] @ centres[second], -1.0, 1.0))
            pairwise_angles[first, second] = math.degrees(math.acos(cosine))

    return {
        "centres": centres,
        "magnitude": magnitude,
        "direction": direction,
        "assignment": best,
        "confident": confident,
        "cutoff": cutoff,
        "loss": loss,
        "preview_colours": np.asarray(preview_colours),
        "representative_strengths": representative_strengths,
        "pairwise_angles_deg": pairwise_angles,
        "paper_variation_group": paper_variation_group,
        "candidate_cluster_stats": candidate_stats,
    }


def colour_response(
    image: np.ndarray,
    paper: np.ndarray,
    centres: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two mutually exclusive, high-confidence ink-density responses."""
    magnitude, direction = optical_density(image, paper)
    similarity = direction @ centres.T
    assignment = np.argmax(similarity, axis=2)
    ordered = np.sort(similarity, axis=2)
    margin = ordered[:, :, -1] - ordered[:, :, -2]
    confidence = valid & (magnitude >= 0.035) & (np.max(similarity, axis=2) >= 0.955)
    confidence &= margin >= 0.006
    responses = np.zeros((*magnitude.shape, 2), np.float32)
    for group in range(2):
        selected = confidence & (assignment == group)
        responses[:, :, group][selected] = magnitude[selected]
    return responses, assignment.astype(np.uint8), confidence


def shift_image(
    image: np.ndarray,
    shift: np.ndarray,
    interpolation: int,
    inverse: bool = True,
) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = np.array([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]]], np.float32)
    flags = interpolation | (cv2.WARP_INVERSE_MAP if inverse else 0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=flags,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def register_one_colour(
    responses: np.ndarray,
    valid: np.ndarray,
    iterations: int,
) -> dict:
    """Independent translation-only refinement for one colour."""
    count, height, width = responses.shape
    initial_values = responses.copy()
    initial_values[~valid] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        template = np.nanmedian(initial_values, axis=0).astype(np.float32)
    template = np.nan_to_num(template)
    shifts = np.zeros((count, 2), np.float32)
    scores = np.full(count, np.nan, np.float32)

    positive = template[template > 0]
    if len(positive) < 500:
        raise RuntimeError("Colour layer has too little support for registration")

    for _iteration in range(iterations):
        positive = template[template > 0]
        threshold = float(np.percentile(positive, 18))
        support = (template >= threshold).astype(np.uint8)
        support = cv2.dilate(support, np.ones((5, 5), np.uint8), iterations=1)
        template_scale = float(np.percentile(positive, 96))
        target = np.clip(template / max(template_scale, 1e-6), 0.0, 1.0)

        for index, moving_raw in enumerate(responses):
            moving_positive = moving_raw[moving_raw > 0]
            if len(moving_positive) < 300:
                continue
            moving_scale = float(np.percentile(moving_positive, 96))
            moving = np.clip(moving_raw / max(moving_scale, 1e-6), 0.0, 1.0)
            mask = (support & valid[index].astype(np.uint8)).astype(np.uint8) * 255
            warp = np.array(
                [[1.0, 0.0, shifts[index, 0]], [0.0, 1.0, shifts[index, 1]]],
                np.float32,
            )
            try:
                score, warp = cv2.findTransformECC(
                    target,
                    moving,
                    warp,
                    cv2.MOTION_TRANSLATION,
                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 120, 1e-7),
                    inputMask=mask,
                    gaussFiltSize=5,
                )
            except cv2.error:
                continue
            candidate = warp[:, 2]
            if np.all(np.isfinite(candidate)) and np.max(np.abs(candidate)) <= 12.0:
                shifts[index] = candidate
                scores[index] = float(score)

        # Retain relative micro-shifts, not an arbitrary common displacement.
        good = np.isfinite(scores)
        if good.any():
            shifts[good] -= np.median(shifts[good], axis=0)

        aligned: list[np.ndarray] = []
        aligned_valid: list[np.ndarray] = []
        for response, mask, shift in zip(responses, valid, shifts):
            aligned.append(shift_image(response, shift, cv2.INTER_CUBIC))
            aligned_valid.append(shift_image(mask.astype(np.uint8), shift, cv2.INTER_NEAREST) > 0)
        stack = np.stack(aligned)
        masks = np.stack(aligned_valid)
        stack[~masks] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            template = np.nanmedian(stack, axis=0).astype(np.float32)
        template = np.nan_to_num(template)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        initial = np.nanmedian(initial_values, axis=0).astype(np.float32)
    initial = np.nan_to_num(initial)

    positive = template[template > 0]
    threshold = float(np.percentile(positive, 18))
    compare_area = cv2.dilate(
        (template >= threshold).astype(np.uint8), np.ones((5, 5), np.uint8)
    ) > 0

    def cosine_similarity(first: np.ndarray, second: np.ndarray, area: np.ndarray) -> float:
        a = first[area].astype(np.float64)
        b = second[area].astype(np.float64)
        denominator = math.sqrt(float(a @ a) * float(b @ b))
        return float((a @ b) / denominator) if denominator > 1e-12 else float("nan")

    similarity_before: list[float] = []
    similarity_after: list[float] = []
    for response, shift in zip(responses, shifts):
        similarity_before.append(cosine_similarity(response, template, compare_area))
        moved = shift_image(response, shift, cv2.INTER_CUBIC)
        similarity_after.append(cosine_similarity(moved, template, compare_area))
    return {
        "initial": initial,
        "refined": template,
        "shifts": shifts,
        "scores": scores,
        "similarity_before": np.asarray(similarity_before),
        "similarity_after": np.asarray(similarity_after),
    }


def render_density(density: np.ndarray, paper: np.ndarray, direction: np.ndarray) -> np.ndarray:
    od = density[:, :, None] * direction[None, None, :]
    image = (paper[None, None, :] + 2.0) * np.exp(-od) - 2.0
    return np.clip(image, 0, 255).astype(np.uint8)


def render_combined(
    layers: list[np.ndarray], paper: np.ndarray, centres: np.ndarray
) -> np.ndarray:
    od = np.zeros((*layers[0].shape, 3), np.float32)
    for layer, centre in zip(layers, centres):
        od += layer[:, :, None] * centre[None, None, :]
    image = (paper[None, None, :] + 2.0) * np.exp(-od) - 2.0
    return np.clip(image, 0, 255).astype(np.uint8)


def density_preview(density: np.ndarray) -> np.ndarray:
    positive = density[density > 0]
    scale = float(np.percentile(positive, 98)) if len(positive) else 1.0
    grey = 255.0 * (1.0 - np.clip(density / max(scale, 1e-6), 0.0, 1.0))
    return grey.astype(np.uint8)


def add_label(image: np.ndarray, text: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 38), (245, 245, 245), -1)
    cv2.putText(output, text, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (25, 25, 25), 2)
    return output


def write_outputs(
    output: Path,
    numbers: list[int],
    reference: np.ndarray,
    paper: np.ndarray,
    learned: dict,
    results: list[dict],
    downsample: int,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output / "registered_colour_median_before.png"), reference.astype(np.uint8))

    initial_layers = [result["initial"] for result in results]
    refined_layers = [result["refined"] for result in results]
    before = render_combined(initial_layers, paper, learned["centres"])
    after = render_combined(refined_layers, paper, learned["centres"])
    cv2.imwrite(str(output / "three_colour_before.png"), before)
    cv2.imwrite(str(output / "three_colour_refined.png"), after)

    # A 1:1 central crop makes the registration change visible without the
    # application shrinking the whole 1392x1608 canvas to fit the screen.
    height, width = before.shape[:2]
    x0, x1 = int(width * 0.28), int(width * 0.72)
    y0, y1 = int(height * 0.12), int(height * 0.68)
    detail = np.hstack(
        [
            add_label(before[y0:y1, x0:x1], "before"),
            add_label(after[y0:y1, x0:x1], "after independent ink shifts"),
        ]
    )
    cv2.imwrite(str(output / "central_detail_before_after.png"), detail)

    panels = [add_label(reference.astype(np.uint8), "registered colour median")]
    panels.append(add_label(before, "2 inks + paper before independent shifts"))
    panels.append(add_label(after, "2 inks + paper after independent shifts"))

    colour_rows: list[dict] = []
    for group, result in enumerate(results):
        number = group + 1
        layer = render_density(result["refined"], paper, learned["centres"][group])
        cv2.imwrite(str(output / f"colour_{number}_refined.png"), layer)
        cv2.imwrite(
            str(output / f"colour_{number}_density.png"),
            density_preview(result["refined"]),
        )
        colour_hex = bgr_hex(learned["preview_colours"][group])
        panels.append(add_label(layer, f"ink {number}: {colour_hex}"))
        shifts_full = result["shifts"].astype(np.float64) * downsample
        similarity_before = float(np.nanmedian(result["similarity_before"]))
        similarity_after = float(np.nanmedian(result["similarity_after"]))
        colour_rows.append(
            {
                "ink": number,
                "hex": colour_hex,
                "representative_bgr": learned["preview_colours"][group].tolist(),
                "od_direction_bgr": learned["centres"][group].tolist(),
                "representative_od": learned["representative_strengths"][group],
                "confident_reference_pixels": int(
                    np.sum(learned["confident"] & (learned["assignment"] == group))
                ),
                "shift_span_x_px": float(np.ptp(shifts_full[:, 0])),
                "shift_span_y_px": float(np.ptp(shifts_full[:, 1])),
                "ecc_median": float(np.nanmedian(result["scores"])),
                "median_template_similarity_before": similarity_before,
                "median_template_similarity_after": similarity_after,
                "template_similarity_change_percent": float(
                    100.0 * (similarity_after / max(similarity_before, 1e-9) - 1.0)
                ),
            }
        )

    paper_panel = np.empty_like(reference, dtype=np.uint8)
    paper_panel[:] = np.clip(np.round(paper), 0, 255).astype(np.uint8)
    panels.append(add_label(paper_panel, f"paper: {bgr_hex(paper)}"))

    rows = []
    for index, stamp in enumerate(numbers):
        row: dict[str, float | int] = {"stamp": stamp}
        for group, result in enumerate(results):
            row[f"ink_{group + 1}_dx_px"] = float(result["shifts"][index, 0] * downsample)
            row[f"ink_{group + 1}_dy_px"] = float(result["shifts"][index, 1] * downsample)
            row[f"ink_{group + 1}_ecc"] = float(result["scores"][index])
        rows.append(row)
    with (output / "per_stamp_shifts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    panel_height, panel_width = reference.shape[:2]
    montage_rows = []
    for start in range(0, len(panels), 3):
        row_panels = panels[start : start + 3]
        while len(row_panels) < 3:
            row_panels.append(np.full((panel_height, panel_width, 3), 245, np.uint8))
        montage_rows.append(np.hstack(row_panels))
    montage = np.vstack(montage_rows)
    cv2.imwrite(str(output / "comparison_montage.png"), montage)

    report = {
        "method": "two optical-density inks plus paper; independent translation-only ECC; robust median",
        "downsample": downsample,
        "stamp_count": len(numbers),
        "stamps": numbers,
        "paper_bgr": paper.tolist(),
        "paper_hex": bgr_hex(paper),
        "colour_cutoff_od": learned["cutoff"],
        "clustering_loss": learned["loss"],
        "discarded_paper_variation_cluster": learned["paper_variation_group"],
        "candidate_cluster_stats": learned["candidate_cluster_stats"],
        "pairwise_colour_angles_deg": learned["pairwise_angles_deg"].tolist(),
        "ink_colours": colour_rows,
        "three_colours": [
            {"role": "ink_1", "hex": colour_rows[0]["hex"]},
            {"role": "ink_2", "hex": colour_rows[1]["hex"]},
            {"role": "paper", "hex": bgr_hex(paper)},
        ],
        "caveat": (
            "Exactly three total colours are an explicit prior: two inks and paper. "
            "Pixels that do not match an ink ray confidently are excluded, not "
            "assigned by complement. Paper is never registered as a moving layer."
        ),
    }
    (output / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("scan", type=Path)
    parser.add_argument("--perf", type=Path)
    parser.add_argument("--design", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--max-colour-samples", type=int, default=180000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scan = args.scan.resolve()
    perf_path = args.perf or scan.with_name(f"data_{scan.stem}_perf.json")
    design_path = args.design or scan.with_name(f"data_{scan.stem}_design.json")
    output = args.output or scan.parents[1] / "experiments" / f"{scan.stem}_three_colour"

    perf = json.loads(perf_path.read_text(encoding="utf-8"))
    design = json.loads(design_path.read_text(encoding="utf-8"))
    images, valid, numbers = aligned_collection(scan, perf, design, args.downsample)
    covered = valid.sum(axis=0) >= max(3, math.ceil(0.6 * len(images)))
    reference = masked_colour_median(images, valid)
    ring = paper_ring(design, reference.shape[:2], args.downsample)
    papers = estimate_papers(images, valid, ring)
    paper = np.median(papers, axis=0)
    learned = learn_ink_directions(
        reference, covered, paper, ring, args.max_colour_samples
    )

    response_stacks = [list() for _ in range(2)]
    for image, mask, stamp_paper in zip(images, valid, papers):
        responses, _assignment, _confidence = colour_response(
            image, stamp_paper, learned["centres"], mask
        )
        for group in range(2):
            response_stacks[group].append(responses[:, :, group])
    results = [
        register_one_colour(np.stack(stack), valid, args.iterations)
        for stack in response_stacks
    ]
    report = write_outputs(
        output,
        numbers,
        reference,
        paper,
        learned,
        results,
        args.downsample,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
