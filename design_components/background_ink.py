"""Extract the separately printed background ink of a two-pass stamp.

For 1K the background pass is a screen of fine ochre dots laid down before the
vermilion main design.  Because the two passes register independently, the
screen shifts from impression to impression and is lost when the collection is
averaged into a golden.  This recovers it from each impression on its own, so
no amount of inter-pass shift can wash it out.

Three ideas carry the method.

The scan is linearised before optical density is taken.  Density is additive in
linear reflectance, not in gamma-encoded sRGB; taking the log of sRGB bends the
apparent hue of weak pixels and will otherwise manufacture a second ink out of
a pale impression of the first.

The background ink's ray is measured from the periodic component alone.  The
screen has a pitch that nothing else on the stamp shares: a postmark is not
periodic, paper tone is not, and a thin impression of the main ink is not.  The
per-channel Fourier coefficient at the screen's wave vector is therefore
produced by the background ink and nothing else.  The three channels sharing a
single phase is the check that one ink, rather than noise, causes it.

Separation is then a projection, not a clustering.  The main ray and the
neutral ray that any black postmark must lie on span a plane; in three channels
exactly one direction escapes it, and a background ink of a different hue is
the only thing that can occupy it.

Averaging across impressions is deliberately not offered.  ``--cross-check``
measures whether the fine screen reproduces between stamps at all; on 1K it
does not, because a quasi-periodic screen has no unique cross-stamp
registration, so the single-impression layers are the honest output.

Experimental.  Reads only the JSON written by the production pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from three_colour_refinement import add_label, bgr_hex, estimate_papers, paper_ring
from two_pass_decomposition import aligned_collection, masked_colour_median

NEUTRAL = np.ones(3) / math.sqrt(3.0)
SCREEN_WINDOW = 512


def to_linear(values: np.ndarray) -> np.ndarray:
    channel = np.clip(values.astype(np.float64) / 255.0, 0.0, 1.0)
    return np.where(channel <= 0.04045, channel / 12.92, ((channel + 0.055) / 1.055) ** 2.4)


def to_srgb(values: np.ndarray) -> np.ndarray:
    channel = np.clip(values, 0.0, 1.0)
    encoded = np.where(
        channel <= 0.0031308, channel * 12.92, 1.055 * channel ** (1 / 2.4) - 0.055
    )
    return np.clip(encoded * 255.0, 0, 255)


def optical_density(image: np.ndarray, paper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Density against paper, computed in linear reflectance."""
    linear = to_linear(image)
    reference = to_linear(np.asarray(paper, np.float64).reshape(1, 1, 3))
    density = -np.log10(np.clip(linear / np.maximum(reference, 1e-9), 1e-4, 1.0))
    density = np.clip(density, 0.0, None)
    magnitude = np.linalg.norm(density, axis=2)
    return density, magnitude


def main_ray(density: np.ndarray, magnitude: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """The main ink ray, from pixels that are dense *and* strongly coloured.

    The chromatic test matters: on a heavily cancelled impression the densest
    pixels are the postmark, and a ray fitted to those is simply neutral.
    """
    chroma = magnitude - 0.999 * (density @ NEUTRAL)
    dense = (
        valid
        & (magnitude > np.percentile(magnitude[valid], 90))
        & (chroma > np.percentile(chroma[valid], 90))
    )
    if dense.sum() < 500:
        raise RuntimeError("No dense coloured pixels to define the main ray")
    direction = density / np.maximum(magnitude[:, :, None], 1e-9)
    ray = np.median(direction[dense], axis=0)
    return ray / np.linalg.norm(ray)


def orthonormal_frame(ray: np.ndarray) -> np.ndarray:
    """Main ink, neutral residue, and the one direction that escapes them."""
    second = NEUTRAL - (NEUTRAL @ ray) * ray
    second /= np.linalg.norm(second)
    third = np.cross(ray, second)
    third /= np.linalg.norm(third)
    return np.stack([ray, second, third])


def screen_window(field: np.ndarray, shape: tuple[int, int]) -> tuple[int, int]:
    """Top-left corner of the analysis window, shared by every impression.

    The screen phase is only comparable between stamps when it is read in the
    same window, so the caller fixes one window for the whole collection.
    """
    ys, xs = np.nonzero(field)
    if len(xs) < 15000:
        raise RuntimeError("No field clear enough to measure the screen")
    cy, cx = int(np.median(ys)), int(np.median(xs))
    y0 = max(0, min(shape[0] - SCREEN_WINDOW, cy - SCREEN_WINDOW // 2))
    x0 = max(0, min(shape[1] - SCREEN_WINDOW, cx - SCREEN_WINDOW // 2))
    return y0, x0


def screen_patch(
    density: np.ndarray, field: np.ndarray, corner: tuple[int, int]
) -> np.ndarray:
    """The windowed, mean-removed density patch the screen is measured in."""
    size = SCREEN_WINDOW
    y0, x0 = corner
    box = (slice(y0, y0 + size), slice(x0, x0 + size))
    weight = field[box].astype(np.float64)
    patch = density[box].astype(np.float64)
    patch = patch - patch[weight > 0].mean(axis=0)
    return patch * (np.outer(np.hanning(size), np.hanning(size)) * weight)[:, :, None]


def screen_coefficient(patch: np.ndarray, wave: np.ndarray) -> np.ndarray:
    """Per-channel Fourier coefficient at one arbitrary wave vector.

    ``screen_ray`` reports the peak on the FFT grid, and neighbouring
    impressions can land in neighbouring bins.  Phases read in different bins
    are not comparable, so cross-impression work evaluates every phase here, at
    a single shared frequency, off the grid if need be.
    """
    index = np.arange(patch.shape[0])
    rows = np.exp(-2j * math.pi * wave[1] * index)
    columns = np.exp(-2j * math.pi * wave[0] * index)
    return np.einsum("y,x,yxc->c", rows, columns, patch, optimize=True)


def screen_ray(
    density: np.ndarray,
    field: np.ndarray,
    pitch_range: tuple[float, float],
    tilt: float,
    corner: tuple[int, int] | None = None,
) -> dict:
    """Ink ray, pitch, angle and phase of the screen, from its Fourier peak.

    Only the periodic component contributes, so a postmark, paper tone and a
    thin main impression cannot bias the result.
    """
    size = SCREEN_WINDOW
    y0, x0 = screen_window(field, density.shape[:2]) if corner is None else corner
    box = (slice(y0, y0 + size), slice(x0, x0 + size))
    weight = field[box].astype(np.float64)
    if weight.sum() < 15000:
        raise RuntimeError("Too little clear field under the screen window")

    window = np.outer(np.hanning(size), np.hanning(size)) * weight
    spectra = []
    for channel in range(3):
        patch = density[box][:, :, channel]
        patch = patch - patch[weight > 0].mean()
        spectra.append(np.fft.fftshift(np.fft.fft2(patch * window)))
    stack = np.stack(spectra)
    total = np.abs(stack.sum(axis=0))

    centre = size // 2
    yy, xx = np.mgrid[:size, :size]
    radius = np.hypot(xx - centre, yy - centre)
    band = (radius > size / pitch_range[1]) & (radius < size / pitch_range[0])
    bearing = np.abs(np.degrees(np.arctan2(yy - centre, xx - centre)))
    band &= np.abs(bearing - 90.0) < tilt
    if not band.any():
        raise RuntimeError("No admissible screen frequency in range")

    peak = int(np.argmax(np.where(band, total, 0.0)))
    py, px = divmod(peak, size)
    pitch = size / max(float(np.hypot(px - centre, py - centre)), 1e-9)
    angle = float(np.degrees(np.arctan2(py - centre, px - centre)))

    coefficients = stack[:, py, px]
    phase = np.angle(coefficients.sum())
    amplitude = np.real(coefficients * np.exp(-1j * phase))
    coherence = float(abs(coefficients.sum()) / max(np.sum(np.abs(coefficients)), 1e-12))
    ray = amplitude / max(np.linalg.norm(amplitude), 1e-12)
    if ray.sum() < 0:
        ray = -ray
        phase += math.pi
    return {
        "ray": ray,
        "pitch": pitch,
        "angle": angle,
        "coherence": coherence,
        "snr": float(total[py, px] / max(float(np.median(total[band])), 1e-9)),
        # Cycles per pixel, in the shared window's frame, plus the phase there.
        "wave": np.array([(px - centre) / size, (py - centre) / size], np.float64),
        "phase": float(phase),
        "corner": (y0, x0),
    }


def separate(
    image: np.ndarray,
    paper: np.ndarray,
    valid: np.ndarray,
    tilt: float,
    corner: tuple[int, int] | None = None,
) -> dict:
    """Split one impression into main ink, postmark and background ink."""
    density, magnitude = optical_density(image, paper)
    ray = main_ray(density, magnitude, valid)
    frame = orthonormal_frame(ray)
    along_main = density @ frame[0]
    along_neutral = density @ frame[1]
    remainder = density @ frame[2]

    postmark = cv2.dilate(
        (along_neutral > np.percentile(along_neutral[valid], 97)).astype(np.uint8),
        np.ones((7, 7), np.uint8),
    ) > 0
    heavy = cv2.dilate(
        (along_main > np.percentile(along_main[valid], 75)).astype(np.uint8),
        np.ones((5, 5), np.uint8),
    ) > 0
    field = valid & ~postmark & ~heavy

    screen = screen_ray(density, field, (7.5, 10.5), tilt, corner)
    separability = float(screen["ray"] @ frame[2])
    if abs(separability) < 0.02:
        raise RuntimeError(
            f"Background ink is only {separability:+.3f} clear of the main plane"
        )

    # Coverage in the background ink's own density units.  Dividing by the
    # separability multiplies the noise by the same factor, so a matched
    # smoothing at the scale of one screen dot follows.  Both steps are linear
    # and signed, so noise is never rectified into pattern.
    coverage = cv2.GaussianBlur((remainder / separability).astype(np.float32), (0, 0), 0.8)

    strength = float(np.percentile(magnitude[field], 90))
    ink_linear = to_linear(np.asarray(paper, np.float64)) * 10.0 ** (-screen["ray"] * strength)
    main_linear = to_linear(np.asarray(paper, np.float64)) * 10.0 ** (
        -ray * float(np.percentile(magnitude[valid], 99))
    )
    return {
        "frame": frame,
        "density": density,
        "magnitude": magnitude,
        "along_main": along_main,
        "along_neutral": along_neutral,
        "remainder": remainder,
        "coverage": coverage,
        "field": field,
        "postmark": postmark,
        "main_ray": ray,
        "ink_ray": screen["ray"],
        "pitch": screen["pitch"],
        "angle": screen["angle"],
        "coherence": screen["coherence"],
        "snr": screen["snr"],
        "wave": screen["wave"],
        "phase": screen["phase"],
        "separability": separability,
        "hue_angle": float(np.degrees(np.arccos(np.clip(ray @ screen["ray"], -1, 1)))),
        "ink_colour": to_srgb(ink_linear),
        "main_colour": to_srgb(main_linear),
    }


def render_ink(coverage: np.ndarray, paper: np.ndarray, ray: np.ndarray) -> np.ndarray:
    density = np.clip(coverage, 0.0, None)[:, :, None] * ray[None, None, :]
    linear = to_linear(np.asarray(paper, np.float64))[None, None, :] * 10.0 ** (-density)
    return to_srgb(linear).astype(np.uint8)


def display_limits(values: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    inside = values[mask]
    if not len(inside):
        return 0.0, 1.0
    low, high = np.percentile(inside, [10, 99.0])
    return float(low), float(high)


def contrast(
    values: np.ndarray, mask: np.ndarray, limits: tuple[float, float], gamma: float = 0.9
) -> np.ndarray:
    low, high = limits
    scaled = np.clip((values - low) / max(high - low, 1e-6), 0, 1) ** gamma
    grey = (255.0 * (1.0 - scaled)).astype(np.uint8)
    grey[~mask] = 255
    return grey


def grid(panels: list[np.ndarray], columns: int) -> np.ndarray:
    items = list(panels)
    while len(items) % columns:
        items.append(np.full_like(items[0], 245))
    return np.vstack([np.hstack(items[i : i + columns]) for i in range(0, len(items), columns)])


def swatch(colour: np.ndarray, shape: tuple[int, int], text: str) -> np.ndarray:
    panel = np.empty((*shape, 3), np.uint8)
    panel[:] = np.clip(np.round(colour), 0, 255).astype(np.uint8)
    return add_label(panel, text)


def cross_check(layers: list[np.ndarray], fields: list[np.ndarray], limit: int = 26) -> dict:
    """Does the fine screen reproduce between impressions, and at what offset?"""
    shape = layers[0].shape
    pad = [cv2.getOptimalDFTSize(shape[0]), cv2.getOptimalDFTSize(shape[1])]

    def spectrum(array):
        padded = np.zeros(pad, np.float64)
        padded[: shape[0], : shape[1]] = array
        return np.fft.rfft2(padded)

    prepared = []
    for layer, mask in zip(layers, fields):
        band = cv2.GaussianBlur(layer, (0, 0), 1.2) - cv2.GaussianBlur(layer, (0, 0), 5.0)
        weight = mask.astype(np.float64)
        value = np.where(weight > 0, band, 0.0).astype(np.float64)
        value -= value.sum() / max(weight.sum(), 1.0) * weight
        value = np.where(weight > 0, value, 0.0)
        prepared.append((spectrum(value), spectrum(value * value), spectrum(weight)))

    rows = []
    for i in range(len(layers)):
        for j in range(i + 1, len(layers)):
            fa, fa2, fma = prepared[i]
            fb, fb2, fmb = prepared[j]
            numerator = np.fft.irfft2(fa * np.conj(fb), pad)
            energy_a = np.fft.irfft2(fa2 * np.conj(fmb), pad)
            energy_b = np.fft.irfft2(fma * np.conj(fb2), pad)
            denominator = np.sqrt(np.maximum(energy_a, 0) * np.maximum(energy_b, 0))
            surface = np.fft.fftshift(
                np.where(denominator > 1e-9, numerator / np.maximum(denominator, 1e-9), 0.0)
            )
            cy, cx = pad[0] // 2, pad[1] // 2
            window = surface[cy - limit : cy + limit + 1, cx - limit : cx + limit + 1]
            index = int(np.argmax(window))
            dy, dx = divmod(index, window.shape[1])
            rim = np.ones(window.shape, bool)
            rim[dy - 3 : dy + 4, dx - 3 : dx + 4] = False
            floor = float(np.median(window[rim]))
            sigma = float(np.std(window[rim]))
            rows.append(
                {
                    "pair": (i, j),
                    "offset": (dx - limit, dy - limit),
                    "peak": float(window[dy, dx]),
                    "z": (float(window[dy, dx]) - floor) / max(sigma, 1e-9),
                }
            )
    return {
        "pairs": rows,
        "median_z": float(np.median([row["z"] for row in rows])),
        "offsets": (2 * limit + 1) ** 2,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract the separately printed background ink of a two-pass stamp."
    )
    parser.add_argument("scan", type=Path)
    parser.add_argument("--perf", type=Path, help="Default: data_<stem>_perf.json")
    parser.add_argument("--design", type=Path, help="Default: data_<stem>_design.json")
    parser.add_argument("--output", type=Path, help="Default: <stem>_background_ink.png")
    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument(
        "--screen-tilt", type=float, default=12.0,
        help="Degrees the screen may lean off horizontal (default: 12)",
    )
    parser.add_argument(
        "--zoom-stamps", default="",
        help="Comma-separated stamps for the magnified strip (default: the four clearest)",
    )
    parser.add_argument("--zoom", type=int, default=5)
    parser.add_argument(
        "--cross-check", action="store_true",
        help="Measure whether the fine screen reproduces between impressions",
    )
    parser.add_argument("--panel-width", type=int, default=440)
    args = parser.parse_args()
    if args.downsample < 1:
        parser.error("--downsample must be positive")
    return args


def main() -> None:
    args = parse_args()
    scan = args.scan.resolve()
    perf_path = args.perf or scan.with_name(f"data_{scan.stem}_perf.json")
    design_path = args.design or scan.with_name(f"data_{scan.stem}_design.json")
    output = args.output or scan.with_name(f"{scan.stem}_background_ink.png")

    perf = json.loads(perf_path.read_text(encoding="utf-8"))
    design = json.loads(design_path.read_text(encoding="utf-8"))
    images, valid, numbers = aligned_collection(scan, perf, design, args.downsample)
    print(f"Scan: {scan}\n  {len(numbers)} stamps, canvas {images.shape[2]}x{images.shape[1]}")

    ring = paper_ring(design, images.shape[1:3], args.downsample)
    papers = estimate_papers(images, valid, ring)
    reference = masked_colour_median(images, valid)

    usable = []
    for number, image, mask, paper in zip(numbers, images, valid, papers):
        try:
            result = separate(image, paper, mask, args.screen_tilt)
        except RuntimeError as error:
            print(f"  stamp {number:2d}: {error}")
            continue
        result.update(paper=paper, number=number)
        result["limits"] = display_limits(result["coverage"], result["field"])
        usable.append(result)
        print(
            f"  stamp {number:2d}: background {bgr_hex(result['ink_colour'])} "
            f"vs main {bgr_hex(result['main_colour'])}, {result['hue_angle']:4.1f}deg apart; "
            f"screen {result['pitch'] * args.downsample:5.2f}px @{result['angle']:+6.1f}deg "
            f"snr {result['snr']:4.1f} coherence {result['coherence']:.2f}; "
            f"separability {result['separability']:+.3f}"
        )
    if not usable:
        raise RuntimeError("No impression yielded a background layer")

    pitches = [r["pitch"] for r in usable]
    consensus = float(np.median(pitches))
    print(
        f"  consensus screen pitch {consensus * args.downsample:.2f} px "
        f"(spread {np.ptp(pitches) * args.downsample:.2f}); "
        f"median hue separation {np.median([r['hue_angle'] for r in usable]):.1f} deg"
    )

    # Rank on how well the screen stands out and how much a postmark leaves
    # readable.  Separability is not a reward: a heavy postmark inflates it.
    def quality(result: dict) -> float:
        if abs(result["pitch"] - consensus) > 0.5:
            return 0.0
        clear = result["field"].sum() / max(result["field"].size, 1)
        return result["snr"] * result["coherence"] * clear

    ranked = sorted(usable, key=quality, reverse=True)
    if args.zoom_stamps:
        wanted = [int(v) for v in args.zoom_stamps.split(",") if v.strip()]
        chosen = [r for number in wanted for r in usable if r["number"] == number]
    else:
        chosen = ranked[:4]
    print("  clearest: " + ", ".join(str(r["number"]) for r in ranked[:4]))

    shape = reference.shape[:2]
    panels = [
        add_label(np.clip(reference, 0, 255).astype(np.uint8), "colour median (golden)"),
        swatch(np.median(papers, axis=0), shape, f"paper {bgr_hex(np.median(papers, axis=0))}"),
        swatch(
            np.median([r["main_colour"] for r in usable], axis=0), shape,
            f"main ink {bgr_hex(np.median([r['main_colour'] for r in usable], axis=0))}",
        ),
        swatch(
            np.median([r["ink_colour"] for r in usable], axis=0), shape,
            f"background ink {bgr_hex(np.median([r['ink_colour'] for r in usable], axis=0))}",
        ),
    ]
    for result in usable:
        panels.append(
            add_label(
                cv2.cvtColor(
                    contrast(result["coverage"], result["field"], result["limits"]),
                    cv2.COLOR_GRAY2BGR,
                ),
                f"stamp {result['number']} background; {result['pitch'] * args.downsample:.2f}px screen",
            )
        )
    for result in usable:
        panels.append(
            add_label(
                render_ink(result["coverage"], result["paper"], result["ink_ray"]),
                f"stamp {result['number']} background in its own ink",
            )
        )

    scaled = []
    for panel in panels:
        factor = args.panel_width / panel.shape[1]
        scaled.append(
            cv2.resize(
                panel,
                (args.panel_width, max(1, int(round(panel.shape[0] * factor)))),
                interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_CUBIC,
            )
        )
    sheet = grid(scaled, 4)

    rectangle = design["golden"]["rectangle"]
    bx = int(rectangle["left_px"] / args.downsample) + int(60 / args.downsample)
    by = int(rectangle["top_px"] / args.downsample) + int(540 / args.downsample)
    bw, bh = int(150 / args.downsample), int(104 / args.downsample)
    strip = []
    for result in chosen:
        index = numbers.index(result["number"])
        crop = np.clip(images[index][by : by + bh, bx : bx + bw], 0, 255).astype(np.uint8)
        layer = contrast(
            result["coverage"][by : by + bh, bx : bx + bw],
            result["field"][by : by + bh, bx : bx + bw],
            result["limits"],
        )
        pair = np.hstack([crop, cv2.cvtColor(layer, cv2.COLOR_GRAY2BGR)])
        pair = cv2.resize(pair, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_NEAREST)
        strip.append(add_label(pair, f"stamp {result['number']}: scan | background ink only"))
    if strip:
        width = max(sheet.shape[1], strip[0].shape[1])
        sheet = np.vstack(
            [
                cv2.copyMakeBorder(
                    block, 0, 0, 0, width - block.shape[1], cv2.BORDER_CONSTANT, value=(245, 245, 245)
                )
                for block in [sheet] + strip
            ]
        )

    if not cv2.imwrite(str(output), sheet):
        raise RuntimeError(f"Cannot write image: {output}")
    print(f"Wrote {output} ({sheet.shape[1]}x{sheet.shape[0]})")

    if args.cross_check:
        print("\nDoes the fine screen reproduce between impressions?")
        report = cross_check([r["coverage"] for r in ranked[:6]], [r["field"] for r in ranked[:6]])
        for row in report["pairs"]:
            first = ranked[row["pair"][0]]["number"]
            second = ranked[row["pair"][1]]["number"]
            print(
                f"  {first:2d}-{second:2d}: peak {row['peak']:+.3f} at "
                f"({row['offset'][0]:+3d},{row['offset'][1]:+3d})  z {row['z']:4.1f}"
            )
        expected = math.sqrt(2.0 * math.log(report["offsets"]))
        print(
            f"  median z {report['median_z']:.1f}; chance alone gives about {expected:.1f} "
            f"over {report['offsets']} offsets"
        )
        if report["median_z"] < expected:
            print(
                "  The screen does not reproduce between impressions above chance.\n"
                "  Averaging cannot sharpen it; the single-impression layers are the result."
            )


if __name__ == "__main__":
    main()
