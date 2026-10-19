"""Find the major inks of a stamp and split each impression between them.

Some denominations are struck from two, three or four plates in as many colours,
each pass a separate trip through the press with its own registration.  Those
need naming before they can be treated separately, and the naming has to come
off the scan: the number of colours is not known in advance.

An ink is a direction in optical density, fixed by the pigment and independent
of how heavily it printed, so the inks of an issue are a few tight clusters on
the unit sphere and a pixel's colour is a non-negative sum of them.  Density
adds, which reflectance does not, so all of this happens in density.

The postmark is carried as one more neutral candidate.  A genuine near-neutral
printing ink may also be named, but its usable area is then restricted to dark
structure that repeats across registered impressions.  This spatial check is
what separates an achromatic plate from an equally achromatic postmark.

Reads nothing from disk.  ``--probe`` names the palette of one scan.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from background_ink import NEUTRAL, optical_density, to_linear, to_srgb
from three_colour_refinement import bgr_hex, estimate_papers, paper_ring
from two_pass_decomposition import aligned_collection

# An ink has to stand this far off every other ink, and has to cover this share
# of the inked area.  Measured, not guessed: at 25 degrees every single-colour
# sheet here collapses to exactly one mode, shading and faint screen tint and
# all, while all three two-colour sheets keep a second one holding 1.6 to 3% of
# the ink.  At 18 degrees the screen tint of 2K passes for a plate.  A rose
# centre in a blue frame is a small part of the design but a whole press run.
SEPARATION_DEG = 25.0
MINOR_SHARE = 0.01
# A scanner-space black is rarely exactly parallel to ``NEUTRAL``: tinted
# paper, channel response and ageing move it several degrees.  Ten degrees is
# still safely below the coloured inks in the reference sheets, while it
# includes the black plate in V/5K (about six degrees from neutral).
ACHROMATIC_DEG = 10.0
ACHROMATIC_SHARE = 0.05


def ink_directions(
    images: np.ndarray,
    valid: np.ndarray,
    papers: np.ndarray,
    stride: int = 3,
    minimum_density: float = 0.0,
) -> np.ndarray:
    """Unit density directions of the pixels that are dense and coloured enough
    to name an ink, pooled over the collection.

    The chromatic test is what keeps the postmark out: its pixels are the
    densest on a cancelled impression, and a cluster fitted to them is grey.
    """
    pooled = []
    for image, mask, paper in zip(images[:, ::stride, ::stride],
                                  valid[:, ::stride, ::stride],
                                  papers):
        if mask.sum() < 500:
            continue
        density, magnitude = optical_density(image, paper)
        chroma = magnitude - 0.999 * (density @ NEUTRAL)
        density_cut = max(float(np.percentile(magnitude[mask], 70)), minimum_density)
        inked = (
            mask
            & (magnitude > density_cut)
            & (chroma > np.percentile(chroma[mask], 70))
        )
        if inked.sum() < 200:
            continue
        pooled.append(density[inked] / magnitude[inked][:, None])
    if not pooled:
        raise RuntimeError("No dense coloured pixels to name an ink")
    return np.concatenate(pooled)


def achromatic_directions(
    images: np.ndarray,
    valid: np.ndarray,
    papers: np.ndarray,
    stride: int = 3,
    angle: float = ACHROMATIC_DEG,
    minimum_density: float = 0.60,
) -> np.ndarray:
    """Dense near-neutral directions used by the first palette stage."""
    pooled = []
    cosine = math.cos(math.radians(angle))
    for image, mask, paper in zip(
        images[:, ::stride, ::stride], valid[:, ::stride, ::stride], papers
    ):
        if mask.sum() < 500:
            continue
        density, magnitude = optical_density(image, paper)
        direction_cosine = np.divide(
            density @ NEUTRAL,
            magnitude,
            out=np.zeros_like(magnitude),
            where=magnitude > 1e-6,
        )
        selected = mask & (magnitude >= minimum_density) & (direction_cosine >= cosine)
        if selected.sum() < 100:
            continue
        pooled.append(density[selected] / magnitude[selected][:, None])
    if not pooled:
        raise RuntimeError("No dense achromatic pixels to name an ink")
    return np.concatenate(pooled)


def modes(
    directions: np.ndarray,
    separation: float = SEPARATION_DEG,
    limit: int = 4,
    share: float = MINOR_SHARE,
) -> tuple[np.ndarray, np.ndarray]:
    """Peel the direction cloud one dense mode at a time.

    A k-way partition is the wrong instrument here.  The rose centre of a
    blue-framed stamp is a few percent of the inked area, so the split that
    minimises scatter divides the blue in half and never names the rose at
    all.  What marks an ink is local density, not a share of the variance.

    One radius does the whole job: directions within ``separation`` of an ink
    belong to that ink, and a further ink has to stand further off than that
    from every ink already named.  So a faint second ink twenty degrees from
    the first is read as a shading of it, which on a screened single-colour
    stamp is the truth.
    """
    cosine = math.cos(math.radians(separation))
    sample = directions[:: max(1, len(directions) // 12000)]
    live = np.ones(len(sample), bool)
    centres: list[np.ndarray] = []
    weights: list[float] = []
    while len(centres) < limit and live.sum() >= share * len(sample):
        pool = sample[live]
        probes = pool[:: max(1, len(pool) // 1500)]
        neighbours = (probes @ pool.T) >= cosine
        densest = int(np.argmax(neighbours.sum(axis=1)))
        centre = np.median(pool[neighbours[densest]], axis=0)
        centre /= max(np.linalg.norm(centre), 1e-12)
        close = (sample @ centre) >= cosine
        weight = float((close & live).sum()) / len(sample)
        if weight < share:
            break
        centres.append(centre)
        weights.append(weight)
        live &= ~close
    if not centres:
        centre = np.median(sample, axis=0)
        return (centre / np.linalg.norm(centre))[None, :], np.array([1.0])
    return np.stack(centres), np.array(weights)


def find_palette(
    directions: np.ndarray,
    limit: int = 4,
    separation: float = SEPARATION_DEG,
    share: float = MINOR_SHARE,
    achromatic_angle: float = ACHROMATIC_DEG,
    achromatic_share: float = ACHROMATIC_SHARE,
) -> dict:
    """The major inks of the issue, in order of how much they cover.

    The ordinary mode finder is deliberately left unchanged for coloured inks.
    An achromatic ink is additional evidence, not a replacement colour model:
    an existing near-neutral mode is labelled as such, or a sufficiently large
    near-neutral population is appended when the coloured modes did not name
    it.  The spatial, collection-wide check that distinguishes a printed black
    plate from black cancellation ink is performed by ``golden_components``.
    """
    centres, weights = modes(directions, separation, limit, share)
    neutral_cosine = math.cos(math.radians(achromatic_angle))
    achromatic = (centres @ NEUTRAL) >= neutral_cosine
    achromatic_added = np.zeros(len(centres), bool)

    neutral_members = (directions @ NEUTRAL) >= neutral_cosine
    neutral_weight = float(np.mean(neutral_members))
    if (
        not achromatic.any()
        and len(centres) < limit
        and neutral_weight >= achromatic_share
    ):
        neutral_centre = np.median(directions[neutral_members], axis=0)
        neutral_centre /= max(float(np.linalg.norm(neutral_centre)), 1e-12)
        # Avoid adding a duplicate just across the labelling boundary.
        nearest = float(np.max(centres @ neutral_centre))
        if nearest < math.cos(math.radians(achromatic_angle)):
            centres = np.vstack([centres, neutral_centre])
            weights = np.append(weights, neutral_weight)
            achromatic = np.append(achromatic, True)
            achromatic_added = np.append(achromatic_added, True)

    order = np.argsort(-weights)
    centres, weights = centres[order], weights[order]
    achromatic = np.asarray(achromatic, bool)[order]
    achromatic_added = achromatic_added[order]
    if len(centres) > 1:
        gram = np.clip(centres @ centres.T, -1.0, 1.0)
        angles = np.degrees(np.arccos(gram))
        apart = float(np.min(angles[~np.eye(len(centres), dtype=bool)]))
    else:
        apart = float("inf")
    return {
        "rays": centres,
        "shares": weights,
        "count": len(centres),
        "separation_deg": apart,
        "achromatic": achromatic,
        "achromatic_added": achromatic_added,
        "achromatic_angle_deg": achromatic_angle,
    }


def find_coloured_palette(
    directions: np.ndarray,
    limit: int = 4,
    separation: float = SEPARATION_DEG,
    share: float = MINOR_SHARE,
    achromatic_angle: float = ACHROMATIC_DEG,
) -> dict:
    """Name chromatic inks after achromatic pixels have been removed.

    Neutral directions are rejected explicitly rather than being offered as a
    competing mode again.  This makes the second stage answer only how many
    coloured printing inks remain.
    """
    coloured = directions[
        (directions @ NEUTRAL) < math.cos(math.radians(achromatic_angle))
    ]
    if len(coloured) < 200:
        raise RuntimeError("No coloured ink remains after achromatic extraction")
    centres, weights = modes(coloured, separation, limit, share)
    if len(centres) > 1:
        gram = np.clip(centres @ centres.T, -1.0, 1.0)
        angles = np.degrees(np.arccos(gram))
        apart = float(np.min(angles[~np.eye(len(centres), dtype=bool)]))
    else:
        apart = float("inf")
    return {
        "rays": centres,
        "shares": weights,
        "count": len(centres),
        "separation_deg": apart,
        "achromatic": np.zeros(len(centres), bool),
        "achromatic_added": np.zeros(len(centres), bool),
        "achromatic_angle_deg": achromatic_angle,
    }


def unmix(density: np.ndarray, rays: np.ndarray) -> np.ndarray:
    """Charge every pixel's density to the one ink that best accounts for it.

    One ink per pixel, because the plates print separate parts of the design --
    a frame and a centre -- so outside the hair of pixels along their shared
    border no pixel carries two of them.  Letting a pixel carry a free mixture
    buys nothing real and costs a great deal, because grey tends to lie close
    to the cone the inks span: on the violet-and-green 35K it sits three
    degrees off their plane at almost exactly half of each, so "a postmark
    stroke" and "equal parts of the two plates" become the same measurement.

    Grey is one more thing a pixel can be, but only instead of an ink and never
    alongside one.  If a real ink is itself grey, the caller limits that ink to
    its repeatable spatial support; colour alone cannot distinguish it from a
    cancellation.  Both halves of that matter.  An ink can lie close to grey --
    70K's brown is eleven degrees off it -- and asking how to divide a pixel
    between two nearly parallel directions is ill posed: offered the pair, the
    fit answered "stroke" for a third of that frame and cut the rest to an
    eighth of its density, which is most of the engraving gone.  Asking instead
    which single direction the pixel came from is a comparison, and stays sound
    at eleven degrees.  But grey has to remain on offer, because without it
    every faintly tinted scrap of bare paper is charged to whichever ink points
    nearest, and the minor plate's layer fills with a wash that drowns the
    pattern a correlation would have locked onto.

    Solved exactly rather than iteratively: one projection per candidate,
    applied to the whole image at once, and the closest non-negative fit wins.
    """
    columns = np.vstack([rays, NEUTRAL])
    flat = density.reshape(-1, 3).T
    best = np.zeros((len(columns), flat.shape[1]), np.float32)
    score = np.full(flat.shape[1], np.inf, np.float32)
    for index, column in enumerate(columns):
        amount = column @ flat
        residual = np.linalg.norm(flat - column[:, None] * amount, axis=0)
        take = (amount >= 0.0) & (residual < score)
        score[take] = residual[take]
        best[:, take] = 0.0
        best[index, take] = amount[take]
    return best.reshape(len(columns), *density.shape[:2])


def postmark_mask(
    amounts: np.ndarray, valid: np.ndarray, share: float = 97.0
) -> np.ndarray:
    """The cancellation strokes: the grey that no plate's ink accounts for.

    Keyed on every ink at once, which is the whole point.  Grey measured as
    "what the *dominant* plate cannot be" condemns any second plate lying on
    the grey side of the first, and on the violet-and-green 35K, where grey
    falls almost exactly between the two, that reading threw away four fifths
    of the green centre as cancellation.

    A ranking rather than an absolute test, so the densest few percent are
    taken and spread by a dilation to catch the stroke's soft shoulder.
    """
    grey = amounts[-1]
    strong = (grey > np.percentile(grey[valid], share)).astype(np.uint8)
    return cv2.dilate(strong, np.ones((7, 7), np.uint8)) > 0


def ink_colour(paper: np.ndarray, ray: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """What the ink looks like printed solid on this paper."""
    return to_srgb(to_linear(np.asarray(paper, np.float64)) * 10.0 ** (-ray * strength))


def describe(palette: dict, paper: np.ndarray) -> str:
    achromatic = palette.get("achromatic", np.zeros(len(palette["rays"]), bool))
    inks = ", ".join(
        f"{'achromatic ' if neutral else ''}{bgr_hex(ink_colour(paper, ray))} {weight:.0%}"
        for ray, weight, neutral in zip(palette["rays"], palette["shares"], achromatic)
    )
    apart = palette["separation_deg"]
    tail = "" if not np.isfinite(apart) else f", {apart:.1f} deg apart"
    return f"{palette['count']} major ink{'s' if palette['count'] > 1 else ''}: {inks}{tail}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Name the major inks of a scan.")
    parser.add_argument("scan", type=Path)
    parser.add_argument("--perf", type=Path, help="Default: data_<stem>_perf.json")
    parser.add_argument("--design", type=Path, help="Default: data_<stem>_design.json")
    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument("--limit", type=int, default=4, help="Most inks to look for")
    parser.add_argument("--separation", type=float, default=SEPARATION_DEG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    perf = json.loads(
        (args.perf or args.scan.with_name(f"data_{args.scan.stem}_perf.json"))
        .read_text(encoding="utf-8")
    )
    design = json.loads(
        (args.design or args.scan.with_name(f"data_{args.scan.stem}_design.json"))
        .read_text(encoding="utf-8")
    )
    images, valid, numbers = aligned_collection(args.scan, perf, design, args.downsample)
    papers = estimate_papers(images, valid, paper_ring(design, images.shape[1:3], args.downsample))
    paper = np.median(papers, axis=0)

    directions = ink_directions(images, valid, papers)
    print(f"{args.scan.name}: {len(numbers)} stamps, paper {bgr_hex(paper)}, "
          f"{len(directions)} coloured pixels sampled")
    # Every mode down to one percent, so the radius and the share a real
    # minority ink needs can be read off rather than assumed.
    for radius in (12.0, 18.0, 25.0, 32.0):
        centres, weights = modes(directions, radius, args.limit, share=0.01)
        inks = "  ".join(
            f"{bgr_hex(ink_colour(paper, ray))} {weight:5.1%}"
            for ray, weight in zip(centres, weights)
        )
        print(f"  radius {radius:4.0f} deg: {inks}")
    print("  chosen -> " + describe(find_palette(directions, args.limit, args.separation), paper))


if __name__ == "__main__":
    main()
