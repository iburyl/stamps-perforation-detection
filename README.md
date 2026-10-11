# stamps-perforation-detection

A research tool that measures postage stamp **perforation gauge** (holes per 20 mm) and
**valley-to-valley dimensions** from a flatbed scan of several stamps on dark backing paper.

NumPy, OpenCV and Pillow are required.

```
pip install -r requirements.txt
python segment_stamps.py scan.png --dpi 1200
```

Outputs, written next to the input:

| file | contents |
| --- | --- |
| `scan_detected.jpg` | the scan with fitted edge lines, hole markers and a per-stamp gauge label |
| `scan_perf.json` | every numeric result: per stamp, per side, per hole |

`scan_detected.jpg` is the result meant to be read. Each edge line is coloured by the
phase of the fallback ladder that produced it, so a side that needed a late fallback is
visible without opening anything else:

| colour | phase | method |
| --- | --- | --- |
| green | 1 — main method | `cross_support` |
| yellow | 2 — first fallback | `adaptive_lab_a` / `adaptive_lab_b` |
| orange | 3 — second fallback | `parallel_edge_arc_recovery`, `parallel_normal_scan_recovery` |
| red | 4 — last resort | `sequential_kmeans_lab` |
| grey | — | no measurement on that side |

The per-stamp gauge label takes the worst phase on the stamp, so a sheet can be triaged
at a glance.

`scan_perf.json` is written for the next processing stage and for agent-driven debugging.
It is deliberately exhaustive and is not intended to be read by hand.

### Collection analysis

`analysis/analyze_perforation.py` reads any collection of `*_perf.json` files produced by
the detector. It is not tied to a named study, catalogue issue, file naming scheme, or a
fixed number of clusters. When matching `*_design.json` files are present beside the
perforation results, their printed-design dimensions are joined by source and stamp id.

```bash
python analysis/analyze_perforation.py /path/to/results
```

Directories are searched recursively. For one directory input, the default output is its
`perforation-analysis` subdirectory; for one file, it is beside that file. The analysis
selects the cluster count with the Gap statistic 1-SE rule and separately tests whether
fitted corner-hole phases support frame perforation or are compatible with independent
line-perforator passes.

In the cluster chart, every source JSON has a distinct color and every selected cluster
has a distinct marker shape. The marker outline identifies the worst edge-detection
algorithm, using the same colors as the annotated detector image. The six observations
farthest from their assigned centroid are labelled as `source:stamp_id`. Axis grid lines
are drawn at every visible 0.25-gauge boundary.

The report directory contains:

| file | contents |
| --- | --- |
| `report.md` | readable conclusions, methods, limitations and embedded charts |
| `analysis.json` | machine-readable cluster and frame-versus-line results |
| `summary.csv` | all complete horizontal/vertical gauge pairs |
| `cluster_assignments.csv` | one selected cluster per complete stamp measurement |
| `design_summary.csv` | printed-design dimensions, matched cluster and frame-based spacing estimates |
| `perforation_clusters.png` | selected clustering and centroids |
| `cluster_count_diagnostics.png` | Gap statistic and silhouette by candidate `k` |
| `design_sizes_by_source.png` | printed-design width versus height across Sources |
| `design_spacing_by_perforation_cluster.png` | estimated horizontal and vertical design gaps by Source and cluster |
| `design_spacing_points_by_perforation_cluster.png` | the same design gaps with every retained stamp shown separately |
| `corner_alignment.png` | fitted corner-hole phase evidence |

Under the frame-perforation hypothesis, the analysis also estimates the stamp repeat from
the centres of the extreme holes. It converts gauge to pitch (`20 / gauge`), selects the
smallest whole number of pitch intervals that spans the measured valley-line dimension,
and uses that integer span as the centre-to-centre repeat. Subtracting the printed-design
dimension gives the estimated edge-to-edge spacing between neighboring designs. Hole radii
are not used in this calculation. Within each Source/perforation-cluster group, only stamps
with the unique modal width/height hole-count pair contribute to spacing means and the chart;
non-modal estimates are reported separately as hole-count outliers.

The frame/line result is deliberately conservative. At least 12 usable corners from four
stamps are required; otherwise it reports `insufficient-data`. A `linear-compatible`
result means that the test found no collection-wide phase locking. It is not proof of a
particular perforating machine.

Diagnostics for one stamp/side/hole are written to `scan_details/` when `--stamp`,
`--side` and `--spot` are given. Every file name starts with the phase that produced it
(`0` for stamp localization and rectification, then `1`–`4` as above), so sorting the
directory walks the algorithm in order, and a phase that did not run leaves no files.
Each panel carries a banner naming its phase, its method, its status, and whether it was
the one selected. A diagnostic run selected with `--stamp` (optionally narrowed in the
report with `--side`) executes and records a shadow run of every current fallback
(`algorithm_trials.json`), so the alternatives can be compared on one baseline.
With `--stamp` and `--side`, `1_all_profiles_and_guesses.png` also shows the exact k=3
and k=4 centroid-colour images, all fourteen binary boundary profiles, every pre-arc
valley guess, their scores, cross-support bonus, and a red border around the winner.

---

## The measurand

Perforation gauge is conventionally quoted as the number of holes per 20 mm. This tool
measures the **pitch between hole centres along a fitted edge line** and converts:

```
gauge = 20 * dpi / (25.4 * pitch_px)
```

The edge line is fitted through the **valley bases** — the innermost point of each hole —
not through the tooth tips. Tooth tips are the damaged, variable part of a perforation and
make a poor datum. Reported `valley_width_px` / `valley_height_px` are therefore the
distances between opposite valley-base lines, not paper dimensions.

`--dpi` is taken on trust; it scales every gauge linearly. Image metadata is not read.

## The evidence rule

The design decision that governs everything else: **a pitch is published only when the
geometry came from fitted circular arcs, with at least five of them accepted on that side.**

Perforation holes are round punches, so a circle fitted to the local paper boundary is a
physically justified model, and its apex is a well-defined point. Everything cheaper — the
binary profiles, the autocorrelation period, parabola fits — is treated as a *search
hypothesis* only. In particular the autocorrelation pitch that drives the search is deleted
from the result before output if the side never reached five arcs, so it cannot leak out as
a measurement (`reliable_side()`, and the `pitch_px` removal at the end of `measure_stamp`).

A side that cannot meet this bar reports no pitch, and the corresponding stamp dimension is
suppressed rather than estimated. The cost is that genuine straight edges (booklet, coil)
are reported as failures; see *Known limitations*.

---

## Algorithmic flow

Stages 1–3 are the phase 0 localization common to every side; stages 4–7 are the phase 1
main method; stage 8 is the phase 2–4 fallback ladder.

### 1. Scan-wide stamp detection — `detect_stamps_2d()`

The scan is downscaled to at most `--processing-width` (default 3000 px). The backing sheet
level is estimated as the 20th percentile of grey, and four thresholds at
`level + delta` for `delta` around `--stamp-delta` produce four foreground masks. Connected
components are proposed from each; components touching the image border are discarded.

A reference scale (`typical_width`, `typical_height`, `typical_area`) is taken as the median
of compact, stamp-sized components. Proposals far from that scale, or too sparse to be
paper, are dropped. A component roughly 2× the normal width or height is split on a uniform
grid, which keeps touching pairs separate.

Proposals are consolidated by non-maximum suppression at IoU 0.25, and a proposal must be
supported by **at least two of the four thresholds** to survive, which stops a
single-threshold artefact from becoming a stamp. Elongated objects such as a ruler lying on
the sheet are excluded earlier, by the scale and aspect limits. Survivors are numbered in
reading order.

No assumption is made about a single backing-paper rectangle, and the image is never
projected into rows, so stamps need not form a clean grid.

### 2. Coarse orientation — `estimate_orientation()`

Within each detection box the mask is morphologically closed (kernel ≈ 4% of the short
side, with explicit black padding at the crop edge) so that heavy cancellation ink cannot
split the stamp into disconnected pieces. The largest contour is taken, and
`cv2.minAreaRect` gives a seed angle modulo 90°.

For each of the four sides, only the **central 70%** of the edge is used — this excludes the
corners — and points within 8% of the extreme are fitted with a Huber-robust line, which
suppresses the perforation scallops and local damage. The side angles are unwrapped near
the seed angle and the median is taken; their spread is recorded as `edge_spread_deg`.

The outer envelope is the 1st/99th percentile of the projected contour, so isolated paper
fibres do not inflate it.

### 3. Rectification

Each stamp is rotated into an axis-aligned patch with a margin of 12% of its short side, so
the whole perforated edge plus surrounding background is available. The patch is converted
to grey (Gaussian σ 0.6) and thresholded once with a single scan-global threshold
(10th percentile of grey + `--stamp-delta`). A Lab copy is kept with slightly stronger
smoothing (σ 1.0) for the k-means primary maps and the colour fallback.

**No morphological closing or dilation is applied to the measurement image** — closing would
partially fill the very holes being measured.

### 4. Binary maps and boundary profiles — `cross_support_edge()`

Each side is transposed/flipped so it can be processed as "top": *along* the edge is x,
*inward* is increasing y. Along the edge, 1% to 99% is used. Across the edge the search band
runs from 5.5% of the stamp outside the nominal border to 13% inside.

The old standalone brightness boundary is no longer run. Instead, phase 1 builds seven
peer binary maps: the scan-wide brightness mask plus every legal assignment of the
intermediate Lab centroids at k=3 and k=4. The darkest centroid is always background and
the lightest is always paper, giving two k=3 and four k=4 maps.

Each map produces a `stack` and a `close` depth profile, fourteen profiles in total.
`close` removes narrow outward spikes from the first-sustained-run depth. `stack` also
removes components below the physical perforation scale, keeps paper connected to the
stamp interior, fills enclosed design holes and measures paper count before closing.

### 5. Periodic valley hypothesis and cross-support selection — `measure_profile()`

The profile is detrended against a robust baseline, and its amplitude must exceed
`max(2.5 px, 0.3% of the short side)` or the side is declared to have no distinct valleys.

A normalised autocorrelation over lags from `short/45` to `short/7` gives a **candidate
period**. Local maxima of the detrended profile are then accepted as valley candidates if
their prominence exceeds `max(2 px, 0.22 × amplitude)`, with non-maximum suppression at
0.55 × period. Each candidate gets a sub-pixel position by quadratic interpolation.

The candidates must then fall on a common integer lattice: a phase is chosen by maximum
support, and the lattice is re-fitted three times, discarding points more than 0.15 periods
off. At least five surviving points on at least five distinct lattice nodes are required.
Finally a robust line is fitted through the valley bases.

This stage produces the **seeds** for arc fitting, plus the quality figures `coverage`,
`autocorrelation`, `spacing_rms_px` and `depth_rms_px`.

Every profile is scored as `count × coverage × autocorrelation`. A brightness profile and
a k-means profile receive a symmetric 1.1× bonus when their periods agree within 2% and
their edge positions agree within 0.08 period. The highest adjusted score is selected
before any arcs are fitted. Brightness has no priority and is never fitted separately.

### 6. Sub-pixel arc refinement — `refine_valleys()`, `fit_arc()`, `fit_parabola()`

This is where the actual measurement happens. Exactly one primary arc pass is run per
side, using the selected profile but the original greyscale pixels. Its final lattice is
constrained to within 5% of the selected profile period to prevent a half/double-period
jump. For each seed the side image is blurred
(σ 1.2) and differentiated along the inward axis with a Sobel kernel. For every column in a
window around the seed, the boundary point is the gradient maximum weighted by a Gaussian
prior around the first-pass boundary, refined to sub-pixel by quadratic interpolation on
three gradient samples.

Those boundary points are fitted with:

* a **circle**, by algebraic initialisation followed by 12 IRLS Gauss-Newton iterations.
  The radius must lie between 0.10 and 0.65 of the period; at least 9 points must be
  inliers and 70% of points must fit; the arc must have support on **both** sides of its
  apex (0.4 radius each way) so a one-sided curve cannot be extrapolated into a hole; and
  the apex must be near the seed. The result point is the arc apex.
* a **parabola**, as a cross-check with comparable inlier requirements.

  If both models fit but their apices differ by more than `max(2.5 px, 0.055 × period)` the
  window is discarded. Note that this veto applies only when the parabola *succeeds*: a
  circle whose parabola fit failed outright is accepted with no cross-check at all.

Nine windows are tried per seed (three half-widths × three offsets). A fit is kept only if
another window produced a nearly identical point, so unstable fits are discarded. Circles
are preferred over parabolas.

### 7. Lattice completion and edge line — `refine_edge()`, `refine_edge_from_arc_lattice()`

Before refinement, the lattice is extended to the full side so that **missing holes are also
attempted**: every unoccupied integer node inside the side gets a hypothetical seed.

If five or more circles were found, `refine_edge_from_arc_lattice()` takes over and builds
the geometry from arcs alone. It finds the largest straight-edge consensus among the arcs
(`arc_line_inliers`, so a single bad arc cannot tilt the line), fits an integer lattice to
the arc positions, re-seeds the remaining empty nodes, and re-fits. `geometry_source` is set
to `circle_arcs` — the marker that licenses publication.

Otherwise a weaker path refits spacing and the base line from whatever fits exist. That path
never sets `circle_arcs`, so its pitch is discarded at output.

A side is marked `ok` only if it has ≥6 points, ≥0.65 coverage, ≥75% lattice occupancy,
autocorrelation ≥0.45, `spacing_rms/pitch` <0.08 and `depth_rms` < `max(3, 0.2 × amplitude)`.
Otherwise it is `review`.

### 8. Fallback ladder

If the one-pass cross-support path did not reach five arcs, progressively more assumptive methods are
tried. **Every one of them still has to produce five circular arcs.** See
*Fallback characteristics* below for how much each is used and how much it shifts the answer.

1. **`adaptive_lab_a` / `adaptive_lab_b`** *(phase 2, yellow)* — the paper/mount boundary is recovered from
   colour instead of brightness. Lab channel a or b, both polarities, and ~21 threshold
   levels are tried, restricted to the outer 7% of the stamp so the printed design is not
   mistaken for the edge. A candidate survives only if neighbouring thresholds (±3 levels)
   agree on pitch and line position. Intended for hinges and mounts that are nearly as
   bright as the paper.

2. **`parallel_edge_arc_recovery`** *(phase 3, orange)* — the side's line is reconstructed from the extreme
   confirmed arcs of the two adjacent sides (or one adjacent plus a reliable opposite side
   for the direction), then a lattice is seeded along it and arcs are fitted. A bounded
   search over candidate pitches, phases and normal offsets picks the hypothesis yielding
   the most circles.

3. **`parallel_normal_scan_recovery`** *(phase 3, orange)* — for the case where a bright scanner or background
   strip sits outside the real stamp edge, so the first-light boundary is simply wrong. Using
   the reliable opposite side's pitch, lines are scanned inward in steps of 1/8 period across
   the whole search band, at two candidate slopes and four phases.

4. **`sequential_kmeans_lab`** *(phase 4, red)* — the remaining k=3/4 `stack` and `close`
   profiles are tried in score order. For each profile, arcs are fitted in Lab L, a and b
   order, with the a/b polarity inferred from the colour transition across the proposed
   boundary. Search stops only when five circular arcs survive corner filtering. This phase
   also challenges a geometric result when its pitch differs from the reliable opposite
   side by roughly 0.15 gauge.

After the ladder, `recovery_targets()` decides what still needs rebuilding: any side without
five arcs, plus the weaker member of any opposite pair whose lines differ by more than 1.5°.
Up to two passes are run, so a side recovered in pass 1 can support its neighbour in pass 2.

### 9. Corner holes — `exclude_points_outside_corners()`

The profile search deliberately spans almost the whole bounding box, which helps on damaged
edges but lets a corner hole be claimed by both adjoining sides, biasing each. Once all four
lines exist, their actual intersections define the admissible interval for each side, and
holes outside it are dropped and the side re-fitted. All four intervals are computed from one
immutable geometry snapshot, so re-fitting one side cannot change the coordinate system seen
by sides processed later.

### 10. Orientation refinement and second pass

`refine_orientation_from_measurement()` re-derives the stamp angle from the fitted
perforation lines, which are a far cleaner angular signal than the contour. Only
arc-supported sides with ≥5 points vote; recovered sides are down-weighted. The strongest
mutually agreeing cluster (within 1.5°) wins, at least two sides must agree, and a
correction larger than 10° is refused as implausible.

If the correction exceeds 0.05°, the stamp is **re-measured from scratch** at the new angle.
This is why `measure_stamp()` runs twice per stamp.

### 11. Dimensions and gauge

Opposite lines are intersected with the normal to their mean direction through the midpoint
at the stamp centre, giving `valley_width_px` / `valley_height_px`, a `*_variation_px`
(the change in separation across ±35% of the span) and the angular disagreement of the pair.

`reconcile_perforation()` then settles the two sides of each axis:

* Each side is re-fitted with `fit_extreme_arc_lattice()`, which chooses the **coarsest**
  lattice consistent with *all* in-bounds arcs, and may discard at most one uniquely
  incompatible arc — never a convenient subset.
* If the sides agree within `PERFORATION_AGREEMENT` (0.25 gauge), done.
* Otherwise the other side's period is tried directly, and the exact **×2 and ×3 harmonics**
  are tested — the classic failure where a side locks onto every second or third hole.
* Gauges must fall in `PERFORATION_LIMITS` (5–22), with 8–17 preferred.
* A reconstructed side is not allowed to move a natively measured gauge.

If the two sides still disagree, both values are reported as a pair, e.g. `(14.20/14.50)`,
rather than averaged.

### 12. Parallelism

Stamps are independent, so with `--workers > 1` the scan is copied once into shared memory
and workers attach read-only, returning only their compact measurements. This avoids
pickling a full scan per stamp on Windows, where workers are spawned rather than forked.

---

## Status vocabulary

| value | meaning |
| --- | --- |
| `ok` | all quality gates passed |
| `review` | a result exists but at least one gate failed, or a fallback produced it |
| `unavailable` | no measurement; no pitch and no dimension are reported |

Per-hole `category` in the JSON:

| value | meaning | drawn as |
| --- | --- | --- |
| `perforation` | circular arc fitted **and** accepted into the lattice and line | filled magenta |
| `ignored_perforation` | arc fitted but rejected (outlier, outside corner bounds) | hollow magenta |
| `approximation` | no arc; a working seed only, contributes nothing | orange |

Fitted edge lines are drawn in the phase colour of the method that produced them.

Each side in `scan_perf.json` also carries `status`, `reason`, `phase`, `phase_name` and
`seed_source`. The last of these records whether the arcs behind that side were searched
for against the image (`image`, phases 1–2) or at positions predicted by a pitch
hypothesis (`hypothesis`, phases 3–4). A consumer that needs independent evidence should
filter on `seed_source == 'image'`, because `geometry_source` alone does not distinguish
the two (see below).

---

## Fallback characteristics

The earlier fallback-share and offset figures belonged to the removed brightness-first
pipeline and are no longer production statistics. They must be re-measured after the
cross-support migration. Until then, phase 3–4 results remain provisional because their
arc searches are seeded by a pitch hypothesis rather than discovered independently from
the image.

---

## Performance

The previous timing table described the removed brightness-first pipeline and is
intentionally not reused. What follows is a static cost accounting derived from the code,
not a measurement; the figures are hypothesis counts, which are stable, rather than
wall-clock, which is not.

The unit of cost is one `refine_valleys()` arc-lattice fit. `kmeans_brightness/gate_cost.py`
measured a full-side pass at about 1037 ms; the calls inside the recovery grids carry fewer
seeds and are cheaper, so treat 1 s as an upper anchor. Everything else in the pipeline —
binarisation, boundary profiles, periodicity scoring, threshold sweeps — is negligible
beside it.

| Stage | Method | Arc fits per invocation |
| --- | --- | --- |
| 1 | `cross_support_edge()` | 1, after 14 cheap profile probes |
| 2 | `adaptive_color_edge()` | 1, after 84 cheap threshold probes |
| 3 | `recover_missing_side()` | anchor pairs × pitch candidates × 5 phases × 25 normal offsets ≈ 125–500 |
| 3 | `recover_side_by_parallel_scan()` | ≤2 slopes × ~19 offsets × 4 phases ≈ 150 |
| 4 | `recover_side_by_sequential_kmeans_lab()` | ≤12 profiles × 3 Lab channels ≤ 36 |

So the primary path is close to free and the ladder is two to three orders of magnitude
more expensive. Both phase-3 methods evaluate their *entire* grid with the full fitter and
keep the best hypothesis; neither screens candidates first nor stops early on a good one.

Three structural multipliers apply on top:

* The recovery loop in `measure_stamp()` runs up to two passes, and a **third** geometry
  pass follows the phase-4 challenge, so a side can enter `recover_missing_side()` and
  `recover_side_by_parallel_scan()` three times with unchanged inputs.
* A side that failed `recover_side_by_parallel_scan()` cheaply in pass 1 (its
  `reliable_side(opposite)` guard short-circuits in microseconds) pays the full grid in a
  later pass once the opposite side has been recovered.

Orientation refinement now starts with a primary-only `cross_support` probe. When at least
three of those edges form a tight orientation consensus, the fallback ladder is deferred
until after the refined angle is known. If primary edges cannot establish the angle, the
former full coarse-angle pass is retained because recovered sides may be essential orientation
evidence. This conditional guard avoids discarded fallback work without changing the hard
cases that depend on it.

The worst case observed is roughly 1000 arc fits per `measure_stamp()` call, about 70% of
them spent on a single side that cannot succeed at any price — see the case study below.

`--stamp` debug runs are slower than production **by design**: the `collect_trials` block
runs all three recovery methods on all four sides regardless of whether the side already
succeeded, so that `algorithm_trials.json` records what each method would have produced.
Debug timings are not representative of a batch run.

### Identified but unimplemented speedups

Recorded here so the analysis is not repeated. None of these are in the code.

* **Pre-screen grid hypotheses.** Score each candidate line with a cheap proxy (mean
  gradient magnitude at the predicted seeds, say) and arc-fit only the best handful. The
  strict circle gate still decides, so currently-succeeding cases are unaffected. Highest
  payoff, and the one needing most care: the proxy must not discard the true offset.
* **Reject bands that cannot hold an edge.** If a side's binary band is effectively
  single-class there is no paper/background transition in it, and every in-band method
  (phases 1, 2, the normal scan, and phase 4) is guaranteed to fail. The test is one
  `mean()`. Only `recover_missing_side()` should survive the guard, since it extrapolates
  from neighbours and may legitimately reach outside the box.
* **Memoise failed geometric recoveries** on (side, reliable neighbours, their lines and
  pitches), which is what they are deterministic in, to suppress the repeat passes above.
* **Cap arc fits per side**, so a pathological stamp has a predictable ceiling instead of
  an unbounded one.

## Known limitations

* **No reliable ground truth.** The corpus has ruler-derived 11/16-interval measurements,
  but they are too uncertain to validate or tune the detector. The only external hypothesis
  currently under test is that the horizontal/vertical gauge pair is either 14.50×15.00 or
  14.25×14.75; the class of each stamp is not independently known. Synthetic tests use
  perfect circles — the same model the arc fitter assumes.
* **Straight edges are reported as failures.** A genuine booklet or coil straight edge yields
  `unavailable` and suppresses that dimension. There is no status value meaning
  "imperforate / straight edge", so it is indistinguishable from a detection failure.
* **One global threshold** is used for every stamp in a scan, with no per-stamp adaptation,
  and detection uses the 20th grey percentile as its baseline while measurement uses the
  10th.
* **`--dpi` is unverified** and scales every gauge linearly.
* **Dependency versions are unpinned**, while the numeric output depends on OpenCV's
  interpolation, Sobel and k-means behaviour.
* The per-hole diagnostics in `segment_stamps.py` **re-implement** the fitting loop from
  `refine_perforation.py` rather than calling it, so the two can drift apart.
* **A mis-placed detection box fails silently and expensively.** Nothing downstream checks
  that the stamp is actually inside its own box, so a lateral offset is reported as a
  per-side measurement failure rather than as a detection fault. See the case study below.

## Case study — laterally offset detection box

Worked out from the `--stamp` debug artefacts of id 6 in the 14K scan, without rerunning
detection. It is recorded because the symptoms are misleading: the visible complaints were
"left border is way off" and "everything takes a very long time", and neither is a
perforation-measurement problem.

**The orientation is correct.** Coarse estimation returned 0.000° — it only ever saw the
axis-aligned bounding box — and the refinement corrected it to −4.231°, voted for by top
(−4.297°) and bottom (−4.165°). The rectified stamp is square, so the refinement did its
job. Residual tilt in the refined frame is 0.127° top, 0.009° bottom, 0.174° right, and
2.363° left; the left outlier is a consequence of the real fault, not the cause.

**The box is laterally offset.** The stamp body occupies patch x 260..1056, while the box
claims x 105..977: shifted 117 px left and 76 px too narrow. The search bands inherit the
error, and three of the four then straddle the wrong material:

| side | search band | true edge | consequence |
| --- | --- | --- | --- |
| bottom | 1128..1346 | 1262 | correct — the only side measured at phase 1 (11 arcs, gauge 14.598, `ok`) |
| left | 57..218 | 260 | 42 px **outside** the stamp; the band is pure backing paper |
| right | 864..1025 | 1056 | 31 px **outside**; the band lands in the printed design |
| top | 40..258 | ~165 | inside, but the outer half is the scan's bright border strip |

Everything else follows. Left produced one arc against the 5-arc bar and published nothing;
its raw phase-1 reading (pitch 58.5 px, gauge 16.144, coverage 0.51, autocorrelation 0.29,
from the `k31`/`close` profile) is noise fitted to backing paper, and the line it drew sits
about 110 px too far left at the top and 60 px at the bottom. Every left fallback returned
`unavailable`, which is the correct answer — no method can find an edge in a band that
contains no edge. Top and right were rescued geometrically from their neighbours at phase 3
(14.506 and 14.942), with `seed_source == 'hypothesis'`; right's recovery line lies at scan
x 6216..6298, i.e. **outside** the box's own right edge at 6184. Only bottom carries
`seed_source == 'image'`, so this stamp rests on a single trustworthy side.

Note the asymmetry worth knowing about: `recover_missing_side()` reached outside the box to
save right, but there is no equivalent reach on left, whose rescue depends on an opposite
side that never became reliable in time.

**Probable cause of the bad box**, stated as a hypothesis because confirming it needs a
detection rerun: the stamp abuts the scan's bright top border. The patch's top-right corner
sits at scan y=98, inside the known ~143 px border strip, and the box top is at scan y=129.

The runtime is the same fault seen from the other end — three of four sides fail phase 1 and
climb the whole ladder. This stamp does not have enough primary edges to establish its
orientation, so the safe orientation guard retains its full coarse-angle pass. It still
needs the early-exit guards described in *Performance* to fail fast.

---

# Design size — `design_size/`

A second, independent measurand: the size of the **printed design** on each stamp, as
opposed to the perforation that surrounds it. It consumes a scan plus the `*_perf.json`
written by the detector.

```
python design_size/measure_design.py scan.jpg --dpi 1200
```

| file | contents |
| --- | --- |
| `scan_design_size.jpg` | the scan with each design rectangle, its corners and its size in mm |
| `scan_design.json` | per-stamp design rectangle, fitted scales and quality figures |
| `scan_golden.png` | the averaged design with every detected frame rule drawn |
| `scan_cancel.jpg` | the scan with the shared design subtracted, leaving the cancellations |
| `scan_uncancelled.jpg` | the scan with the cancellations taken off and the design restored |

`scan_design_size.jpg` is the result meant to be read, and is the counterpart of
`scan_detected.jpg`. Line weights, font scale and label position follow it, so the two can
be compared at the same zoom. A normal rectangle is red, a recovered result needing review
is orange, its corners are cyan, and the label carries the stamp number and the size to two
decimals. A stamp the perforation stage could not close has no design rectangle and is
outlined in grey as `not measured`. A rejected registration is likewise grey and has no
published size, so neither kind of omission is silent.

Because neither the fitted transform nor the placement shears, the drawn quadrilateral's
side lengths *are* the reported width and height — the annotation is the measurement, not
an illustration of it.

One scan is one population. Different scans carry different designs and are never matched
to each other; nothing in the tool compares across scans.

## The measurand

The design is bounded by one or more printed **rules**. The reported rectangle is the
separation of opposite rule **centrelines**, counted outwards-in, with `--rule 1` the
outermost. A rule's width grows with inking pressure, roughly symmetrically, so its edges
move with the printing and its middle does not. This is the same reasoning that puts the
perforation datum at the valley bases rather than the tooth tips.

All rules found on all four sides are recorded in `rules`, so the choice is auditable and a
different one can be published without re-deriving anything. Their separation is also a
free consistency check: two genuine rules keep a constant gap, and on the 5K scan they do,
19.1 px horizontally against 18.6 px vertically. Not every design has a second rule — the
high values with coloured centres have only a thin outer rule — so a `rules` entry beyond
the first is not evidence that one exists.

## Why an average, and why it is not optional

The design is not centred inside its perforation. Measured across 28 stamps, the gap from
the valley line to the design runs from 0 to 38 px horizontally and 0 to 60 px vertically,
and the perforation frequently cuts the design away entirely at a corner.

Two consequences follow, and they point in the same direction:

* **A per-stamp rectangle fit is not reliable.** The feature being measured is partly
  absent on many stamps.
* **Averaging on the perforation frame alone does not work.** A de-centring of 38 px is
  wider than the frame rule itself, so the rules of different stamps do not overlap and the
  average has no rule to detect.

So the average and the per-stamp fit are *the same operation*: the stamps are jointly
registered against their own running average until it converges (congealing), and the
average is simply the fixed point. Scale is then measured against the whole design — the
oval, the lettering, the ornaments — rather than against an edge that may not be there.

## Algorithmic flow

### 1. Common frame

Each stamp's valley-base quadrilateral is intersected from the four `line_image` entries in
the perforation JSON. A stamp missing any side is skipped: three lines cannot bound a
stamp, and reconstructing the fourth would import the perforation measurement's error into
this one.

Each stamp is then placed on a shared canvas by **rotation and translation only**. No
scaling is applied at this stage, because scale is the measurand. The mask is the valley
quadrilateral eroded by `--erode`, every pixel of which is paper by construction. Grey is
taken from Lab L and normalised to zero mean and unit variance per stamp, since ink density
and paper tone vary and only geometry is being compared.

### 2. Congealing

Four pyramid levels, 1/8 to 1/1, with two or three rounds each. Each round rebuilds the
average, re-registers every stamp against it, re-fixes the gauge, and recomputes the
outlier weights. The coarse levels exist to carry the de-centring, which is far larger than
the frame rule is wide.

Registration is `cv2.findTransformECC` with an affine model. Maximising the correlation
coefficient is invariant to linear brightness and contrast change, which is needed because
cancellation, fading and scan exposure differ between stamps, and it returns a continuous
sub-pixel transform. A discrete size sweep would be both slower and less precise.

All stamps propose their transform before any proposal is accepted. Scale is compared with
the simultaneous cohort by median and MAD, with a minimum four-percent tolerance; residual
rotation over 3 degrees, free shear over 0.05, and very broad absolute scale limits are also
guards against a cancellation becoming a geometrically convincing local optimum. A rejected
ordinary proposal is retried once with the previous round's residual outliers replaced by
the current golden prediction. The coordinate offset is deliberately not bounded: a design
can genuinely be printed off-centre relative to its perforation.

### 3. Five degrees of freedom, not six

`MOTION_AFFINE` has six. The sixth is shear, and nothing shears a printed design: shrinkage
and plate differences scale it along its own axes, and the platen is rigid. Fitting shear
anyway only lets registration noise leak into the two scales being measured, so each
estimate is projected back onto scale, rotation and translation.

The shear the unconstrained fit asked for is kept as `unconstrained_shear`, and the
residual angle as `residual_rotation_deg`. Both should be near zero — the perforation
geometry already removed the rotation — so both are independent evidence that a stamp
registered, not parameters of the answer.

### 4. Gauge fixing

Congealing determines the stamps only up to one transform shared by all of them. Left
alone, the whole set drifts. Each round pins the geometric mean scale to 1 and the mean
rotation and translation to 0. That also makes the golden's own rectangle the collection
mean, so it can be read directly, and keeps the per-stamp angle usable as the check above.

### 5. Robust weighting

Cancellation ink, hinge remnants and tears are uncorrelated between stamps, so they stand
out as large local residuals against the average. Pixels beyond `--outlier-cutoff` robust
sigma stop contributing to the average. They are also replaced by the golden for the robust
retry of an implausible ECC proposal; normal proposals retain the original pixels because
masking every residual edge measurably biases scale. The surviving fraction is reported per
stamp as `kept_fraction`; on the three scans tested it runs from 0.93 to 1.00.

Final scales are checked once more against the collection. An isolated scale outlier, or a
stamp for which no affine refinement was ever accepted, gets
`registration_status: rejected`; its diagnostic candidate is retained under
`candidate_design_*`, but the publishable `design_width_*` and `design_height_*` fields are
omitted. Recovered ECC failures/rejections and unusually large residual rotation, shear or
RMS produce `registration_status: review` without suppressing the measurement.

### 6. Reading the golden

Profiles are taken over the central `--profile-span` of each axis and reduced with a
median, because ornaments and value text interrupt the frame on any single row.

A pixel is only read if at least `--min-coverage` of the stamps reach it, so that the
average rather than any individual stamp is the thing being measured. Near the canvas
margin only one or two stamps reach a pixel, and whatever they happen to carry there
survives averaging intact. Pixels below the quorum are dimmed in `scan_golden.png`.

The quorum is deliberately low. It was once 0.6 and was the only defence against margin
artefacts, but a quorum tight enough to exclude them also clipped the canvas straight
through the 70K scan's left-hand outer rule, which is how that scan came to report
15.99 mm instead of 16.40. The four-side agreement below rejects such artefacts on the
evidence instead, so the quorum's remaining job is only to mark where the profile is
defined at all. Each side's search is then restricted to that defined span — a profile
cannot be padded out to full length without the padding itself forming a step that reads
as a feature.

### 7. Finding the rules

The four sides of a design are one printed rectangle observed four times. That is the only
premise this stage needs, and it replaces every per-side threshold.

Working inwards from each edge over `--rule-search-span`, each local maximum of the
darkness profile is described by two numbers:

* **Prominence** — how far one must descend from the peak before being able to climb
  higher. It is invariant to any constant added to the profile, so it judges a peak against
  its own neighbourhood rather than against a global level. Divided by the profile's own
  noise (the robust scatter against a lightly smoothed copy) it becomes a dimensionless
  significance, which is what `--rule-significance` bounds.
* **Width at half prominence**, again measured from the peak's own saddles. Unlike a
  threshold-crossing width this does not depend on how strongly the line happened to print,
  which is precisely the property needed to compare one side against another.

The centre is the darkness-weighted centroid of the peak over that half-prominence support,
taken on the unsmoothed profile.

Candidates are then matched across sides, outermost first, and a set of four is accepted
only if its widest and narrowest member differ by no more than `--rule-width-tolerance`.
This is the whole of the decision. A faint rule and a strong rule of the same line agree on
half-prominence width even when their prominences differ by a factor of two, so a side that
prints weakly is no longer at risk of being skipped; and a peak that only one side shows is
rejected however strong it is, because the other three cannot corroborate it. If one side
genuinely cannot see the outermost rule, no family containing it exists and all four step
inwards together rather than reporting a mixture.

The tolerance has an empirical gap to sit in. Across the eight scans the accepted families
span width ratios of 1.02 to 1.24, while the 70K mixture that had to be refused was 1.75,
so the default 1.5 is not a tuned value. `--rule-significance` is a permissive floor that
only bounds the candidate list: real rules there run from 31 to 123 sigma and spurious peaks
up to 34, so the two overlap and significance alone could not have made this decision.

### 8. Output

Each stamp's rectangle is the golden rectangle multiplied by that stamp's own fitted
`scale_x` and `scale_y`.

To place it back on the scan, the golden rectangle's corners are carried through the
stamp's fitted transform into its canvas and then through the inverse of the rigid
placement that put the canvas there, giving `design_corners_image` in original scan
pixels. The perforation stage's own detection box is reused only to position the label.

## Revealing the cancellations

Congealing separates every stamp into what it shares with the others and what is only its
own. The design size is measured from the first part; `scan_cancel.jpg` is the second,
which is very largely the postmark. It is written on every run, not behind a flag, because
it is a by-product of work already done rather than a separate job.

Nothing extra is estimated. The golden warped into each stamp's own frame is already
computed, today only to report `residual_rms`, and it is already the basis on which the
outlier weighting recognises cancellation ink. Subtracting it costs one more pass.

`scan_cancel.jpg` keeps the scan's geometry, so it can be laid beside `scan_detected.jpg`.
Outside the measured stamps the scan is left as a pale ghost, which keeps the layout and
the perforation readable without competing with what has been revealed.

Two corrections do the real work:

* **A gain and an offset per stamp**, fitted on inlier pixels only. The normalisation in
  `stamp_patch` equalises each stamp's mean and spread but not its ink density, so without
  this the dark parts of the design do not cancel. Fitting on inliers is what keeps a heavy
  postmark from dragging the gain and leaving the design behind.
* **Local-shift deghosting.** A misregistration of a fraction of a pixel turns the design
  into design + d·∇design, which is why the oval and the lettering survive a plain
  subtraction while the paper between them does not. That two-parameter model is fitted
  over a sliding `--reveal-window` and subtracted, removing the part of the residual that
  any local shift could explain. A postmark is not a shifted copy of the design, so it
  stays; in the tests the model removes at least 60% of a 0.3 px misregistration while
  leaving an added mark intact.

`--reveal-fade` sets how much of the design to take out, from 0 to 1. Full removal reads
the postmark most clearly; leaving a quarter behind shows where on the design it sits.

The output is plain greyscale and carries no annotation, so everything in it is the
subtraction's own result and can be judged as such. Two things in it are not cancellation:

* **Where the design prints solid there is no signal to recover**, because the postmark
  there is ink on ink. Those areas come out blank, and blank means "nothing could have
  been seen here" rather than "no cancellation here".
* **Fine design detail comes back as texture.** Stipple and engraving at the resolution
  limit do not register identically across stamps, so the average blurs them and each
  stamp keeps its own version, which no subtraction can remove. It shows as a faint ghost
  of the design and a fine mottle.

## Taking the cancellations off

`scan_uncancelled.jpg` is the same decomposition read the other way: where the residual
says a pixel is not part of the design, the design is put back. Only marked pixels change,
so away from a postmark the output is the original scan down to the bit — the change is
computed and added, rather than the stamp being resampled and pasted.

Deciding which pixels are marked is the whole difficulty, and a threshold on the residual
cannot do it. The texture above is not small, and a stroke's soft edge is as faint as it
is. Two things are true of a postmark and of nothing else on the stamp: **it is connected,
and it is large.** So each mark is followed outwards from a core of four robust sigma down
to one and a half, and is believed only if its core covers enough of the stamp to have
been meant. A deep isolated excursion of the texture is neither connected to a core nor
large, and is left alone.

That selectivity is the point, and it is measurable. On the 3K scan the heavily cancelled
stamps have 3 to 8% of their area replaced, while the one that carries almost no postmark
has 0.0% — it comes through untouched. A plain 2 sigma threshold cleaned the heavy stamps
slightly better but replaced 8% of the clean one, which is not removing a cancellation but
substituting the average for the stamp.

Lightness under a mark comes from the golden, carried into the stamp's own exposure by the
same gain and offset. Colour cannot be taken from anywhere, since black ink destroys the
stamp's own hue as well, so it is rebuilt from the restored lightness using the relation
between the two over the uncancelled part of that stamp. It holds as long as lightness
predicts hue, which on these stamps it does: it carries a one-colour print on tinted paper
exactly, and the two-colour 70K well enough that the brown frame and the orange centre
both come back correctly. Two inks of similar lightness and different hue would not be
told apart, and would be reconstructed as whatever the regression averages them to.

`--uncancel-fade` sets how much of the mark to take off, from 0 to 1.

**It cannot tell a postmark from any other individual blemish.** A tear, a thin spot, a
pen stroke or a plate flaw peculiar to one stamp all satisfy the same description, and all
will be restored away. The operation is "make this stamp agree with the others", and
reading it as "remove the cancellation" is an interpretation of the result, not a property
of the method. **Nothing restored is evidence about the stamp** — the detail under a mark
is the average of the others, invented for this stamp.

## What the numbers are worth

**Relative sizes are precise; absolute sizes are not.** Every stamp's figure is the golden
rectangle scaled, so the absolute value rests on one rule detection on one image, and a
bias there moves every stamp in the scan equally. Relative differences within a scan are
unaffected. For sorting a scan into groups this is the right trade; for quoting a
traceable millimetre figure it is not.

**The precision is real and was measured.** Re-running a scan with a different mask
erosion, canvas padding and congealing schedule moves the fitted scales by a run-to-run sd
of 319 ppm in x and 96 ppm in y, against a between-stamp sd of 3798 ppm and 2349 ppm. The
method therefore resolves the observed spread about twelve times over. In millimetres, on
the 5K scan, that is roughly 0.005 mm of method noise against a 0.063 mm spread.

**The error budget is physical, not algorithmic.** At 1200 dpi one pixel is 21 µm, and
sub-pixel registration over a frame perimeter of a few thousand pixels is far below that.
What remains is paper shrinkage, ink spread and scanner geometry.

## Known limitations

* **Nothing checks the scanner.** A flatbed whose true x and y scale differ, or whose scale
  varies along the platen, produces exactly the signature this tool is designed to find.
  Measured size should be regressed against position along the scan before any difference
  is believed. `--dpi` is taken on trust here as it is elsewhere.
* **Shrinkage is not separated from plate differences.** Sheets were perforated after
  printing, so drying shrinkage moves the design size and the perforation pitch together.
  Correlating the two against `*_perf.json` would test this; it has not been done.
* **A size difference is assumed to be the whole story.** Genuinely different dies would
  also register as a scale change. The residual map between two groups distinguishes them —
  pure scale gives residuals growing radially from the centre, different dies give local
  ones — but the tool does not compute it.
* **Three stamps minimum**, and few stamps make a weak average; the quorum is clamped to
  two stamps so a small scan still measures something.
* **The four-side agreement needs four sides.** A design whose frame is genuinely
  interrupted on one side — cut by the sheet margin, or overprinted — leaves no family to
  accept, and the scan reports nothing rather than guessing from three.
* **`--rule 1` is not a physical identity across scans.** It is the outermost rule *that
  this scan could see on all four sides*. If a scan loses its outer rule to the quorum or to
  damage, its rule 1 is the next line in, and comparing it with another scan's rule 1
  compares two different printed lines. The recorded half-prominence widths are what make
  the two distinguishable.

## Tests

```
python -m unittest discover -s tests -t .
```

About one minute. The real hinge regression uses a cropped fixture stored in
`tests/fixtures/`, so the suite has no external scan dependency and does not skip in CI.

`tests/test_design_size.py` closes the loop on the design measurement with a synthetic set
that is de-centred by more than its frame rule is wide and defaced with cancellation bars:
known independent x and y scales are recovered to 0.2%, and the rectangle to 1.5 px.

Four of its cases are regressions for failures seen on real scans: a margin artefact read
as a rule; coverage accumulated in units of 255 rather than stamps, which silently disabled
the quorum that prevents it; the 3K scan, where the rule printed faintly on two sides and a
threshold-based detector dropped through to the band behind it; and the 70K scan, where one
side could not see the outermost rule at all and the four sides had to step inwards
together. The last two are reproduced from the real candidate lists, so they fail if the
selection rule regresses to anything per-side.

---

## Method experiments — `kmeans_brightness/`

A reproducible study of replacements for the former phase 1 `brightness` path. Production
and the experiments now share the `stack`/`close` implementations in `edge_profiles.py`;
the experiment scripts contain the stored comparisons and reporting code.

What is compared:

* **Binarization** — `brightness` (one global threshold, as shipped) against a k-means
  clustering of the edge band in Lab, where the darkest centroid is fixed to background,
  the brightest to paper, and every assignment of the remaining centroids is tried: two
  splits for k=3, four for k=4, six for both together (`kmeans3`, `kmeans4`, `kmeans34`).
  **`kmeans34` is the chosen direction**, because a global threshold cannot separate a
  stamp hinge from the paper at all, while a centroid split can, recovering holes that are
  otherwise unmeasurable.
* **Ridge extraction** — eight ways to turn an edge band into a boundary depth profile,
  from the shipped `raw` first-sustained-run rule to an area-and-pinhole gate, an
  interior-connectivity gate, a paper-count depth, an asymmetric closing of the depth
  signal, the straight-line light-majority experiment, and combinations (`ridges.py`).

### Production primary — one-pass cross-support

Production now uses the strict one-pass selector. The former standalone brightness
method is gone: its binary mask survives only as one peer among the seven maps and never
receives its own arc pass or priority.

1. build the brightness peer plus all six k=3/4 centroid assignments;
2. extract `stack` and `close` profiles from every map;
3. score all fourteen profiles before fitting any arcs;
4. apply a symmetric 1.1× bonus when a brightness and k-means profile agree in period and
   edge position;
5. select the maximum adjusted score and run exactly one greyscale arc pass, with a 5%
   profile-pitch constraint on the final lattice.

The sequential k-means-only arc search remains an experiment, not the production method.
It publishes two more sides on `worst.jpg`, but needs multiple arc passes. Pixel voting,
median-depth fusion, general pitch/line consensus, same-map `stack`/`close` agreement and
the line-majority ridge all performed worse than cross-support.

A less circular check first clusters the 65 complete stamps into two groups without
supplying either theoretical pair. For the strict one-pass results, the fitted
horizontal/vertical centres are `(14.266, 14.747)` and `(14.535, 14.968)`, with cluster
sizes 34 and 31, separation 0.348 and within-cluster RMS 0.095. The sequential results
produce virtually the same centres. Only after fitting are they compared with the
theoretical `(14.25, 14.75)` and `(14.50, 15.00)` pairs. This is evidence compatible
with the two-class hypothesis, but not proof: a shared detector bias can move both
methods together and there is still no independent class label per stamp.

The one-pass selector publishes 302/320 collection sides. Of the 65 complete
four-side stamps, 61 assign both axes to the same nearest theoretical pair; opposite-side
p90 is 0.153 and no published side is more than 0.5 from both axis-appropriate theoretical
values. It agrees very closely with the sequential selector where both publish: median
gauge difference 0.001, p90 0.027, maximum 0.099 over 302 collection sides. On
`worst.jpg` it publishes 94/108 sides, versus 96/108 for the sequential multi-pass
selector. Across 18 nearby bonus/tolerance settings, coverage stays at 301–302/320 on the
collection and 94/108 on `worst.jpg`, with no side more than 0.5 from both axis-appropriate
theoretical values.

### Profile roughness follow-up — recorded, not yet production

An October 2026 follow-up investigated a false primary edge on `3K`, stamp 8, left side,
and checked candidate-selection changes on the eight available nominal scans (`1K`, `2K`,
`3K`, `5K`, `7K`, `14K`, `35K`, and `70K`). The sweep contained 98 stamps, 390 sides with
at least one usable cheap-profile candidate, 194 complete opposite-side pairs, and 387
sides with an independent reference made from at least five accepted circular arcs. The
source scans and scratch evaluator outputs are external to this repository, so the figures
below are an experiment record rather than a committed reproducible benchmark.

The failure was unusually clear. The current score selected `k410/close`, pitch 25.851 px,
14 guesses, on the problem side. Its boundary followed image texture rather than the stamp
edge. A correct profile family was near 63–64 px: the opposite side measured 63.750 px and
the corresponding side in the older scan measured about 64.08 px.

Three selector changes were compared:

* **Count cap.** Capping the count contribution at six selected `brightness/close`, pitch
  63.070 px, on the problem side. A global cap is not safe, however: on `7K`, stamp 11,
  right side, it replaced the correct `k401/close` result at 64.55 px (64.49 px arc
  reference) with a doubled-period `k411/stack` result at 123.54 px. `cap6` must therefore
  not be introduced globally.
* **Continuous smoothness penalty.** Multiplying the current score by
  `exp(-weight * roughness)` removed the gross half-period alias at weight 1.0, but changed
  15 selections and still chose `k30/close`, pitch 55.503 px, for the problem side. This is
  14.82% away from its opposite side and does not solve the edge-selection error.
* **Hard roughness gate.** For a profile depth sequence `d`, the tested dimensionless
  measure was

  ```text
  roughness = P90(abs(d - GaussianBlur(d, sigma=1.6))) / pitch
  ```

  The current winners had median roughness 0.02, 95th percentile 0.03, and 99th percentile
  0.06. The false `3K` winner was 0.639; the next-highest winner was 0.18. Thresholds 0.20
  and 0.30 therefore flagged only the known bad side in this dataset.

The meaning of the hard gate is important. Removing the rough candidate and selecting the
next candidate by the unchanged score chooses the incorrect 55.503 px profile. At threshold
0.20 this changes one of 390 sides and removes the only opposite-side alias above 1.5×, but
the number of pairs within 10% remains 193/194:

| selector | changed sides | pairs within 5% | pairs within 10% | aliases >= 1.5x |
| --- | ---: | ---: | ---: | ---: |
| current | 0 | 191/194 | 193/194 | 1 |
| reject rough candidates, then reuse current score (`roughness <= 0.20`) | 1 | 191/194 | 193/194 | 0 |

The promising interpretation is instead to reject the **whole primary result** when its
winner exceeds 0.20 and hand the side to the existing fallback ladder. One targeted full
pipeline run did this without `cap6` and without an opposite-side score weight. The existing
`parallel_edge_arc_recovery` recovered the left side at 63.882 px from 13 points, only 0.21%
from the right side's 63.750 px. No other primary winner crossed the 0.20 threshold in the
eight-scan selector sweep, including the correct `7K` side.

This gate is not implemented yet. If continued, the conservative next experiment is:

1. compute roughness only for the already selected winner, keeping the normal primary path
   linear in profile length and requiring no extra arc pass;
2. when roughness exceeds 0.20, publish no primary edge but preserve all profile candidates
   for the existing recovery methods;
3. run the complete end-to-end dataset before changing production selection.

Opposite-side pitch agreement may remain weak supporting evidence, but it must not dominate
selection: opposing perforation is often similar, not guaranteed identical. The same applies
to agreement between different scans of the same stamp design. Neither was added as a score
term in this follow-up.

### Input

The test scan is `worst.jpg`: 27 worst-case stamps cropped from eight real scans and
composited onto backing paper, at 1200 dpi. It is 47 MB and is **not** in the repository.
Point the experiments at your copy:

```
set STAMPS_WORST_SCAN=C:\path\to\worst.jpg
```

Without it, the default path in `experiment.py` is used.

### Rerunning

Run from the repository root, in this order. The two experiments use 10 worker processes
and write the JSON the reports read, so a report can be regenerated in seconds without
repeating a sweep.

```
python kmeans_brightness/experiment.py          # -> results.json        ~3.5 min
python kmeans_brightness/ridge_experiment.py    # -> ridge_results.json  ~12 min
python kmeans_brightness/report.py       > kmeans_brightness/report.txt
python kmeans_brightness/ridge_report.py > kmeans_brightness/ridge_report.txt
python kmeans_brightness/degradations.py        # -> degradation_sheets/  ~6 min
python kmeans_brightness/arc_strategy_experiment.py
python kmeans_brightness/arc_strategy_report.py
python kmeans_brightness/collection_experiment.py
python kmeans_brightness/collection_report.py
python kmeans_brightness/selection_policy_report.py
python kmeans_brightness/candidate_search_experiment.py
python kmeans_brightness/candidate_search_report.py
python kmeans_brightness/candidate_search_collection.py
python kmeans_brightness/candidate_search_collection_report.py
python kmeans_brightness/one_pass_experiment.py
python kmeans_brightness/one_pass_collection.py
python kmeans_brightness/one_pass_report.py
python kmeans_brightness/one_pass_tuning.py
python kmeans_brightness/one_pass_source_report.py
python kmeans_brightness/theoretical_validation_report.py
```

`degradations.py` renders a `--stamp`-style sheet for every side where a k-means method
does worse than `brightness`, showing the intermediate ridge, the intermediate spots and
the final arc spots for all six splits, plus a second sheet crossing the chosen splits
with every ridge extractor. It writes about 316 MB, so the output is left out of the
repository and regenerated on demand.

Supporting scripts, none of which are needed to reproduce the reports:

| script | purpose | runtime |
| --- | --- | --- |
| `check_ridges.py` | asserts the `raw` extractor reproduces `boundary_profile` exactly | ~40 s |
| `smoke.py [stamp]` | one stamp through all four methods and all six splits | ~1 min |
| `cost.py` | per-stamp cost of the four methods as they would run in production | ~4 min |
| `gate_cost.py` | per-side cost of each ridge extractor against one arc fit | ~1 min |

`common.py` (shared loading and the definition of a degradation) and `ridges.py` (the
eight extractors) are modules, not entry points.

### Stored results

`results.json` and `ridge_results.json` are the raw measurements; `report.txt` and
`ridge_report.txt` are the rendered analyses, regenerated from the JSON by the two report
scripts. All four are committed, so the conclusions can be checked without the scan.

The `worst.jpg` studies measure the primary path only, with the phase 2–4 fallback ladder
disabled. `theoretical_validation_report.py` intentionally does not read the manual CSV.
It reports coverage, within-stamp opposite-side agreement, consistency of the horizontal
and vertical assignments with the same nearest theoretical pair, and agreement between
the one-pass and sequential selectors. Legacy report scripts can still print the rough
`Prf.U` and `Prf.L` ruler measurements for diagnosis, but those numbers are not ground
truth and must not be used to validate or tune a selector.
