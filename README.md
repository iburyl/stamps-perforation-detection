# stamps-perforation-detection

A research tool that measures postage stamp **perforation gauge** (holes per 20 mm) and
**valley-to-valley dimensions** from a flatbed scan of several stamps on dark backing paper.

Only NumPy and OpenCV are required.

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

Diagnostics for one stamp/side/hole are written to `scan_details/` when `--stamp`,
`--side` and `--spot` are given. Every file name starts with the phase that produced it
(`0` for stamp localization and rectification, then `1`–`4` as above), so sorting the
directory walks the algorithm in order, and a phase that did not run leaves no files.
Each panel carries a banner naming its phase, its method, its status, and whether it was
the one selected. A diagnostic run selected with `--stamp` (optionally narrowed in the
report with `--side`) executes and records a shadow run of every current fallback
(`algorithm_trials.json`), so the alternatives can be compared on one baseline.

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

Phase 1 computes fourteen inexpensive profiles but performs only one greyscale arc pass
per side. The previous timing table described the removed brightness-first pipeline and
is intentionally not reused. The fallback ladder can still dominate stamps whose primary
pass fails, and orientation refinement can cause the complete measurement to run twice.

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

## Tests

```
python -m unittest discover -s tests -t .
```

44 tests, about one minute. The real hinge regression uses a cropped fixture stored in
`tests/fixtures/`, so the suite has no external scan dependency and does not skip in CI.

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
