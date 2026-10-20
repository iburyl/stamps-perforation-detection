# Design Components TODO

These observations concern the current component-reconstruction path implemented by
`design_components/golden_components.py` and invoked from
`design_size/measure_design.py`. The older
`design_components/background_pattern_refinement.py` prototype is not the active
integration path.

## Implemented — palette/background guards and cell selection

The 17 October follow-up implements the first corrective pass from the visual
review of 1K, 2K, 3K, 5K, and 7K:

- A near-neutral candidate is no longer accepted merely because it repeats. If
  it lies within 18 degrees of the coloured ray and their blurred gradient maps
  reproduce the same plate, the split is collapsed and the physical ink colour
  is re-estimated from all chromatic observations. This rejects the false grey
  components on 2K, 5K, and 7K while retaining the genuine black plate on V/5K.
- A recovered background is independently compared with the accepted main
  plate. A close-colour candidate with duplicated structure is flagged in the
  sidecar. Visual review showed that this guard was too aggressive for 7K, so
  it is diagnostic rather than an automatic rejection.
- Main/background density is now conserved: the recovered background's
  projection is removed from the main coefficient before the two rays are
  reconstructed. A synthetic regression test covers the calculation.
- Anchor and consensus cell registrations are scored by pairwise agreement
  outside the central fitting window. Visual review across all five issues found
  consensus to be the most reliable universal default, including cases where
  the held-out score preferred the anchor. The panel therefore always uses the
  consensus rebuild and fixed `averaged | consensus | legacy | best | second |
  anchor` columns; the held-out scores remain diagnostic.
- The same cell-consensus registration is now available for an explicitly selected
  colour plate through `--screen-ink`. Sparse colour plates choose their own shared
  phase window and adapt the bearing-window size to the amount of surviving
  evidence. Registration is estimated on the cleaned periodic field, but its
  transforms are applied to the complete colour layer so the strong ornament is
  not discarded with the fitting mask. For a colour plate, the
  preservation-ranked best impression is authoritative wherever it is readable;
  registered peers only fill its unreadable regions. This prevents a strong
  screen in a badly faded stamp from replacing the visibly superior complete
  design. Backward selection also falls back to the strongest singleton whenever
  a multi-impression merge would weaken the screen.
  On V/5K this admits 15 impressions at a 128/64 px geometry window; held-out cell
  agreement prefers consensus (+0.017) over the strongest-anchor solution (+0.006).
- Component reconstruction no longer inherits the historical five-impression
  minimum from `two_pass_decomposition`. Its production call accepts every
  non-empty aligned collection, and the screen route likewise continues with any
  non-empty separable set; quorum, validation, and cell consensus degrade according
  to the actual sample count. I/10K now completes from four stamps, using all four
  colour observations and reporting +0.937 half-set agreement.

The final full-resolution runs take 52 s (1K), 60 s (2K), 34 s (3K), 61 s
(5K), and 103 s (7K) on the current machine. The held-out validation adds two
full-layer feature/warp evaluations; the dominant scaling cost remains the
pairwise cell resolver, especially for 21-stamp 7K.

## Done — Correct density conservation in the single-colour screen path

`background_ink.separate()` stores `along_main` as the projection of the complete
pixel density onto the main-ink ray. That projection already includes the part of
the background ink parallel to the main ray. `screen_passes()` currently combines
this uncorrected value with the complete recovered background ray, so the parallel
part of the background ink is counted twice during `reconstruct()`.

- Corrected the main coefficient before merging as:
  `main = along_main - background_coverage * dot(background_ray, main_ray)`.
- Recompute the main coefficient in the collection-wide basis rather than retaining
  per-impression projections made with slightly different main rays.
- Added a synthetic conservation test in which known main and background densities
  are separated and reconstructed. The reconstructed optical density should match
  the input within interpolation tolerance.
- Check the correction on the existing 1K and 5K results, where the background and
  main rays are especially close.

Relevant code:

- `design_components/background_ink.py`, `separate()` and `coverage_against()`
- `design_components/golden_components.py`, `screen_passes()` and `reconstruct()`

## P1 — Reject unsupported or contaminated background-screen detections

`screen_ray()` always returns the strongest admissible Fourier bin, even when the
bin is only noise. The collection path also calls `separate()` with
`min_separability=0.0`. The later shared-basis separability check tests only colour
geometry; it does not establish that a repeatable printed screen exists.

Existing 1K and 5K component panels show a substantial ghost of the main design in
the reported background pass. The result is nevertheless emitted as a successful
separation.

- Define minimum collection-level SNR, channel coherence, and separability criteria.
- Require an independent half-set agreement threshold before publishing a
  background golden.
- Compare aligned agreement with an unaligned or frequency-shifted control so that
  periodic noise cannot pass merely because the target frequency was selected from
  the same data.
- If the checks fail, emit only the main pass and report that the background layer
  could not be separated.
- Add negative tests using a single-ink image with texture/noise but no periodic
  background screen.

Relevant code:

- `design_components/background_ink.py`, `screen_ray()`
- `design_components/golden_components.py`, `screen_passes()` and
  `background_golden()`

## P1 — Stop the postmark mask from punching holes in worn colour ink

The staged local-colour path offers each pixel either the local colour ray or the
neutral ray, then treats the densest neutral residuals as cancellation and dilates
them by 7x7 pixels. On 5K stamp 3, 236,549 pixels are assigned to the colour after
black exclusion, but the postmark mask subsequently removes 71,350 of those colour
pixels (about 30%). Visual stage comparison confirms that intact horizontal lines
inside the oval and many background-screen points disappear at this step.

- Do not infer cancellation solely from being closer to the neutral ray when the
  printed colour is faded or locally desaturated.
- Require stroke-like spatial evidence or collection-level non-repeatability before
  declaring a neutral residual to be a postmark.
- Replace unconditional 7x7 dilation with evidence-aware growth; it currently turns
  isolated classification errors into conspicuous holes.
- Keep separate diagnostics for colour assigned before black exclusion, after black
  exclusion, and after cancellation removal.
- Add a regression fixture in which faded chromatic lines remain present beside a
  genuine dark cancellation stroke.

Relevant code:

- `design_components/ink_palette.py`, `unmix()` and `postmark_mask()`
- `design_components/golden_components.py`, `colour_layers()`

## P1 — Disambiguate periodic peaks when registering colour layers

The current ECC registration can converge to the nearest local peak of a periodic
colour pattern rather than the globally consistent phase. For 5K stamps 3 and 2,
the registered pair has direct correlation 0.243. Applying an additional shift of
approximately (+6, +8) source pixels raises it to 0.638. The competing positions
are about 10 pixels apart, matching one repeat of the printed ornament. The oval
and outer field independently prefer essentially the same correction, so this is
not a compromise between different regions.

The analytic background path has the same ambiguity. Its measured wave has an
8.31 px period and `phase_shifts()` retains only the fractional shift modulo that
period. For stamps 3 and 2, the resulting background-layer correlation is 0.365;
an additional neighbouring-cell shift of (+8, +8) raises it to 0.710. The method
also cannot observe displacement along the ruling, so phase alignment alone is not
enough to position the two-dimensional ornament.

The existing 1K background reconstruction shows the same failure in a larger
collection. Its measured period is 8.796 px. Using the strongest single background
impression, stamp 2, as the reference, 11 of the other 14 impressions prefer a
residual vertical displacement of +8 or +9 px after the current phase-only
alignment; only three prefer the same period cell. Their mean direct correlation
in the clean screen window rises from 0.399 to 0.469 when the best neighbouring
two-dimensional cell is selected. The current Fourier screen-strength score rises
only from 117.04 to 117.49, demonstrating that this metric is almost invariant to
an integer-period error and therefore cannot validate the chosen cell by itself.

An evaluation implementation now builds masked high-pass two-dimensional
correlation surfaces for every pair, then emits both a strongest-impression
anchor solution and a robust global pairwise solution. The old phase-only merge,
the best and second-best single impressions, their pair, and both new solutions
remain side by side in the component panels. The core resolver takes 0.28--4.72 s
per tested issue (12.21 s total for 1K, 2K, 3K, 5K, and 7K); most of the larger
end-to-end diagnostic cost comes from extracting the optional screen plate and
encoding the additional full-resolution panel columns.

Before making either cell-resolved candidate the default:

- Compare the anchor and consensus candidates with a held-out, cell-sensitive
  two-dimensional score rather than the period-invariant Fourier strength.
- Add a confidence gate for impressions whose best and second cell are nearly
  tied, and validate the chosen cell in more than one clean window.
- Investigate 2K separately: only four selected colour impressions currently
  retain enough clear field to fit the screen bearing, and the legacy selector
  keeps only two of those, so registration is not its main limiting factor.

Visual acceptance of the five expanded panels gives a more specific routing
decision. On 1K both cell-resolved candidates improve on phase-only registration,
with global consensus slightly preferable. On 3K both improve, with the
strongest-impression anchor slightly preferable. On 2K the ordinary colour plate
is recovered well, but every attempted background reconstruction substantially
duplicates it; this is a decomposition failure, not a cell-registration failure.
On 5K and 7K the staged palette invents a large grey/achromatic component from a
single coloured plate, so every downstream background variant is invalid. The
original 7K colour merge remains reasonably good.

- Do not select a background candidate until its density is demonstrably distinct
  from the already accepted colour plates. A screen that reproduces their design
  edges should be rejected as duplicated ink.
- Estimate a weak screen from positive optical-density residual after reconstructing
  the accepted major plates, instead of reinterpreting an entire selected colour
  layer as background.
- Reject a proposed achromatic plate when it is spatially coextensive and highly
  correlated with a nearby coloured ray. For the current 2K, 5K, and 7K palettes,
  the alleged neutral and coloured rays are only 10.8, 12.3, and 13.9 degrees apart;
  angle alone is not sufficient, but the small angle plus shared design structure
  is strong evidence for one ink observed with varying saturation.
- Keep the 1K consensus and 3K strongest-anchor results as the current preferred
  registration candidates, but derive that choice from a held-out score rather
  than hard-coding issue names.

- Search several translation seeds covering at least one full pattern period
  before the sub-pixel ECC refinement.
- In the background path, use analytic phase only for sub-period refinement; pick
  the integer period and along-ruling displacement from aperiodic structure or a
  two-dimensional lattice/template correlation.
- Rank candidates by direct pair/template agreement after warping, rather than by
  the optimizer's local ECC score alone.
- Use aperiodic anchors or low-frequency plate structure to select the correct
  repeat, then use the periodic detail only for sub-pixel refinement.
- Reject or flag a merge when the chosen solution is substantially worse than a
  neighbouring period-shifted candidate.
- Add a synthetic periodic-pattern test whose correct alignment lies one repeat
  away from the initial position.

Relevant code:

- `design_components/golden_components.py`, `register_pattern()` and
  `colour_passes()`
- `design_components/golden_components.py`, `phase_shifts()` and
  `background_golden()`

## P1 — Do not crop artwork with reviewed perforation boundaries

`aligned_collection()` constructs each validity polygon directly from the four
perforation valley lines, even when a side is explicitly marked `review`. On 5K
stamp 3 the top-side detector reports `fewer than five circular arc fits` and puts
its valley line through the printed ornament and denomination circles. The aligned
valid mask consequently starts at y=159, versus roughly y=40--86 for the other
stamps. It removes 75,968 pixels from the top relative to common collection
coverage, or 8.8% of the working stamp area.

The same mask reaches both background routes. Restoring only the missing top support
for stamp 3 increases readable background-field coverage from 404,022 to 435,566
pixels (+7.8%), including 20,207 to 51,077 pixels in the top 220 rows. The detected
screen pitch stays essentially unchanged (8.258 px in the integrated path) because
the FFT window lies below the bad boundary, but the missing field is still forced to
zero before the background golden is merged.

- Validate that every perforation boundary lies outside the measured printed-design
  corners by a plausible paper margin.
- Do not use a `review` side as a hard validity boundary without that geometric
  check.
- Fall back to the orientation polygon, source-image support, or a design-corner
  mask with a conservative outward margin when a side is unreliable.
- Keep image sampling and trusted-validity masks conceptually separate: available
  source pixels should not be discarded merely because perforation measurement is
  uncertain.
- Add a regression case with a damaged or folded top margin whose detected
  perforation line crosses otherwise intact artwork.

Relevant code:

- `design_components/two_pass_decomposition.py`, `aligned_collection()` and
  `valley_quad()`
- `design_components/golden_components.py`, `screen_passes()`
- `design_components/background_ink.py`, `main()` and `separate()`
- `design_components/background_pattern_refinement.py`, `main()`

## P2 — Make half-set validation reproduce the production merge

The production background golden is a matched-filter weighted mean. However,
`report_halves()` calls `half_set_agreement()` without weights, which makes both
halves use `robust_golden()` and therefore validates a different estimator.

- Preserve the weights returned by `window_scorer()` in the merge result.
- Pass the selected weights to both the aligned and unaligned half-set checks.
- Prefer fitting selection and weights independently within each half, or clearly
  label the current check as conditional on full-dataset selection.
- Add a regression test where weighted-mean and median results intentionally differ.

Relevant code:

- `design_components/golden_components.py`, `select_and_merge()`,
  `half_set_agreement()`, and `report_halves()`

## P2 — Fix the `best_single` metric and panel text

`choose_impressions()` starts `trail` with the score of the complete collection.
The sidecar currently writes `trail[0]` as `best_single`, and the component panel
uses it as the single-impression comparison. For the current 1K output this reports
117.04 even though the maximum value in `strength_alone` is 71.86.

- Write `max(back["alone"])` as `best_single`.
- Use the same value in the background-pass panel label.
- Rename the trail values explicitly if both the all-impressions baseline and the
  backward-elimination sequence need to be retained.
- Add a unit test for the sidecar fields and displayed comparison values.

Relevant code:

- `design_components/golden_components.py`, `choose_impressions()` and
  `screen_passes()`

## P2 — Expand direct test coverage for `design_components`

Initial tests now cover achromatic palette retention, repeatable neutral support,
solid achromatic-mask extraction, exclusion of neutral ink from the second-stage
colour search, per-impression colour-variant estimation, and the averaged-layer
quality fallback. The rest of the component reconstruction path still lacks
focused regression coverage.

Add focused tests for:

- optical-density separation and reconstruction conservation;
- known translation and rigid-motion recovery;
- screen phase and bearing recovery;
- rejection of non-periodic noise and main-design leakage;
- weighted merge and half-set validation consistency;
- sidecar metric semantics;
- handling of rejected design registrations and partially unreadable impressions.

## P2 — Let weaker colour impressions fill uncovered regions asymmetrically

The staged colour path now finds a well-preserved group from a conspicuous gap in
per-impression ink evidence; it does not assume a fixed number of stamps and it
continues down to a single-impression best effort. The current merge deliberately
excludes the weaker group, which protects recovered ink from being voted away but
also discards any genuine detail that survives only in one of those impressions.

- Build the primary template and its ink support from the adaptively selected
  well-preserved group.
- Register weaker impressions against that fixed template rather than allowing
  them to redefine it.
- Let weaker impressions add positive evidence only in regions inadequately
  covered by the primary group; their zero or washed-out pixels must never erase
  primary ink.
- Require spatial agreement between multiple weaker observations where the sample
  count permits it, while retaining an explicitly labelled single-observation
  best effort for small collections.
- Add synthetic tests with varying strong-group sizes, complementary occlusions,
  and a majority of washed impressions.

Relevant code:

- `design_components/golden_components.py`, `colour_preservation_selection()` and
  `colour_passes()`

## Documentation

Update `design_components/README.md` to identify `golden_components.py` and
`background_ink.py` as the current implementation invoked by the production design
measurement command. Keep the older experiments clearly labelled as historical
baselines.

## Regression guardrails for background routing

The IX-X 7K background decision proved unstable when it used the exact crossing
of two noisy angle estimates as a hard boundary. Rebuilding the masks moved the
measurement to 4.141 degrees of tilt against 4.152 degrees of scatter and silently
changed a useful two-pass reconstruction into a single-ink fallback. IX-X 5K is a
different case (3.524 against 4.132 degrees) where the fallback is desirable.

- Replace the current narrow uncertainty allowance with a confidence estimate or
  a downstream reconstruction-quality comparison when enough representative
  sheets are available.
- Keep regression fixtures for both IX-X 5K (must fall back) and IX-X 7K (must
  retain the background pass).
- Keep integration fixtures for automatic colour-screen routing: V 1K, V 3K, and
  V 5K must select screen consensus without a palette index, while VIII 7K must
  retain its ordinary solid-red registration.
- Calibrate the current strength and held-out thresholds on more issues. The
  initial corpus separates at best-single screen strengths 20.65, 7.55, and 5.36
  with held-out consensus scores +0.128, +0.066, and +0.017, versus VIII 7K's
  accidental 3.60 peak and effectively zero held-out score.
- Add a direct held-out comparison against the ordinary colour reconstruction.
  Merge gain alone is unsuitable: the approved V 5K screen consensus is useful
  even at 0.99x.

## Fixed-count unified ink model

The required `--ink-count N` option bypasses the staged achromatic/colour palette and seeds
exactly `N` primary inks together on the full optical-density sphere. The modes
come from the registered median so moving cancellations do not become palette
centres, while per-impression rays preserve genuine batch, fading, and scanner
colour differences. `N=1` deliberately leaves the existing main/background
separation and reconstruction unchanged; the background is not counted in `N`.
Plate identity is now spatially anchored to the nearest high-confidence median
core. Per-impression rays control density and rendered colour inside that fixed
form, but cannot move a pixel from one plate to another merely because its hue
changed. This is important for inks damaged by water or represented by several
colour batches.

Current full-resolution checkpoints:

- IX-X 2K with `N=1` follows the existing background route and gains 6.4% edge
  energy over the averaged golden.
- IX-X 35K with `N=2` finds violet and green plates, with half-set agreements
  +0.902 and +0.936 and a 10.7% edge-energy gain.
- V 5K with `N=2` finds neutral and blue-violet plates without a separate
  achromatic reconstruction. Spatial ownership raises the weak colour plate's
  half-set agreement from +0.153 to +0.596 while the neutral plate remains at
  +0.966; the final reconstruction gains 26.5% edge energy.
- Full comparison panels were generated for all 12 configured scans in V, VIII,
  and IX-X. V and VIII use `N=2`; IX-X 1/2/3/5/7K use `N=1` plus the unchanged
  background route, and IX-X 14/35/70K use `N=2`. The complete run took 1039 s
  on the current machine. VIII 7K was the slowest case at 241 s because it has
  32 impressions; V 5K took 174 s for 16 impressions.
- IX-X 5K correctly rejects the unresolved background candidate and keeps a
  single ink. IX-X 3K is the only current comparison whose rebuilt edge-energy
  score is below the averaged golden (-2.3%); its strict agreed-background mask
  is nearly empty and still needs visual/metric review.

Next checks:

- Replace the current fixed neutral-nuisance angle with a validation based on
  spatial repeatability and held-out reconstruction quality.
- Review whether IX-X 3K should fall back from its weak agreed-background result,
  using held-out structure rather than edge energy alone.
- Reduce the quadratic pairwise registration cost for large collections such as
  VIII 7K without changing the selected consensus.

## Impression agreement, and why the V colour plate is one impression

Edge energy answers only whether the rebuild is crisp, and a single un-averaged
impression satisfies that better than any merge. XXI 3K scores the corpus's
largest gain, +85.6%, from a plate rebuilt out of one of nine impressions with
its postmark attached, while IX-X 3K scores -2.6% on a visibly better result. The
rebuild is now also scored by how well it predicts each impression it was built
from: `impression_agreement` reports the median correlation over the edge band
and the gap between the closest impression and that median. A plate left
authoritative shows up as a wide gap, because it is being reproduced rather than
estimated. On V 3K the rebuild agrees at +0.617 against the averaged golden's
+0.636, so the extra edge energy there is not a better model of the sheet.

`best_preserving_colour_merge` deliberately keeps one impression authoritative
wherever it is readable, which is why all three V colour grounds carry the stains
and uneven inking of a single stamp: V 1K #10, V 3K #1, V 5K #7. Replacing that
with a merge, and letting the best impression fill only what no two impressions
corroborated, was tried and backed out. It does remove the stains and even the
coverage, and edge energy even rose on V 3K from +16.2% to +18.6%, but at 4x
magnification the burelage is gone: the triangular ground on V 5K and the
ornamental ground on V 3K are smeared into a flat wash. One global shift per
impression cannot hold an 8 px ruling in phase right across a stamp, so the
median cancels the narrowband component while the stains, being broadband,
average constructively. The screen-strength figures do not show this because
they are measured on a different, phase-aware merge inside the 256 px window.

To have both, split the two regimes rather than choosing between them: take the
broadband part from the merge of all impressions, and the narrowband part at the
known pitch from a phase-coherent source. Notes for whoever picks this up:

- The phase-aware merge already exists; `add_variant` fills `candidate["golden"]`
  from it before `best_preserving_colour_merge` overwrites it. V 5K measures 45.33
  best single against 69.60 for that merge, so the screen does reinforce when the
  cells are respected.
- The removed `narrow_band` demodulate/low-pass/remodulate helper is the shape of
  the extraction needed. It was dropped from the background route because the low
  band there carries the background's own design envelope; here the low band is
  wanted, so that objection does not apply.
- Keep fixtures for both directions: V 5K must keep a resolvable triangular
  ground, and V 1K must stop reporting stamp #10's stains as the plate.
- The three V panels also mislabel this. The screen route rewrites
  `comparison_notes` to bare `legacy`/`best`/`second`/`anchor`, discarding the
  stamp number the candidate note already carried, and the top row joins notes
  positionally across passes whose variant lists are unrelated. Column 4 of V 5K
  therefore reads "second-best single stamp #9" over an ink 2 image sourced from
  #7, because #9 is ink 1's ranking. Give each comparison column one owning pass.
