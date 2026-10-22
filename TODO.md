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

The earlier `best_preserving_colour_merge` response made one impression
authoritative wherever it was readable. It made the V colour grounds look
plausible, but only by hiding the multi-impression reconstruction failure behind
V 1K #10, V 3K #1, or V 5K #7. That is not an acceptable production fallback.
The main path must remain a genuine collection reconstruction whenever two or
more usable observations exist; the best singleton is now a separate, stable
`from best (#...)` diagnostic column. The known failure is therefore visible
again: at 4x magnification the triangular ground on V 5K and the ornamental
ground on V 3K can smear or acquire phase shadows. One global shift per
impression cannot hold an 8 px ruling in phase right across a stamp, so a merge
can cancel or double the narrowband component while broadband stains average
constructively. The screen-strength figures do not show the whole failure
because they are measured inside one phase-aware 256 px window.

Everything described from here to the end of this section is implemented on the
branch `screen-band-separation`, not on `main`. `main` carries these notes and
`research.md` only. The branch holds `narrow_band`, `band_separated_golden`,
`masked_agreement` and `broadband_branch_margin` in `golden_components.py`, the
extra panel column, the new reporting, and `ScreenBandSeparationTests`,
`BestPreservingColourMergeTests` and `BroadbandBranchMarginTests`. It is kept
separate because the branch-margin measurement below shows the registration it
builds on to be wrong often enough that none of it should be published yet.

Both regimes are now split rather than traded against each other.
`best_preserving_colour_merge` takes the broadband plate from the merge of all
preservation-selected impressions, and swaps the ruling back in at the known
pitch from the cell-aware merge already standing in `candidate["golden"]` — the
only source whose phase is coherent, and the one V 5K measures at 69.60 against
45.33 for its best single impression. The best impression now fills only what no
two impressions corroborated. The older behaviour is still what runs when no
screen source is passed, so the ordinary colour merge is unchanged.

`narrow_band` is back for the extraction, demodulating against the measured
wave, low-passing, and remodulating. The low pass is what makes this work here:
the envelope is free to follow a phase that drifts across the stamp, which is
exactly what one global shift per impression leaves behind. It was dropped from
the background route because the low band there carries the background's own
design envelope; here the low band is wanted, so that objection does not apply.
`SCREEN_ENVELOPE_PERIODS` sets how far the envelope may vary.

`ScreenBandSeparationTests` and `BestPreservingColourMergeTests` pin both
directions on a synthetic sheet whose ruling, design and stains are known: the
ground must survive a merge that cancels it without band separation, and the
best impression's stain must not reach the published plate. The variant is
published as a seventh `band-separated` panel column beside `anchor`, and the
screen strength of all three routes is printed per pass, so the trade is visible
without replacing anything.

The three V sheets confirm the stain half and contradict the ground half. Ink 2
screen strength rises on every pass — 1K anchor 50.81 to 59.62, 1K consensus
37.42 to 44.78, 3K anchor 39.08 to 40.70, 3K consensus 35.26 to 39.80, 5K
consensus 48.31 to 50.10 — and the horizontal ruling inside the oval comes back
at full contrast on all three. Stamp #10's stains leave the 1K plate, and the
postmark blobs leave the 3K and 5K plates. But the frame's ornamental ground
goes soft: the 3K cartouches stop reading as a "3" and the 5K triangles lose
most of their contrast. Only the one measured wave vector is swapped back, so a
2D ornament keeps all its harmonics in the broadband part, where they stay
smeared. Screen strength cannot see this, because it measures only the
fundamental that was restored. 5K anchor is the one pass that falls, 53.60 to
50.90, which is the same effect reaching the fundamental.

The cells themselves are wrong, and that is now measured rather than suspected.
`best_screen_cell` restricts every candidate to a whole period of the one
measured wave, which is the frame ground's. The oval's horizontal ruling is a
second periodic structure with its own pitch, and a displacement that is a whole
period of the ground is in general not a whole period of the oval, so "right
ground cell, oval a line or two off" is inside the allowed candidate set and
scores almost the same. `broadband_branch_margin` strips the ruling with
`narrow_band` and asks the aperiodic remainder, against a leave-one-out median
of the other impressions, whether it would rather sit a whole period away. A
period is exactly one turn of the carrier, so no measurement read off the ruling
can answer that; the design can.

It disputes a great many impressions. V 5K anchor: 10 of 16, the worst being
stamp #19 at +0.176 over a kept +0.258, and stamp #6 wanting two periods. V 1K
anchor: 4 of 10, stamp #8 at +0.188 over a kept +0.448. V 3K anchor: 3 of 8,
stamp #4 at +0.173 over a kept +0.423. Consensus is better everywhere but not
clean: 7 of 16, 2 of 10, 2 of 8. Two further readings matter. On V 1K the screen
strength prefers anchor, 83.76 against consensus's 72.07, while the design
disputes twice as many of anchor's cells — the metric that picks between the two
variants is blind to the error being compared. And on V 5K consensus five
impressions all want -1, which is not five independent mistakes but one gauge:
`recentre_screen_cells` takes the zero from the median of the branches, and the
design says the whole set should sit a period the other way.

`median_cell_gap` is now printed too, having been computed and discarded since
the resolver was written. It is the margin the ruling gave the winning cell over
the runner-up, and it is small: median 0.184 on V 1K, 0.164 on V 3K, 0.109 on
V 5K, with the least decisive impression at 0.038, 0.102 and 0.031. Those are
coin tosses, and they are being read off evidence that cannot see the question.

The approach that should fix this is integer-ambiguity resolution, as used for
phase unwrapping and for carrier-phase positioning: let the unambiguous but
imprecise measurement choose the integer, and the precise but ambiguous one
refine within it. Both halves are already here and are wired the wrong way
round. `phase_shifts` is the precise-ambiguous half and is correct as written,
giving sub-pixel position modulo the pitch. The aperiodic design, once
`narrow_band` has taken the ruling out of it, is the unambiguous half, good to a
pixel or two with no periodic ambiguity at all, and it currently does no work.
Three changes follow, two of them deletions:

- Compute the correlation surface on `layer - narrow_band(layer, ...)` rather
  than on `screen_cell_feature`, so the comb of equal peaks is gone before
  anything is ranked.
- Delete the whole-period restriction in `best_screen_cell`. It exists only
  because the carrier could not be trusted beyond one period, and on broadband
  evidence it is actively harmful: it pins the answer to a lattice whose spacing
  is the ground's period, which is exactly how the oval ends up a line off at no
  cost. Replace it with a snap to the phase branch nearest the broadband peak.
- Delete the median gauge in `recentre_screen_cells`. The uniform -1 across most
  of V 5K consensus is that hack failing; once branches are decided on absolute
  evidence the gauge is determined rather than guessed.

A second constraint is available if the two rulings have different pitches. The
oval's ruling and the frame ground both constrain the same global translation,
and their ambiguity lattices differ, so jointly they are ambiguous only at the
beat period -- for pitches near 8.3 and 9.0 px that is about 75 px, outside the
search limit, and the ambiguity disappears. It does not matter that the two
rulings occupy different parts of the stamp, since a global translation is
constrained by both wherever each is measured. The attraction is that it uses
only narrowband evidence, which has far better signal here than the design does:
the engraved ink registers at ECC 0.80 while the ink carrying the ruling manages
0.410 on V 1K, 0.259 on V 3K and 0.315 on V 5K.

Do the broadband unwrapping first anyway. The margins above prove that evidence
discriminates on these exact sheets, and `research.md` section 8 records that
the fringe-projection literature, which has compared the two families against a
noise model, finds hierarchical coarse-to-fine unwrapping the most reliable of
the three and the beat approach the most susceptible to noise-induced errors --
which is this choice exactly. Before building the dual-pitch version at all,
check its published precondition: robust Chinese-remainder reconstruction is
only safe while the phase error stays below a quarter of the greatest common
divisor of the two pitches. Measure that first; if it fails, the idea is dead
and the work is saved.

Also still to do:

- Fix only the cells that can be fixed. Partial ambiguity resolution is the
  standard answer to a third of the set being undecidable: merge the ruling from
  the impressions whose cell is settled and let the disputed ones contribute
  their broadband content alone. We currently force a cell on all of them.
- Give `cell_gap` a calibrated threshold. It is the ratio test, and the GNSS
  literature shows the ratio test with a fixed critical value -- our hard-coded
  `tolerance` -- does not test what it appears to. The fixed-failure-rate form
  derives the threshold from the covariance and a declared tolerable failure
  rate, which would turn gaps of 0.03 to 0.18 into a decision instead of a guess.
- Replace the envelope workaround with unbending. `narrow_band`'s low pass lets
  the amplitude absorb a phase error that one global shift leaves behind;
  electron crystallography corrects it instead, cross-correlating against a
  tightly masked transform of the image itself to get a distortion vector and a
  goodness for every unit cell, then re-interpolating onto an ideal lattice.
  That also yields the per-cell confidence map `broadband_branch_margin`
  currently reduces to one number per impression.
- Stop judging a variant by the quantity its own alignment maximised. The anchor
  variant aligns to one impression by correlating the ruling, and
  `screen_strength` scores the result by its Fourier amplitude at that same
  wave; aligned averages are known to inherit the reference's Fourier phases, so
  part of the anchor's score is guaranteed by construction. V 1K shows it
  directly -- anchor 83.76 against consensus 72.07 while twice as many of
  anchor's cells are disputed. Score on a held-out band or on the broadband
  remainder instead.

- Stop remodulating the carrier where no ruling was printed. The band-separated
  column paints a faint ruling over flat paper, because `band_separated_golden`
  swaps the band across the whole field.
- Swap back every frequency at which the impressions agree, not just the
  measured wave. `repeatability_gain` already computes that per-frequency
  coherence and `apply_gain` already applies it, so the ornament's harmonics
  can travel with the fundamental instead of staying in the smeared part.
- The three V panels also mislabel this. The screen route rewrites
  `comparison_notes` to bare `legacy`/`best`/`second`/`anchor`, discarding the
  stamp number the candidate note already carried, and the top row joins notes
  positionally across passes whose variant lists are unrelated. Column 4 of V 5K
  therefore reads "second-best single stamp #9" over an ink 2 image sourced from
  #7, because #9 is ink 1's ranking. Give each comparison column one owning pass.

Interim implementation for review:

- Every panel is limited to `as averaged | main path | from best (#...) |
  from best 2 (...) | spectral hybrid`.
  Component rows under `as averaged` show the corresponding best source crop as
  observed, while the next two cells show the accepted multi-stamp component and
  its explicit singleton control.
- `choose_impressions()` and colour preservation keep at least two observations
  in the main route when two are available. A sharper singleton is reported but
  cannot silently replace the reconstruction.
- `from best 2` is built independently for every printing pass. Its heading
  lists, per ink, the pair actually used; it no longer changes one ink while
  silently leaving another at its many-stamp result.
- The one retained non-production spectral column splits the colour screen into
  a robust broadband median and a complex-demodulated narrow band at the
  measured wave.
- Only a scalar 75th-percentile density normalization is borrowed from the
  accepted consensus; no pixels or spatial structure come from the singleton.
- The production/main-path column now exposes the genuine multi-impression
  consensus. The spectral hybrids remain non-production review columns.

V 1K cancellation-colour check:

- The unified path must not model cancellation as a neutral-grey nuisance.
  Stamp #10 has a strong reddish-brown cancellation (a representative residual
  colour is about `#8A694E`), while #7 and #8 contain darker brown/black marks.
  Cancellation colour is therefore arbitrary external evidence, not another
  printing ink and not necessarily close to the neutral direction.
- The former 50-degree colour gate was wider than the 44.9-degree separation
  between the two V 1K inks. Spatial ownership could consequently override the
  observed colour and project the cancellation onto the orange ray. For stamp
  #7, pixels actually closer to ink 1 supplied 38.1% of the old ink-2 density.
- The experimental correction accepts a pixel only when its nearest local ink
  direction agrees with the collection's spatial owner and the ink's repeatable
  support. The colour tolerance is derived from inter-ink separation (20.2
  degrees for V 1K, clipped to 12--25 degrees); every other dense pixel is an
  arbitrary-colour residual and casts no vote for any plate.
- On the corrected V 1K run, #7, #8, and #10 have residual fractions 22.7%,
  20.2%, and 18.9%. Their cancellation marks disappear from both reconstructed
  inks, while #10 remains usable for both passes. Ink 2 keeps nine impressions,
  its global cell consensus wins held-out agreement (+0.070 versus +0.037), and
  the final multi-impression rebuild gains 11.8% edge energy over the averaged
  golden. Keep this case as a regression fixture before generalising the rule.
