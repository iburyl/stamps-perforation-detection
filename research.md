# Related research: separating printing passes from a set of slightly shifted impressions

This note answers one question: **does anybody else solve the problem that
`design_components/` solves, and under what name?**

The answer is that no published work addresses stamps printed in several passes, but the
problem decomposes into five sub-problems that are each a mature research field in some
other domain. Nothing here is a stamp paper. Everything here is something we can borrow a
formulation, a failure mode, or an evaluation protocol from.

## The abstract problem

Stripped of philately, what `golden_components.py` and
`background_pattern_refinement.py` do is this:

Given `N` scans of nominally identical objects, each produced by stacking `K` unknown
template layers with per-instance sub-pixel rigid offsets, observed through a
non-linear ink/paper mixing model, with a slowly varying support signal and gross
sparse outliers:

1. **Decide `K`** — how many printing passes this denomination actually had.
2. **Unmix** each pixel into per-layer contributions, including where layers overlap.
3. **Segment** by *which layer a pixel belongs to*, where the available evidence is
   colour, motion across the set, or both.
4. **Register each layer independently** at sub-pixel accuracy, without a clean
   reference.
5. **Fuse** the `N` observations of each layer into one sharp template, rejecting
   cancellations and damage.

Written that way, there is a lot of prior art. It is just filed under five different
names.

---

## 1. Segmentation by differential motion across a set of superimposed images

This is the closest match to the question as asked: *can you segment an image into layers
because the layers move differently between frames?* Yes, and it is one of the oldest
threads in computer vision.

- **Wang & Adelson, "Representing Moving Images with Layers", IEEE TIP 3(5), 1994**
  ([PDF](https://persci.mit.edu/pub_pdfs/wang_tr279.pdf),
  [DOI](https://doi.org/10.1109/83.334981)). The founding layered representation:
  intensity map + alpha map + velocity map per layer, recovered by computing local optical
  flow, fitting affine motions, and iteratively reassigning pixels to the motion model
  that explains them best. Our `two_pass_decomposition.py` is a rediscovery of exactly
  this algorithm, with "frames" replaced by "stamps from one sheet".
- **Szeliski, Avidan & Anandan, "Layer Extraction from Multiple Images Containing
  Reflections and Transparency", CVPR 2000**
  ([DOI](https://doi.org/10.1109/cvpr.2000.855826)). The *additive transparency* case,
  which is much nearer to printed ink over paper than the occlusion case: the observation
  is a sum of component layers moving relative to each other, and the component images
  themselves — not just their motions — are recovered by constrained least squares, plus
  min/max-composites over aligned frames as bounds. The min-composite as an initialiser
  for a layer that is never fully observed is directly applicable to our pale ground
  under the dark ink.
- **Weiss, "Deriving Intrinsic Images from Image Sequences", ICCV 2001**
  ([PDF](https://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Weiss01.pdf),
  [DOI](https://doi.org/10.1109/iccv.2001.937606)). **The most directly useful paper in
  this list.** Weiss shows that a naive temporal median over raw aligned frames gives the
  wrong answer, and that the correct maximum-likelihood estimator takes the median *of
  derivative-filter responses* and then reconstructs the layer from those, under a
  sparse-filter-output prior. This is precisely the failure we hit when the faint third
  ink "largely cancelled in the median" and had to be rescued with a band-pass response
  before averaging — our band-pass workaround is a special case of the right estimator,
  and the paper gives the principled version. The paper also explicitly works in log
  space, which matches our optical-density formulation.
- **Xue, Rubinstein, Liu & Freeman, "A Computational Approach for Obstruction-Free
  Photography", SIGGRAPH 2015**
  ([PDF](https://people.csail.mit.edu/mrub/papers/ObstructionFreePhotography_SIGGRAPH2015.pdf),
  [project](https://sites.google.com/site/obstructionfreephotography/)). Modern treatment
  of the same two-layer problem from ~5 frames with a few centimetres of baseline. Two
  design decisions transfer: per-pixel flow fields rather than a parametric motion model,
  and **initialising motion from an "edge flow" estimate**, because edges survive the
  blending of two layers while flat interiors do not. Our pipeline already seeds from
  "repeatable design edges" for the same reason; this is the published justification, and
  the alternating motion/layer refinement is the part we have not yet implemented.
- **Jojic & Frey, "Learning Flexible Sprites in Video Layers", CVPR 2001**
  ([IEEE](https://ieeexplore.ieee.org/document/990476)) and **Frey & Jojic,
  "Transformation-Invariant Clustering Using the EM Algorithm", TPAMI 25(1), 2003**
  ([DOI](https://doi.org/10.1109/tpami.2003.1159942)). The generative version: each
  observation is a layered composition of latent appearance maps and masks, with the
  per-frame transformation treated as a latent variable and learned jointly by
  (variational) EM. This is item 1 of our own "next improvements" list — three shared ink
  templates, a smooth paper field, and three transforms per stamp — already written down
  and solved in 2001–2003. Worth reading before writing that optimiser.
- **Gandelsman, Shocher & Irani, "Double-DIP", CVPR 2019**
  ([CVPR open access](https://openaccess.thecvf.com/content_CVPR_2019/html/Gandelsman_Double-DIP_Unsupervised_Image_Decomposition_via_Coupled_Deep-Image-Priors_CVPR_2019_paper.html),
  [arXiv](https://arxiv.org/abs/1812.00467),
  [code](https://github.com/yossigandelsman/DoubleDIP)). The learning-free deep analogue:
  couple two or more deep image priors so that their outputs sum to the input, add an
  exclusion/decorrelation loss between layers, and layers separate with no training data
  beyond the single input. Relevant mostly as a sanity baseline for the hardest case —
  one faint periodic layer under one dense aperiodic layer, where the two have very
  different internal statistics, which is the exact condition Double-DIP exploits.

**Why our motion-only experiment failed, in the literature's terms.** The negative result
recorded in `design_components/README.md` (motion-only masks agreeing with colour masks at
chance level, 0.5–0.6) is the expected outcome, not a bug. Motion segmentation by
clustering trajectories is well studied — see **Elhamifar & Vidal, Sparse Subspace
Clustering** ([CVPR 2009 PDF](https://www.cis.jhu.edu/~ehsan/Downloads/SSC-CVPR09-Ehsan.pdf),
[TPAMI 2013](https://doi.org/10.1109/tpami.2013.57)) and the Hopkins155 benchmark — and it
works because each rigid motion spans a distinct low-dimensional subspace of trajectory
space. With `N≈10` impressions and offsets of a few pixels, our "trajectories" are
2-vectors with sub-pixel separation between clusters, well below the noise injected by
paper deformation, dot gain and scanner texture. The subspaces are not separable at that
SNR. The literature's answer is consistent with what we concluded independently: use a
different cue (colour) to get the mask, and use motion only to *measure* the offset once
the mask exists.

---

## 2. Joint alignment of a batch of instances without a clean reference

Our open problem of "which stamp do we anchor on, and does the anchor bias the result"
is the central question of an entire sub-field.

- **Learned-Miller, "Data Driven Image Models through Continuous Joint Alignment",
  TPAMI 28(2), 2006** ([DOI](https://doi.org/10.1109/tpami.2006.34)) — *congealing*:
  align a whole ensemble by minimising the summed per-pixel entropy of the stack, so no
  member is privileged and the result is an unbiased group centre rather than a
  registration to an arbitrarily chosen member. **Cox et al., "Least Squares Congealing",
  CVPR 2008** ([DOI](https://doi.org/10.1109/cvpr.2008.4587573),
  [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC2919820/)) replaces the entropy criterion
  with SSD-to-the-group-mean solved by Gauss–Newton, estimating all warps simultaneously
  with no step-size tuning — closer to what we would bolt onto the existing ECC code.
- **Peng, Ganesh, Wright, Xu & Ma, "RASL: Robust Alignment by Sparse and Low-Rank
  Decomposition", TPAMI 34(11), 2012**
  ([PDF](https://people.eecs.berkeley.edu/~yima/psfile/RASL_PAMI.pdf),
  [project](https://people.eecs.berkeley.edu/~yima/matrix-rank/rasl.html),
  [Python](https://github.com/welch/rasl)). This is the formulation that matches our data
  best of anything in this document. Seek per-image transforms such that the stack
  decomposes into a **low-rank** aligned component plus a **sparse** error component,
  minimising nuclear norm + ℓ1. The low-rank part is the shared printing plate; the sparse
  part is cancellations, tears, thins and fingerprints. It explicitly targets "gross
  corruption such as occlusion", handles batches of a hundred-plus images, and replaces
  both our anchor choice *and* our robust median with one convex program. The cancellation
  mask we listed as "next improvement 2" is RASL's sparse component, obtained for free.
- **Cryo-EM reference-free 2D class averaging** — see SPIDER's
  [reference-free alignment](https://spider-em.github.io/SPIDER/docs/reffreealign.html),
  RELION's [2D classification](https://relion.readthedocs.io/en/latest/SPA_tutorial/Class2D.html),
  and **"Pre-pro is a Fast Pre-processor for Single-Particle Cryo-EM by Enhancing 2D
  Classification", *Communications Biology* 3, 2020**
  ([paper](https://www.nature.com/articles/s42003-020-01229-0)). Structural biology solves
  our exact fusion problem at vastly worse SNR: thousands of noisy, randomly shifted and
  rotated views of one object, averaged into class averages with no template. Three things
  are worth importing. (i) The field's standard argument against a chosen reference —
  "the resulting parameters are biased by the choice of the reference image", and features
  of the reference survive into the average — is the formal statement of the risk in our
  manual `--pattern-anchor 5`. (ii) Their bootstrap builds an initial average by
  incrementally registering randomly ordered members and then refines leave-one-out, which
  is a cheap, directly implementable replacement for picking an anchor by hand. (iii)
  Alignment parameters estimated on *denoised* surrogates and then applied to the
  *original* images is exactly our "estimate registration on the cleaned periodic field,
  apply the transform to the complete colour layer" trick, independently arrived at.
  Their cautionary literature on spurious structure emerging from aligning pure noise is
  worth reading before we trust any faint recovered layer.
- Modern deep equivalents, for context rather than immediate use: **GANgealing**
  ([CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Peebles_GAN-Supervised_Dense_Visual_Alignment_CVPR_2022_paper.pdf),
  [arXiv](https://arxiv.org/abs/2112.05143)) and **Neural Congealing**
  ([CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Ofri-Amar_Neural_Congealing_Aligning_Images_to_a_Joint_Semantic_Atlas_CVPR_2023_paper.pdf),
  [arXiv](https://arxiv.org/abs/2302.03956)), which learn a joint atlas plus per-image
  dense mappings from as few as 25 images. Both operate at semantic scale, far coarser
  than the sub-pixel regime we need, so they are the wrong tool for the measurement but
  the right precedent for "atlas + per-instance deformation" as a model.

The medical-imaging equivalent of the same idea is unbiased atlas construction / population
registration; see the congealing extension to 3D volumes
([PDF](https://people.cs.umass.edu/~elm/papers/congeal_3D.pdf)), which states the
"eliminate the need for selecting a reference frame a priori" argument in the cleanest form.

---

## 3. Recovering detail from sub-pixel shifts rather than fighting them

Our residual registration errors are a nuisance to be measured. One field treats the same
signal as the *asset*.

- **Wronski et al., "Handheld Multi-Frame Super-Resolution", SIGGRAPH 2019**
  ([arXiv](https://arxiv.org/abs/1905.03277),
  [ACM](https://dl.acm.org/doi/10.1145/3306346.3323024)). Involuntary hand tremor provides
  random sub-pixel offsets across a burst; aligning and adaptively merging those shifted
  samples yields resolution *above* the sensor's, and the per-frame-varying aliasing is
  turned from an artefact into information. The relevant lesson for us: ten impressions
  with independent sub-pixel offsets are ten independent samplings of the same plate, so
  the plate can in principle be reconstructed at higher resolution than any single 1200 dpi
  scan — not merely denoised. Their robustness machinery (per-pixel merge weights that
  reject frames disagreeing locally with the reference) is a drop-in alternative to a plain
  robust median and degrades more gracefully than one.
- Astronomy's **drizzle** (Fruchter & Hook, "Drizzle: A Method for the Linear
  Reconstruction of Undersampled Images", PASP 114, 2002) solves the same
  dithered-undersampled-exposures problem with variable-pixel linear reconstruction, and is
  the canonical reference if we ever want a principled resampling kernel instead of
  interpolate-then-average.

Sub-pixel estimation itself is settled: **Evangelidis & Psarakis, ECC, TPAMI 30(10), 2008**
([PDF](http://xanthippi.ceid.upatras.gr/people/evangelidis/george_files/PAMI_2008.pdf),
[DOI](https://doi.org/10.1109/tpami.2008.113)) — which is what `cv2.findTransformECC` and
therefore our pipeline already uses, chosen for its invariance to photometric distortion,
which is the right property for faded and differently-toned paper. The complementary tool
for translation-only cases is upsampled-DFT phase cross-correlation
(Guizar-Sicairos, Thurman & Fienup, *Opt. Lett.* 33(2), 2008; implemented as
[`skimage.registration.phase_cross_correlation`](https://github.com/scikit-image/scikit-image/blob/v0.25.0/skimage/registration/_phase_cross_correlation.py)),
which is cheaper than ECC and gives a cleaner confidence peak on the periodic ground —
with the caveat that on a periodic signal the peak repeats every raster period, which is the
documented reason our search had to be clamped to ±4 px.

---

## 4. Unmixing inks in optical-density space

Our three-ink-plus-paper decomposition in optical density, with k-means on normalised
density directions and non-negative per-pixel contributions, is a known method in three
other fields. We are using the standard tool; what is non-standard is only the
application.

- **Histopathology stain separation.** **Ruifrok & Johnston, "Quantification of
  Histochemical Staining by Color Deconvolution", *Anal. Quant. Cytol. Histol.* 23(4),
  2001** ([PDF](https://helios2.mi.parisdescartes.fr/~lomn/Cours/CV/BME/HistoPatho/Quantification_of_histochemical_staining.pdf))
  is our algorithm, for up to three stains: convert to OD = −log₁₀(I/I₀), characterise each
  stain by a unit OD vector, and solve for per-stain concentrations — specifically *because*
  hard "one pixel, one colour" classification destroys information wherever stains
  co-localise, which is the same argument our README makes about ink overlap.
  **Macenko et al., ISBI 2009** ([PDF](https://www.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf))
  then estimates the stain vectors *from each image* via the OD-space covariance
  eigenvectors and robust angular percentiles, rather than assuming them — the same job our
  `spherical_kmeans` over OD directions does, with a more stable estimator and an explicit
  treatment of the per-slide variation that corresponds to our per-stamp paper tone. The
  field's standing caution is also ours: these are device-dependent estimates, not
  colorimetry, which is why the `#4E3B3D`-style values in our README are correctly labelled
  as scanner RGB.
- **Hyperspectral unmixing.** **Bioucas-Dias & Plaza, "Hyperspectral Unmixing:
  Geometrical, Statistical, and Sparse Regression-Based Approaches", SPIE 2010**
  ([PDF](https://www2.umbc.edu/rssipl/people/aplaza/Papers/Conferences/2010.SPIE.Unmixingreview.pdf))
  is the reference survey for the linear mixing model with non-negative abundances. Two
  points matter here. First, it explains why ICA is the wrong family for this problem —
  abundances are non-negative and sum-constrained, so they are not statistically
  independent — which rules out a tempting shortcut. Second, **collaborative/joint sparse
  regression** ([Iordache et al., TGRS 52(1), 2014](https://www2.umbc.edu/rssipl/people/aplaza/Papers/Journals/2014.TGRS.Collaborative.pdf))
  imposes the constraint that the *same* small set of endmembers is active across all
  pixels, which is precisely the physical fact that one sheet was printed with one set of
  inks and would stabilise the "is this near-neutral candidate a real plate?" decisions the
  TODO describes as fragile.
- **Soft colour segmentation.** **Aksoy et al., ACM TOG 36(2), 2017**
  ([PDF](https://yaksoy.github.io/papers/TOG17-scs.pdf)) and **Akimoto et al., "Fast Soft
  Color Segmentation", CVPR 2020**
  ([PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Akimoto_Fast_Soft_Color_Segmentation_CVPR_2020_paper.pdf))
  decompose an image into RGBA layers of homogeneous colour, with alpha carrying the
  overlap, under an explicit constraint that recompositing reproduces the input. That
  reconstruction constraint is a test we do not currently apply to our three ink layers and
  could adopt cheaply as a per-stamp residual check.
- **Historical-document source separation.** Bleed-through and palimpsest work solves
  "two overlapping writings, one surface" with the same linear-mixture machinery:
  **Tonazzini et al., DAS 2004**
  ([PDF](http://muscle.ercim.eu/images/DocumentPDF/MP_361_Tonazzini_2004_LNCS3163.pdf)) and
  **Tonazzini, Salerno & Bedini, IJDAR 10(1), 2007**
  ([DOI](https://doi.org/10.1007/s10032-006-0015-z)) treat the RGB channels as independent
  "views" of a mixture of foreground text, bleed-through and support, and separate them by
  decorrelation or ICA — including the observation that the naive linear model breaks where
  ink saturates, later addressed by an explicitly
  [non-linear, non-stationary model](https://www.eurasip.org/Proceedings/Eusipco/Eusipco2012/Conference/papers/1569583261.pdf).
  Our own three-ink model has the same saturation problem under dense dark ink. The
  Archimedes Palimpsest programme
  ([EUSIPCO 2011](https://eurasip.org/Proceedings/Eusipco/Eusipco2011/papers/1569422597.pdf))
  is the same task with multispectral capture, and is the argument for the calibration
  target in our own improvement list: more bands make the unmixing well-posed instead of
  marginal.
- **Art investigation.** Separating the mixed X-ray of a double-sided panel is our problem
  with `K=2` and no motion cue at all, solved by bringing in a second modality as side
  information: **Deligiannis et al., IEEE TIP 2017**
  ([DOI](https://doi.org/10.1109/tip.2016.2623484),
  [arXiv](https://arxiv.org/html/1605.06474)) with coupled dictionary learning, and
  **Sabetsarvestani et al., *Science Advances*, 2019**
  ([paper](https://www.science.org/doi/10.1126/sciadv.aaw7416)) with a self-supervised CNN
  trained only against the constraint that the two outputs sum to the observed mixture.
  The self-supervised "sum-to-observation" loss is a general recipe for layer separation
  with no ground truth, which is our situation exactly.

---

## 5. Measuring print registration error as a quantity

Our per-layer shifts are not just an intermediate: they are a measurement about the press.
Two industries measure them routinely.

- **Printing.** **Yang et al., "Measuring CMYK Color Plane Misregistration from Scanned
  Printed Customer Content Image", IS&T Electronic Imaging 2022**
  ([DOI](https://doi.org/10.2352/ei.2022.34.15.color-286),
  [ResearchGate](https://www.researchgate.net/publication/364176532_Measuring_CMYK_color_plane_misregistration_from_scanned_printed_customer_content_image))
  determines direction, magnitude and *identity* of the misregistered plane from an
  arbitrary scanned print with no prior knowledge of content, via per-channel FFT
  cross-correlation against the digital original, at sub-pixel accuracy. The structure of
  their pipeline is ours, minus the luxury of a digital original — which is the entire
  reason we have to build a golden template first. Earlier, **"Estimating Printer
  Misregistration from Color Shifts: a New Paradigm", SPIE/IS&T Electronic Imaging 2008**
  ([DOI](https://doi.org/10.1117/12.766539)) inverts the relationship instead, recovering
  sub-pixel misregistration from *macroscopic colour shifts* in a designed halftone patch,
  using a reflectance printer model — a useful reminder that colour shift at overlaps
  carries registration information independently of any geometric fit.
- **Semiconductor lithography.** Layer-to-layer misregistration is "overlay", measured to
  nanometre precision on every wafer; see KLA's
  [imaging-based overlay metrology](https://www.kla.com/products/chip-manufacturing/metrology)
  and e.g. [US10817999B2](https://patents.google.com/patent/US10817999B2/en). Two concepts
  are worth stealing even though the physics differs. The field distinguishes **relative
  overlay** (layer B against layer A) from **absolute alignment** (each layer against the
  ideal grid), and it reports overlay as a **spatial fingerprint across the field** fitted
  by a low-order correction model, separating systematic from residual error. Our current
  output is a scalar X/Y shift per layer per stamp; reporting it as a per-sheet fingerprint
  with a fitted systematic component and a residual would say considerably more about how
  the sheet was printed.

---

## 6. Periodic ground: lattice and wallpaper-pattern analysis

The faint ruled/periodic security ground is a *near-regular texture*, and the fact that our
correlation search "sometimes jumped to the neighbouring repeat and produced false 6–13 px
shifts" is the defining difficulty of that literature.

- **Hays, Leordeanu, Efros & Liu, "Discovering Texture Regularity as a Higher-Order
  Correspondence Problem", ECCV 2006**
  ([PDF](https://faculty.cc.gatech.edu/~hays/papers/texture_matching.pdf)) formulates
  lattice discovery as correspondence under joint visual-similarity and
  geometric-consistency constraints, followed by thin-plate-spline rectification.
- **Park, Brocklehurst, Collins & Liu, "Deformed Lattice Detection in Real-World Images
  Using Mean-Shift Belief Propagation", TPAMI 31(10), 2009**
  ([DOI](https://doi.org/10.1109/tpami.2009.73),
  [project](http://vision.cse.psu.edu/research/deformedLattice/Deformed_Lattice_Detection.html))
  encodes 2D crystallographic-group constraints in a degree-4 MRF and tracks the lattice,
  roughly doubling detection rates over the above with a ~10× speed-up.

Both matter for the same reason: they recover a lattice *with its deformation field*, which
is both a principled replacement for our hand-clamped ±4 px search window — phase is
estimated modulo a known basis instead of by unconstrained correlation — and the natural way
to attack the open problem of filling the ground where opaque dark ink hides it. Our current
approach (register the periodic field against an analytic wave at its own measured pitch,
correcting only the component across the ruling because the along-ruling component is a
symmetry of the screen) is the one-dimensional special case of exactly this reasoning, and
is correct as far as it goes.

---

## 7. Comparing many impressions of the same original: the cultural-heritage thread

This is the community whose *goals* most resemble ours, if not its methods. The shared
premise is that multiple impressions from one plate/die/block differ in informative ways.

- **Optical and digital collation.** Bibliographers have superimposed copies of the same
  printed page since the Hinman collator to find press variants. The digital successors are
  the [Traherne Digital Collator](https://oxfordtraherne.web.ox.ac.uk/traherne-digital-collator)
  and VGG's [Image Compare](https://robots.ox.ac.uk/~vgg/software/image-compare/), which
  register two page images for scale, perspective and page curvature and present
  transparent-overlay / two-colour-channel / toggle visualisations of the differences;
  earlier automated work is **van Beusekom et al., "Image-Matching for Revision Detection
  in Printed Historical Documents", DAGM 2007**
  ([PDF](http://vanbeusekom.lu/data/documents/publications/beusekom--image-matching-for-revision-detection-in-printed-historical-documents--DAGM--2007.pdf)).
  Our `background_pattern_before_after.png` is a collation visualisation; this is the field
  that has thought about which overlay presentation actually communicates a difference.
- **Type-imprint matching.** **Vogler et al., "Contrastive Attention Networks for
  Attribution of Early Modern Print", AAAI 2023**
  ([PDF](https://kartikgo.github.io/AAAI_2023_Matching_Damaged_Type.pdf),
  [DOI](https://doi.org/10.1609/aaai.v37i4.25659)) identifies anonymous printers by matching
  uniquely damaged metal type across books. Structurally this is our problem: extract many
  instances of one nominal glyph, normalise rotation/offset/size/scale with a learned
  generative alignment model, then compare at a level of detail finer than the alignment
  noise, while remaining robust to inking variation and digitisation artefacts. Their
  synthetic-damage data-augmentation scheme — simulating bends, fractures and inking
  variation of the printing process to compensate for scarce labels — is a usable template
  for the synthetic-known-shift validation in our own improvement list.
- **Numismatic die studies.** **"Automatic Die Studies for Ancient Numismatics"**
  ([arXiv 2407.20876](https://arxiv.org/html/2407.20876),
  [code](https://github.com/ClementCornet/Auto-Die-Studies)) clusters thousands of coin
  images by which die struck them: robust local-descriptor matching, MAGSAC++ filtering, a
  pairwise similarity graph, and label-propagation clustering whose critical threshold is
  chosen by an *intrinsic* criterion (silhouette coefficient) requiring no ground-truth
  labels. The intrinsic-criterion trick is worth noting generally: we have the same problem
  of choosing thresholds (ink-direction margins, confidence cutoffs) with no labelled data.
  [Riedones3D](https://github.com/humanpose1/riedones3d) does the 3D-scan version.
- **Woodblock prints.** Japanese *ukiyo-e* is multi-pass colour printing with physical
  registration marks (*kentō*) and per-impression misregistration, i.e. our problem with
  bigger tolerances. **"Color Decomposition of Overlapped Watercolors", ICPR 2004**
  ([DOI](https://doi.org/10.1109/icpr.2004.1334409)) inverts a particle-density pigment
  model to **estimate the woodblocks from a print** — explicitly the same goal as
  reconstructing a printing plate from impressions, and notable for modelling the fact that
  overlapping watercolour depends on *printing order*, a non-commutativity our linear
  density model ignores. A modern take uses soft colour separation to recover per-colour
  block layers ([ICCC 2021](https://computationalcreativity.net/iccc21/wp-content/uploads/2021/09/ICCC_2021_paper_9.pdf)),
  and [ukiyo-e.org](https://ukiyo-e.org/about) automatically groups and aligns multiple
  copies of the same print for comparison across 200,000+ images.
- **Philately itself.** The published computational work is classification, not
  reconstruction: **"Deep Learning for Philately Understanding", IEEE WNYISPW 2017**
  ([DOI](https://doi.org/10.1109/wnyipw.2017.8356251),
  [code](https://github.com/rohand24/Stamp_Identification)) recognises Penny Red plate
  number and corner letters from ~130,000 photographs at >95% row/column and 90% plate
  accuracy despite heavy cancellations, and hobbyist CNN plating tools now exist for the
  1851 3¢ Washington ([PlateAI](https://stampplating.com/plateai)). These are 2,600-way
  classifiers over whole stamps. **No published philatelic work separates printing passes
  or reconstructs a plate from registered impressions.**
- **Security printing.** The "Sound-of-Intaglio" line of work authenticates banknotes from
  the intrinsic micro-structure of intaglio printing using shift-invariant wavelet packet
  decompositions and higher-order frequency statistics, robust to wear
  ([overview](https://platform.keesingtechnologies.com/renaissance-of-intaglio/)), with a
  public 1200 dpi intaglio-vs-offset dataset
  ([Zenodo](https://doi.org/10.5281/zenodo.11119013)). Same substrate, same resolution,
  same "is this impression from the genuine plate" question, different mathematics —
  texture statistics rather than geometric reconstruction. If plate reconstruction is ever
  aimed at authentication, this is the baseline to beat.

---

## What is actually novel in our setting

Combining the above, the parts of our problem that are *not* covered by existing work:

1. **The frames are different physical objects, not frames of a video.** Layer-separation
   work assumes one scene observed repeatedly; we have `N` impressions each with its own
   paper deformation, ageing, cancellation and wear. The "shift" is press registration
   error, not camera motion, and it is a property of the artefact worth measuring in itself.
2. **The number of layers is unknown and varies per denomination.** Essentially all the
   layer and unmixing literature takes `K` as given. Our pipeline reads `K` off the scan
   and has to defend against false splits — the hardest open issue in `TODO.md`, and the
   one with the least prior art to borrow from.
3. **The shift magnitude is at or below the noise floor of the cue that would reveal it.**
   Burst super-resolution gets shifts for free from a clean sensor; motion segmentation gets
   subspaces from large displacements. We need to detect layer membership from offsets of
   0.1–8 px in the presence of paper distortion of comparable size. This is why colour must
   carry the segmentation and motion only the measurement.
4. **One layer is largely unobservable.** The pale ground under the dark ink is never seen;
   no amount of registration recovers it. Only a periodicity model, a plate model, or an
   explicit "observed / model-completed" coverage map can fill it — and reporting that
   distinction honestly is a requirement the heritage-imaging literature takes seriously and
   the computer-vision literature mostly does not.

## Shortlist: what to read, and why

| Our open problem | Closest prior work | What it gives us |
| --- | --- | --- |
| Faint layer vanishes in the median | Weiss, ICCV 2001 | The correct ML estimator — median of *filtered* responses, not of frames |
| Manual anchor stamp, possible anchor bias | Cryo-EM reference-free alignment; congealing | Bootstrap + leave-one-out refinement; unbiased group centre; the formal bias argument |
| Cancellation mask "instead of hoping for the median" | RASL, TPAMI 2012 | Low-rank plate + sparse corruption, jointly with the warps, as one convex program |
| Joint generative model (3 templates + per-stamp transforms) | Jojic & Frey, CVPR 2001 / TPAMI 2003 | The model already written down and solved by variational EM |
| Periodic ground: false jumps to the next repeat | Park et al., TPAMI 2009; Hays et al., ECCV 2006 | Lattice + deformation field; phase modulo a known basis instead of free correlation |
| Validation on synthetic known shifts and hold-out | Vogler et al., AAAI 2023 | A print-process-aware synthetic degradation scheme for exactly this validation |
| Ink vectors estimated per stamp, stability of the split | Macenko et al., ISBI 2009; Iordache et al., TGRS 2014 | Robust per-image OD-vector estimation; a shared-endmember constraint across the sheet |
| Reporting shifts as a result | CMYK misregistration (EI 2022); overlay metrology | Per-sheet registration fingerprint with systematic + residual, not a scalar per stamp |
| Reconstructing the plate at better than scan resolution | Wronski et al., SIGGRAPH 2019 | Sub-pixel offsets as signal; robust per-pixel merge weights |
| Sanity baseline with no model at all | Double-DIP, CVPR 2019 | Unsupervised two-layer split driven only by differing internal statistics |

## Search terms that find this literature

Useful for follow-up, since none of it indexes under "stamps": *layered motion
segmentation*, *transparency / reflection separation*, *intrinsic images from image
sequences*, *batch image alignment*, *congealing*, *groupwise / population registration*,
*reference-free 2D class averaging*, *multi-frame super-resolution*, *colour deconvolution*,
*stain separation*, *spectral unmixing*, *soft colour segmentation*, *bleed-through removal*,
*blind source separation of document images*, *colour plane misregistration*, *overlay
metrology*, *near-regular texture*, *deformed lattice detection*, *digital collation*,
*type-imprint analysis*, *die study*.
