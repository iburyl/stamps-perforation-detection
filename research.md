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

## 8. Deciding which repeat you are on: the integer-ambiguity literature

Section 6 notes that our correlation search "sometimes jumped to the neighbouring repeat".
`broadband_branch_margin` has since measured how often: on V 5K's anchor variant, 10 of 16
impressions sit on a cell the design disputes, one of them by two periods. That turns a
nuisance into the main open problem, and it has a name outside imaging. A measurement that
is precise but periodic determines a quantity only modulo one period; recovering the
integer is **ambiguity resolution**, and four separate fields have mature machinery for it.
This is the most directly useful addition to this document since Weiss.

- **Temporal phase unwrapping in fringe projection.** **Huntley & Saldner, "Temporal
  phase-unwrapping algorithm for automated interferogram analysis", *Appl. Opt.* 32(17),
  3047, 1993** ([abstract](https://opg.optica.org/ao/abstract.cfm?uri=ao-32-17-3047)) and
  **Saldner & Huntley, *Appl. Opt.* 36(13), 2770, 1997**
  ([abstract](https://opg.optica.org/ao/abstract.cfm?uri=ao-36-13-2770)) resolve the 2π
  ambiguity by measuring phase at several fringe pitches and unwrapping each pixel
  independently over the sequence, coarsest first, so errors cannot propagate spatially.
  The comparative review in *Optics and Lasers in Engineering* 85, 2016
  ([DOI](https://doi.org/10.1016/j.optlaseng.2016.04.022)) sorts the field into three
  families — multi-frequency (hierarchical), multi-wavelength (heterodyne/beat), and
  number-theoretic — and ranks them against a stochastic noise model. **Its conclusion
  decides our open design question: multi-frequency hierarchical unwrapping is the most
  reliable, and the multi-wavelength beat approach is the most susceptible to
  noise-induced unwrapping errors.** `TODO.md` proposes exactly these two options — resolve
  the branch from the broadband design (coarse, unambiguous, then refine with the carrier)
  versus resolve it from the beat between the oval's pitch and the ground's. The review
  says to do the first and treat the second with suspicion, which is also what our own
  signal-to-noise figures suggest. See also the follow-up on choosing the pattern sequence,
  *Opt. Express* 25(17), 20381, 2017
  ([full text](https://opg.optica.org/fulltext.cfm?uri=oe-25-17-20381)).
- **GNSS carrier-phase ambiguity resolution.** The closest formal match to
  `solve_screen_cell_graph`, which is an integer least-squares problem whether or not it
  was written as one. **Teunissen, "The least-squares ambiguity decorrelation adjustment",
  *J. Geodesy* 70(1–2), 65, 1995** ([DOI](https://doi.org/10.1007/bf00863419)) is the LAMBDA
  method: decorrelate the float ambiguities by a Z-transformation, *then* search the integer
  space, because searching a correlated lattice directly is both slow and error-prone. The
  [LAMBDA toolbox](https://www.tudelft.nl/en/ceg/about-faculty/departments/geoscience-remote-sensing/research/lambda/lambda)
  is open source, and **LAMBDA 4.0, *GPS Solutions*, 2024**
  ([DOI](https://doi.org/10.1007/s10291-024-01738-z)) is the current version. Three things
  transfer directly:
  1. **Our `cell_gap` is the ratio test**, best over second-best. "On GNSS Ambiguity
     Acceptance Tests" (Teunissen, 2007,
     [PDF](https://gnss.curtin.edu.au/wp-content/uploads/sites/21/2016/04/Teunissen2007GNSS.pdf);
     journal version *Survey Review*, 2009, [DOI](https://doi.org/10.1179/003962609x390058))
     shows that the ratio test **is not a test of whether the integer solution is correct**,
     and that using it with a fixed critical value — which is what our hard-coded
     `tolerance` in `best_screen_cell` does — has known shortcomings. The replacement is the
     fixed-failure-rate ratio test, which derives the threshold from the covariance and a
     declared tolerable failure rate; ready-made fitting functions are published in
     *Sensors*, 2016 ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4969832/)). This would
     give our cell gaps of 0.03–0.18 an actual meaning instead of a guess.
  2. **Partial ambiguity resolution** — fix only the subset of integers that can be fixed
     at the required confidence and leave the rest floating — is the field's standard answer
     to precisely our situation, where most impressions are confident and a third are not.
     Applied here it means merging the ruling only from impressions whose cell is decided,
     and letting the disputed ones contribute their broadband content alone. We currently
     force a cell on every impression.
  3. **Integer bootstrapping and best-integer-equivariant estimation** are the graceful
     degradations between "fix" and "do not fix", which is the choice our code treats as
     binary.
- **Number-theoretic unwrapping, robustly.** If we do pursue two pitches, plain
  Chinese-remainder reconstruction is a trap: a small remainder error produces a large
  integer error. **Xia & Wang, "Phase Unwrapping and A Robust Chinese Remainder Theorem",
  *IEEE SPL* 14(4), 247, 2007** ([DOI](https://doi.org/10.1109/lsp.2006.884898),
  [PDF](https://www.eecis.udel.edu/~xxia/RobustCRT.pdf)) and **Wang & Xia, *IEEE TSP*,
  2009** ([DOI](https://doi.org/10.1109/tsp.2009.2025079)) give the robust version together
  with the condition that makes it safe: the reconstruction error is bounded by the
  remainder error provided that error stays **below a quarter of the greatest common divisor
  of the moduli**. That is a concrete, checkable feasibility test for the dual-pitch idea
  before any of it is built — measure the phase residual, compare it with a quarter of the
  gcd of the two pitches, and if it fails, the idea is dead and we have saved the work. For
  choosing pitches rather than inheriting them, see **"Optimum frequency selection in
  multifrequency interferometry", *Opt. Lett.* 28(11), 887, 2003**
  ([DOI](https://doi.org/10.1364/ol.28.000887)) and the method of excess fractions
  ([theory](https://doi.org/10.1364/ao.50.005484),
  [analytical solution](https://doi.org/10.1364/ao.52.005758)).
- **Registration of aliased images.** **Vandewalle, Süsstrunk & Vetterli, "Registration of
  aliased images for super-resolution imaging", SPIE/IS&T Electronic Imaging 2006**
  ([DOI](https://doi.org/10.1117/12.644032), book chapter
  [PDF](https://pixeltje.be/wp-content/uploads/VandewalleSV10.pdf)) is the same architecture
  we arrived at, published in the super-resolution setting: when part of the spectrum is
  free of aliasing, **estimate the shift from that part alone — equivalently, low-pass
  before registering — and only then disambiguate the aliased part.** Their analysis also
  describes our comb exactly: the objective splits into orthogonal Fourier subspaces, each
  with "periodically repeating minima", and subspaces that are fully ambiguous can simply be
  skipped because any offset fits them. That is the formal statement of why scoring cells on
  the carrier is scoring nothing.
- **Moiré overlay metrology.** Two overlaid gratings of pitch `p₁` and `p₂` beat at
  `p₁p₂/|p₁−p₂|`, and a small relative displacement shifts the moiré fringes by an amplified
  amount — 5× to 120× in published imaging-overlay targets
  ([SPIE 2021](https://doi.org/10.1117/12.2583733),
  [Opt. Eng. 53(8), 084101, 2014](https://doi.org/10.1117/1.oe.53.8.084101),
  and for field stitching **Murphy et al., JVST B 18, 3287, 2000**
  [PDF](https://www.photonics.umd.edu/wp-content/uploads/pubs/ja-5/Murphy_JVSTB_18_3287_2000.pdf)).
  Measurement *error* is divided by the same factor, which is why the technique is used at
  all. The catch, stated plainly in the stitching paper, is that it needs a reference
  pattern showing where the fringes would lie at zero error. We have no such reference — we
  are trying to build it — so this is a measurement we could only make between two
  impressions, never against truth. Worth knowing as the principled version of the
  dual-pitch idea, and as the reason it is a comparison and not a calibration.

---

## 9. Averaging a periodic pattern over many deformed copies: crystallographic image processing

Section 6's lattice-detection papers find a deformed lattice in *one* image. There is an
older field that then **averages many such images into one high-resolution unit cell**,
which is our periodic-ground problem end to end, and it has had production software for
thirty years: the MRC programs, **2dx** (**"2dx_merge: data management and merging for 2D
crystal images", *J. Struct. Biol.*, 2007**,
[PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC2157552/)), its successor
[FOCUS](https://lbem-focus.epfl.ch/wiki/doku.php?id=1_0%3A2dx_image_processing), the
[Methods chapter](https://doi.org/10.1007/978-1-62703-176-9_10), and **CRISP**
(Hovmöller, *Ultramicroscopy* 41, 1992, [PDF](http://www.fos.su.se/~svenh/CRISP.pdf));
overview at [Crystallographic image processing](https://en.wikipedia.org/wiki/Crystallographic_image_processing).
Three pieces of it are of dedicated value.

- **Crystal unbending is the answer to our global-shift problem.** `TODO.md` records that
  "one global shift per impression cannot hold an 8 px ruling in phase right across a
  stamp", and `narrow_band`'s low-passed envelope is our workaround — it lets the amplitude
  absorb a phase error rather than correcting it. Unbending corrects it: cross-correlate the
  image against a *tightly masked* version of its own transform, which acts as a locally
  perfect reference, obtain a **distortion vector plus a correlation goodness for every
  single unit cell**, then re-interpolate the image onto an ideal lattice. The literature
  estimates unbending roughly doubles the resolution at which shape can be recovered. This
  is strictly stronger than what we do, and it is also how to get the per-cell confidence
  map that `broadband_branch_margin` currently approximates with one number per impression.
- **Phase-origin refinement is our anchor-versus-consensus question, already solved.**
  Merging many images of one 2D crystal requires putting them on a common phase origin, and
  2dx's `Merge & Refine` does this **iteratively over the whole dataset**, then regenerates
  image maps to verify the origins it chose. CRISP is specifically noted for superior
  phase-origin refinement, obtained by testing symmetry constraints and moving the origin to
  the best centre. Our `recentre_screen_cells` takes the gauge from the median of the
  branches it was handed, which `TODO.md` now records failing on V 5K consensus, where five
  impressions all want to move one period together. A symmetry- or consensus-driven origin
  refinement is the published alternative.
- **Per-reflection quality, and iterating against the merged result.** These pipelines
  record an `IQ` value per diffraction spot — a per-frequency reliability, which is what
  `repeatability_gain` computes — and `MAKETRAN` re-unbends each image against a *synthetic
  reference generated from the merged dataset*, which is our "register against the running
  template" done in Fourier space with the lattice imposed. Both are cheap to adopt.

The caveat is the usual one: these programs assume a genuine crystal with a plane-group
symmetry to exploit, and a printed ruling has a one-dimensional repeat and no symmetry
constraint beyond translation. The unbending and origin-refinement machinery does not depend
on that symmetry; only the plane-group steps do.

---

## 10. How much of a merged result is just the reference?

Section 2 flags the cryo-EM community's argument against a chosen reference. That argument
has since been quantified, and the quantitative form speaks directly to a number we measured.

**Henderson, "Avoiding the pitfalls of single particle cryo-electron microscopy: Einstein
from noise", *PNAS* 110(45), 2013** ([paper](https://doi.org/10.1073/pnas.1314449110)) is the
canonical statement: align pure noise to a template by cross-correlation, average, and the
template reappears. **"Quantification of model bias underlying the phenomenon of Einstein
from Noise", *Statistica Sinica*** ([DOI](https://doi.org/10.5705/ss.202020.0334)) derives
the size of the bias in terms of the sample count, dimension and selection size.
**"Einstein from Noise: Statistical Analysis", arXiv 2407.05277, 2024**
([arXiv](https://arxiv.org/abs/2407.05277)) proves the result that matters here: **the
Fourier *phases* of the aligned-and-averaged estimator converge to the Fourier phases of the
template**, which is why the structure reappears at all.

Read that against our own pipeline. The anchor variant aligns every impression to one
strongest impression by maximising correlation of the ruling, and `screen_strength` then
scores the merge by its Fourier amplitude **at that same wave**. The anchor's carrier phase
is therefore partly imposed on the result, and the statistic used to judge the result is the
one most sensitive to it. A high anchor score is to some degree guaranteed by construction.
That is not a hypothetical: on V 1K the anchor variant scores 83.76 against consensus's
72.07 while `broadband_branch_margin` disputes twice as many of anchor's cells. The
literature predicted the discrepancy, and it argues for judging any variant on evidence the
alignment did not optimise — a held-out band, or the broadband remainder — rather than on
the quantity the alignment maximised.

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
| Impressions placed on the wrong whole-period cell | Temporal phase unwrapping review, OLEN 2016 | Which of our two candidate fixes to build: hierarchical beats beat-frequency |
| `cell_gap` used as a fixed-threshold acceptance test | Teunissen 2007/2009; FFRT, *Sensors* 2016 | Our ratio test does not test what we think; a failure-rate-calibrated threshold instead |
| A third of impressions on disputed cells | Partial ambiguity resolution (LAMBDA) | Fix only the confident cells; let the rest contribute broadband only |
| Dual-pitch (oval plus ground) branch resolution | Robust CRT, IEEE SPL 2007 / TSP 2009 | A checkable precondition — phase error below ¼ gcd of the pitches — before building it |
| Registering a signal whose spectrum repeats | Vandewalle et al., SPIE EI 2006 | Estimate the shift on the unambiguous band, then disambiguate the rest |
| One global shift cannot hold the ruling in phase | Crystal unbending (MRC / 2dx / FOCUS) | Per-unit-cell distortion vectors and goodness, then re-interpolation onto an ideal lattice |
| Which gauge, and anchor versus consensus | Phase-origin refinement (2dx `Merge & Refine`, CRISP) | Iterative dataset-wide origin refinement with verification, not a median of branches |
| Is a high `screen_strength` real? | Einstein-from-noise, PNAS 2013; arXiv 2407.05277 | Aligned averages inherit the reference's Fourier phases; judge on evidence alignment did not optimise |
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

For the ambiguity problem specifically, which indexes under none of the above: *temporal
phase unwrapping*, *multi-frequency / hierarchical unwrapping*, *fringe order
determination*, *integer ambiguity resolution*, *integer least squares*, *LAMBDA*, *ratio
test*, *partial ambiguity resolution*, *robust Chinese remainder theorem*, *method of excess
fractions*, *synthetic wavelength*, *registration of aliased images*, *moiré metrology*,
*crystal unbending*, *phase origin refinement*, *crystallographic image processing*,
*reference bias*, *Einstein from noise*.
