# Bottom-mua dynamic-tail TPSF model

`train_bottom_mua_dynamic_tail.py` predicts a bottom-layer absorption
coefficient (`bottom_absorption_mul`) from time-of-flight photon
distributions (TPSFs), using a per-wavelength dynamic "late tail" window
defined by a `LATE_START` index.

This is a standalone training script (paths in `TRAIN_DIR`,
`WAVELENGTH_FILE`, and `MODEL_PATH` are local Windows paths meant to be
edited for your machine) and is unrelated to the spleen-ultrasound
pipeline described in the top-level `README.md`; it lives in its own
folder for that reason.

## Changes made to reduce validation error

Starting from the original full-TPSF + raw-tail convolutional model, this
version adds:

1. **Fixed an augmentation wiring bug.** The training `Dataset` was always
   constructed with `augment=False`, so the amplitude-jitter/additive-noise
   /time-shift augmentation implemented in `augment_scaled_tpsf` never ran
   during training regardless of `USE_TRAINING_AUGMENTATION`. It is now
   wired to the flag, and the flag defaults to `True`.
2. **Statistical-moment features.** `compute_moment_features` derives
   total counts, mean time-of-flight, temporal standard deviation,
   skewness, peak amplitude, and peak time from both the full TPSF and the
   raw tail, and these are concatenated into the fusion vector alongside
   the convolutional features. These are the same low-order moments used
   by generalized method-of-moments absorption estimators in diffuse
   optics, so they give the regression head a physically grounded, low-
   variance signal that does not depend solely on what the CNN encoders
   manage to learn from a limited number of training images.
3. **Raw wavelength as a direct scalar.** Each TPSF's true (un-normalized)
   wavelength is concatenated straight into the fusion vector, with no
   learned encoder in between, so the regression head is supervised
   directly by the actual physical wavelength rather than by an
   intermediate representation a small sub-network invents on its own. An
   earlier version of this file routed the wavelength through a
   `Linear(1, 8) -> ReLU -> Linear(8, 4)` embedding first; that's been
   removed to match the original direct-scalar design.
4. **Larger per-step image batch.** `IMAGE_BATCH_SIZE` was raised from 1 to
   4 so that each optimizer step averages gradients over TPSFs drawn from
   several different phantoms/images, reducing gradient noise from
   within-image correlation (169 wavelengths from the same phantom share
   the same bottom-layer geometry).

## Further options worth trying on your data

These were not applied because they need to be tuned against real
training curves / target scale, which are not available here:

- If validation MAE is dominated by a few high-`mua` outliers, try
  `nn.SmoothL1Loss(beta=...)` in place of `RawMuaLoss`'s `F.l1_loss` (the
  validation MAE/RMSE metrics are computed independently of the training
  criterion, so this only changes optimization dynamics, not what is
  reported).
- If training loss is still falling but validation loss has plateaued,
  increase `HEAD_DROPOUT`/`TEMPORAL_DROPOUT` or `WEIGHT_DECAY` slightly.
- If the model is instead underfitting (train and validation MAE both
  high and close together), increase `TEMPORAL_FILTERS_PER_KERNEL` or
  `TEMPORAL_FEATURE_DIM` before adding more regularization.
- Consider k-fold cross-validation across images instead of a single
  80/20 split if the number of training images is small, since a single
  split can make the reported validation MAE noisy from run to run.

## Depth-resolved variant

`train_bottom_mua_depth_resolved.py` replaces the raw-tail convolutional
encoder with an explicit depth-resolved sequence model, and imports its
data loading / dataset / full-TPSF encoder / loss from
`train_bottom_mua_dynamic_tail.py` rather than duplicating them.

Motivation: in a diffusive medium, photons collected at progressively
later times in a DTOF have, on average, travelled deeper into the tissue
before returning to the detector, and for late times the tail's
log-amplitude decay slope is itself a direct estimator of absorption
(`ln I(t) ~= -mua * v * t + const`). The tail is therefore not one blob of
"late-time information" — it's an ordered sequence of increasingly
deep-sampling segments — but the original model pooled the whole tail
into a single global feature vector and threw that structure away.

The depth-resolved model instead:

1. Slices each tail `[LATE_START:300]` into an ordered sequence of
   overlapping windows (`WINDOW_BINS` wide, `WINDOW_STRIDE` apart), from
   the shallowest (right at `LATE_START`) to the deepest (ending at the
   last time bin, bin 300, for every sample regardless of its
   `LATE_START`).
2. Computes a physics-informed feature per window: the local
   log-amplitude decay slope, mean amplitude, and peak amplitude
   (`compute_window_slope_features`).
3. Runs a GRU over that window sequence, producing a running hidden state
   that gets refined as deeper (later) windows are folded in.
4. Reads out a non-negative absorption estimate at every step, giving an
   explicit, inspectable `depth_profile` per TPSF.

Only one label (bottom-mua) exists per DTOF, so the profile can't be
supervised at every depth. Instead `depth_profile_auxiliary_losses`
applies deep supervision to just the deepest `NUM_SUPERVISED_TAIL_WINDOWS`
points of the profile, with weights that increase toward the deepest
window, plus a variance ("plateau") penalty encouraging those deep points
to stabilize near the label instead of continuing to swing — consistent
with the physical expectation that late-enough tail segments should be
dominated by the bottom layer. The GRU's final hidden state also feeds the
main regression head (alongside the full-TPSF and wavelength features), so
this deep supervision also strengthens gradient flow into the tail
encoder.

To inspect a sample's depth profile directly (e.g. to plot how the
estimate evolves and converges as more of the tail is included), call
`model(tpsf, wavelength, late_start)` — it returns
`(prediction, supervised_profiles)` rather than a single tensor.

Tunable knobs specific to this variant: `WINDOW_BINS`, `WINDOW_STRIDE`,
`NUM_SUPERVISED_TAIL_WINDOWS`, `LATE_WEIGHT_POWER`,
`AUX_DEPTH_LOSS_WEIGHT`, `PLATEAU_LOSS_WEIGHT`.

## Spectral-smoothing variant

`train_bottom_mua_spectral_smoothing.py` addresses a gap shared by both
models above: every one of the 169 wavelength-specific TPSFs in an image
is regressed completely independently, so a given wavelength's prediction
never sees any information from its neighbors. On real predicted-vs-truth
spectra this shows up as three symptoms: high-frequency jitter across
wavelength (nothing enforces the physically expected spectral smoothness),
compressed peak/trough amplitude (each wavelength's estimate has to stand
entirely on its own TPSF instead of pooling evidence with neighbors), and
drift at the spectral edges (a single dataset-wide TPSF input scale
assumes uniform amplitude/SNR across wavelength, which real
source/detector responses rarely have).

This variant keeps `BottomMuaDepthResolvedNet`'s tail encoder unchanged
and adds:

1. **Per-wavelength TPSF input scale** (`estimate_per_wavelength_input_scale`,
   `PerWavelengthNormalizedDataset`) — a length-169 scale vector instead of
   one dataset-wide scalar, so each wavelength is normalized against its
   own typical amplitude rather than a global one.
2. **`SpectralSmoother`** — a residual depthwise-separable 1D convolution
   applied across the wavelength axis, over the fused per-wavelength
   feature vectors of one image, immediately before the regression head.
   Every wavelength's prediction gets access to a local neighborhood
   (`SPECTRAL_SMOOTHING_KERNEL_SIZE` wide) of the other wavelengths'
   evidence. Its pointwise mixing weights are zero-initialized, so the
   module starts as the identity function and can only begin contributing
   once training shows it reduces the loss.
3. An optional, off-by-default spectral total-variation loss
   (`SPECTRAL_TV_LOSS_WEIGHT`) for experimentation — left at `0.0` because
   a hard smoothness penalty can flatten genuine peaks rather than just
   removing jitter, whereas `SpectralSmoother` gives the network a
   structural way to be smooth without that risk.

`train_bottom_mua_spectral_smoothing.py` is fully standalone (no imports
from the other two files in this folder) and, like them, concatenates each
TPSF's raw wavelength directly into the fusion vector rather than through
a learned embedding. Spectral mixing requires every wavelength of an
image to be present together in a batch, which is automatically true here
since each image always contributes exactly `N_WAVELENGTHS` contiguous
rows after `flatten_image_batch`, regardless of `IMAGE_BATCH_SIZE`.

`inference_bottom_mua_spectral_smoothing.py` is the matching standalone
inference script: it rebuilds `BottomMuaSpectralNet` from scratch, loads a
checkpoint this training script produced, divides each test TPSF by its
own per-wavelength scale, and predicts on one complete test image (all
169 wavelengths) at a time, since `SpectralSmoother` requires that. The
model's `forward` returns `(prediction, depth_profile)`; the script saves
both.

## Relative-error training loss

`train_bottom_mua_spectral_smoothing.py` now exposes a `LOSS_MODE`
setting (`"raw"` or `"relative"`, default `"relative"`) selecting the
primary regression loss via `build_loss()`:

- `"raw"` (`RawMuaLoss`) -- plain L1/MAE in physical mua units, as before.
- `"relative"` (`RelativeMuaLoss`) -- mean absolute percentage error
  against the true value: `|prediction - target| / target`.

This addresses a case where validation MAE looks good in absolute terms
but generalization on real test phantoms is still poor: when the
bottom-mua target range spans more than an order of magnitude (as it does
here), plain absolute-unit L1 loss lets the optimizer minimize the average
error mostly by fitting the numerically larger targets, leaving
proportionally much worse accuracy at the low end of the range -- e.g. a
fixed absolute error of ~7e-4 is a rounding error against a target of
3e-2 but a ~26% relative error against a target of 2.7e-3. Since low-mua
phantoms are exactly where earlier predicted-vs-true plots showed the
largest divergence from ground truth, `RelativeMuaLoss` makes every
wavelength's proportional accuracy count equally in the training
objective, regardless of its absolute magnitude. Validation MAE/RMSE are
still reported in raw physical units either way; only the optimization
target changes. The checkpoint records which mode was used
(`checkpoint["loss_mode"]`).

Switching `LOSS_MODE` back to `"raw"` reproduces the original loss
exactly; the two are not meant to be blended; if `"relative"` turns out to
overcorrect (visibly worse fit at the high end of the range), that is the
signal to try `"raw"` again or a compromise such as `nn.SmoothL1Loss` on
log-transformed targets instead.

## Improving generalization beyond the loss function

If switching to `LOSS_MODE = "relative"` does not fully close the gap
between validation error and real-test-phantom accuracy, the loss
function was only ever addressing the "wide target range" symptom, not
every possible cause of a train/test mismatch. Worth checking next, in
priority order:

1. **Split by phantom/source, not by file.** If `TRAIN_DIR` mixes several
   simulated/experimental phantom families, a random 80/20 file split
   still lets near-duplicate images from the same phantom land on both
   sides, so validation MAE can look good even when the model has not
   learned to generalize across phantoms. Hold out entire phantoms (or
   entire simulation batches) for validation instead.
2. **Check per-wavelength/per-oxygenation target coverage.** If the
   training set's true bottom-mua values never reach the range a held-out
   test phantom's oxygenation level produces at some wavelengths, no loss
   function or architecture change can fix that -- it is a coverage gap,
   not a fitting problem. Compare the training set's raw-target histogram
   against the test phantom's per-wavelength true values directly.
3. **Re-run the same held-out oxygenation-level test phantoms after
   retraining** with `LOSS_MODE = "relative"` to confirm the low-mua
   region actually improved, rather than assuming it from validation MAE
   alone -- validation MAE is computed on the same narrow distribution as
   training and will not surface a coverage gap.

## Tail-weighted variant (a simpler alternative)

`train_bottom_mua_tail_weighted.py` is a standalone alternative to
`train_bottom_mua_spectral_smoothing.py`'s depth-resolved GRU branch,
built after that design's *training* loss (not just validation loss)
stayed high -- a sign of optimization difficulty from the architecture
itself, not a generalization problem worth more tuning. The GRU branch
added six interacting tail-specific hyperparameters (`WINDOW_BINS`,
`WINDOW_STRIDE`, `NUM_SUPERVISED_TAIL_WINDOWS`, `LATE_WEIGHT_POWER`,
`AUX_DEPTH_LOSS_WEIGHT`, `PLATEAU_LOSS_WEIGHT`) and a recurrent network to
optimize; this variant removes all of it.

Instead of a second sequence-model branch over sliced tail windows, this
model runs a single multi-kernel convolution encoder over the *entire*
raw TPSF (so the top-layer-dominated early part is never discarded), and
changes only how that encoder's features are pooled into a fixed-size
vector:

- **Weighted-average pooling** with weight 1.0 for every time bin before
  that sample's own `LATE_START`, ramping linearly up to
  `1.0 + TAIL_WEIGHT_BOOST` at the final bin -- directly encoding "later
  photons carry more information about the bottom-layer absorption"
  (`ln I(t) ~= -mua * v * t + const` for late-enough `t`) as one
  physically legible number instead of a windowed sequence model.
- **Plain (unweighted) max pooling** alongside it, so the early peak
  position/amplitude -- which still carries top-layer information needed
  to separate the top layer's contribution from the bottom layer's -- is
  not lost just because it falls outside the tail-weighted region.

This collapses the previous variant's six tail-specific hyperparameters
into one (`TAIL_WEIGHT_BOOST`, default `3.0` -- the last time bin counts
4x as much as an early bin in the weighted average) and removes the GRU
entirely, at the cost of not producing an inspectable per-depth profile
the way `DepthResolvedTailEncoder` did. `SpectralSmoother` and the
auxiliary/plateau losses are both dropped for this first pass; they can
be layered back in once this simpler backbone is confirmed to fit the
training set well, since a spectral smoother is a separate concern
(cross-wavelength coherence) from the tail-weighting question this
variant addresses.

Like the other variants, this one keeps per-wavelength TPSF input
normalization, the raw wavelength scalar (no learned embedding), and the
`LOSS_MODE` ("raw"/"relative") choice, since none of those were
implicated in the high training loss.

## Returning to the depth-resolved variant: a tuning pass

The tail-weighted variant above did not train well in practice, so
`train_bottom_mua_spectral_smoothing.py` (GRU depth-resolved branch,
`SpectralSmoother`, `LOSS_MODE = "relative"`) remains the primary model.
This pass reverts the two settings that had made results worse, restores
capacity that had been cut too far, and fixes an augmentation-wiring
regression:

1. **`NUM_SUPERVISED_TAIL_WINDOWS`: `20 -> 5`, `SPECTRAL_SMOOTHING_KERNEL_SIZE`: `11 -> 7`.**
   Both were raised in an earlier experiment and made results worse --
   `20` pulled 54-71% of each tail toward the single bottom-mua label
   (see the earlier tail-window-coverage analysis), and the wider
   smoothing kernel likely flattened genuine spectral peaks/troughs along
   with the jitter it was meant to remove.
2. **`TEMPORAL_FEATURE_DIM`: `32 -> 48`.** A capacity-rebalancing
   experiment had narrowed the full-TPSF branch to widen the tail branch.
   Both train *and* validation MAE got worse, which is the signature of
   an actual capacity bottleneck (the full-TPSF branch still needs to
   characterize the top layer well enough for the head to separate its
   contribution from the bottom layer's), not overfitting. Restored.
3. **`DEPTH_FEATURE_DIM`: `16 -> 24`.** This is the per-window embedding
   dimension actually fed to the GRU at every step -- narrower than
   `DEPTH_HIDDEN_DIM` (64), so it was compressing each window's 24-dim
   raw input (20 bins + 3 slope/amplitude features + 1 depth-position
   coordinate) more than the GRU's hidden state could make use of. A
   wider hidden state cannot recover detail already discarded upstream,
   so this widens the actual bottleneck rather than the buffer around it.
4. **`TEMPORAL_DROPOUT`/`HEAD_DROPOUT`: `0.15 -> 0.10`.** With both train
   and validation MAE elevated together (not a train/val gap), the
   symptom points at under-capacity/over-regularization rather than
   overfitting, so dropout was eased alongside restoring capacity above.
5. **`USE_TRAINING_AUGMENTATION`: re-enabled, and a wiring bug fixed.**
   A locally-edited copy of this file had reintroduced the exact
   augmentation-wiring bug this project originally started by fixing:
   `train_dataset` was constructed with `augment=False` hardcoded,
   bypassing the `USE_TRAINING_AUGMENTATION` flag entirely. It is wired
   back to `augment=USE_TRAINING_AUGMENTATION`, and the flag is set back
   to `True` -- it is the only regularizer left once dropout was eased,
   and was part of the originally-validated baseline.

`DEPTH_HIDDEN_DIM=64` is kept at its widened value; it was never
implicated in the regression and has independent justification (a longer
window sequence needs a wider hidden state to avoid lossy compression).

## Source-aware train/validation split

`TRAIN_DIR` mixes three data sources with a real, physically-motivated
domain gap between them (different IRFs): pure simulated data, pure
experimental data, and a small simulated set specifically calibrated to
be closer to the experimental data. A plain random 80/20 split over every
file lets validation's source mix fall out by chance -- with roughly 64%
of all files simulated, a random split's validation set ends up mostly
testing simulated-data fit even though the real test set is three
experimental phantoms. A validation MAE that looks good under that split
was never lying, it just wasn't answering the question that matters for
deployment.

`build_source_lookup` and `stratified_train_val_split` replace the random
split with one that partitions each source separately:

- `SOURCE_SIMULATED_END` / `SOURCE_EXPERIMENTAL_END` mark the boundaries
  between the three sources in the sorted file list. `build_source_lookup`
  prints the filename at each boundary on every run specifically so you
  can confirm the sorted order actually matches those file-number ranges
  before trusting anything downstream -- if `TRAIN_DIR`'s file count ever
  changes, it also prints a warning that the boundaries need updating.
- `SOURCE_VAL_FRACTION_OVERRIDES` lets the small, precious
  `simulated_close_to_experimental` set (42 files) use a much smaller
  validation share (default 10%) than the other two sources (20%), so
  holding it out for validation doesn't remove most of its value from
  training. Set it to `0.0` to keep all of it in training and rely
  entirely on the real experimental test phantoms for that source's
  held-out check.
- `SOURCE_OVERSAMPLE_WEIGHTS` optionally oversamples a source in the
  training `DataLoader` via `WeightedRandomSampler`, for when experimental
  and close-to-experimental data (currently ~36% of files combined) are
  getting outweighed by the numerically larger simulated set during
  training. Left at `1.0` for every source (no-op, identical to plain
  shuffling) until the per-source validation breakdown below shows it's
  needed.
- `summarize_late_start_by_source` prints `LATE_START` statistics
  separately per source -- a large shift between sources here is the IRF
  difference showing up directly in the one quantity the tail-weighting
  and window mechanics depend on most.
- Every epoch now also prints Val MAE/RMSE separately for each source
  (via `per_source_val_loaders`), alongside the combined Val MAE that
  still drives the learning-rate scheduler and best-checkpoint selection
  exactly as before. This is diagnostic first: it turns one blended
  validation number into three, so you can see directly whether the model
  is fitting simulated data much better than experimental before deciding
  whether oversampling or anything architectural is actually needed.

The first real run under this split (best epoch 92) showed exactly the
gap this was built to surface: `simulated` MAE 7.12e-4, `experimental`
MAE 7.85e-4, `simulated_close_to_experimental` MAE 1.54e-3 -- more than
double either other source, and one of the two domains the real test set
is made of. That result led directly to the two changes below.

## Removing the now-redundant (and likely harmful) per-wavelength scale

The incoming TPSF is already area-under-curve normalized per wavelength
(each row's own integral is fixed) before it reaches this script. That
already accomplishes what `estimate_per_wavelength_input_scale` /
`PerWavelengthNormalizedDataset` were built for -- correcting non-uniform
source/detector power across wavelength -- so applying it again is
redundant at best.

It is very plausibly worse than redundant, and the per-source validation
gap above is consistent with why: `estimate_per_wavelength_input_scale`
rescales by each wavelength's **peak** (99th-percentile) amplitude, not
its integral, computed once from all training files combined (~64%
simulated). Once the integral is already fixed, peak height is part of
the physically meaningful pulse shape -- a more absorbing/scattering
wavelength produces a broader, lower-peaked pulse at the same total
photon budget -- so a peak-based scale estimated mostly from simulated
data can systematically mis-scale the sources it least reflects. That
predicts exactly the observed ordering: simulated best (the scale fits
it), experimental worse, and the smallest source
(`simulated_close_to_experimental`, with the least influence on that
shared scale despite needing the most correction) worst of all.

`USE_PER_WAVELENGTH_INPUT_SCALE` (default `False`) turns this off without
deleting the function, so it can still be re-tested if needed --
`estimate_per_wavelength_input_scale` is still computed and recorded in
the checkpoint for reference either way.

`AMPLITUDE_JITTER_STD` is set to `0.0` for a related reason: multiplying
a whole TPSF row by a random factor breaks the fixed-integral property
the input already has by construction, training on rows whose integral
is no longer constant -- a regime that never occurs in real,
always-normalized data. `MAX_TIME_SHIFT` and `ADDITIVE_NOISE_STD` are
unaffected by AUC normalization (timing jitter and per-bin measurement
noise are still real regardless of integral scale) and are kept.

## Fine-tuning toward the test set's domains

The real test set is experimental phantoms plus a subset of
`simulated_close_to_experimental` -- exactly the two weaker-performing
sources above, and not `simulated`, which is only there to teach the
model general DTOF-to-mua structure from an abundantly large dataset.
`fine_tune_on_target_domains()` is a second training phase: it loads the
checkpoint `train_spectral_model()` already produced, then continues
training it using only `FINE_TUNE_SOURCES = ("experimental",
"simulated_close_to_experimental")`, at a much lower learning rate
(`FINE_TUNE_LEARNING_RATE = 2e-6`) so the broad representation learned
from the numerically larger simulated set is specialized rather than
overwritten. See "Fine-tuning always starts a fresh optimizer -- LR
mismatch check" below for why this needs to stay below whatever LR the
base run actually converged at, not just below its starting
`LEARNING_RATE`.

It reuses the exact same stratified train/val file assignment as
`train_spectral_model` (same `SEED`), just filtered down to the two
target sources, so the fine-tuning validation files are the identical
ones already reported on in the general run -- before/after numbers stay
directly comparable. Early stopping and best-checkpoint selection during
fine-tuning use the **combined** validation set across both target
sources, not `simulated_close_to_experimental` alone, since that source's
validation split is only a handful of files -- too few for a reliable
stopping decision by itself, even though it's still reported separately
each epoch via `per_source_val_loaders`.

## Fine-tuning's sampler was over-equalizing the tiny source

The two `FINE_TUNE_SOURCES` are themselves imbalanced roughly 30:1
(`experimental` vs. `simulated_close_to_experimental`). The fine-tuning
sampler originally weighted each sample by `1 / (its source's file
count)`, giving every source exactly equal **total** sampling mass per
epoch regardless of size -- at a 30:1 imbalance that meant each
`simulated_close_to_experimental` file was drawn roughly 27x more often
per epoch than each `experimental` file.

A real run showed exactly the failure that predicts: after fixing the
LR mismatch above, `simulated_close_to_experimental`'s validation MAE
still improved during fine-tuning while `experimental`'s -- the actual
real-phantom target -- did not move at all. That's consistent with the
model memorizing the ~38 heavily-repeated `simulated_close_to_experimental`
training files (which correlates with its own tiny held-out validation
split, since both are drawn from the same narrow distribution) rather
than genuinely improving on `experimental`, which was simultaneously
getting less than half its previous per-epoch exposure to make room for
that oversampling.

Fixed by reusing the exact same fixed per-source multipliers
`SOURCE_OVERSAMPLE_WEIGHTS` already uses for the main run (`experimental:
2.0`, `simulated_close_to_experimental: 3.0`, applied per file rather than
divided by count) instead of a second, far more aggressive full-equalization
scheme that was never validated on its own. This keeps
`simulated_close_to_experimental` oversampled (~1.5x per file relative to
`experimental` here, vs. ~27x under the old full equalization) without
suppressing `experimental`'s exposure enough to stall its own learning.

Set `RUN_FINE_TUNING = True` and run this file to fine-tune the checkpoint
already saved at `SPECTRAL_MODEL_PATH` instead of training a new one from
scratch; the result is saved separately to `FINE_TUNE_MODEL_PATH`
(`SPECTRAL_MODEL_PATH` with `_finetuned` appended), so both the general
and fine-tuned checkpoints are kept for comparison.

### Fine-tuning source imbalance fix

`FINE_TUNE_SOURCES` are themselves badly imbalanced -- `experimental` has
roughly 30x as many files as `simulated_close_to_experimental` -- so a
plain shuffled loader over both would fill nearly every batch with
`experimental` rows and barely touch the very source with the worst
validation error. `fine_tune_on_target_domains()` now builds a
`WeightedRandomSampler` for the fine-tuning train loader, weighting each
sample by the inverse of its own source's file count so both target
sources contribute roughly equal total mass per epoch regardless of how
many files each has.

## Code review fixes (no training-behavior change from the above)

A pass over `train_bottom_mua_spectral_smoothing.py` after the fine-tuning
and AUC-normalization changes above found two more issues, fixed
alongside the fine-tuning sampler rebalancing:

- **Wasted startup I/O**: `estimate_per_wavelength_input_scale` was still
  called unconditionally at the start of both training and fine-tuning,
  re-reading and scanning every training file to compute a scale that
  `USE_PER_WAVELENGTH_INPUT_SCALE = False` throws away unused. Both call
  sites now go through `maybe_estimate_per_wavelength_input_scale()`,
  which returns an all-ones scale immediately when the flag is off and
  only pays the full-scan cost when the flag is on.
- **Stale "300 gates" references**: four leftover error messages and a
  docstring still referred to a hardcoded `300`-time-gate design from
  before `N_TIME_GATES` was introduced (one more than originally found,
  in `load_late_start_indices`'s out-of-range error message). All now
  reference `N_TIME_GATES` so they stay correct if the crop window
  changes.

## Visualizing training progress

Two ways to see the train/val MAE curve:

**Live, while training is running** -- `update_training_curve_plot()`
overwrites a PNG (`TRAINING_CURVE_PATH`, next to `SPECTRAL_MODEL_PATH`;
`FINE_TUNE_TRAINING_CURVE_PATH` during fine-tuning) after every epoch, so
opening that file mid-run shows current progress without waiting for
training to finish. Controlled by `PLOT_TRAINING_CURVE` (default `True`);
requires `matplotlib` and is silently skipped (with one warning) if it
isn't installed, so it can never crash a training run.

**After the fact, from a saved checkpoint** -- `plot_training_curves.py`
loads a checkpoint and plots the same curve from its saved
`train_bottom_mua_mae_history` / `val_bottom_mua_mae_history` (or
`fine_tune_*` equivalents, auto-detected):

```
python plot_training_curves.py --checkpoint "C:\path\to\bottom_mua_spectral_model.pth"
```

Both read the exact same history lists, so they always agree; the live
PNG is just faster feedback during a run in progress.

## A note on ADDITIVE_NOISE_STD

`USE_TRAINING_AUGMENTATION` gates two active augmentations now that
`AMPLITUDE_JITTER_STD = 0.0`: `MAX_TIME_SHIFT` (timing jitter, physically
justified by the real IRF differences between sources) and
`ADDITIVE_NOISE_STD` (flat per-bin Gaussian noise). The latter is worth
checking against real data rather than assuming it's calibrated correctly:
since the TPSF is AUC-normalized, a "typical" bin averages roughly
`1 / N_TIME_GATES`, but the tail -- the exact region
`DepthResolvedTailEncoder` depends on for its log-amplitude decay slope --
decays to values well below that average. A flat `std = 0.001` applied
everywhere can be comparable to or larger than real tail amplitudes,
which risks swamping the physics-based signal the tail encoder is built
to extract. Worth checking actual tail-bin magnitudes on real data before
trusting the current value; if the tail sits near or below that noise
floor, consider lowering it or scaling it relative to each row's own
amplitude instead of using one fixed constant.

## Critical fix: source classification was scrambled by string sorting

`build_source_lookup` classified each file by its **position in
`sorted(glob(...))`**, on the assumption that sorting the file list would
recover numeric order (file #1 through file #3615). It does not:
`sorted()` on filenames sorts them as plain strings, so e.g.
`"DTOF_961.mat"` lands alphabetically among all the `"DTOF_9xx.mat"` /
`"DTOF_9xxx.mat"` names rather than next to its true numeric neighbors
`DTOF_960.mat` and `DTOF_962.mat`. Confirmed directly from real output:
the file landing at sorted-position #2300 (the intended
simulated/experimental boundary) was `DTOF_3068.mat`, and the file at
sorted-position #3574 (the intended experimental/close-to-experimental
boundary) was `DTOF_961.mat` -- both far from their intended numeric
boundaries. Every source label assigned since the source-aware split was
introduced was therefore wrong for most files.

This means the per-source validation numbers reported so far (e.g. the
epoch-92 breakdown showing `simulated_close_to_experimental` at roughly
2x the MAE of the other two sources) were computed against these
scrambled groups, not the real ones -- they do not reliably describe
per-source performance. Likewise, `fine_tune_on_target_domains()` runs so
far fine-tuned on whatever scrambled mix of files happened to carry the
`experimental` / `simulated_close_to_experimental` labels, not the
correct target-domain files.

**Fix**: `extract_file_number()` parses each file's own numeric index
directly out of its filename (e.g. `DTOF_137.mat` -> `137`) via regex,
and `classify_source_by_file_number()` classifies using that number
against `SOURCE_SIMULATED_END` / `SOURCE_EXPERIMENTAL_END` directly --
independent of file-list order entirely. `build_source_lookup`'s sanity
check now prints the actual file-number range observed in each source
(instead of boundary filenames at specific sorted positions), so a
misclassification is now visible immediately as an out-of-range number
in the wrong bucket rather than needing to be inferred from filenames at
arbitrary sorted positions.

**Action needed**: re-run `train_spectral_model()` (and
`fine_tune_on_target_domains()` afterward) from scratch with this fix.
Any checkpoint or per-source validation numbers from before this fix
should not be trusted or compared against future runs -- the underlying
groups were different.

## Anti-overfitting pass (after the classification fix)

With the split fixed, a real run showed train MAE falling fast (3.75e-3 ->
1.62e-3 in 6 epochs) while overall val MAE rose *above* the constant
baseline within 2 epochs -- classic overfitting, and unusually fast for
this architecture. The per-source breakdown pinpointed the cause: with
`SOURCE_OVERSAMPLE_WEIGHTS` all at `1.0`, `simulated` (~64% of training
files) dominated every epoch's gradient signal, so `experimental` and
`simulated_close_to_experimental` -- the real test set's actual domains --
degraded 2-3x faster than `simulated` over the same epochs.

Changes made, none touching `TemporalEncoder` / `DepthResolvedTailEncoder`
/ `SpectralSmoother` (those were already tuned through several rounds of
prior experiments, and narrowing them previously made both train AND val
worse -- a capacity problem, not overfitting):

- **`SOURCE_OVERSAMPLE_WEIGHTS`** -- `experimental` and
  `simulated_close_to_experimental` raised to `2.0`/`3.0` so they get
  roughly equal gradient mass per epoch regardless of file count, directly
  countering the majority-source dominance seen above.
- **`LEARNING_RATE`** `1e-4` -> `5e-5`, **`WEIGHT_DECAY`** `1e-5` -> `1e-4`
  -- slows the early optimizer steps and penalizes large weights more.
- **`TEMPORAL_DROPOUT`** `0.10` -> `0.20`, **`HEAD_DROPOUT`** `0.10` ->
  `0.30` -- less room for the model to memorize per-file idiosyncrasies.
- **`PLATEAU_LOSS_WEIGHT`** `0.05` -> `0.0` -- removed as a genuinely
  unnecessary block: the weighted-L1 term in `depth_profile_auxiliary_losses`
  already pulls every supervised tail window toward the same single
  bottom-mua label, which already pushes those windows toward agreeing
  with each other as a side effect. The explicit variance penalty on top
  of that was redundant, and was one more term the optimizer could exploit
  to overfit each training tail's exact shape rather than the underlying
  decay-slope relationship. The function still computes it (for anyone who
  wants to re-enable it), it just no longer contributes to the loss.

**Action needed**: re-run `train_spectral_model()` with these settings and
compare the new per-epoch curve (via the live PNG) against the pattern
above -- val MAE should track train MAE much more closely for longer
before diverging, and the `experimental`/`simulated_close_to_experimental`
per-source numbers should stop blowing up disproportionately to
`simulated`.

## Exponential moving average of weights (EMA)

A real run with the anti-overfitting pass above showed the runaway
divergence was fixed, but the resulting curve wasn't healthy either: train
MAE fell smoothly to ~1e-3 over 55 epochs while val MAE oscillated between
roughly 2.5e-3 and 5.8e-3 for the *entire* run, with only one epoch (36,
out of 55) landing meaningfully below the constant baseline. That's not
overfitting -- it's optimization noise: the weights are still bouncing
around a decent solution rather than settling into one, so "the epoch
with the lowest val MAE" is largely a lucky noise draw rather than a
reliable checkpoint to keep.

`USE_EMA` (default `True`) tracks a separate, frozen copy of the model's
weights (`eval_model`) that gets updated after every optimizer step as
`ema_weight <- EMA_DECAY*ema_weight + (1-EMA_DECAY)*current_weight`
(`create_ema_model` / `update_ema_model`, section 12c). Validation, the
scheduler, and best-checkpoint selection all use `eval_model` instead of
the raw in-training `model`, so the saved checkpoint is a smoothed average
over recent steps rather than one single noisy snapshot. `EMA_DECAY =
0.999` gives an averaging half-life of roughly 700 steps -- about one
epoch at this dataset's size -- long enough to smooth out the observed
epoch-to-epoch swings without lagging so far behind that it can't track
real, sustained improvement. `fine_tune_on_target_domains()` mirrors the
same setup, seeding its own `eval_model` from the loaded (already-EMA)
checkpoint. Setting `USE_EMA = False` reproduces the exact prior
behavior (`eval_model` becomes `model` itself).

This adds no new loss term and no architecture change -- it only changes
*which* weights get validated and saved.

## Simplification pass: removing dead and redundant code

The file had accumulated several toggles that were tried, found not to
help (or found actively harmful), and left in place as off-by-default
switches "in case they're needed again." With none of them adding value
and all of them adding surface area to read through, they were removed
outright rather than just left disabled -- none of this touches the
3-source split, per-source validation, or fine-tuning, which are kept in
full:

- **Per-wavelength input rescale** (`USE_PER_WAVELENGTH_INPUT_SCALE`,
  `estimate_per_wavelength_input_scale`,
  `maybe_estimate_per_wavelength_input_scale`) -- established redundant
  once the input TPSF is already AUC-normalized per wavelength upstream,
  and plausibly harmful to the domains it was least calibrated for.
  `PerWavelengthNormalizedDataset` (which threaded a `per_wavelength_scale`
  array through every dataset/loader/checkpoint call site even though it
  was always a no-op array of ones) is renamed `SpectralTPSFDataset` and
  no longer takes that parameter at all.
- **Plateau variance penalty** (`PLATEAU_LOSS_WEIGHT`) -- was already
  zeroed out as redundant with the weighted-L1 depth-supervision term;
  now the computation itself is gone.
  `depth_profile_auxiliary_losses` (returned an unused second value) is
  now `depth_profile_auxiliary_loss` (returns just the one loss value
  actually used).
- **Spectral total-variation loss** (`SPECTRAL_TV_LOSS_WEIGHT`,
  `spectral_total_variation_loss`) -- an experimental extra smoothing
  penalty that was never actually turned on across this project's
  history; `SpectralSmoother` already gives the network a structural way
  to produce smooth spectra.
- **Amplitude-jitter augmentation** (`AMPLITUDE_JITTER_STD`) -- not an
  experimental toggle but a permanently-wrong augmentation for this
  dataset: it multiplies a whole TPSF row by a random factor, which
  breaks the fixed-integral (AUC-normalized) property this data has by
  construction. `MAX_TIME_SHIFT` and `ADDITIVE_NOISE_STD` remain, since
  both model real physical variation (timing jitter, measurement noise).

Kept as-is, since they serve the stated goal of validating per-source and
fine-tuning toward the test set's domains: the source-aware split
(`build_source_lookup`, `stratified_train_val_split`), per-source
validation reporting (`build_per_source_val_loaders`), source
oversampling (`SOURCE_OVERSAMPLE_WEIGHTS`), `fine_tune_on_target_domains`,
EMA, the live training-curve plot, and the core architecture
(`TemporalEncoder`, `DepthResolvedTailEncoder`, `SpectralSmoother`).

## Faster LR decay after the validation minimum

A real (EMA-smoothed) run showed val MAE bottom out at epoch 5 -- just
below the constant baseline -- then climb for 5 straight epochs back past
that baseline, while train MAE kept falling smoothly the entire time.
That's real, sustained overfitting resuming after the minimum, not
epoch-to-epoch noise. The cause: `SCHEDULER_PATIENCE = 10` meant the LR
would not have halved until epoch 15 (10 epochs past the epoch-5 best),
so the optimizer kept taking full-sized steps for 10 more epochs while
actively overfitting.

`SCHEDULER_PATIENCE` lowered `10 -> 3` so the LR backs off much sooner
after validation stops improving, before the model has had that much room
to drift past its best point. `EARLY_STOPPING_PATIENCE` scaled down
`20 -> 12` to match (roughly 4x the new scheduler patience, so about 3 LR
halvings still get a chance to find a better minimum before training
gives up for good, rather than burning most of the patience budget at a
single, too-high LR). Note that the checkpoint actually saved was never
at risk from this climb -- `best_state` only updates on improvement -- so
this change is about training efficiency and giving the optimizer a real
chance at a better minimum, not about correctness of what gets saved.

## Normalizing the wavelength input

The wavelength scalar was being concatenated into the fused feature
vector as its raw physical value (e.g. ~650-950 nm), while everything
else in that vector (`full_features`, `tail_final_features`) had already
passed through `LayerNorm` and sat at roughly unit scale. That left one
channel of every downstream layer 2-3 orders of magnitude out of scale
with the rest -- most consequentially for `SpectralSmoother`'s depthwise
convolution, which has no normalization before it at all, so that
channel's kernel had to learn an unusually tiny weight just to keep its
contribution from dominating.

`load_wavelength_vector()` already computes a `[0,1]` min-max-normalized
version of the wavelengths (`normalized_wavelengths`) -- it just wasn't
being used for training, only saved into the checkpoint for reference.
Both `train_spectral_model()` and `fine_tune_on_target_domains()` now
pass `normalized_wavelengths` into `SpectralTPSFDataset` and
`build_per_source_val_loaders` instead of `raw_wavelengths`. This doesn't
change the original design intent (the model is still directly supervised
by each TPSF's actual wavelength, not a learned embedding) -- min-max
normalization is a fixed, invertible rescaling of the same physical
quantity, not a learned intermediate representation. `raw_wavelengths` is
still computed and recorded in the checkpoint for reference.

Since `load_wavelength_vector()` is deterministic given the same
`WAVELENGTH_FILE`, `fine_tune_on_target_domains()` recomputes the exact
same `[0,1]` scale the base checkpoint was trained with -- no drift
between phase 1 and phase 2. If you have a separate inference/prediction
script outside this file that feeds the raw wavelength value to a loaded
checkpoint, it needs to normalize the same way before this change's
checkpoints will predict correctly.

## Larger batch size for smoother gradients

`IMAGE_BATCH_SIZE` doubled `4 -> 8`. Each step previously saw only 4
independent images (676 rows, but those rows are 4 groups of 169
highly-correlated wavelengths, not 676 independent samples) -- a fairly
noisy per-step gradient estimate, and part of why validation bounced so
much before EMA was added. A larger batch also interacts with
`SOURCE_OVERSAMPLE_WEIGHTS`: with only 4 images per step, the minority
sources' 2x/3x upweighting doesn't reliably show up in every batch; a
larger batch makes that representation more consistent step to step
instead of arriving in bursts.

Not a free win in general -- very large batches are empirically
associated with converging to sharper, worse-generalizing minima unless
compensated with other changes (LR scaling, warmup) -- but at a modest
doubling for a ~75K-parameter model, that risk is unlikely to matter.
Left `LEARNING_RATE` untouched to isolate this as one variable, per the
usual practice in this file; lower `IMAGE_BATCH_SIZE` back toward 4 if
this doesn't fit your GPU's memory.

## Per-source raw MAE isn't a fair cross-source comparison by itself

A run showed `experimental`'s raw validation MAE lower than `simulated`'s
-- easy to read as "the model does better on experimental," but
`bottom_mua_mae` is an *absolute* error in raw physical mua units. If
`experimental`'s target values simply sit at a smaller absolute magnitude
than `simulated`'s wider synthetic sweep, a lower raw MAE there doesn't
mean better *proportional* accuracy -- it can just reflect smaller
targets. Since bottom-mua targets are known to span more than an order of
magnitude (the whole reason `RelativeMuaLoss` exists), this ambiguity is
real, not hypothetical, and the code had no way to tell the two apart.

Two additions close that gap:

- **`summarize_raw_target_by_source()`** -- prints each source's target
  min/median/max/mean once at the start of training, so you can directly
  check whether `experimental`'s targets are just numerically smaller.
- **`validate_spectral()`** now also returns `bottom_mua_relative_mae`
  (mean `|error|/target`, the same quantity `RelativeMuaLoss` computes) --
  scale-independent, so it's the fair number for comparing accuracy
  *across* sources with potentially different target magnitudes. Printed
  as `Val RelMAE=...%` alongside the existing raw `Val MAE`/`Val RMSE` in
  both the combined and per-source lines, in both `train_spectral_model()`
  and `fine_tune_on_target_domains()`. The combined `val_loader`'s raw MAE
  still drives the scheduler and best-checkpoint selection, unchanged --
  RelMAE is diagnostic, matching the existing per-source reporting's role.

## Measuring bias, not just error magnitude

MAE/RMSE/RelMAE all average `|error|`, which erases the sign -- a model
that is consistently biased in one direction (always over- or
under-predicting) and one that is unbiased but noisy can show the exact
same MAE. That distinction matters directly for fine-tuning: the reason
to expect fine-tuning to help is that a systematic domain bias (e.g. from
`experimental`/`simulated_close_to_experimental` being underrepresented
during general training) is a structured, one-directional error that
gradient descent corrects efficiently, unlike random scatter. There was
no way to check that empirically, since nothing measured sign.

`validate_spectral()` now also returns `bottom_mua_bias` (mean signed
`prediction - target`) and `bottom_mua_relative_bias` (the same divided
by target, matching `RelativeMuaLoss`'s scale). Printed as
`Val RelBias=+/-...%` in the per-source lines (both `train_spectral_model()`
and `fine_tune_on_target_domains()`) -- a large `|RelBias|` alongside a
similar RelMAE points to a systematic, correctable bias rather than noise;
comparing it between the general checkpoint and the fine-tuned one on the
same source directly shows whether fine-tuning actually reduced that bias
rather than just moved the scatter around.

## Fine-tuning always starts a fresh optimizer -- LR mismatch check

`fine_tune_on_target_domains()` creates a brand-new `AdamW` optimizer at a
fixed `FINE_TUNE_LEARNING_RATE` -- it never reads or inherits whatever LR
the main run's `ReduceLROnPlateau` schedule had decayed to by the time it
stopped. With `SCHEDULER_PATIENCE=3` and `EARLY_STOPPING_PATIENCE=12`, the
main run can go through up to 4 LR halvings before stopping (`5e-5 ->
2.5e-5 -> 1.25e-5 -> 6.25e-6 -> 3.1e-6`), so it's entirely plausible its
LR ends up *below* `FINE_TUNE_LEARNING_RATE = 1e-5` by the time it stops.
If that happens, fine-tuning would take **bigger** steps than the base
model had already converged with -- the opposite of "a much lower
learning rate so the general representation isn't overwritten," and the
two phases had no way to notice this mismatch.

Fixed by recording `best_epoch_learning_rate` (the LR in effect at the
epoch actually saved as best) into the checkpoint in
`train_spectral_model()`. `fine_tune_on_target_domains()` now compares
`FINE_TUNE_LEARNING_RATE` against that recorded value right after loading
the base checkpoint and prints a `WARNING` if fine-tuning's LR isn't
actually lower -- a diagnostic, not an automatic override, so you can
decide whether to lower `FINE_TUNE_LEARNING_RATE` based on what the base
run actually converged at.

**This mismatch was confirmed on a real run, not just theoretical**: the
base checkpoint's best epoch had already decayed to `6.25e-6`, while
`FINE_TUNE_LEARNING_RATE` was `1e-5` -- larger, not smaller. That run's
fine-tuning validation MAE never beat its own epoch 1 (see the patience
section below), consistent with the first fine-tune steps being large
enough to kick the weights out of the minimum the base checkpoint was
already sitting in, rather than gently specializing it.
`FINE_TUNE_LEARNING_RATE` was lowered to `2e-6` -- below that run's
converged `6.25e-6` -- specifically because of this. If your own base
checkpoint converges at a different LR, check its
`best_epoch_learning_rate` and keep `FINE_TUNE_LEARNING_RATE` below it.

## Tightening fine-tuning's own scheduler/early-stopping patience

A real fine-tuning run showed train MAE decreasing while val MAE didn't --
the same overfits-after-the-minimum pattern already found and fixed in
the main training loop (`SCHEDULER_PATIENCE`/`EARLY_STOPPING_PATIENCE`),
just never carried over to `fine_tune_on_target_domains()`, which still
had the original `FINE_TUNE_SCHEDULER_PATIENCE=5` /
`FINE_TUNE_EARLY_STOPPING_PATIENCE=10`. If anything this pattern is more
likely during fine-tuning, not less: its training pool is only two
sources, and the inverse-frequency `WeightedRandomSampler` gives the tiny
`simulated_close_to_experimental` group (~38 train files) equal total
gradient mass to the much larger `experimental` group, meaning the model
gets shown that small set's exact files repeatedly -- a strong setup for
memorizing training files fast while genuinely held-out validation files
don't improve.

`FINE_TUNE_SCHEDULER_PATIENCE` lowered `5 -> 2`, `FINE_TUNE_EARLY_STOPPING_PATIENCE`
lowered `10 -> 8`, mirroring the main run's fix so the fine-tuning LR
backs off sooner after validation stops improving instead of continuing
to overfit at full LR for several more epochs.
