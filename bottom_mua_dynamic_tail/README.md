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
