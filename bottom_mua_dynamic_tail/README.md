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

## Global-context variant

Evaluating a checkpoint across phantoms at five different oxygenation
levels showed the same qualitative error shape everywhere (compressed
peak, undershooting trough, diverging tail), but its *magnitude* changed
systematically with the phantom's oxygenation level, worst at the
extremes. That pattern points at something `SpectralSmoother` cannot
provide: real oxy-/deoxy-hemoglobin extinction spectra make the whole
shape of a 169-wavelength curve informative about which regime a phantom
is in, but a local convolution kernel has a bounded receptive field and
structurally cannot let evidence at wavelength 0-20 correct a prediction
at wavelength 140-169.

`train_bottom_mua_spectral_smoothing_global_context.py` and its matching
`inference_bottom_mua_spectral_smoothing_global_context.py` are copies of
the single-block spectral-smoothing pair with one addition:
`GlobalSpectralContext`. It pools the fused per-wavelength features
across *all* 169 wavelengths of one image into a single per-image context
vector (mean + max pooling), and uses it to apply a FiLM-style
(feature-wise linear modulation) affine correction -- `features * (1 +
scale) + shift`, with `scale`/`shift` computed once per image and
broadcast to every wavelength -- before `SpectralSmoother`'s local
smoothing refines the result further. Like every other correction module
in this codebase, its final layer is zero-initialized, so it starts as
the identity transform and only begins contributing once training shows
it reduces the loss.

This is again a separate, independently trainable file (architecture
identifier `bottom_mua_depth_resolved_spectral_smoothing_global_context`,
its own default `SPECTRAL_MODEL_PATH`/`MODEL_PATH`), not a drop-in
replacement for `train_bottom_mua_spectral_smoothing.py` -- keep both
around to compare whether global conditioning meaningfully reduces the
oxygenation-dependent bias the local-only smoother leaves behind. If it
doesn't, that would point to a training-data coverage gap (the training
set never showing the backbone a phantom whose true bottom-mua reached a
given value at a given wavelength) rather than an architecture limitation
-- worth checking directly against your training set's per-wavelength
target distribution.
