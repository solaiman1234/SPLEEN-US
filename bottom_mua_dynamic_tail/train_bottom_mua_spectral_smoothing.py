# -*- coding: utf-8 -*-
"""
Depth-resolved, spectrally-smoothed bottom-mua model. Fully standalone --
does not import from train_bottom_mua_dynamic_tail.py or
train_bottom_mua_depth_resolved.py.

Each image contains:
    - 169 wavelength-specific TPSFs
    - N_TIME_GATES selected time bins per TPSF
    - 169 bottom_absorption_mul targets
    - 169 LATE_START indices

For each wavelength-specific TPSF:
    full raw TPSF -> full convolution encoder (TemporalEncoder)
    raw TPSF tail -> depth-resolved sequence encoder (DepthResolvedTailEncoder)
    raw wavelength -> one direct scalar, no wavelength encoder

The wavelength is concatenated into the fused feature vector as a direct
[0,1] min-max-normalized scalar (see load_wavelength_vector), not a
learned embedding, so the regression head is still directly supervised by
each TPSF's actual wavelength rather than by an intermediate
representation invented by a small sub-network trained on nothing but
that one scalar. Min-max normalization (rather than the raw physical
value, e.g. ~650-950 nm) matters here because this scalar is concatenated
with LayerNorm'd encoder outputs that sit at roughly unit scale --
feeding in the raw value left one channel of every downstream layer
(including SpectralSmoother's un-normalized depthwise convolution) 2-3
orders of magnitude out of scale with the rest.

Depth-resolved tail encoding: in a diffusive medium, photons collected at
progressively later times in a DTOF have, on average, travelled deeper
into the tissue before returning to the detector, and for late times the
tail's log-amplitude decay slope is a direct, low-variance estimator of
the local absorption coefficient (ln I(t) ~= -mua * v * t + const). The
tail [LATE_START:N_TIME_GATES] is therefore not one blob of "late-time
information": it is an ordered sequence of increasingly deep-sampling
segments. DepthResolvedTailEncoder slices the tail into overlapping,
ordered windows, extracts a windowed decay-slope feature per window, and
runs a GRU over the sequence (shallow -> deep), reading out an explicit
depth-resolved absorption profile at every step. Only one label (the
bottom-layer mua) exists per DTOF, so the deepest few points of that
profile are supervised directly (with depth-increasing weights) and a
plateau penalty encourages them to stabilize instead of continuing to
swing, consistent with late-enough tail segments being dominated by the
bottom-layer absorption.

Spectral coherence: all 169 wavelengths of an image are otherwise
regressed completely independently, which on real predicted-vs-truth
spectra shows up as high-frequency jitter, compressed peak/trough
amplitude, and drift at the spectral edges. SpectralSmoother is a
zero-initialized residual depthwise-separable 1D convolution applied
across the wavelength axis on the fused per-wavelength feature vectors,
immediately before the regression head, so every wavelength's prediction
can pool evidence from a local neighborhood of other wavelengths instead
of standing entirely on its own TPSF. It starts as the identity function
and only begins contributing once training shows it reduces the loss.
A per-wavelength input rescale (dividing each wavelength's TPSF by its
own peak amplitude) was tried for the same reason -- real source/detector
responses aren't uniform across wavelength -- and removed: the input TPSF
is already area-under-curve normalized per wavelength before it reaches
this script, which already accounts for that, and the extra rescale was
found to plausibly hurt the domains it was least calibrated for.
"""

import copy
import os
import random
import re
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    import matplotlib
    matplotlib.use("Agg")  # No GUI needed; this only ever saves a PNG to disk.
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ============================================================
# 1. USER SETTINGS
# ============================================================

TRAIN_DIR = r"D:\Simulation_TPSF_Convolution_Animal_IRF\DTOF_training_simu_and_exp_resampled"
WAVELENGTH_FILE = r"D:\Simulation_TPSF_Convolution_Animal_IRF\wavelength_exp_index.mat"
WAVELENGTH_KEY = "wavelengths_exp"

SPECTRAL_MODEL_PATH = r"D:\DTOF_exp_simu_depth_resolved_spectral_bottom_mua_v2.pth"

TPSF_KEY = "dtof_auc"
BOTTOM_MUA_KEY = "bottom_absorption_mul"
LATE_START_KEY = "End_start"

# MATLAB indexing normally starts at 1. Set False for zero-based stored indices.
LATE_START_IS_MATLAB_ONE_BASED = True

N_WAVELENGTHS = 169
N_TIME_GATES = 450
TIME_GATE_START = 0
TIME_GATE_END = 450

# Doubled from 4: each step only saw 4 independent images (676 rows, but
# those rows are 4 groups of 169 highly-correlated wavelengths, not 676
# independent samples), which is a noisy per-step gradient estimate and
# part of why validation bounced so much before EMA. A larger batch also
# gives SOURCE_OVERSAMPLE_WEIGHTS' minority-source upweighting a better
# chance of actually showing up within every step instead of being spread
# thin across occasional batches. The model is small (~75K parameters)
# and each image is a small tensor, so this is safe on essentially any
# GPU with a few GB free -- lower it back to 4 if you hit an
# out-of-memory error on your hardware.
IMAGE_BATCH_SIZE = 8
NUM_EPOCHS = 200
TRAIN_FRACTION = 0.80

# TRAIN_DIR mixes three data sources with a known, physically real domain
# gap between them (different IRFs), identified by each file's own numeric
# index parsed from its filename (e.g. DTOF_137.mat -> 137) -- NOT by its
# position in sorted(glob(...)), which sorts filenames as plain strings
# ("DTOF_1.mat" < "DTOF_10.mat" < "DTOF_100.mat" < "DTOF_1000.mat" < ... <
# "DTOF_11.mat") and so does not follow numeric order at all. See
# extract_file_number.
#   file numbers 1..SOURCE_SIMULATED_END                          -> "simulated"
#   file numbers SOURCE_SIMULATED_END+1..SOURCE_EXPERIMENTAL_END   -> "experimental"
#   file numbers SOURCE_EXPERIMENTAL_END+1..end                    -> "simulated_close_to_experimental"
# A plain random 80/20 split over all files lets validation's source mix
# fall out by chance -- with ~64% of files simulated, validation ends up
# mostly testing simulated-data fit even though the real test set is
# experimental phantoms. The split below instead partitions each source
# separately, so validation always contains a deliberate share of every
# source rather than whatever a random shuffle happens to produce.
SOURCE_SIMULATED_END = 2300
SOURCE_EXPERIMENTAL_END = 3573
SOURCE_TOTAL_EXPECTED = 3615

# Per-source override of the validation fraction; anything not listed
# here uses (1 - TRAIN_FRACTION). "simulated_close_to_experimental" is
# small (42 files) and specifically bridges the sim/experimental domain
# gap -- holding out the usual 20% would remove ~8 of those 42 files from
# training for comparatively little validation signal, so it defaults to
# a smaller share. Set it to 0.0 to keep all of it in training and rely
# entirely on the real experimental test phantoms for that source's
# held-out check.
SOURCE_VAL_FRACTION_OVERRIDES = {
    "simulated_close_to_experimental": 0.10,
}

# Per-source oversampling weight for the training sampler. Turned on: a
# real run's per-source validation MAE showed exactly the pattern this
# was meant to catch -- with uniform sampling, "simulated" (~64% of
# training files) dominates every epoch's gradient signal, so the model
# drifted toward fitting it while "experimental" and
# "simulated_close_to_experimental" (the real test set's domains)
# degraded 2-3x faster over the same epochs. Weighting the minority
# sources higher gives them roughly equal total gradient mass per epoch
# regardless of file count, directly countering that.
SOURCE_OVERSAMPLE_WEIGHTS = {
    "simulated": 1.0,
    "experimental": 2.0,
    "simulated_close_to_experimental": 3.0,
}

# Fine-tuning: after train_spectral_model() produces a general checkpoint
# from all three sources, fine_tune_on_target_domains() continues training
# that checkpoint using only the sources the real test set is made of
# (experimental phantoms + simulated_close_to_experimental), at a much
# lower learning rate so the broad representation learned from the
# numerically larger simulated set isn't overwritten, only specialized.
# It reuses the exact same stratified train/val file assignment as
# train_spectral_model (same SEED), just filtered to these two sources,
# so the per-source validation numbers already seen for those sources
# stay directly comparable before/after fine-tuning.
FINE_TUNE_SOURCES = ("experimental", "simulated_close_to_experimental")
FINE_TUNE_MODEL_PATH = SPECTRAL_MODEL_PATH.replace(".pth", "_finetuned.pth")
# Lowered from 1.0e-5: a real run's base checkpoint had its best epoch at an
# LR of 6.25e-6 (ReduceLROnPlateau had already decayed it down from
# LEARNING_RATE=5.0e-5 by then), so the old 1.0e-5 fine-tune LR was actually
# LARGER than the step size the optimizer had already converged to -- the
# opposite of "gentle specialization." That run's fine-tuning never beat its
# epoch-1 validation MAE at all, consistent with those first fine-tune steps
# being big enough to kick the weights out of the minimum the base
# checkpoint was already sitting in. Set below the base run's own converged
# LR so fine-tuning steps are always smaller than what training had settled
# into -- check your own base checkpoint's "best_epoch_learning_rate" (only
# present if you're on the version of this script with the LR-mismatch
# warning) and keep this below it.
FINE_TUNE_LEARNING_RATE = 2.0e-6
FINE_TUNE_NUM_EPOCHS = 50

# Lowered to match the same fix applied to SCHEDULER_PATIENCE/
# EARLY_STOPPING_PATIENCE above: a real run showed the main training loop
# overfitting fast after its validation minimum because the LR stayed
# high for too many further epochs. Fine-tuning's own pool is smaller
# than the main run's (only two sources, and the WeightedRandomSampler
# gives the tiny simulated_close_to_experimental group equal gradient
# mass to the much larger experimental group), so the same fast-overfit
# pattern is if anything more likely here, not less -- this was never
# revisited when the main run's patience was tightened.
FINE_TUNE_EARLY_STOPPING_PATIENCE = 8
FINE_TUNE_SCHEDULER_PATIENCE = 2

# Set True and run this file to fine-tune the checkpoint already saved at
# SPECTRAL_MODEL_PATH instead of training a new one from scratch.
RUN_FINE_TUNING = False

# Overwrites a PNG of the train/val MAE curve after every epoch, so
# progress can be checked by opening the file while training is still
# running rather than waiting for it to finish. Requires matplotlib;
# silently skipped (with one warning) if it is not installed.
PLOT_TRAINING_CURVE = True
TRAINING_CURVE_PATH = SPECTRAL_MODEL_PATH.replace(".pth", "_training_curve.png")
FINE_TUNE_TRAINING_CURVE_PATH = FINE_TUNE_MODEL_PATH.replace(".pth", "_training_curve.png")

# A real run showed train MAE falling fast while val MAE rose above the
# constant baseline within 2 epochs -- overfitting starting far earlier
# than this architecture should need. LEARNING_RATE and WEIGHT_DECAY were
# both loosened for that run; tightened back to slow the early optimizer
# steps and penalize large weights more, alongside the dropout increases
# below.
LEARNING_RATE = 5.0e-5
WEIGHT_DECAY = 1.0e-4

# A real run after the anti-overfitting pass above showed train MAE
# falling smoothly to ~1e-3 while val MAE oscillated between roughly
# 2.5e-3 and 5.8e-3 for the entire run, with only one lucky epoch (out of
# 55) landing meaningfully below the constant baseline -- not runaway
# overfitting anymore, but the weights are still bouncing around a
# decent solution rather than settling into one, so picking "the epoch
# with the lowest val MAE" is largely picking a noise draw. An
# exponential moving average of the weights, evaluated and checkpointed
# instead of the raw in-training weights, averages that noise out
# without adding any new loss term or architecture.
USE_EMA = True
EMA_DECAY = 0.999

NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()

# A real run showed val MAE bottom out at epoch 5 (just below the
# constant baseline), then climb for 5 straight epochs back past that
# baseline while train MAE kept falling smoothly the whole time -- real,
# sustained overfitting resuming after the minimum, not noise.
# SCHEDULER_PATIENCE=10 meant the LR would not have halved until epoch 15
# (10 epochs past best epoch 5), so the optimizer kept taking full-sized
# steps for 10 epochs while actively overfitting. Lowered so the LR backs
# off much sooner after validation stops improving, before the model has
# had that much room to drift. EARLY_STOPPING_PATIENCE scaled down to
# match (roughly 4x the scheduler patience, so ~3 LR halvings still get a
# chance before training gives up for good).
EARLY_STOPPING_PATIENCE = 12
SCHEDULER_PATIENCE = 3
MIN_DELTA = 1.0e-8
GRADIENT_CLIP_NORM = 2.0

SEED = 42
EPS = 1.0e-8

TEMPORAL_FILTERS_PER_KERNEL = 12
TEMPORAL_POOL_BINS = 4

# Full-TPSF branch capacity. A prior experiment narrowed this to 32 (while
# widening DEPTH_HIDDEN_DIM) to rebalance capacity toward the tail branch;
# both train AND validation MAE got worse, which is the signature of a
# genuine capacity bottleneck (the full-TPSF branch still needs enough
# room to characterize the top layer's contribution so the head can
# separate it from the bottom layer's), not of overfitting. Restored to
# its original width.
TEMPORAL_FEATURE_DIM = 48
# Raised from 0.10 alongside the WEIGHT_DECAY/LEARNING_RATE changes above
# to fight the same fast-overfitting pattern -- dropout in both the
# per-branch encoders and the head gives the model less room to memorize
# per-file idiosyncrasies rather than the underlying TPSF-to-mua relationship.
TEMPORAL_DROPOUT = 0.20

HEAD_DROPOUT = 0.30

USE_TRAINING_AUGMENTATION = True
# MAX_TIME_SHIFT (timing jitter) and ADDITIVE_NOISE_STD (per-bin
# measurement noise) model real physical variation between the sources'
# different IRFs and detectors. A multiplicative amplitude-jitter
# augmentation was tried too, but removed: it breaks the AUC-normalization
# property (each row's integral == a fixed constant) the input data has
# by construction, training on rows whose integral is no longer fixed --
# a regime that never occurs in this real, always-normalized data.
ADDITIVE_NOISE_STD = 0.001
MAX_TIME_SHIFT = 1

# Depth-resolved tail encoder settings. DEPTH_HIDDEN_DIM is widened beyond
# its original 32/48 to 64 to give the GRU enough capacity to carry a long
# window sequence (up to ~44 windows at N_TIME_GATES=450) without lossy
# compression; this was not implicated in the regression above, so it is
# kept. DEPTH_FEATURE_DIM -- the per-window embedding actually fed to the
# GRU at each step -- was the narrower of the two: at 16 it compressed
# each window's 24-dim raw input (20 bins + 3 slope/amplitude features +
# 1 depth-position coordinate) down more than the 64-dim hidden state it
# feeds would need, so a wider hidden state alone could not recover detail
# already discarded upstream. Widened to relieve that bottleneck.
WINDOW_BINS = 20
WINDOW_STRIDE = 10
DEPTH_FEATURE_DIM = 24
DEPTH_HIDDEN_DIM = 64

# How many of the deepest windows in each tail get direct supervision
# against the bottom-mua label. NUM_SUPERVISED_TAIL_WINDOWS=20 pulled
# 54-71% of each tail toward the single bottom-mua label and made results
# worse; reverted to 5, anchoring deep supervision to just the windows
# closest to the DTOF's final bin.
NUM_SUPERVISED_TAIL_WINDOWS = 5

# Exponent controlling how much more the auxiliary loss weights the
# deepest supervised window relative to the shallowest of the supervised
# set (weights increase with depth).
LATE_WEIGHT_POWER = 2.0

AUX_DEPTH_LOSS_WEIGHT = 0.3
# A separate variance penalty across the supervised windows (on top of the
# weighted-L1 term above) was tried and removed: the L1 term already pulls
# every supervised window toward the same single bottom-mua label, which
# already pushes those windows toward agreeing with each other as a side
# effect, so the variance penalty was redundant and just gave the
# optimizer one more way to overfit each training tail's exact shape.

# Spectral-smoothing settings. SPECTRAL_SMOOTHING_KERNEL_SIZE=11 also made
# results worse (likely flattening genuine peaks/troughs by pooling too
# wide a wavelength neighborhood); reverted to 7.
SPECTRAL_SMOOTHING_KERNEL_SIZE = 7

# "raw": plain L1/MAE in physical mua units (RawMuaLoss). Treats "off by X"
# the same regardless of whether the true value is near the low or high
# end of the target range, so on a wide dynamic range (bottom-mua targets
# here can span more than an order of magnitude) the optimizer can lower
# the average absolute error mostly by fitting the numerically larger
# targets, leaving proportionally worse accuracy at the low end.
# "relative": mean absolute percentage error against the true value
# (RelativeMuaLoss). Every wavelength's proportional accuracy counts
# equally regardless of its absolute magnitude, which matters when test
# phantoms span a wide mua range including low values.
LOSS_MODE = "relative"


# ============================================================
# 2. CONFIGURATION AND REPRODUCIBILITY
# ============================================================

if TIME_GATE_START != 0 or TIME_GATE_END != N_TIME_GATES:
    raise ValueError(
        f"This version must use the first {N_TIME_GATES} TPSF columns."
    )

if TEMPORAL_POOL_BINS < 1:
    raise ValueError("TEMPORAL_POOL_BINS must be at least one.")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 3. DATA LOADING
# ============================================================

def load_wavelength_vector():
    if not os.path.isfile(WAVELENGTH_FILE):
        raise FileNotFoundError(f"Wavelength file was not found:\n{WAVELENGTH_FILE}")

    mat = loadmat(WAVELENGTH_FILE)

    if WAVELENGTH_KEY not in mat:
        available = [key for key in mat if not key.startswith("__")]
        raise KeyError(
            f"'{WAVELENGTH_KEY}' was not found. Available variables: {available}"
        )

    wavelengths = np.asarray(mat[WAVELENGTH_KEY], dtype=np.float32).reshape(-1)

    if wavelengths.size != N_WAVELENGTHS:
        raise ValueError(
            f"Wavelength vector contains {wavelengths.size} values; "
            f"expected {N_WAVELENGTHS}."
        )

    if not np.all(np.isfinite(wavelengths)):
        raise FloatingPointError("Non-finite wavelengths were found.")

    wavelength_min = float(wavelengths.min())
    wavelength_max = float(wavelengths.max())

    if wavelength_max <= wavelength_min:
        raise ValueError("Wavelength range must be greater than zero.")

    normalized = (wavelengths - wavelength_min) / (wavelength_max - wavelength_min)

    raw = wavelengths.reshape(N_WAVELENGTHS, 1).astype(np.float32)
    normalized = normalized.reshape(N_WAVELENGTHS, 1).astype(np.float32)

    print("\nWavelength information")
    print(f"Variable: {WAVELENGTH_KEY}")
    print(f"Range: {wavelength_min:.6f} to {wavelength_max:.6f}")

    return raw, normalized


def clean_tpsf_array(x, file_path):
    x = np.asarray(x)

    if x.ndim != 2:
        raise ValueError(f"TPSF in '{file_path}' must be 2D; received {x.shape}.")

    if x.shape[0] == N_WAVELENGTHS:
        oriented = x
    elif x.shape[1] == N_WAVELENGTHS:
        oriented = x.T
    else:
        raise ValueError(
            f"TPSF in '{file_path}' has shape {x.shape}; one dimension "
            f"must equal {N_WAVELENGTHS}."
        )

    if oriented.shape[1] < N_TIME_GATES:
        raise ValueError(
            f"TPSF in '{file_path}' has only {oriented.shape[1]} time gates; "
            f"at least {N_TIME_GATES} are required."
        )

    oriented = oriented[:, TIME_GATE_START:TIME_GATE_END]
    oriented = np.asarray(oriented, dtype=np.float32)
    oriented = np.nan_to_num(oriented, nan=0.0, posinf=0.0, neginf=0.0)
    oriented = np.clip(oriented, 0.0, None)

    expected = (N_WAVELENGTHS, N_TIME_GATES)
    if oriented.shape != expected:
        raise RuntimeError(f"Selected TPSF shape is {oriented.shape}; expected {expected}.")

    return oriented


def load_training_tpsf(mat, file_path):
    if TPSF_KEY not in mat:
        raise KeyError(f"'{TPSF_KEY}' was not found in:\n{file_path}")
    return clean_tpsf_array(mat[TPSF_KEY], file_path)


def load_bottom_mua_target(mat, file_path):
    if BOTTOM_MUA_KEY not in mat:
        raise KeyError(f"'{BOTTOM_MUA_KEY}' was not found in:\n{file_path}")

    target = np.asarray(mat[BOTTOM_MUA_KEY])

    if target.ndim == 1:
        target = target.reshape(-1, 1)
    elif target.ndim == 2 and target.shape[0] == 1:
        target = target.T

    if target.ndim != 2 or target.shape[0] < N_WAVELENGTHS:
        raise ValueError(
            f"'{BOTTOM_MUA_KEY}' in '{file_path}' has shape {target.shape}; "
            f"expected at least [{N_WAVELENGTHS},1]."
        )

    target = target[:N_WAVELENGTHS, :1].astype(np.float32)
    target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)

    if np.any(target <= 0.0):
        raise ValueError(
            "Bottom-mua labels must be strictly positive. "
            f"Non-positive values were found in:\n{file_path}"
        )

    return target


def load_late_start_indices(mat, file_path):
    """Load one late-start index for each of the 169 TPSFs.

    Returned values are zero-based indices relative to the selected
    [0:N_TIME_GATES] time window and have shape [169,1].
    """

    if LATE_START_KEY not in mat:
        available = [key for key in mat if not key.startswith("__")]
        raise KeyError(
            f"'{LATE_START_KEY}' was not found in:\n{file_path}\n"
            f"Available variables: {available}"
        )

    late_start = np.asarray(mat[LATE_START_KEY])

    if late_start.ndim == 1:
        late_start = late_start.reshape(-1, 1)
    elif late_start.ndim == 2 and late_start.shape[0] == 1:
        late_start = late_start.T

    if late_start.ndim != 2 or late_start.shape[0] < N_WAVELENGTHS:
        raise ValueError(
            f"'{LATE_START_KEY}' in '{file_path}' has shape "
            f"{late_start.shape}; expected [169,1] or [1,169]."
        )

    late_start = late_start[:N_WAVELENGTHS, :1].astype(np.float64)

    if not np.all(np.isfinite(late_start)):
        raise FloatingPointError(f"Non-finite late-start indices were found in:\n{file_path}")

    rounded = np.rint(late_start)

    if np.max(np.abs(late_start - rounded)) > 1.0e-4:
        raise ValueError(f"Late-start values must be integer indices in:\n{file_path}")

    late_start = rounded.astype(np.int64)

    if LATE_START_IS_MATLAB_ONE_BASED:
        late_start = late_start - 1

    # Convert indices on the original time axis to indices inside the crop.
    late_start = late_start - TIME_GATE_START

    if np.any(late_start < 0) or np.any(late_start >= N_TIME_GATES):
        invalid_min = int(late_start.min())
        invalid_max = int(late_start.max())
        raise ValueError(
            f"Converted late-start indices in '{file_path}' range from "
            f"{invalid_min} to {invalid_max}; valid Python indices are "
            f"0 to {N_TIME_GATES - 1}. Check whether your indices are "
            f"MATLAB one-based and whether they refer to the first {N_TIME_GATES} gates."
        )

    return late_start.reshape(N_WAVELENGTHS, 1)


# ============================================================
# 4. TARGET SUMMARY STATISTICS
# ============================================================

def summarize_raw_target(file_list):
    """Print raw bottom-mua statistics. No target transform is applied."""
    all_targets = []

    for file_path in file_list:
        all_targets.append(load_bottom_mua_target(loadmat(file_path), file_path))

    targets = np.concatenate(all_targets, axis=0).astype(np.float32)

    print("\nBottom-mua raw target statistics")
    print(f"Minimum:        {targets.min():.6e}")
    print(f"Maximum:        {targets.max():.6e}")
    print(f"Mean:           {targets.mean():.6e}")
    print(f"Median:         {np.median(targets):.6e}")
    print(f"Standard dev.:  {targets.std():.6e}")

    if targets.std() < 1.0e-8:
        raise RuntimeError(
            "Bottom-mua variation is almost zero; meaningful regression is not possible."
        )


def summarize_late_start_indices(file_list):
    values = []

    for file_path in file_list:
        mat = loadmat(file_path)
        values.append(load_late_start_indices(mat, file_path))

    values = np.concatenate(values, axis=0).reshape(-1)

    print("\nLate-start statistics after conversion to Python indices")
    print(f"Minimum: {values.min()}")
    print(f"Median:  {np.median(values):.1f}")
    print(f"Maximum: {values.max()}")
    print(f"Mean:    {values.mean():.2f}")

    return {
        "minimum": int(values.min()),
        "median": float(np.median(values)),
        "maximum": int(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }


def calculate_constant_baseline(train_files, val_files):
    train_values = []
    val_values = []

    for file_path in train_files:
        train_values.append(load_bottom_mua_target(loadmat(file_path), file_path))

    for file_path in val_files:
        val_values.append(load_bottom_mua_target(loadmat(file_path), file_path))

    train_values = np.concatenate(train_values, axis=0)
    val_values = np.concatenate(val_values, axis=0)

    constant_prediction = float(np.median(train_values))
    baseline_mae = float(np.mean(np.abs(val_values - constant_prediction)))

    return constant_prediction, baseline_mae


# ============================================================
# 4b. SOURCE-AWARE SPLIT
# ============================================================

FILE_NUMBER_PATTERN = re.compile(r"(\d+)(?=\.mat$)")


def extract_file_number(file_path):
    """Extract the integer index embedded in a filename like DTOF_137.mat.

    Source classification must key off this actual number, not off
    position in a sorted(glob(...)) list: Python's sorted() on these
    paths sorts lexicographically as strings, so e.g. "DTOF_961.mat"
    lands alphabetically among all the "DTOF_9xx.mat"/"DTOF_9xxx.mat"
    names rather than next to its true numeric neighbors DTOF_960.mat and
    DTOF_962.mat. That silently scrambles any classification based on
    sorted-list position -- exactly what SOURCE_SIMULATED_END /
    SOURCE_EXPERIMENTAL_END assume they can use.
    """

    basename = os.path.basename(file_path)
    match = FILE_NUMBER_PATTERN.search(basename)
    if match is None:
        raise ValueError(
            f"Could not find a numeric file index in filename: {basename}"
        )
    return int(match.group(1))


def classify_source_by_file_number(file_number):
    """Maps a file's own numeric index to its data source, per the ranges
    documented next to SOURCE_SIMULATED_END above.
    """

    if file_number <= SOURCE_SIMULATED_END:
        return "simulated"
    if file_number <= SOURCE_EXPERIMENTAL_END:
        return "experimental"
    return "simulated_close_to_experimental"


def build_source_lookup(all_files):
    """Returns {file_path: source_label}, classifying each file by the
    numeric index parsed from its own filename (see extract_file_number),
    and prints a sanity check of the file-number range actually observed
    in each source so you can confirm it against what you expect before
    trusting the split.
    """

    file_numbers = {file_path: extract_file_number(file_path) for file_path in all_files}

    if len(all_files) != SOURCE_TOTAL_EXPECTED:
        print(
            f"\nWARNING: found {len(all_files)} training files but "
            f"SOURCE_TOTAL_EXPECTED={SOURCE_TOTAL_EXPECTED}. The source "
            f"boundaries (SOURCE_SIMULATED_END, SOURCE_EXPERIMENTAL_END) "
            f"were set for a different file count -- update them or this "
            f"split will misclassify files."
        )

    lookup = {
        file_path: classify_source_by_file_number(file_numbers[file_path])
        for file_path in all_files
    }

    print("\nSource boundary sanity check (by each file's own parsed number,")
    print("not sorted-list position -- confirm these ranges match what you expect):")
    for label in ("simulated", "experimental", "simulated_close_to_experimental"):
        numbers = sorted(file_numbers[f] for f in all_files if lookup[f] == label)
        if numbers:
            print(f"  {label}: {len(numbers)} files, numbers {numbers[0]} to {numbers[-1]}")
        else:
            print(f"  {label}: 0 files")

    return lookup


def stratified_train_val_split(all_files, source_lookup, train_fraction=TRAIN_FRACTION, seed=SEED):
    """Splits each data source separately so validation always contains a
    deliberate share of every source, instead of whatever a single random
    shuffle across all files happens to produce.
    """

    groups = {}
    for file_path in all_files:
        groups.setdefault(source_lookup[file_path], []).append(file_path)

    rng = random.Random(seed)
    train_files, val_files = [], []

    print("\nSource-aware train/validation split")
    for label, files in sorted(groups.items()):
        files = list(files)
        rng.shuffle(files)

        val_fraction = SOURCE_VAL_FRACTION_OVERRIDES.get(label, 1.0 - train_fraction)
        n_val = int(round(val_fraction * len(files)))
        if val_fraction > 0.0:
            n_val = max(1, n_val)
        n_val = min(n_val, len(files) - 1) if len(files) > 1 else 0

        group_val = files[:n_val]
        group_train = files[n_val:]

        train_files.extend(group_train)
        val_files.extend(group_val)

        print(f"  {label}: {len(files)} files total -> "
              f"{len(group_train)} train / {len(group_val)} val")

    rng.shuffle(train_files)
    rng.shuffle(val_files)

    return train_files, val_files, groups


def summarize_late_start_by_source(all_files, source_lookup):
    """Prints LATE_START statistics separately per source. A large shift
    between sources here is the IRF difference showing up directly in the
    one quantity the tail-weighting/window mechanics depend on most.
    """

    by_source = {}
    for file_path in all_files:
        mat = loadmat(file_path)
        values = load_late_start_indices(mat, file_path).reshape(-1)
        by_source.setdefault(source_lookup[file_path], []).append(values)

    print("\nLate-start statistics by source")
    for label, value_list in sorted(by_source.items()):
        values = np.concatenate(value_list)
        print(
            f"  {label}: min={values.min()}, median={np.median(values):.1f}, "
            f"max={values.max()}, mean={values.mean():.2f}"
        )


def summarize_raw_target_by_source(all_files, source_lookup):
    """Prints bottom-mua target statistics separately per source.

    Raw per-source validation MAE (printed each epoch) is an absolute
    error in physical mua units. If one source's targets simply sit at a
    smaller absolute magnitude than another's, its raw MAE will read
    lower even at the same *relative* accuracy -- this print lets you
    check that directly instead of assuming "lower raw MAE" means "more
    accurate" when comparing across sources.
    """

    by_source = {}
    for file_path in all_files:
        target = load_bottom_mua_target(loadmat(file_path), file_path).reshape(-1)
        by_source.setdefault(source_lookup[file_path], []).append(target)

    print("\nBottom-mua raw target statistics by source")
    for label, value_list in sorted(by_source.items()):
        values = np.concatenate(value_list)
        print(
            f"  {label}: min={values.min():.6e}, median={np.median(values):.6e}, "
            f"max={values.max():.6e}, mean={values.mean():.6e}"
        )


# ============================================================
# 5. DATASET
# ============================================================

def shift_time_axis(tpsf, shift):
    if shift == 0:
        return tpsf

    shifted = np.zeros_like(tpsf)

    if shift > 0:
        shifted[:, shift:] = tpsf[:, :-shift]
    else:
        shifted[:, :shift] = tpsf[:, -shift:]

    return shifted


def augment_scaled_tpsf(tpsf, late_start):
    augmented = np.asarray(tpsf, dtype=np.float32).copy()
    adjusted_late_start = np.asarray(late_start, dtype=np.int64).copy()

    if MAX_TIME_SHIFT > 0:
        shift = int(np.random.randint(-MAX_TIME_SHIFT, MAX_TIME_SHIFT + 1))
        augmented = shift_time_axis(augmented, shift)
        adjusted_late_start = np.clip(adjusted_late_start + shift, 0, N_TIME_GATES - 1)

    if ADDITIVE_NOISE_STD > 0.0:
        noise = np.random.normal(0.0, ADDITIVE_NOISE_STD, size=augmented.shape).astype(np.float32)
        augmented += noise

    augmented = np.clip(augmented, 0.0, None).astype(np.float32)
    return augmented, adjusted_late_start


class SpectralTPSFDataset(Dataset):
    """Loads one image's [N_WAVELENGTHS, N_TIME_GATES] TPSF (already
    AUC-normalized per wavelength upstream), target, and LATE_START
    indices, with optional augmentation.
    """

    def __init__(self, file_list, wavelength_values, augment=False):
        self.file_list = list(file_list)
        self.augment = bool(augment)

        wavelength_values = np.asarray(wavelength_values, dtype=np.float32)
        expected_wavelength_shape = (N_WAVELENGTHS, 1)
        if wavelength_values.shape != expected_wavelength_shape:
            raise ValueError(
                f"wavelength_values has shape {wavelength_values.shape}; "
                f"expected {expected_wavelength_shape}."
            )
        self.wavelengths = torch.from_numpy(wavelength_values)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_path = self.file_list[index]
        mat = loadmat(file_path)

        tpsf = load_training_tpsf(mat, file_path)
        target = load_bottom_mua_target(mat, file_path)
        late_start = load_late_start_indices(mat, file_path)

        if self.augment:
            tpsf, late_start = augment_scaled_tpsf(tpsf, late_start)

        return (
            torch.from_numpy(tpsf),
            torch.from_numpy(target),
            self.wavelengths.clone(),
            torch.from_numpy(late_start.astype(np.int64)),
        )


def flatten_image_batch(tpsf, target=None, wavelength=None, late_start=None):
    tpsf_flat = tpsf.reshape(-1, N_TIME_GATES)

    target_flat = None if target is None else target.reshape(-1, 1)
    wavelength_flat = None if wavelength is None else wavelength.reshape(-1, 1)
    late_start_flat = None if late_start is None else late_start.reshape(-1, 1)

    return tpsf_flat, target_flat, wavelength_flat, late_start_flat


def build_per_source_val_loaders(val_files, source_lookup, wavelength_values, labels=None):
    """One DataLoader per data source found in val_files (or just the
    sources in `labels`, if given), for per-source diagnostic reporting.
    """

    loaders = {}
    label_set = labels if labels is not None else sorted(set(source_lookup.values()))
    for label in label_set:
        source_val_files = [f for f in val_files if source_lookup[f] == label]
        if not source_val_files:
            continue
        source_val_dataset = SpectralTPSFDataset(
            file_list=source_val_files,
            wavelength_values=wavelength_values,
            augment=False,
        )
        loaders[label] = DataLoader(
            source_val_dataset, batch_size=IMAGE_BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False,
        )
    return loaders


# ============================================================
# 6. RAW TPSF CHANNEL
# ============================================================

def build_raw_tpsf_channel(tpsf):
    """Convert [N,N_TIME_GATES] raw TPSFs to [N,1,N_TIME_GATES]."""

    if tpsf.ndim != 2 or tpsf.shape[1] != N_TIME_GATES:
        raise ValueError(f"Expected [N,{N_TIME_GATES}], received {tuple(tpsf.shape)}.")

    tpsf = torch.nan_to_num(tpsf, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.clamp(tpsf, min=0.0).unsqueeze(1)


# ============================================================
# 7. FULL-TPSF CONVOLUTION ENCODER
# ============================================================

def choose_group_count(number_of_channels):
    for candidate in (4, 3, 2):
        if number_of_channels % candidate == 0:
            return candidate
    return 1


class TemporalEncoder(nn.Module):
    """Multi-kernel 1D convolution encoder for the full raw TPSF."""

    def __init__(
        self,
        filters_per_kernel=TEMPORAL_FILTERS_PER_KERNEL,
        pool_bins=TEMPORAL_POOL_BINS,
        feature_dim=TEMPORAL_FEATURE_DIM,
        dropout=TEMPORAL_DROPOUT,
    ):
        super().__init__()

        groups = choose_group_count(filters_per_kernel)

        def branch(kernel_size):
            return nn.Sequential(
                nn.Conv1d(
                    in_channels=1,
                    out_channels=filters_per_kernel,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=False,
                ),
                nn.GroupNorm(groups, filters_per_kernel),
                nn.ReLU(),
            )

        self.small_filter = branch(3)
        self.medium_filter = branch(7)
        self.large_filter = branch(15)
        self.pool_bins = int(pool_bins)

        projection_input = 3 * 2 * filters_per_kernel * self.pool_bins

        self.output_projection = nn.Sequential(
            nn.Linear(projection_input, 96),
            nn.LayerNorm(96),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(96, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
        )

    def pool(self, x):
        average = F.adaptive_avg_pool1d(x, self.pool_bins)
        maximum = F.adaptive_max_pool1d(x, self.pool_bins)
        return torch.cat([average, maximum], dim=1).flatten(1)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError(f"TemporalEncoder expects [N,1,T], received {tuple(x.shape)}.")

        features = torch.cat(
            [
                self.pool(self.small_filter(x)),
                self.pool(self.medium_filter(x)),
                self.pool(self.large_filter(x)),
            ],
            dim=1,
        )

        return self.output_projection(features)


# ============================================================
# 8. DEPTH-RESOLVED TAIL ENCODER
# ============================================================

def build_tail_windows(raw_tail, window_bins=WINDOW_BINS, stride=WINDOW_STRIDE):
    """Slice a raw tail [N,T] into an ordered window sequence [N,K,window_bins].

    Windows are ordered from the earliest (shallowest) part of the tail to
    the latest (deepest) part. If the tail is shorter than one window, a
    single left-zero-padded window covering the whole tail is returned.
    """

    if raw_tail.ndim != 2:
        raise ValueError(f"Expected [N,T], received {tuple(raw_tail.shape)}.")

    n_samples, tail_length = raw_tail.shape

    if tail_length <= window_bins:
        padded = raw_tail.new_zeros((n_samples, window_bins))
        padded[:, window_bins - tail_length:] = raw_tail
        return padded.unsqueeze(1)

    starts = list(range(0, tail_length - window_bins + 1, stride))
    if starts[-1] != tail_length - window_bins:
        starts.append(tail_length - window_bins)

    windows = torch.stack(
        [raw_tail[:, start:start + window_bins] for start in starts],
        dim=1,
    )

    return windows


def compute_window_slope_features(windows, eps=EPS):
    """Physics-informed per-window features.

    windows: [N,K,window_bins]
    returns: [N,K,3] -> (log-amplitude decay slope, mean amplitude, peak
    amplitude). The slope is a direct diffusion-theory estimator of
    absorption for late enough windows: ln(I(t)) ~= -mua * v * t + const.
    """

    n_samples, n_windows, window_bins = windows.shape

    log_amplitude = torch.log1p(windows.clamp_min(0.0))

    time_index = torch.arange(window_bins, device=windows.device, dtype=windows.dtype)
    time_centered = time_index - time_index.mean()
    time_variance = time_centered.pow(2).sum().clamp_min(eps)

    mean_log_amplitude = log_amplitude.mean(dim=2, keepdim=True)
    slope = (
        (log_amplitude - mean_log_amplitude) * time_centered.view(1, 1, -1)
    ).sum(dim=2) / time_variance

    mean_amplitude = windows.mean(dim=2)
    peak_amplitude = windows.amax(dim=2)

    features = torch.stack([slope, mean_amplitude, peak_amplitude], dim=2)

    return torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


class DepthResolvedTailEncoder(nn.Module):
    """Sequence model over ordered tail windows producing a depth profile."""

    def __init__(
        self,
        window_bins=WINDOW_BINS,
        depth_feature_dim=DEPTH_FEATURE_DIM,
        hidden_dim=DEPTH_HIDDEN_DIM,
        dropout=TEMPORAL_DROPOUT,
    ):
        super().__init__()

        # Raw window samples + 3 slope/amplitude features + 1 normalized
        # depth-position coordinate.
        window_input_dim = window_bins + 3 + 1

        self.window_encoder = nn.Sequential(
            nn.Linear(window_input_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, depth_feature_dim),
            nn.LayerNorm(depth_feature_dim),
            nn.ReLU(),
        )

        self.sequence_model = nn.GRU(
            input_size=depth_feature_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.depth_readout = nn.Linear(hidden_dim, 1)

    def forward(self, raw_tail):
        """raw_tail: [N,T] (no channel dimension).

        Returns:
            final_feature: [N, hidden_dim], the deepest hidden state.
            depth_profile: [N, K], a non-negative absorption estimate at
                every window, ordered shallow -> deep.
        """

        windows = build_tail_windows(raw_tail)
        n_samples, n_windows, window_bins = windows.shape

        slope_features = compute_window_slope_features(windows)

        depth_position = torch.linspace(
            0.0, 1.0, n_windows, device=raw_tail.device, dtype=raw_tail.dtype
        ).view(1, n_windows, 1).expand(n_samples, -1, -1)

        window_input = torch.cat([windows, slope_features, depth_position], dim=2)
        window_features = self.window_encoder(window_input)

        hidden_sequence, _ = self.sequence_model(window_features)

        depth_profile = F.softplus(self.depth_readout(hidden_sequence)).squeeze(-1)
        final_feature = hidden_sequence[:, -1, :]

        return final_feature, depth_profile


def depth_profile_auxiliary_loss(supervised_profiles, target, late_weight_power=LATE_WEIGHT_POWER):
    """Deep-supervision loss on the deepest windows of the depth profile.

    supervised_profiles: [N, L], shallow-to-deep ordering (column -1 is the
        single deepest window).
    target: [N, 1], the bottom-mua label.

    Pulls every supervised window toward the bottom-mua label, with
    weight increasing toward the deepest window.
    """

    n_windows = supervised_profiles.shape[1]

    weights = torch.linspace(
        1.0, float(n_windows), n_windows,
        device=supervised_profiles.device, dtype=supervised_profiles.dtype,
    ).pow(late_weight_power)
    weights = weights / weights.sum()

    error = supervised_profiles - target
    return (weights.view(1, -1) * error.abs()).sum(dim=1).mean()


# ============================================================
# 9. SPECTRAL SMOOTHER
# ============================================================

class SpectralSmoother(nn.Module):
    """Residual depthwise-separable smoothing across the wavelength axis.

    Operates on the fused per-wavelength feature vectors of one image at a
    time (shape [N_WAVELENGTHS, feature_dim]), giving every wavelength's
    prediction access to a local neighborhood of the other wavelengths'
    evidence. Zero-initialized so it starts as the identity function.
    """

    def __init__(self, feature_dim, kernel_size=SPECTRAL_SMOOTHING_KERNEL_SIZE, n_wavelengths=N_WAVELENGTHS):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for symmetric padding.")

        self.feature_dim = int(feature_dim)
        self.n_wavelengths = int(n_wavelengths)
        padding = kernel_size // 2

        self.depthwise = nn.Conv1d(
            feature_dim, feature_dim, kernel_size=kernel_size,
            padding=padding, groups=feature_dim, bias=False,
        )
        self.norm = nn.GroupNorm(choose_group_count(feature_dim), feature_dim)
        self.pointwise = nn.Conv1d(feature_dim, feature_dim, kernel_size=1)

        # Zero-initialized pointwise mixing: at the start of training the
        # residual branch contributes nothing, so this module is exactly
        # the identity until training shows the mixing helps.
        nn.init.zeros_(self.pointwise.weight)
        nn.init.zeros_(self.pointwise.bias)

    def forward(self, fused_features):
        n_total, feature_dim = fused_features.shape

        if feature_dim != self.feature_dim:
            raise ValueError(
                f"Expected feature dimension {self.feature_dim}, received {feature_dim}."
            )

        if n_total % self.n_wavelengths != 0:
            raise ValueError(
                f"Batch size {n_total} is not a multiple of {self.n_wavelengths} "
                "wavelengths; spectral mixing requires whole images (all "
                "wavelengths of an image present together in the batch)."
            )

        n_images = n_total // self.n_wavelengths

        sequence = fused_features.view(n_images, self.n_wavelengths, feature_dim)
        sequence = sequence.transpose(1, 2)  # [n_images, feature_dim, n_wavelengths]

        residual = self.pointwise(F.relu(self.norm(self.depthwise(sequence))))
        mixed = sequence + residual

        mixed = mixed.transpose(1, 2).reshape(n_total, feature_dim)
        return mixed


# ============================================================
# 10. FULL MODEL
# ============================================================

class BottomMuaSpectralNet(nn.Module):
    """Full-TPSF features + depth-resolved tail sequence + raw wavelength
    scalar, spectrally smoothed across the wavelength axis before the
    regression head.

    The wavelength enters the head as a direct scalar rather than through a
    learned embedding, so the regression head is supervised directly by the
    true wavelength of each TPSF instead of by an intermediate learned
    representation the model invents on its own.
    """

    def __init__(
        self,
        temporal_feature_dim=TEMPORAL_FEATURE_DIM,
        filters_per_kernel=TEMPORAL_FILTERS_PER_KERNEL,
        pool_bins=TEMPORAL_POOL_BINS,
        temporal_dropout=TEMPORAL_DROPOUT,
        head_dropout=HEAD_DROPOUT,
        depth_hidden_dim=DEPTH_HIDDEN_DIM,
        num_supervised_windows=NUM_SUPERVISED_TAIL_WINDOWS,
        spectral_kernel_size=SPECTRAL_SMOOTHING_KERNEL_SIZE,
    ):
        super().__init__()

        self.tail_hidden_dim = int(depth_hidden_dim)
        self.num_supervised_windows = int(num_supervised_windows)

        self.full_encoder = TemporalEncoder(
            filters_per_kernel=filters_per_kernel,
            pool_bins=pool_bins,
            feature_dim=temporal_feature_dim,
            dropout=temporal_dropout,
        )

        self.tail_sequence_encoder = DepthResolvedTailEncoder(
            hidden_dim=depth_hidden_dim,
            dropout=temporal_dropout,
        )

        # +1 for the raw wavelength scalar, concatenated directly (no
        # learned encoder) below.
        fused_dim = temporal_feature_dim + depth_hidden_dim + 1

        self.spectral_smoother = SpectralSmoother(fused_dim, kernel_size=spectral_kernel_size)

        self.head = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        nn.init.normal_(self.head[-1].weight, mean=0.0, std=1.0e-3)
        nn.init.constant_(self.head[-1].bias, 0.0)

    def encode_raw_tails(self, tpsf, late_start):
        """Group TPSFs sharing a LATE_START and run the depth-resolved
        sequence encoder on each group's exact raw tail.

        Returns:
            final_features: [N, tail_hidden_dim]
            supervised_profiles: [N, num_supervised_windows], the deepest
                windows of each sample's depth profile, right-aligned so
                column -1 is always the single deepest window regardless
                of how many windows a given tail produced.
        """

        starts = late_start.reshape(-1).to(device=tpsf.device, dtype=torch.long)
        sample_count = tpsf.shape[0]

        if starts.numel() != sample_count:
            raise ValueError(
                "The number of LATE_START values does not match the number of TPSFs."
            )

        if torch.any(starts < 0) or torch.any(starts >= N_TIME_GATES):
            raise ValueError(f"LATE_START values must be within 0..{N_TIME_GATES - 1}.")

        final_features = tpsf.new_zeros((sample_count, self.tail_hidden_dim))
        supervised_profiles = tpsf.new_zeros((sample_count, self.num_supervised_windows))

        for unique_start in torch.unique(starts, sorted=True):
            start = int(unique_start.item())

            sample_indices = torch.nonzero(starts == unique_start, as_tuple=False).reshape(-1)

            raw_tail = tpsf[sample_indices, start:]

            final_feature, depth_profile = self.tail_sequence_encoder(raw_tail)

            n_windows = depth_profile.shape[1]
            target_length = self.num_supervised_windows

            if n_windows >= target_length:
                group_supervised = depth_profile[:, -target_length:]
            else:
                pad_width = target_length - n_windows
                left_pad = depth_profile[:, :1].expand(-1, pad_width)
                group_supervised = torch.cat([left_pad, depth_profile], dim=1)

            final_features = final_features.index_copy(0, sample_indices, final_feature)
            supervised_profiles = supervised_profiles.index_copy(0, sample_indices, group_supervised)

        return final_features, supervised_profiles

    def forward(self, tpsf, wavelength, late_start):
        full_channels = build_raw_tpsf_channel(tpsf)
        full_features = self.full_encoder(full_channels)

        tail_final_features, supervised_profiles = self.encode_raw_tails(tpsf, late_start)

        wavelength_input = wavelength.to(dtype=full_features.dtype)

        fused_features = torch.cat(
            [full_features, tail_final_features, wavelength_input], dim=1
        )
        fused_features = self.spectral_smoother(fused_features)

        prediction = self.head(fused_features)

        return prediction, supervised_profiles


# ============================================================
# 11. RAW TARGET LOSS
# ============================================================

class RawMuaLoss(nn.Module):
    """Direct MAE loss in the original bottom-mua physical units."""

    def forward(self, prediction, target):
        return F.l1_loss(prediction, target, reduction="mean")


class RelativeMuaLoss(nn.Module):
    """Mean absolute percentage error against the true bottom-mua value.

    Bottom-mua targets here can span more than an order of magnitude (see
    summarize_raw_target). Plain L1/MAE in raw units treats "off by X" the
    same regardless of whether the true value is near the low or high end
    of that range, so the optimizer has no incentive to be proportionally
    accurate at the low end -- it can minimize the average absolute error
    mostly by fitting the numerically larger targets. Dividing by the true
    value instead makes every wavelength's relative accuracy count
    equally, independent of its absolute magnitude.
    """

    def __init__(self, eps=EPS):
        super().__init__()
        self.eps = float(eps)

    def forward(self, prediction, target):
        relative_error = (prediction - target).abs() / target.clamp_min(self.eps)
        return relative_error.mean()


def build_loss(loss_mode=LOSS_MODE):
    if loss_mode == "raw":
        return RawMuaLoss()
    if loss_mode == "relative":
        return RelativeMuaLoss()
    raise ValueError(f"Unknown LOSS_MODE: {loss_mode!r}. Expected 'raw' or 'relative'.")


# ============================================================
# 12. VALIDATION
# ============================================================

@torch.no_grad()
def validate_spectral(model, loader, criterion, device):
    """Note on comparing bottom_mua_mae across sources: it is an absolute
    error in raw physical mua units, so a lower value for one source does
    not by itself mean the model is more accurate there -- it could just
    mean that source's targets sit at a smaller absolute magnitude (see
    summarize_raw_target_by_source). bottom_mua_relative_mae (mean
    |error|/target, matching RelativeMuaLoss) is scale-independent and is
    the fairer number for cross-source comparisons.

    bottom_mua_bias is the mean SIGNED error (prediction - target, not
    absolute), and bottom_mua_relative_bias is the same divided by target.
    MAE/RMSE/RelMAE all average |error|, which erases the sign -- a model
    that is consistently biased in one direction and one that is unbiased
    but noisy can show the same MAE. Bias is the number that actually
    tells them apart, e.g. to check whether fine-tuning corrected a
    systematic over/under-prediction rather than just changed the scatter.
    """

    model.eval()

    loss_sum = 0.0
    absolute_error_sum = 0.0
    squared_error_sum = 0.0
    relative_error_sum = 0.0
    signed_error_sum = 0.0
    signed_relative_error_sum = 0.0
    value_count = 0

    for tpsf, target, wavelength, late_start in loader:
        tpsf = tpsf.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        wavelength = wavelength.to(device, non_blocking=True)
        late_start = late_start.to(device, non_blocking=True)

        tpsf, target, wavelength, late_start = flatten_image_batch(
            tpsf, target, wavelength, late_start
        )

        prediction, _ = model(tpsf, wavelength, late_start)
        loss = criterion(prediction, target)

        error = prediction - target
        count = error.numel()

        loss_sum += loss.item() * count
        absolute_error_sum += torch.abs(error).sum().item()
        squared_error_sum += torch.square(error).sum().item()
        relative_error_sum += (torch.abs(error) / target.clamp_min(EPS)).sum().item()
        signed_error_sum += error.sum().item()
        signed_relative_error_sum += (error / target.clamp_min(EPS)).sum().item()
        value_count += count

    return {
        "raw_loss": loss_sum / value_count,
        "bottom_mua_mae": absolute_error_sum / value_count,
        "bottom_mua_rmse": np.sqrt(squared_error_sum / value_count),
        "bottom_mua_relative_mae": relative_error_sum / value_count,
        "bottom_mua_bias": signed_error_sum / value_count,
        "bottom_mua_relative_bias": signed_relative_error_sum / value_count,
    }


# ============================================================
# 12b. LIVE TRAINING CURVE PLOT
# ============================================================

_warned_matplotlib_missing = False


def update_training_curve_plot(train_history, val_history, best_epoch, output_path, title):
    """Overwrite a PNG of train vs. val MAE so far.

    Called at the end of every epoch (see train_spectral_model and
    fine_tune_on_target_domains below), so opening output_path while
    training is still running shows current progress -- no need to wait
    for training to finish or reload a checkpoint afterward.
    """

    global _warned_matplotlib_missing

    if not PLOT_TRAINING_CURVE:
        return

    if not MATPLOTLIB_AVAILABLE:
        if not _warned_matplotlib_missing:
            print("\nmatplotlib is not installed; skipping live training curve plot.")
            _warned_matplotlib_missing = True
        return

    epochs = list(range(1, len(train_history) + 1))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(epochs, train_history, label="Train MAE", color="tab:blue")
    ax.plot(epochs, val_history, label="Val MAE", color="tab:orange")

    if best_epoch and 1 <= best_epoch <= len(epochs):
        ax.axvline(best_epoch, color="tab:green", linestyle="--", alpha=0.6,
                    label=f"Best epoch ({best_epoch})")

    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Bottom-mua MAE (log scale)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ============================================================
# 12c. EXPONENTIAL MOVING AVERAGE OF WEIGHTS
# ============================================================

def create_ema_model(model):
    """A frozen, eval-mode copy of model to hold the running weight
    average. See USE_EMA's comment above for why this exists.
    """

    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for param in ema_model.parameters():
        param.requires_grad_(False)
    return ema_model


@torch.no_grad()
def update_ema_model(ema_model, model, decay=EMA_DECAY):
    """Called once per optimizer step: ema_weight <- decay*ema_weight +
    (1-decay)*current_weight. Buffers (e.g. GroupNorm running stats, if
    any existed) are copied directly rather than averaged.
    """

    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.mul_(decay).add_(param.detach(), alpha=1.0 - decay)
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.copy_(buffer)


# ============================================================
# 13. TRAINING
# ============================================================

def train_spectral_model():
    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(f"Training directory was not found:\n{TRAIN_DIR}")

    raw_wavelengths, normalized_wavelengths = load_wavelength_vector()

    wavelength_absolute_path = os.path.normcase(os.path.abspath(WAVELENGTH_FILE))

    all_files = []
    for file_path in sorted(glob(os.path.join(TRAIN_DIR, "*.mat"))):
        if os.path.normcase(os.path.abspath(file_path)) == wavelength_absolute_path:
            continue
        all_files.append(file_path)

    if not all_files:
        raise FileNotFoundError(f"No training .mat files were found in:\n{TRAIN_DIR}")

    source_lookup = build_source_lookup(all_files)
    train_files, val_files, _ = stratified_train_val_split(all_files, source_lookup)

    if not train_files or not val_files:
        raise RuntimeError("Training or validation file list is empty.")

    summarize_raw_target(train_files)
    summarize_raw_target_by_source(all_files, source_lookup)
    late_start_statistics = summarize_late_start_indices(train_files)
    summarize_late_start_by_source(all_files, source_lookup)

    constant_prediction, constant_baseline_mae = calculate_constant_baseline(train_files, val_files)
    print(f"\nConstant bottom-mua prediction: {constant_prediction:.6e}")
    print(f"Constant validation MAE:         {constant_baseline_mae:.6e}")

    train_dataset = SpectralTPSFDataset(
        file_list=train_files,
        wavelength_values=normalized_wavelengths,
        augment=USE_TRAINING_AUGMENTATION,
    )
    val_dataset = SpectralTPSFDataset(
        file_list=val_files,
        wavelength_values=normalized_wavelengths,
        augment=False,
    )

    # Spectral mixing needs every wavelength of an image present together in
    # the batch. This is automatically satisfied here since each image
    # always contributes exactly N_WAVELENGTHS contiguous rows after
    # flatten_image_batch, regardless of IMAGE_BATCH_SIZE.
    if any(weight != 1.0 for weight in SOURCE_OVERSAMPLE_WEIGHTS.values()):
        sample_weights = [
            SOURCE_OVERSAMPLE_WEIGHTS[source_lookup[file_path]] for file_path in train_files
        ]
        train_sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(train_files), replacement=True
        )
        train_loader = DataLoader(
            train_dataset, batch_size=IMAGE_BATCH_SIZE, sampler=train_sampler,
            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False,
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=IMAGE_BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False,
        )
    val_loader = DataLoader(
        val_dataset, batch_size=IMAGE_BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False,
    )

    # Separate per-source validation loaders purely for diagnostic
    # reporting each epoch -- the combined val_loader above still drives
    # the scheduler and best-checkpoint selection, so switching this on
    # does not change what "best" means.
    per_source_val_loaders = build_per_source_val_loaders(
        val_files, source_lookup, normalized_wavelengths
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = BottomMuaSpectralNet().to(device)
    criterion = build_loss().to(device)

    # eval_model is what gets validated and checkpointed. With USE_EMA it
    # is a separately-tracked running average of model's weights (see
    # USE_EMA's comment); with USE_EMA off it is just model itself, so
    # disabling the flag reproduces the exact prior behavior.
    eval_model = create_ema_model(model) if USE_EMA else model

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=SCHEDULER_PATIENCE, min_lr=1.0e-7
    )

    best_val_mae = float("inf")
    best_epoch = 0
    best_state = None
    best_epoch_learning_rate = None
    epochs_without_improvement = 0

    train_mae_history = []
    val_mae_history = []

    for epoch in range(NUM_EPOCHS):
        model.train()

        absolute_error_sum = 0.0
        sample_count = 0

        for tpsf, target, wavelength, late_start in train_loader:
            tpsf = tpsf.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            wavelength = wavelength.to(device, non_blocking=True)
            late_start = late_start.to(device, non_blocking=True)

            tpsf, target, wavelength, late_start = flatten_image_batch(
                tpsf, target, wavelength, late_start
            )

            optimizer.zero_grad(set_to_none=True)

            prediction, supervised_profiles = model(tpsf, wavelength, late_start)

            primary_loss = criterion(prediction, target)
            aux_loss = depth_profile_auxiliary_loss(supervised_profiles, target)

            loss = primary_loss + AUX_DEPTH_LOSS_WEIGHT * aux_loss

            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss detected.")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_NORM)
            optimizer.step()

            if USE_EMA:
                update_ema_model(eval_model, model)

            raw_error = prediction.detach() - target
            count = raw_error.numel()
            absolute_error_sum += torch.abs(raw_error).sum().item()
            sample_count += count

        train_mae = absolute_error_sum / sample_count
        metrics = validate_spectral(eval_model, val_loader, criterion, device)

        scheduler.step(metrics["bottom_mua_mae"])
        current_lr = optimizer.param_groups[0]["lr"]

        train_mae_history.append(train_mae)
        val_mae_history.append(metrics["bottom_mua_mae"])

        print(
            f"Epoch [{epoch + 1:03d}/{NUM_EPOCHS}] | "
            f"Train MAE={train_mae:.6e} | "
            f"Val MAE={metrics['bottom_mua_mae']:.6e} | "
            f"Val RMSE={metrics['bottom_mua_rmse']:.6e} | "
            f"Val RelMAE={metrics['bottom_mua_relative_mae'] * 100:.2f}% | "
            f"LR={current_lr:.2e}"
        )
        # RelMAE (mean |error|/target) is scale-independent, unlike raw
        # MAE/RMSE -- use it, not raw MAE, to compare accuracy across
        # sources whose target magnitudes may differ (see
        # summarize_raw_target_by_source and validate_spectral's docstring).
        for label, loader in per_source_val_loaders.items():
            source_metrics = validate_spectral(eval_model, loader, criterion, device)
            print(
                f"    [{label}] Val MAE={source_metrics['bottom_mua_mae']:.6e} | "
                f"Val RMSE={source_metrics['bottom_mua_rmse']:.6e} | "
                f"Val RelMAE={source_metrics['bottom_mua_relative_mae'] * 100:.2f}% | "
                f"Val RelBias={source_metrics['bottom_mua_relative_bias'] * 100:+.2f}%"
            )

        if metrics["bottom_mua_mae"] < best_val_mae - MIN_DELTA:
            best_val_mae = metrics["bottom_mua_mae"]
            best_epoch = epoch + 1
            best_epoch_learning_rate = current_lr
            epochs_without_improvement = 0
            best_state = copy.deepcopy(eval_model.state_dict())
            print(f"  -> Best model updated at epoch {best_epoch}.")
        else:
            epochs_without_improvement += 1

        update_training_curve_plot(
            train_mae_history, val_mae_history, best_epoch,
            TRAINING_CURVE_PATH, "Training progress: bottom_mua MAE",
        )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print("\nEarly stopping activated.")
            break

    if best_state is None:
        raise RuntimeError("No best model state was captured.")

    checkpoint = {
        "model_state_dict": best_state,
        "architecture": "bottom_mua_depth_resolved_spectral_smoothing",
        "loss_mode": LOSS_MODE,
        "n_wavelengths": N_WAVELENGTHS,
        "n_time_gates": N_TIME_GATES,
        "time_gate_start": TIME_GATE_START,
        "time_gate_end_exclusive": TIME_GATE_END,
        "late_start_key": LATE_START_KEY,
        "late_start_is_matlab_one_based": LATE_START_IS_MATLAB_ONE_BASED,
        "late_start_statistics": late_start_statistics,
        "source_simulated_end": SOURCE_SIMULATED_END,
        "source_experimental_end": SOURCE_EXPERIMENTAL_END,
        "source_val_fraction_overrides": SOURCE_VAL_FRACTION_OVERRIDES,
        "source_oversample_weights": SOURCE_OVERSAMPLE_WEIGHTS,
        "source_file_counts": {
            label: sum(1 for f in all_files if source_lookup[f] == label)
            for label in sorted(set(source_lookup.values()))
        },
        "temporal_filters_per_kernel": TEMPORAL_FILTERS_PER_KERNEL,
        "temporal_pool_bins": TEMPORAL_POOL_BINS,
        "temporal_feature_dim": TEMPORAL_FEATURE_DIM,
        "temporal_dropout": TEMPORAL_DROPOUT,
        "head_dropout": HEAD_DROPOUT,
        "depth_feature_dim": DEPTH_FEATURE_DIM,
        "depth_hidden_dim": DEPTH_HIDDEN_DIM,
        "window_bins": WINDOW_BINS,
        "window_stride": WINDOW_STRIDE,
        "num_supervised_tail_windows": NUM_SUPERVISED_TAIL_WINDOWS,
        "spectral_smoothing_kernel_size": SPECTRAL_SMOOTHING_KERNEL_SIZE,
        "tail_extraction": "depth_resolved_grouped_by_late_start",
        "wavelength_usage": "normalized_0_1_scalar_direct",
        "use_ema": USE_EMA,
        "ema_decay": EMA_DECAY,
        "wavelength_file": WAVELENGTH_FILE,
        "wavelength_key": WAVELENGTH_KEY,
        "raw_wavelengths": torch.from_numpy(raw_wavelengths),
        "normalized_wavelengths": torch.from_numpy(normalized_wavelengths),
        "best_epoch": best_epoch,
        "best_validation_bottom_mua_mae": best_val_mae,
        "best_epoch_learning_rate": best_epoch_learning_rate,
        "constant_validation_bottom_mua_mae": constant_baseline_mae,
        "train_bottom_mua_mae_history": train_mae_history,
        "val_bottom_mua_mae_history": val_mae_history,
    }

    torch.save(checkpoint, SPECTRAL_MODEL_PATH)

    print("\nTraining complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation bottom-mua MAE: {best_val_mae:.6e}")
    print(f"Constant validation baseline MAE: {constant_baseline_mae:.6e}")
    print(f"Checkpoint saved to:\n{SPECTRAL_MODEL_PATH}")


# ============================================================
# 14. FINE-TUNING ON THE TEST SET'S DOMAINS
# ============================================================

def fine_tune_on_target_domains():
    """Continues training an existing SPECTRAL_MODEL_PATH checkpoint using
    only FINE_TUNE_SOURCES, at FINE_TUNE_LEARNING_RATE.

    Reuses the exact same stratified split (same SEED) as
    train_spectral_model, filtered down to the target sources, so the
    fine-tuning validation files are the same ones already reported on in
    the general run -- results stay directly comparable before/after.
    """

    if not os.path.isfile(SPECTRAL_MODEL_PATH):
        raise FileNotFoundError(
            f"No checkpoint found at:\n{SPECTRAL_MODEL_PATH}\n"
            f"Run train_spectral_model() first to produce one to fine-tune."
        )

    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(f"Training directory was not found:\n{TRAIN_DIR}")

    # Same normalization as train_spectral_model: load_wavelength_vector()
    # is deterministic given the same WAVELENGTH_FILE, so this recomputes
    # the identical [0,1] scale the base checkpoint was trained with.
    _, normalized_wavelengths = load_wavelength_vector()

    wavelength_absolute_path = os.path.normcase(os.path.abspath(WAVELENGTH_FILE))
    all_files = []
    for file_path in sorted(glob(os.path.join(TRAIN_DIR, "*.mat"))):
        if os.path.normcase(os.path.abspath(file_path)) == wavelength_absolute_path:
            continue
        all_files.append(file_path)

    source_lookup = build_source_lookup(all_files)
    train_files, val_files, _ = stratified_train_val_split(all_files, source_lookup)

    fine_tune_train_files = [f for f in train_files if source_lookup[f] in FINE_TUNE_SOURCES]
    fine_tune_val_files = [f for f in val_files if source_lookup[f] in FINE_TUNE_SOURCES]

    if not fine_tune_train_files or not fine_tune_val_files:
        raise RuntimeError("No fine-tuning training or validation files found for FINE_TUNE_SOURCES.")

    print(f"\nFine-tuning on sources: {FINE_TUNE_SOURCES}")
    print(f"  Fine-tune train files: {len(fine_tune_train_files)}")
    print(f"  Fine-tune val files:   {len(fine_tune_val_files)}")

    train_dataset = SpectralTPSFDataset(
        file_list=fine_tune_train_files,
        wavelength_values=normalized_wavelengths,
        augment=USE_TRAINING_AUGMENTATION,
    )
    val_dataset = SpectralTPSFDataset(
        file_list=fine_tune_val_files,
        wavelength_values=normalized_wavelengths,
        augment=False,
    )

    # FINE_TUNE_SOURCES are themselves wildly imbalanced (experimental
    # outnumbers simulated_close_to_experimental roughly 30:1). This
    # previously weighted each sample by 1/(its source's file count), which
    # gives every source exactly equal TOTAL sampling mass regardless of
    # size -- with a 30:1 file-count imbalance that means each
    # simulated_close_to_experimental file was drawn roughly 27x more often
    # per epoch than each experimental file. A real run showed exactly the
    # failure mode that implies: simulated_close_to_experimental's
    # validation MAE improved (consistent with the model memorizing its
    # ~38 heavily-repeated training files) while experimental's validation
    # MAE -- the actual real-phantom target -- did not improve at all,
    # since it was getting less than half its previous per-epoch exposure.
    # Reusing the same fixed per-source multipliers SOURCE_OVERSAMPLE_WEIGHTS
    # already uses for the main run (a per-file weight, not divided by
    # count) keeps the same well-tested, much milder oversampling ratio
    # (~1.5x per file here, vs. ~27x under full equalization) instead of a
    # second, far more aggressive scheme that was never validated on its own.
    fine_tune_sample_weights = [
        SOURCE_OVERSAMPLE_WEIGHTS[source_lookup[file_path]]
        for file_path in fine_tune_train_files
    ]
    fine_tune_sampler = WeightedRandomSampler(
        fine_tune_sample_weights, num_samples=len(fine_tune_train_files), replacement=True
    )
    train_loader = DataLoader(
        train_dataset, batch_size=IMAGE_BATCH_SIZE, sampler=fine_tune_sampler,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False,
    )
    # The combined loader above -- both FINE_TUNE_SOURCES together -- drives
    # early stopping and best-checkpoint selection, since
    # simulated_close_to_experimental alone is too small (a handful of
    # files) for that decision to be reliable on its own.
    val_loader = DataLoader(
        val_dataset, batch_size=IMAGE_BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False,
    )

    per_source_val_loaders = build_per_source_val_loaders(
        fine_tune_val_files, source_lookup, normalized_wavelengths,
        labels=FINE_TUNE_SOURCES,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    base_checkpoint = torch.load(SPECTRAL_MODEL_PATH, map_location=device)

    model = BottomMuaSpectralNet().to(device)
    model.load_state_dict(base_checkpoint["model_state_dict"])
    print(
        f"Loaded checkpoint from epoch {base_checkpoint.get('best_epoch', '?')} "
        f"(validation MAE {base_checkpoint.get('best_validation_bottom_mua_mae', float('nan')):.6e})."
    )

    # fine_tune_on_target_domains() always starts a brand-new optimizer at
    # FINE_TUNE_LEARNING_RATE -- it does not read or inherit whatever LR
    # the main run's ReduceLROnPlateau schedule had decayed to by the time
    # it stopped. The whole point of fine-tuning at "a much lower learning
    # rate" is to nudge the base checkpoint's weights gently rather than
    # overwrite them; if the base run's LR had already decayed below
    # FINE_TUNE_LEARNING_RATE before it stopped, fine-tuning would actually
    # take BIGGER steps than the base model had converged with, working
    # against that goal. Warn if that's the case so it isn't silent.
    base_epoch_lr = base_checkpoint.get("best_epoch_learning_rate")
    if base_epoch_lr is not None and FINE_TUNE_LEARNING_RATE >= base_epoch_lr:
        print(
            f"\nWARNING: FINE_TUNE_LEARNING_RATE ({FINE_TUNE_LEARNING_RATE:.2e}) is not "
            f"lower than the LR the base checkpoint's best epoch was actually trained "
            f"at ({base_epoch_lr:.2e}). Fine-tuning would take steps at least as large "
            f"as what training had already converged to, rather than gently "
            f"specializing it -- consider lowering FINE_TUNE_LEARNING_RATE below "
            f"{base_epoch_lr:.2e}."
        )

    # See USE_EMA's comment above train_spectral_model's identical setup.
    eval_model = create_ema_model(model) if USE_EMA else model

    criterion = build_loss().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=FINE_TUNE_LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=FINE_TUNE_SCHEDULER_PATIENCE, min_lr=1.0e-7
    )

    best_val_mae = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    train_mae_history = []
    val_mae_history = []

    for epoch in range(FINE_TUNE_NUM_EPOCHS):
        model.train()

        absolute_error_sum = 0.0
        sample_count = 0

        for tpsf, target, wavelength, late_start in train_loader:
            tpsf = tpsf.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            wavelength = wavelength.to(device, non_blocking=True)
            late_start = late_start.to(device, non_blocking=True)

            tpsf, target, wavelength, late_start = flatten_image_batch(
                tpsf, target, wavelength, late_start
            )

            optimizer.zero_grad(set_to_none=True)

            prediction, supervised_profiles = model(tpsf, wavelength, late_start)

            primary_loss = criterion(prediction, target)
            aux_loss = depth_profile_auxiliary_loss(supervised_profiles, target)

            loss = primary_loss + AUX_DEPTH_LOSS_WEIGHT * aux_loss

            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite fine-tuning loss detected.")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_NORM)
            optimizer.step()

            if USE_EMA:
                update_ema_model(eval_model, model)

            raw_error = prediction.detach() - target
            count = raw_error.numel()
            absolute_error_sum += torch.abs(raw_error).sum().item()
            sample_count += count

        train_mae = absolute_error_sum / sample_count
        metrics = validate_spectral(eval_model, val_loader, criterion, device)

        scheduler.step(metrics["bottom_mua_mae"])
        current_lr = optimizer.param_groups[0]["lr"]

        train_mae_history.append(train_mae)
        val_mae_history.append(metrics["bottom_mua_mae"])

        print(
            f"[Fine-tune] Epoch [{epoch + 1:03d}/{FINE_TUNE_NUM_EPOCHS}] | "
            f"Train MAE={train_mae:.6e} | "
            f"Val MAE={metrics['bottom_mua_mae']:.6e} | "
            f"Val RMSE={metrics['bottom_mua_rmse']:.6e} | "
            f"Val RelMAE={metrics['bottom_mua_relative_mae'] * 100:.2f}% | "
            f"LR={current_lr:.2e}"
        )
        for label, loader in per_source_val_loaders.items():
            source_metrics = validate_spectral(eval_model, loader, criterion, device)
            print(
                f"    [{label}] Val MAE={source_metrics['bottom_mua_mae']:.6e} | "
                f"Val RMSE={source_metrics['bottom_mua_rmse']:.6e} | "
                f"Val RelMAE={source_metrics['bottom_mua_relative_mae'] * 100:.2f}% | "
                f"Val RelBias={source_metrics['bottom_mua_relative_bias'] * 100:+.2f}%"
            )

        if metrics["bottom_mua_mae"] < best_val_mae - MIN_DELTA:
            best_val_mae = metrics["bottom_mua_mae"]
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            best_state = copy.deepcopy(eval_model.state_dict())
            print(f"  -> Best fine-tuned model updated at epoch {best_epoch}.")
        else:
            epochs_without_improvement += 1

        update_training_curve_plot(
            train_mae_history, val_mae_history, best_epoch,
            FINE_TUNE_TRAINING_CURVE_PATH, "Fine-tuning progress: bottom_mua MAE",
        )

        if epochs_without_improvement >= FINE_TUNE_EARLY_STOPPING_PATIENCE:
            print("\nFine-tuning early stopping activated.")
            break

    if best_state is None:
        raise RuntimeError("No best fine-tuned model state was captured.")

    fine_tune_checkpoint = dict(base_checkpoint)
    fine_tune_checkpoint.update({
        "model_state_dict": best_state,
        "architecture": "bottom_mua_depth_resolved_spectral_smoothing_finetuned",
        "fine_tune_sources": FINE_TUNE_SOURCES,
        "fine_tune_learning_rate": FINE_TUNE_LEARNING_RATE,
        "fine_tune_base_checkpoint": SPECTRAL_MODEL_PATH,
        "fine_tune_best_epoch": best_epoch,
        "fine_tune_best_validation_bottom_mua_mae": best_val_mae,
        "fine_tune_train_bottom_mua_mae_history": train_mae_history,
        "fine_tune_val_bottom_mua_mae_history": val_mae_history,
    })

    torch.save(fine_tune_checkpoint, FINE_TUNE_MODEL_PATH)

    print("\nFine-tuning complete.")
    print(f"Best fine-tune epoch: {best_epoch}")
    print(f"Best fine-tune validation bottom-mua MAE: {best_val_mae:.6e}")
    print(f"Fine-tuned checkpoint saved to:\n{FINE_TUNE_MODEL_PATH}")


if __name__ == "__main__":
    if RUN_FINE_TUNING:
        fine_tune_on_target_domains()
    else:
        train_spectral_model()
