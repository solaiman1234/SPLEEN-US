# -*- coding: utf-8 -*-
"""
Tail-weighted bottom-mua model. Fully standalone -- does not import from
any other file in this folder.

This is a deliberately simpler alternative to
train_bottom_mua_spectral_smoothing.py's depth-resolved GRU branch. That
design used a separate sequence encoder (sliding windows, a GRU, deep
supervision on the deepest windows, a plateau penalty) to make late-tail
information dominate the prediction. That is six interacting
hyperparameters (WINDOW_BINS, WINDOW_STRIDE, NUM_SUPERVISED_TAIL_WINDOWS,
LATE_WEIGHT_POWER, AUX_DEPTH_LOSS_WEIGHT, PLATEAU_LOSS_WEIGHT) and a
recurrent network to optimize, and when training loss itself is high (not
just validation loss), that points at optimization difficulty from the
architecture, not at a generalization problem worth more tuning.

Each image contains:
    - 169 wavelength-specific TPSFs
    - N_TIME_GATES selected time bins per TPSF
    - 169 bottom_absorption_mul targets
    - 169 LATE_START indices (the "end index": where each TPSF's late,
      diffusion-theory-valid tail begins)

For each wavelength-specific TPSF:
    full raw TPSF -> one multi-kernel convolution encoder, pooled with a
        per-sample time weighting that favors bins at and after LATE_START
    raw wavelength -> one direct scalar, no wavelength encoder

Tail weighting, not a separate tail branch: in a diffusive medium, photons
collected at progressively later times have, on average, travelled deeper
into the tissue, so late time bins carry more direct information about the
bottom-layer absorption than early ones (ln I(t) ~= -mua * v * t + const
for late-enough t). Instead of building a second sequence model to exploit
this, TailWeightedTemporalEncoder still convolves over the *entire* raw
TPSF (so the top-layer-dominated early part is not thrown away -- the
model still needs it to separate the top layer's contribution from the
bottom layer's), but pools each convolution branch two ways:

    - a weighted average over time, with weight 1.0 for every bin before
      that sample's own LATE_START and ramping linearly up to
      (1 + TAIL_WEIGHT_BOOST) at the final bin, so the tail dominates the
      pooled feature without discarding the rest of the trace;
    - a plain (unweighted) max, so the early peak position/amplitude --
      which still carries top-layer information -- is not lost just
      because it is not in the weighted-late-time region.

This replaces the previous design's six tail-specific hyperparameters with
one (TAIL_WEIGHT_BOOST) and removes the GRU entirely.
"""

import copy
import os
import random
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset


# ============================================================
# 1. USER SETTINGS
# ============================================================

TRAIN_DIR = r"D:\Simulation_TPSF_Convolution_Animal_IRF\DTOF_training_simu_and_exp_resampled"
WAVELENGTH_FILE = r"D:\Simulation_TPSF_Convolution_Animal_IRF\wavelength_exp_index.mat"
WAVELENGTH_KEY = "wavelengths_exp"

MODEL_PATH = r"D:\DTOF_exp_simu_tail_weighted_bottom_mua.pth"

TPSF_KEY = "dtof_auc"
BOTTOM_MUA_KEY = "bottom_absorption_mul"
LATE_START_KEY = "End_start"

# MATLAB indexing normally starts at 1. Set False for zero-based stored indices.
LATE_START_IS_MATLAB_ONE_BASED = True

N_WAVELENGTHS = 169
N_TIME_GATES = 450
TIME_GATE_START = 0
TIME_GATE_END = 450

IMAGE_BATCH_SIZE = 4
NUM_EPOCHS = 200
TRAIN_FRACTION = 0.80

LEARNING_RATE = 1.0e-4
WEIGHT_DECAY = 1.0e-5

NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()

EARLY_STOPPING_PATIENCE = 20
SCHEDULER_PATIENCE = 10
MIN_DELTA = 1.0e-8
GRADIENT_CLIP_NORM = 2.0

SEED = 42
EPS = 1.0e-8

TEMPORAL_FILTERS_PER_KERNEL = 12
TEMPORAL_FEATURE_DIM = 48
TEMPORAL_DROPOUT = 0.15

HEAD_DROPOUT = 0.15

USE_TRAINING_AUGMENTATION = True
AMPLITUDE_JITTER_STD = 0.01
ADDITIVE_NOISE_STD = 0.001
MAX_TIME_SHIFT = 1

# How much more weight the last time bin gets relative to every bin before
# LATE_START, in the weighted-average pooling. Weight ramps linearly from
# 1.0 at LATE_START to (1.0 + TAIL_WEIGHT_BOOST) at the final bin, and is
# flat at 1.0 for every bin before LATE_START. TAIL_WEIGHT_BOOST=3.0 means
# the last bin counts 4x as much as an early bin in the weighted average.
TAIL_WEIGHT_BOOST = 3.0

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
        f"This version requires TIME_GATE_START=0 and TIME_GATE_END=N_TIME_GATES "
        f"({N_TIME_GATES})."
    )

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

    raw = wavelengths.reshape(N_WAVELENGTHS, 1).astype(np.float32)

    print("\nWavelength information")
    print(f"Variable: {WAVELENGTH_KEY}")
    print(f"Range: {wavelength_min:.6f} to {wavelength_max:.6f}")

    return raw


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
    """Load one late-start ("end index") for each of the 169 TPSFs.

    Returned values are zero-based Python indices into the selected
    [TIME_GATE_START:TIME_GATE_END] window and have shape [169,1].
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

    late_start = late_start - TIME_GATE_START

    if np.any(late_start < 0) or np.any(late_start >= N_TIME_GATES):
        invalid_min = int(late_start.min())
        invalid_max = int(late_start.max())
        raise ValueError(
            f"Converted late-start indices in '{file_path}' range from "
            f"{invalid_min} to {invalid_max}; valid Python indices are "
            f"0 to {N_TIME_GATES - 1}. Check whether your indices are "
            f"MATLAB one-based and whether they refer to the selected "
            f"time-gate window."
        )

    return late_start.reshape(N_WAVELENGTHS, 1)


# ============================================================
# 4. SCALE AND TARGET SUMMARY STATISTICS
# ============================================================

def estimate_per_wavelength_input_scale(file_list):
    """Per-wavelength (not dataset-wide) 99th-percentile TPSF amplitude.

    Returns a [N_WAVELENGTHS, 1] float32 array: the median, across
    training images, of each wavelength's own 99th-percentile amplitude.
    Using one scale per wavelength instead of a single dataset-wide scalar
    accounts for wavelength-dependent source power / detector response
    instead of forcing every wavelength through the same normalization.
    """

    per_file_percentiles = []

    for file_path in file_list:
        tpsf = load_training_tpsf(loadmat(file_path), file_path)
        percentile = np.percentile(tpsf, 99.0, axis=1)
        per_file_percentiles.append(percentile)

    stacked = np.stack(per_file_percentiles, axis=0)
    scale = np.median(stacked, axis=0)
    scale = np.clip(scale, EPS, None).astype(np.float32)

    print("\nPer-wavelength TPSF input scale")
    print(f"Minimum: {scale.min():.6e}")
    print(f"Median:  {np.median(scale):.6e}")
    print(f"Maximum: {scale.max():.6e}")

    return scale.reshape(N_WAVELENGTHS, 1)


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

    if AMPLITUDE_JITTER_STD > 0.0:
        amplitude_factor = float(np.exp(np.random.normal(0.0, AMPLITUDE_JITTER_STD)))
        augmented *= amplitude_factor

    if ADDITIVE_NOISE_STD > 0.0:
        noise = np.random.normal(0.0, ADDITIVE_NOISE_STD, size=augmented.shape).astype(np.float32)
        augmented += noise

    augmented = np.clip(augmented, 0.0, None).astype(np.float32)
    return augmented, adjusted_late_start


class PerWavelengthNormalizedDataset(Dataset):
    """Divides each wavelength's TPSF row by its own scale instead of one
    dataset-wide scalar.
    """

    def __init__(self, file_list, per_wavelength_scale, wavelength_values, augment=False):
        self.file_list = list(file_list)
        self.augment = bool(augment)

        per_wavelength_scale = np.asarray(per_wavelength_scale, dtype=np.float32)
        expected_scale_shape = (N_WAVELENGTHS, 1)
        if per_wavelength_scale.shape != expected_scale_shape:
            raise ValueError(
                f"per_wavelength_scale has shape {per_wavelength_scale.shape}; "
                f"expected {expected_scale_shape}."
            )
        self.per_wavelength_scale = per_wavelength_scale

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
        tpsf = (tpsf / self.per_wavelength_scale).astype(np.float32)

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


# ============================================================
# 6. RAW TPSF CHANNEL
# ============================================================

def build_raw_tpsf_channel(tpsf):
    """Convert [N,T] raw TPSFs to [N,1,T]."""

    if tpsf.ndim != 2 or tpsf.shape[1] != N_TIME_GATES:
        raise ValueError(f"Expected [N,{N_TIME_GATES}], received {tuple(tpsf.shape)}.")

    tpsf = torch.nan_to_num(tpsf, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.clamp(tpsf, min=0.0).unsqueeze(1)


def choose_group_count(number_of_channels):
    for candidate in (4, 3, 2):
        if number_of_channels % candidate == 0:
            return candidate
    return 1


# ============================================================
# 7. TAIL-WEIGHTED POOLING
# ============================================================

def compute_tail_weight(late_start, n_time_gates, tail_boost, device, dtype):
    """Per-sample, per-time-bin importance weight for weighted-average pooling.

    late_start: [N] integer index of where each sample's tail begins.
    Returns: [N, n_time_gates] weight, equal to 1.0 for every bin before
    that sample's own late_start, then ramping linearly up to
    (1.0 + tail_boost) at the final bin. Bins right at late_start start the
    ramp at 1.0, so there is no discontinuity -- only a change in slope.
    """

    time_index = torch.arange(n_time_gates, device=device, dtype=dtype).unsqueeze(0)  # [1,T]
    late_start = late_start.to(device=device, dtype=dtype).view(-1, 1)  # [N,1]

    span = (float(n_time_gates - 1) - late_start).clamp_min(1.0)
    progress = ((time_index - late_start) / span).clamp(min=0.0, max=1.0)

    return 1.0 + float(tail_boost) * progress


def weighted_average_pool(features, weight, eps=EPS):
    """features: [N,C,T], weight: [N,T] (need not sum to 1). Returns [N,C]."""

    weight = weight.unsqueeze(1)  # [N,1,T]
    weighted_sum = (features * weight).sum(dim=2)
    weight_total = weight.sum(dim=2).clamp_min(eps)
    return weighted_sum / weight_total


# ============================================================
# 8. TAIL-WEIGHTED TEMPORAL ENCODER
# ============================================================

class TailWeightedTemporalEncoder(nn.Module):
    """Multi-kernel 1D convolution over the full raw TPSF, pooled with a
    per-sample time weighting that favors the tail.

    Every branch runs over the entire [N,1,T] trace (the early,
    top-layer-dominated part is never discarded), then is reduced to a
    fixed-size vector with two pooling operations: a weighted average that
    favors bins at and after LATE_START (see compute_tail_weight), and a
    plain unweighted max that preserves early-peak information regardless
    of where it falls relative to LATE_START.
    """

    def __init__(
        self,
        filters_per_kernel=TEMPORAL_FILTERS_PER_KERNEL,
        feature_dim=TEMPORAL_FEATURE_DIM,
        dropout=TEMPORAL_DROPOUT,
        tail_weight_boost=TAIL_WEIGHT_BOOST,
    ):
        super().__init__()

        groups = choose_group_count(filters_per_kernel)
        self.tail_weight_boost = float(tail_weight_boost)

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

        # weighted-mean + plain-max pooled vector per branch, 3 branches.
        projection_input = 3 * 2 * filters_per_kernel

        self.output_projection = nn.Sequential(
            nn.Linear(projection_input, 96),
            nn.LayerNorm(96),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(96, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
        )

    def pool(self, x, weight):
        weighted_mean = weighted_average_pool(x, weight)
        plain_max = x.amax(dim=2)
        return torch.cat([weighted_mean, plain_max], dim=1)

    def forward(self, x, late_start):
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError(
                f"TailWeightedTemporalEncoder expects [N,1,T], received {tuple(x.shape)}."
            )

        weight = compute_tail_weight(
            late_start.reshape(-1), n_time_gates=x.shape[-1],
            tail_boost=self.tail_weight_boost, device=x.device, dtype=x.dtype,
        )

        features = torch.cat(
            [
                self.pool(self.small_filter(x), weight),
                self.pool(self.medium_filter(x), weight),
                self.pool(self.large_filter(x), weight),
            ],
            dim=1,
        )

        return self.output_projection(features)


# ============================================================
# 9. FULL MODEL
# ============================================================

class BottomMuaTailWeightedNet(nn.Module):
    """Tail-weighted TPSF features + raw wavelength scalar -> bottom-mua.

    The wavelength enters the head as a direct scalar rather than through a
    learned embedding, so the regression head is supervised directly by the
    true wavelength of each TPSF instead of by an intermediate learned
    representation the model invents on its own.
    """

    def __init__(
        self,
        temporal_feature_dim=TEMPORAL_FEATURE_DIM,
        filters_per_kernel=TEMPORAL_FILTERS_PER_KERNEL,
        temporal_dropout=TEMPORAL_DROPOUT,
        head_dropout=HEAD_DROPOUT,
        tail_weight_boost=TAIL_WEIGHT_BOOST,
    ):
        super().__init__()

        self.encoder = TailWeightedTemporalEncoder(
            filters_per_kernel=filters_per_kernel,
            feature_dim=temporal_feature_dim,
            dropout=temporal_dropout,
            tail_weight_boost=tail_weight_boost,
        )

        # +1 for the raw wavelength scalar, concatenated directly (no
        # learned encoder) below.
        fused_dim = temporal_feature_dim + 1

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

    def forward(self, tpsf, wavelength, late_start):
        channels = build_raw_tpsf_channel(tpsf)
        features = self.encoder(channels, late_start)

        wavelength_input = wavelength.to(dtype=features.dtype)
        fused_features = torch.cat([features, wavelength_input], dim=1)

        return self.head(fused_features)


# ============================================================
# 10. RAW TARGET LOSS
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
# 11. VALIDATION
# ============================================================

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()

    loss_sum = 0.0
    absolute_error_sum = 0.0
    squared_error_sum = 0.0
    value_count = 0

    for tpsf, target, wavelength, late_start in loader:
        tpsf = tpsf.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        wavelength = wavelength.to(device, non_blocking=True)
        late_start = late_start.to(device, non_blocking=True)

        tpsf, target, wavelength, late_start = flatten_image_batch(
            tpsf, target, wavelength, late_start
        )

        prediction = model(tpsf, wavelength, late_start)
        loss = criterion(prediction, target)

        error = prediction - target
        count = error.numel()

        loss_sum += loss.item() * count
        absolute_error_sum += torch.abs(error).sum().item()
        squared_error_sum += torch.square(error).sum().item()
        value_count += count

    return {
        "raw_loss": loss_sum / value_count,
        "bottom_mua_mae": absolute_error_sum / value_count,
        "bottom_mua_rmse": np.sqrt(squared_error_sum / value_count),
    }


# ============================================================
# 12. TRAINING
# ============================================================

def train_model():
    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(f"Training directory was not found:\n{TRAIN_DIR}")

    raw_wavelengths = load_wavelength_vector()

    wavelength_absolute_path = os.path.normcase(os.path.abspath(WAVELENGTH_FILE))

    all_files = []
    for file_path in sorted(glob(os.path.join(TRAIN_DIR, "*.mat"))):
        if os.path.normcase(os.path.abspath(file_path)) == wavelength_absolute_path:
            continue
        all_files.append(file_path)

    if not all_files:
        raise FileNotFoundError(f"No training .mat files were found in:\n{TRAIN_DIR}")

    split_rng = random.Random(SEED)
    split_rng.shuffle(all_files)

    split_index = int(TRAIN_FRACTION * len(all_files))
    train_files = all_files[:split_index]
    val_files = all_files[split_index:]

    if not train_files or not val_files:
        raise RuntimeError("Training or validation file list is empty.")

    per_wavelength_scale = estimate_per_wavelength_input_scale(train_files)
    summarize_raw_target(train_files)
    late_start_statistics = summarize_late_start_indices(train_files)

    constant_prediction, constant_baseline_mae = calculate_constant_baseline(train_files, val_files)
    print(f"\nConstant bottom-mua prediction: {constant_prediction:.6e}")
    print(f"Constant validation MAE:         {constant_baseline_mae:.6e}")

    train_dataset = PerWavelengthNormalizedDataset(
        file_list=train_files,
        per_wavelength_scale=per_wavelength_scale,
        wavelength_values=raw_wavelengths,
        augment=USE_TRAINING_AUGMENTATION,
    )
    val_dataset = PerWavelengthNormalizedDataset(
        file_list=val_files,
        per_wavelength_scale=per_wavelength_scale,
        wavelength_values=raw_wavelengths,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=IMAGE_BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=IMAGE_BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = BottomMuaTailWeightedNet().to(device)
    criterion = build_loss().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=SCHEDULER_PATIENCE, min_lr=1.0e-7
    )

    best_val_mae = float("inf")
    best_epoch = 0
    best_state = None
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

            prediction = model(tpsf, wavelength, late_start)
            loss = criterion(prediction, target)

            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss detected.")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_NORM)
            optimizer.step()

            raw_error = prediction.detach() - target
            count = raw_error.numel()
            absolute_error_sum += torch.abs(raw_error).sum().item()
            sample_count += count

        train_mae = absolute_error_sum / sample_count
        metrics = validate(model, val_loader, criterion, device)

        scheduler.step(metrics["bottom_mua_mae"])
        current_lr = optimizer.param_groups[0]["lr"]

        train_mae_history.append(train_mae)
        val_mae_history.append(metrics["bottom_mua_mae"])

        print(
            f"Epoch [{epoch + 1:03d}/{NUM_EPOCHS}] | "
            f"Train MAE={train_mae:.6e} | "
            f"Val MAE={metrics['bottom_mua_mae']:.6e} | "
            f"Val RMSE={metrics['bottom_mua_rmse']:.6e} | "
            f"LR={current_lr:.2e}"
        )

        if metrics["bottom_mua_mae"] < best_val_mae - MIN_DELTA:
            best_val_mae = metrics["bottom_mua_mae"]
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            best_state = copy.deepcopy(model.state_dict())
            print(f"  -> Best model updated at epoch {best_epoch}.")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print("\nEarly stopping activated.")
            break

    if best_state is None:
        raise RuntimeError("No best model state was captured.")

    checkpoint = {
        "model_state_dict": best_state,
        "architecture": "bottom_mua_tail_weighted",
        "loss_mode": LOSS_MODE,
        "n_wavelengths": N_WAVELENGTHS,
        "n_time_gates": N_TIME_GATES,
        "time_gate_start": TIME_GATE_START,
        "time_gate_end_exclusive": TIME_GATE_END,
        "late_start_key": LATE_START_KEY,
        "late_start_is_matlab_one_based": LATE_START_IS_MATLAB_ONE_BASED,
        "late_start_statistics": late_start_statistics,
        "temporal_filters_per_kernel": TEMPORAL_FILTERS_PER_KERNEL,
        "temporal_feature_dim": TEMPORAL_FEATURE_DIM,
        "temporal_dropout": TEMPORAL_DROPOUT,
        "head_dropout": HEAD_DROPOUT,
        "tail_weight_boost": TAIL_WEIGHT_BOOST,
        "wavelength_usage": "raw_scalar_direct",
        "wavelength_scale_mode": "per_wavelength",
        "per_wavelength_input_scale": per_wavelength_scale,
        "wavelength_file": WAVELENGTH_FILE,
        "wavelength_key": WAVELENGTH_KEY,
        "raw_wavelengths": torch.from_numpy(raw_wavelengths),
        "best_epoch": best_epoch,
        "best_validation_bottom_mua_mae": best_val_mae,
        "constant_validation_bottom_mua_mae": constant_baseline_mae,
        "train_bottom_mua_mae_history": train_mae_history,
        "val_bottom_mua_mae_history": val_mae_history,
    }

    torch.save(checkpoint, MODEL_PATH)

    print("\nTraining complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation bottom-mua MAE: {best_val_mae:.6e}")
    print(f"Constant validation baseline MAE: {constant_baseline_mae:.6e}")
    print(f"Checkpoint saved to:\n{MODEL_PATH}")


if __name__ == "__main__":
    train_model()
