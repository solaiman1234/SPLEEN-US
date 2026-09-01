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

The wavelength is concatenated into the fused feature vector as-is (the
true physical value, not a normalized or learned-embedded one), so the
regression head is directly supervised by each TPSF's actual wavelength
instead of by an intermediate representation invented by a small
sub-network trained on nothing but that one scalar.

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
Per-wavelength TPSF input normalization (a length-169 scale vector
instead of one dataset-wide scalar) is used for the same reason: real
source/detector responses aren't uniform across wavelength.
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

SPECTRAL_MODEL_PATH = r"D:\DTOF_exp_simu_depth_resolved_spectral_bottom_mua.pth"

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
TEMPORAL_POOL_BINS = 4
TEMPORAL_FEATURE_DIM = 48
TEMPORAL_DROPOUT = 0.15

HEAD_DROPOUT = 0.15

USE_TRAINING_AUGMENTATION = True
AMPLITUDE_JITTER_STD = 0.01
ADDITIVE_NOISE_STD = 0.001
MAX_TIME_SHIFT = 1

# Depth-resolved tail encoder settings. With N_TIME_GATES=450, a tail
# starting at LATE_START=0 produces up to ~44 windows for the GRU to
# process (vs. ~29 when this design was sized around a 300-gate TPSF), so
# DEPTH_HIDDEN_DIM is widened from its original 32 to give the GRU enough
# capacity to carry that much sequential information without lossy
# compression.
WINDOW_BINS = 20
WINDOW_STRIDE = 10
DEPTH_FEATURE_DIM = 16
DEPTH_HIDDEN_DIM = 48

# How many of the deepest windows in each tail get direct supervision
# against the bottom-mua label. Raised from 3 to anchor deep supervision
# across more of what can now be a much longer window sequence.
NUM_SUPERVISED_TAIL_WINDOWS = 5

# Exponent controlling how much more the auxiliary loss weights the
# deepest supervised window relative to the shallowest of the supervised
# set (weights increase with depth).
LATE_WEIGHT_POWER = 2.0

AUX_DEPTH_LOSS_WEIGHT = 0.3
PLATEAU_LOSS_WEIGHT = 0.05

# Spectral-smoothing settings.
SPECTRAL_SMOOTHING_KERNEL_SIZE = 7

# Optional light total-variation penalty on the final predicted spectrum,
# on top of the structural smoothing already performed by SpectralSmoother.
# Kept off by default: too much of this can flatten genuine peaks rather
# than just removing jitter.
SPECTRAL_TV_LOSS_WEIGHT = 0.0

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
    raise ValueError("This version must use the first 300 TPSF columns.")

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

    Returned values are zero-based indices relative to the selected [0:300]
    time window and have shape [169,1].
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
            f"MATLAB one-based and whether they refer to the first 300 gates."
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
    """Convert [N,300] raw TPSFs to [N,1,300]."""

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


def depth_profile_auxiliary_losses(supervised_profiles, target, late_weight_power=LATE_WEIGHT_POWER):
    """Deep-supervision losses on the deepest windows of the depth profile.

    supervised_profiles: [N, L], shallow-to-deep ordering (column -1 is the
        single deepest window).
    target: [N, 1], the bottom-mua label.

    Returns (weighted_l1_loss, plateau_loss). The L1 term pulls every
    supervised window toward the bottom-mua label, with weight increasing
    toward the deepest window. The plateau term penalizes variance across
    those windows, encouraging the profile to have stabilized by the time
    it reaches the label depth.
    """

    n_windows = supervised_profiles.shape[1]

    weights = torch.linspace(
        1.0, float(n_windows), n_windows,
        device=supervised_profiles.device, dtype=supervised_profiles.dtype,
    ).pow(late_weight_power)
    weights = weights / weights.sum()

    error = supervised_profiles - target
    weighted_l1 = (weights.view(1, -1) * error.abs()).sum(dim=1).mean()

    plateau = supervised_profiles.var(dim=1, unbiased=False).mean()

    return weighted_l1, plateau


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


def spectral_total_variation_loss(prediction, n_wavelengths=N_WAVELENGTHS):
    """Mean absolute difference between wavelength-adjacent predictions.

    Off by default (SPECTRAL_TV_LOSS_WEIGHT == 0.0): SpectralSmoother
    already gives the network a structural way to produce smooth spectra,
    and a strong explicit penalty here can suppress genuine peaks rather
    than just jitter. Provided for experimentation.
    """

    n_total = prediction.shape[0]
    if n_total % n_wavelengths != 0:
        raise ValueError(f"Batch size {n_total} is not a multiple of {n_wavelengths}.")

    n_images = n_total // n_wavelengths
    spectrum = prediction.view(n_images, n_wavelengths)
    return (spectrum[:, 1:] - spectrum[:, :-1]).abs().mean()


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

        prediction, _ = model(tpsf, wavelength, late_start)
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

    # Spectral mixing needs every wavelength of an image present together in
    # the batch. This is automatically satisfied here since each image
    # always contributes exactly N_WAVELENGTHS contiguous rows after
    # flatten_image_batch, regardless of IMAGE_BATCH_SIZE.
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

    model = BottomMuaSpectralNet().to(device)
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

            prediction, supervised_profiles = model(tpsf, wavelength, late_start)

            primary_loss = criterion(prediction, target)
            aux_loss, plateau_loss = depth_profile_auxiliary_losses(supervised_profiles, target)

            loss = primary_loss + AUX_DEPTH_LOSS_WEIGHT * aux_loss + PLATEAU_LOSS_WEIGHT * plateau_loss

            if SPECTRAL_TV_LOSS_WEIGHT > 0.0:
                loss = loss + SPECTRAL_TV_LOSS_WEIGHT * spectral_total_variation_loss(prediction)

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
        metrics = validate_spectral(model, val_loader, criterion, device)

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
        "architecture": "bottom_mua_depth_resolved_spectral_smoothing",
        "loss_mode": LOSS_MODE,
        "n_wavelengths": N_WAVELENGTHS,
        "n_time_gates": N_TIME_GATES,
        "time_gate_start": TIME_GATE_START,
        "time_gate_end_exclusive": TIME_GATE_END,
        "late_start_key": LATE_START_KEY,
        "late_start_is_matlab_one_based": LATE_START_IS_MATLAB_ONE_BASED,
        "late_start_statistics": late_start_statistics,
        "temporal_filters_per_kernel": TEMPORAL_FILTERS_PER_KERNEL,
        "temporal_pool_bins": TEMPORAL_POOL_BINS,
        "temporal_feature_dim": TEMPORAL_FEATURE_DIM,
        "temporal_dropout": TEMPORAL_DROPOUT,
        "head_dropout": HEAD_DROPOUT,
        "depth_hidden_dim": DEPTH_HIDDEN_DIM,
        "window_bins": WINDOW_BINS,
        "window_stride": WINDOW_STRIDE,
        "num_supervised_tail_windows": NUM_SUPERVISED_TAIL_WINDOWS,
        "spectral_smoothing_kernel_size": SPECTRAL_SMOOTHING_KERNEL_SIZE,
        "spectral_tv_loss_weight": SPECTRAL_TV_LOSS_WEIGHT,
        "tail_extraction": "depth_resolved_grouped_by_late_start",
        "wavelength_usage": "raw_scalar_direct",
        "wavelength_scale_mode": "per_wavelength",
        "per_wavelength_input_scale": per_wavelength_scale,
        "wavelength_file": WAVELENGTH_FILE,
        "wavelength_key": WAVELENGTH_KEY,
        "raw_wavelengths": torch.from_numpy(raw_wavelengths),
        "normalized_wavelengths": torch.from_numpy(normalized_wavelengths),
        "best_epoch": best_epoch,
        "best_validation_bottom_mua_mae": best_val_mae,
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


if __name__ == "__main__":
    train_spectral_model()
