# -*- coding: utf-8 -*-
"""
Bottom-mua model with one dynamic raw tail for each TPSF.

Each image contains:
    - 169 wavelength-specific TPSFs
    - 300 selected time bins per TPSF
    - 169 bottom_absorption_mul targets
    - 169 LATE_START indices

For each wavelength-specific TPSF:
    full raw TPSF -> full convolution encoder
    raw TPSF tail -> tail convolution encoder
    full/tail TPSF -> low-order statistical moments (mean/variance/skew/
        peak amplitude/peak time/total counts), used directly as auxiliary
        features
    normalized wavelength -> small nonlinear wavelength embedding

The full/tail convolutional features, the full/tail moment features, and the
wavelength embedding are concatenated and used to predict the corresponding
bottom-layer absorption coefficient.

The tail is TPSF[LATE_START:300] exactly. There is no interpolation,
resampling, padding, tail mask, or LATE_START feature encoder.

Changes relative to the original single-scalar-wavelength version:
    1. Fixed a wiring bug where the training dataset always received
       augment=False regardless of USE_TRAINING_AUGMENTATION, so the
       amplitude/noise/time-shift augmentation implemented below never
       actually ran during training.
    2. Added low-order statistical moments of the full TPSF and of the raw
       tail (total counts, mean time-of-flight, temporal std, skewness,
       peak amplitude, peak time) as auxiliary head inputs. These are the
       same summary statistics used by generalized method-of-moments
       absorption estimators in diffuse optics, and give the head a
       numerically stable signal that is not solely dependent on what the
       convolutional encoders learn from a limited number of training
       images.
    3. Replaced the raw wavelength scalar with a small nonlinear MLP
       embedding, since chromophore absorption spectra are not linear in
       wavelength.
    4. Increased the default per-step image batch size so that gradients
       are averaged over TPSFs drawn from more than one phantom/image at a
       time, reducing gradient variance from within-image correlation.
"""



import copy
import os
import random
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# ============================================================
# 1. USER SETTINGS
# ============================================================

TRAIN_DIR = r"D:\Simulation_TPSF_Convolution_Animal_IRF\DTOF_training_simu_and_exp_resampled"
WAVELENGTH_FILE = r"D:\Simulation_TPSF_Convolution_Animal_IRF\wavelength_exp_index.mat"
WAVELENGTH_KEY = 'wavelengths_exp'

MODEL_PATH = r"D:\DTOF_exp_simu_full_and_tail_cropped_exp_wavelength_resampled_raw_target.pth"

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
TARGET_LOG_EPS = 1.0e-8

TPSF_INPUT_CHANNELS = 1
TEMPORAL_FILTERS_PER_KERNEL = 12  ##12
TEMPORAL_POOL_BINS = 4
TEMPORAL_FEATURE_DIM = 48
TEMPORAL_DROPOUT = 0.15

MOMENT_FEATURE_DIM = 6
WAVELENGTH_EMBEDDING_DIM = 4

HEAD_DROPOUT = 0.15

USE_TRAINING_AUGMENTATION = True
AMPLITUDE_JITTER_STD = 0.01
ADDITIVE_NOISE_STD = 0.001
MAX_TIME_SHIFT = 1


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
        raise FileNotFoundError(
            f"Wavelength file was not found:\n{WAVELENGTH_FILE}"
        )

    mat = loadmat(WAVELENGTH_FILE)

    if WAVELENGTH_KEY is not None:
        if WAVELENGTH_KEY not in mat:
            available = [key for key in mat if not key.startswith("__")]
            raise KeyError(
                f"'{WAVELENGTH_KEY}' was not found. "
                f"Available variables: {available}"
            )
        detected_key = WAVELENGTH_KEY
        wavelength_array = np.asarray(mat[WAVELENGTH_KEY])
    else:
        candidates = []
        for key, value in mat.items():
            if key.startswith("__"):
                continue
            array = np.asarray(value)
            if (
                np.issubdtype(array.dtype, np.number)
                and array.size == N_WAVELENGTHS
            ):
                candidates.append((key, array))

        if len(candidates) != 1:
            names = [key for key, _ in candidates]
            raise ValueError(
                f"Expected one numeric {N_WAVELENGTHS}-value wavelength "
                f"variable, found: {names}. Set WAVELENGTH_KEY explicitly."
            )

        detected_key, wavelength_array = candidates[0]

    wavelengths = np.asarray(
        wavelength_array,
        dtype=np.float32,
    ).reshape(-1)

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

    normalized = (
        wavelengths - wavelength_min
    ) / (wavelength_max - wavelength_min)

    raw = wavelengths.reshape(N_WAVELENGTHS, 1).astype(np.float32)
    normalized = normalized.reshape(N_WAVELENGTHS, 1).astype(np.float32)

    print("\nWavelength information")
    print(f"Variable: {detected_key}")
    print(f"Range: {wavelength_min:.6f} to {wavelength_max:.6f}")

    return raw, normalized, detected_key


def clean_tpsf_array(x, file_path):
    x = np.asarray(x)

    if x.ndim != 2:
        raise ValueError(
            f"TPSF in '{file_path}' must be 2D; received {x.shape}."
        )

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
    oriented = np.nan_to_num(
        oriented,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    oriented = np.clip(oriented, 0.0, None)

    expected = (N_WAVELENGTHS, N_TIME_GATES)
    if oriented.shape != expected:
        raise RuntimeError(
            f"Selected TPSF shape is {oriented.shape}; expected {expected}."
        )

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
    target = np.nan_to_num(
        target,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if np.any(target <= 0.0):
        raise ValueError(
            "Bottom-mua labels must be strictly positive. "
            f"labels. Non-positive values were found in:\n{file_path}"
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
        raise FloatingPointError(
            f"Non-finite late-start indices were found in:\n{file_path}"
        )

    rounded = np.rint(late_start)

    if np.max(np.abs(late_start - rounded)) > 1.0e-4:
        raise ValueError(
            f"Late-start values must be integer indices in:\n{file_path}"
        )

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
# 4. SCALE AND TARGET-TRANSFORM ESTIMATION
# ============================================================

def estimate_tpsf_input_scale(file_list):
    percentiles = []

    for file_path in file_list:
        tpsf = load_training_tpsf(loadmat(file_path), file_path)
        value = float(np.percentile(tpsf, 99.0))
        if np.isfinite(value) and value > 0.0:
            percentiles.append(value)

    if not percentiles:
        raise RuntimeError("Could not estimate a positive TPSF input scale.")

    scale = max(float(np.median(percentiles)), EPS)
    print(f"\nFixed TPSF input scale: {scale:.6e}")
    return scale


def summarize_raw_target(file_list):
    """Print raw bottom-mua statistics. No target transform is applied."""
    all_targets = []

    for file_path in file_list:
        all_targets.append(
            load_bottom_mua_target(loadmat(file_path), file_path)
        )

    targets = np.concatenate(all_targets, axis=0).astype(np.float32)

    print("\nBottom-mua raw target statistics")
    print(f"Minimum:        {targets.min():.6e}")
    print(f"Maximum:        {targets.max():.6e}")
    print(f"Mean:           {targets.mean():.6e}")
    print(f"Median:         {np.median(targets):.6e}")
    print(f"Standard dev.:  {targets.std():.6e}")

    if targets.std() < 1.0e-8:
        raise RuntimeError(
            "Bottom-mua variation is almost zero; meaningful regression "
            "is not possible."
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
        shift = int(
            np.random.randint(
                -MAX_TIME_SHIFT,
                MAX_TIME_SHIFT + 1,
            )
        )
        augmented = shift_time_axis(augmented, shift)
        adjusted_late_start = np.clip(
            adjusted_late_start + shift,
            0,
            N_TIME_GATES - 1,
        )

    if AMPLITUDE_JITTER_STD > 0.0:
        amplitude_factor = float(
            np.exp(np.random.normal(0.0, AMPLITUDE_JITTER_STD))
        )
        augmented *= amplitude_factor

    if ADDITIVE_NOISE_STD > 0.0:
        noise = np.random.normal(
            0.0,
            ADDITIVE_NOISE_STD,
            size=augmented.shape,
        ).astype(np.float32)
        augmented += noise

    augmented = np.clip(augmented, 0.0, None).astype(np.float32)
    return augmented, adjusted_late_start


class BottomMuaDataset(Dataset):
    def __init__(
        self,
        file_list,
        tpsf_input_scale,
        normalized_wavelengths,
        augment=False,
    ):
        self.file_list = list(file_list)
        self.tpsf_input_scale = float(tpsf_input_scale)
        self.augment = bool(augment)

        normalized_wavelengths = np.asarray(
            normalized_wavelengths,
            dtype=np.float32,
        )

        expected = (N_WAVELENGTHS, 1)
        if normalized_wavelengths.shape != expected:
            raise ValueError(
                f"normalized_wavelengths has shape "
                f"{normalized_wavelengths.shape}; expected {expected}."
            )

        self.wavelengths = torch.from_numpy(normalized_wavelengths)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_path = self.file_list[index]
        mat = loadmat(file_path)

        tpsf = load_training_tpsf(mat, file_path)
        tpsf = (tpsf / self.tpsf_input_scale).astype(np.float32)

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


def flatten_image_batch(
    tpsf,
    target=None,
    wavelength=None,
    late_start=None,
):
    tpsf_flat = tpsf.reshape(-1, N_TIME_GATES)

    target_flat = None
    if target is not None:
        target_flat = target.reshape(-1, 1)

    wavelength_flat = None
    if wavelength is not None:
        wavelength_flat = wavelength.reshape(-1, 1)

    late_start_flat = None
    if late_start is not None:
        late_start_flat = late_start.reshape(-1, 1)

    return tpsf_flat, target_flat, wavelength_flat, late_start_flat


# ============================================================
# 6. RAW TPSF CHANNEL AND STATISTICAL MOMENT FEATURES
# ============================================================

def build_raw_tpsf_channel(tpsf):
    """Convert [N,300] raw TPSFs to [N,1,300]."""

    if (
        tpsf.ndim != 2
        or tpsf.shape[1] != N_TIME_GATES
    ):
        raise ValueError(
            f"Expected [N,{N_TIME_GATES}], "
            f"received {tuple(tpsf.shape)}."
        )

    tpsf = torch.nan_to_num(
        tpsf,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return torch.clamp(
        tpsf,
        min=0.0,
    ).unsqueeze(1)


def compute_moment_features(tpsf, eps=EPS):
    """Low-order statistical moments of a raw time-of-flight pulse.

    Returns [log(1+total counts), mean time, std time, skewness, peak
    amplitude, peak time], with the time-domain quantities scaled by the
    number of bins so they stay comparable across full and tail windows of
    different lengths. These are the same summary statistics used by
    generalized method-of-moments absorption estimators in diffuse optics.
    """

    if tpsf.ndim != 2:
        raise ValueError(f"Expected [N,T], received {tuple(tpsf.shape)}.")

    n_bins = tpsf.shape[1]
    time_index = torch.arange(
        n_bins,
        device=tpsf.device,
        dtype=tpsf.dtype,
    ).unsqueeze(0)

    total_counts = tpsf.sum(dim=1, keepdim=True)
    safe_total = total_counts.clamp_min(eps)

    mean_time = (tpsf * time_index).sum(dim=1, keepdim=True) / safe_total
    centered = time_index - mean_time
    variance = (tpsf * centered.pow(2)).sum(dim=1, keepdim=True) / safe_total
    std_time = torch.sqrt(variance.clamp_min(eps))
    skewness = (
        (tpsf * centered.pow(3)).sum(dim=1, keepdim=True) / safe_total
    ) / std_time.pow(3).clamp_min(eps)

    peak_amplitude, peak_index = tpsf.max(dim=1, keepdim=True)
    peak_time = peak_index.to(tpsf.dtype)

    features = torch.cat(
        [
            torch.log1p(total_counts),
            mean_time / n_bins,
            std_time / n_bins,
            skewness,
            peak_amplitude,
            peak_time / n_bins,
        ],
        dim=1,
    )

    return torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


# ============================================================
# 7. SIMPLE CONVOLUTION MODEL
# ============================================================

def choose_group_count(number_of_channels):
    for candidate in (4, 3, 2):
        if number_of_channels % candidate == 0:
            return candidate
    return 1


class TemporalEncoder(nn.Module):
    """Shared convolution design for the full TPSF and raw tail."""

    def __init__(
        self,
        filters_per_kernel=TEMPORAL_FILTERS_PER_KERNEL,
        pool_bins=TEMPORAL_POOL_BINS,
        feature_dim=TEMPORAL_FEATURE_DIM,
        dropout=TEMPORAL_DROPOUT,
    ):
        super().__init__()

        groups = choose_group_count(
            filters_per_kernel
        )

        def branch(kernel_size):
            return nn.Sequential(
                nn.Conv1d(
                    in_channels=1,
                    out_channels=filters_per_kernel,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=False,
                ),
                nn.GroupNorm(
                    groups,
                    filters_per_kernel,
                ),
                nn.ReLU(),
            )

        self.small_filter = branch(3)
        self.medium_filter = branch(7)
        self.large_filter = branch(15)
        self.pool_bins = int(pool_bins)

        projection_input = (
            3
            * 2
            * filters_per_kernel
            * self.pool_bins
        )

        self.output_projection = nn.Sequential(
            nn.Linear(
                projection_input,
                96,
            ),
            nn.LayerNorm(96),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                96,
                feature_dim,
            ),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
        )

    def pool(self, x):
        average = F.adaptive_avg_pool1d(
            x,
            self.pool_bins,
        )

        maximum = F.adaptive_max_pool1d(
            x,
            self.pool_bins,
        )

        return torch.cat(
            [average, maximum],
            dim=1,
        ).flatten(1)

    def forward(self, x):
        if (
            x.ndim != 3
            or x.shape[1] != 1
        ):
            raise ValueError(
                "TemporalEncoder expects [N,1,T], "
                f"received {tuple(x.shape)}."
            )

        features = torch.cat(
            [
                self.pool(
                    self.small_filter(x)
                ),
                self.pool(
                    self.medium_filter(x)
                ),
                self.pool(
                    self.large_filter(x)
                ),
            ],
            dim=1,
        )

        return self.output_projection(
            features
        )


class BottomMuaSimpleFullTailNet(nn.Module):
    """Full TPSF features + raw-tail features + moments + wavelength embedding."""

    def __init__(
        self,
        temporal_feature_dim=TEMPORAL_FEATURE_DIM,
        filters_per_kernel=TEMPORAL_FILTERS_PER_KERNEL,
        pool_bins=TEMPORAL_POOL_BINS,
        temporal_dropout=TEMPORAL_DROPOUT,
        head_dropout=HEAD_DROPOUT,
        moment_feature_dim=MOMENT_FEATURE_DIM,
        wavelength_embedding_dim=WAVELENGTH_EMBEDDING_DIM,
    ):
        super().__init__()

        self.temporal_feature_dim = int(
            temporal_feature_dim
        )
        self.moment_feature_dim = int(moment_feature_dim)

        self.full_encoder = TemporalEncoder(
            filters_per_kernel=filters_per_kernel,
            pool_bins=pool_bins,
            feature_dim=temporal_feature_dim,
            dropout=temporal_dropout,
        )

        self.tail_encoder = TemporalEncoder(
            filters_per_kernel=filters_per_kernel,
            pool_bins=pool_bins,
            feature_dim=temporal_feature_dim,
            dropout=temporal_dropout,
        )

        self.wavelength_encoder = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, wavelength_embedding_dim),
        )

        fused_dim = (
            2 * temporal_feature_dim
            + 2 * self.moment_feature_dim
            + wavelength_embedding_dim
        )

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

        nn.init.normal_(
            self.head[-1].weight,
            mean=0.0,
            std=1.0e-3,
        )

        nn.init.constant_(
            self.head[-1].bias,
            0.0,
        )

    def encode_raw_tails(
        self,
        full_channels,
        late_start,
    ):
        """Encode every exact raw tail with no padding or resampling.

        Samples with the same LATE_START are grouped and sent through the
        tail encoder together. Each group uses:
            full_channels[group, :, start:300]
        """

        starts = late_start.reshape(-1).to(
            device=full_channels.device,
            dtype=torch.long,
        )

        sample_count = full_channels.shape[0]

        if starts.numel() != sample_count:
            raise ValueError(
                "The number of LATE_START values does not "
                "match the number of TPSFs."
            )

        if (
            torch.any(starts < 0)
            or torch.any(
                starts >= N_TIME_GATES
            )
        ):
            raise ValueError(
                "LATE_START values must be within 0..299."
            )

        tail_features = full_channels.new_zeros(
            (
                sample_count,
                self.temporal_feature_dim,
            )
        )

        tail_moments = full_channels.new_zeros(
            (
                sample_count,
                self.moment_feature_dim,
            )
        )

        for unique_start in torch.unique(
            starts,
            sorted=True,
        ):
            start = int(
                unique_start.item()
            )

            sample_indices = torch.nonzero(
                starts == unique_start,
                as_tuple=False,
            ).reshape(-1)

            raw_tail = full_channels[
                sample_indices,
                :,
                start:,
            ]

            encoded_tail = self.tail_encoder(
                raw_tail
            )

            tail_features = tail_features.index_copy(
                0,
                sample_indices,
                encoded_tail,
            )

            group_moments = compute_moment_features(
                raw_tail.squeeze(1)
            )

            tail_moments = tail_moments.index_copy(
                0,
                sample_indices,
                group_moments,
            )

        return tail_features, tail_moments

    def forward(
        self,
        tpsf,
        wavelength,
        late_start,
    ):
        full_channels = build_raw_tpsf_channel(
            tpsf
        )

        full_features = self.full_encoder(
            full_channels
        )

        full_moments = compute_moment_features(tpsf)

        tail_features, tail_moments = self.encode_raw_tails(
            full_channels,
            late_start,
        )

        wavelength_embedding = self.wavelength_encoder(
            wavelength.to(dtype=full_features.dtype)
        )

        fused_features = torch.cat(
            [
                full_features,
                tail_features,
                full_moments,
                tail_moments,
                wavelength_embedding,
            ],
            dim=1,
        )

        return self.head(
            fused_features
        )


# ============================================================
# 8. RAW TARGET LOSS
# ============================================================

class RawMuaLoss(nn.Module):
    """Direct MAE loss in the original bottom-mua physical units."""

    def __init__(self):
        super().__init__()

    def forward(self, prediction, target):
        return F.l1_loss(
            prediction,
            target,
            reduction="mean",
        )


# ============================================================
# 9. VALIDATION
# ============================================================

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()

    loss_sum = 0.0
    absolute_error_sum = 0.0
    squared_error_sum = 0.0
    value_count = 0

    predictions = []
    targets = []

    for tpsf, target, wavelength, late_start in loader:
        tpsf = tpsf.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        wavelength = wavelength.to(device, non_blocking=True)
        late_start = late_start.to(device, non_blocking=True)

        tpsf, target, wavelength, late_start = flatten_image_batch(
            tpsf,
            target,
            wavelength,
            late_start,
        )

        prediction = model(
            tpsf,
            wavelength,
            late_start,
        )

        loss = criterion(
            prediction,
            target,
        )

        error = prediction - target
        count = error.numel()

        loss_sum += loss.item() * count
        absolute_error_sum += torch.abs(error).sum().item()
        squared_error_sum += torch.square(error).sum().item()
        value_count += count

        predictions.append(prediction.cpu())
        targets.append(target.cpu())

    predictions = torch.cat(predictions, dim=0)
    targets = torch.cat(targets, dim=0)

    return {
        "raw_loss": loss_sum / value_count,
        "bottom_mua_mae": absolute_error_sum / value_count,
        "bottom_mua_rmse": np.sqrt(squared_error_sum / value_count),
        "prediction_mean": predictions.mean().item(),
        "prediction_std": predictions.std(unbiased=False).item(),
        "target_mean": targets.mean().item(),
        "target_std": targets.std(unbiased=False).item(),
    }


def calculate_constant_baseline(train_files, val_files):
    train_values = []
    val_values = []

    for file_path in train_files:
        train_values.append(
            load_bottom_mua_target(loadmat(file_path), file_path)
        )

    for file_path in val_files:
        val_values.append(
            load_bottom_mua_target(loadmat(file_path), file_path)
        )

    train_values = np.concatenate(train_values, axis=0)
    val_values = np.concatenate(val_values, axis=0)

    constant_prediction = float(np.median(train_values))
    baseline_mae = float(
        np.mean(np.abs(val_values - constant_prediction))
    )

    return constant_prediction, baseline_mae


# ============================================================
# 10. TRAINING
# ============================================================

def train_model():
    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(
            f"Training directory was not found:\n{TRAIN_DIR}"
        )

    raw_wavelengths, normalized_wavelengths, wavelength_key = (
        load_wavelength_vector()
    )

    wavelength_absolute_path = os.path.normcase(
        os.path.abspath(WAVELENGTH_FILE)
    )

    all_files = []
    for file_path in sorted(glob(os.path.join(TRAIN_DIR, "*.mat"))):
        if (
            os.path.normcase(os.path.abspath(file_path))
            == wavelength_absolute_path
        ):
            continue
        all_files.append(file_path)

    if not all_files:
        raise FileNotFoundError(
            f"No training .mat files were found in:\n{TRAIN_DIR}"
        )

    split_rng = random.Random(SEED)
    split_rng.shuffle(all_files)

    split_index = int(TRAIN_FRACTION * len(all_files))
    train_files = all_files[:split_index]
    val_files = all_files[split_index:]

    if not train_files or not val_files:
        raise RuntimeError("Training or validation file list is empty.")

    tpsf_input_scale = estimate_tpsf_input_scale(train_files)
    summarize_raw_target(train_files)
    late_start_statistics = summarize_late_start_indices(train_files)

    constant_prediction, constant_baseline_mae = (
        calculate_constant_baseline(train_files, val_files)
    )

    print(
        f"\nConstant bottom-mua prediction: "
        f"{constant_prediction:.6e}"
    )
    print(
        f"Constant validation MAE:         "
        f"{constant_baseline_mae:.6e}"
    )

    train_dataset = BottomMuaDataset(
        file_list=train_files,
        tpsf_input_scale=tpsf_input_scale,
        normalized_wavelengths=normalized_wavelengths,
        augment=USE_TRAINING_AUGMENTATION,
    )

    val_dataset = BottomMuaDataset(
        file_list=val_files,
        tpsf_input_scale=tpsf_input_scale,
        normalized_wavelengths=normalized_wavelengths,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=IMAGE_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=IMAGE_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )

    print(f"\nTotal images:      {len(all_files)}")
    print(f"Training images:   {len(train_files)}")
    print(f"Validation images: {len(val_files)}")
    print(f"Training TPSFs:    {len(train_files) * N_WAVELENGTHS}")
    print(f"Validation TPSFs:  {len(val_files) * N_WAVELENGTHS}")
    print("Selected gates:    [0:300]")
    print("Late-tail mode:    raw samples + right padding")
    print(f"MATLAB one-based:  {LATE_START_IS_MATLAB_ONE_BASED}")
    print(f"Augmentation:      {USE_TRAINING_AUGMENTATION}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    model = BottomMuaSimpleFullTailNet().to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)

    criterion = RawMuaLoss().to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=SCHEDULER_PATIENCE,
        min_lr=1.0e-7,
    )

    with torch.no_grad():
        test_tpsf, test_target, test_wavelength, test_late_start = next(
            iter(train_loader)
        )

        (
            test_tpsf,
            test_target,
            test_wavelength,
            test_late_start,
        ) = flatten_image_batch(
            test_tpsf.to(device),
            test_target.to(device),
            test_wavelength.to(device),
            test_late_start.to(device),
        )

        test_prediction = model(
            test_tpsf,
            test_wavelength,
            test_late_start,
        )

        print("\nShape check")
        print("TPSF batch:       ", tuple(test_tpsf.shape))
        print("Wavelength batch: ", tuple(test_wavelength.shape))
        print("Late-start batch: ", tuple(test_late_start.shape))
        print("Target batch:     ", tuple(test_target.shape))
        print("Prediction batch: ", tuple(test_prediction.shape))
        print(
            "Late-start range in batch:",
            int(test_late_start.min().item()),
            "to",
            int(test_late_start.max().item()),
        )

    train_raw_loss_history = []
    train_mae_history = []
    val_raw_loss_history = []
    val_mae_history = []

    best_val_mae = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    plt.ion()
    figure, axis = plt.subplots(figsize=(9, 6))
    train_line, = axis.plot([], [], label="Train bottom-mua MAE")
    val_line, = axis.plot([], [], label="Validation bottom-mua MAE")
    axis.axhline(
        constant_baseline_mae,
        linestyle="--",
        label="Constant validation baseline",
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Raw bottom-mua MAE")
    axis.set_title("Dynamic late-start bottom-mua model")
    axis.grid(True)
    axis.legend()

    for epoch in range(NUM_EPOCHS):
        model.train()

        raw_loss_sum = 0.0
        absolute_error_sum = 0.0
        sample_count = 0

        for tpsf, target, wavelength, late_start in train_loader:
            tpsf = tpsf.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            wavelength = wavelength.to(device, non_blocking=True)
            late_start = late_start.to(device, non_blocking=True)

            tpsf, target, wavelength, late_start = flatten_image_batch(
                tpsf,
                target,
                wavelength,
                late_start,
            )

            optimizer.zero_grad(set_to_none=True)

            prediction = model(
                tpsf,
                wavelength,
                late_start,
            )

            loss = criterion(
                prediction,
                target,
            )

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite training loss detected."
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=GRADIENT_CLIP_NORM,
            )

            optimizer.step()

            raw_error = prediction.detach() - target
            count = raw_error.numel()

            raw_loss_sum += loss.item() * count
            absolute_error_sum += torch.abs(raw_error).sum().item()
            sample_count += count

        train_raw_loss = raw_loss_sum / sample_count
        train_mae = absolute_error_sum / sample_count

        metrics = validate(
            model,
            val_loader,
            criterion,
            device,
        )

        scheduler.step(metrics["bottom_mua_mae"])
        current_lr = optimizer.param_groups[0]["lr"]

        train_raw_loss_history.append(train_raw_loss)
        train_mae_history.append(train_mae)
        val_raw_loss_history.append(metrics["raw_loss"])
        val_mae_history.append(metrics["bottom_mua_mae"])

        print(
            f"Epoch [{epoch + 1:03d}/{NUM_EPOCHS}] | "
            f"Train raw loss={train_raw_loss:.6e} | "
            f"Val raw loss={metrics['raw_loss']:.6e} | "
            f"Train MAE={train_mae:.6e} | "
            f"Val MAE={metrics['bottom_mua_mae']:.6e} | "
            f"Val RMSE={metrics['bottom_mua_rmse']:.6e} | "
            f"Pred mean/std="
            f"{metrics['prediction_mean']:.6e}/"
            f"{metrics['prediction_std']:.6e} | "
            f"Target mean/std="
            f"{metrics['target_mean']:.6e}/"
            f"{metrics['target_std']:.6e} | "
            f"LR={current_lr:.2e}"
        )

        if metrics["bottom_mua_mae"] < best_val_mae - MIN_DELTA:
            best_val_mae = metrics["bottom_mua_mae"]
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            if isinstance(model, nn.DataParallel):
                best_state = copy.deepcopy(model.module.state_dict())
            else:
                best_state = copy.deepcopy(model.state_dict())

            print(f"  -> Best model updated at epoch {best_epoch}.")
        else:
            epochs_without_improvement += 1

        x_axis = np.arange(1, len(train_mae_history) + 1)
        train_line.set_xdata(x_axis)
        train_line.set_ydata(train_mae_history)
        val_line.set_xdata(x_axis)
        val_line.set_ydata(val_mae_history)
        axis.relim()
        axis.autoscale_view()
        figure.canvas.draw()
        figure.canvas.flush_events()

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print("\nEarly stopping activated.")
            break

    plt.ioff()

    if best_state is None:
        raise RuntimeError("No best model state was captured.")

    checkpoint = {
        "model_state_dict": best_state,
        "architecture": "bottom_mua_simple_full_tail_moments_wavelength_embedding",
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
        "moment_feature_dim": MOMENT_FEATURE_DIM,
        "wavelength_embedding_dim": WAVELENGTH_EMBEDDING_DIM,
        "tail_extraction": "raw_grouped_by_late_start",
        "wavelength_usage": "nonlinear_embedding",
        "temporal_dropout": TEMPORAL_DROPOUT,
        "head_dropout": HEAD_DROPOUT,
        "tpsf_input_scale": tpsf_input_scale,
        "wavelength_file": WAVELENGTH_FILE,
        "wavelength_key": wavelength_key,
        "raw_wavelengths": torch.from_numpy(raw_wavelengths),
        "normalized_wavelengths": torch.from_numpy(normalized_wavelengths),
        "training_augmentation": USE_TRAINING_AUGMENTATION,
        "best_epoch": best_epoch,
        "best_validation_bottom_mua_mae": best_val_mae,
        "constant_validation_bottom_mua_mae": constant_baseline_mae,
        "train_raw_loss_history": train_raw_loss_history,
        "val_raw_loss_history": val_raw_loss_history,
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
