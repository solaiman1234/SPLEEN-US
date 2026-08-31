# -*- coding: utf-8 -*-
"""
Inference for the global-context variant of the depth-resolved,
spectrally-smoothed bottom-mua model
(train_bottom_mua_spectral_smoothing_global_context.py). Fully standalone
-- rebuilds the exact model architecture here so a checkpoint produced by
that training script can be loaded and run without importing it. Not
interchangeable with checkpoints from train_bottom_mua_spectral_smoothing.py
(no global context) -- this script checks the checkpoint's architecture
identifier and refuses to load a mismatched one.

Each test image contains:
    - 169 wavelength-specific TPSFs
    - 450 selected time bins per TPSF
    - 169 LATE_START indices

For each wavelength-specific TPSF:
    full raw TPSF -> full convolution encoder (TemporalEncoder)
    raw TPSF tail -> depth-resolved sequence encoder (DepthResolvedTailEncoder)
    raw wavelength -> one direct scalar, no wavelength encoder
The fused per-wavelength features are then globally conditioned on the
whole spectrum (GlobalSpectralContext) and locally smoothed across all 169
wavelengths of the image (SpectralSmoother) before the regression head.

Two things differ from the older single-scalar-wavelength checkpoint format
this script's predecessor consumed:

1. The TPSF input is normalized by a per-wavelength scale vector
   (`per_wavelength_input_scale`, one value per wavelength) instead of a
   single dataset-wide scalar, so each of the 169 rows of a test TPSF must
   be divided by its own scale value.
2. The model's forward pass returns a tuple, (prediction, depth_profile),
   not a single tensor: 'prediction' is the bottom-mua estimate and
   'depth_profile' is the model's supervised depth-resolved absorption
   estimate for the deepest few tail windows of each TPSF (see
   train_bottom_mua_spectral_smoothing.py for how it's trained). Both are
   saved to the output file.

Because SpectralSmoother mixes information across the wavelength axis, a
call to the model must always be given a complete image's worth of TPSFs
(all 169 wavelengths at once, in wavelength order) -- never a subset. This
script always loads and predicts one full test image per file, so that
constraint is satisfied automatically.
"""



import os
import re

import numpy as np
from scipy.io import loadmat, savemat

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. USER SETTINGS
# ============================================================

MODEL_PATH = r"D:\DTOF_exp_simu_depth_resolved_spectral_global_context_bottom_mua.pth"

TEST_DIR = (
    r"D:\Simulation_TPSF_Convolution_Animal_IRF"
    r"\Phantom_2_DTOF"
)

TEST_FILE_PREFIX = "DTOF_"
TEST_FILE_EXTENSION = ".mat"
TEST_TPSF_KEY = "dtof_auc"

# Keep None to use the late-start variable name stored in the checkpoint.
TEST_LATE_START_KEY = "End_start"

# Output files are saved as pred_bottom_mua_1.mat,
# pred_bottom_mua_2.mat, pred_bottom_mua_3.mat, ...
OUTPUT_DIR = r"D:\decon_TPSF\pred_bottom_Phantom_2_DTOF"
OUTPUT_FILE_PREFIX = "pred_bottom_mua_"

# When True, one bad file is reported and the remaining files are processed.
CONTINUE_ON_ERROR = True

# Fallback only when wavelengths are absent from the checkpoint.
WAVELENGTH_FILE = (
    r"D:\Simulation_TPSF_Convolution_Animal_IRF"
    r"\wavelength_exp_index.mat"
)
WAVELENGTH_KEY = "wavelengths_exp"


# ============================================================
# 2. DEFAULT SETTINGS
# ============================================================
# These must match train_bottom_mua_spectral_smoothing.py's globals. Most
# are overridden per-checkpoint below via build_model_from_checkpoint;
# WINDOW_BINS/WINDOW_STRIDE cannot be (see the note on DepthResolvedTailEncoder
# below), so build_checkpoint validates the checkpoint was trained with
# these exact values instead of silently using the wrong window size.

N_WAVELENGTHS = 169
N_TIME_GATES = 450
TIME_GATE_START = 0
TIME_GATE_END = 450

EPS = 1.0e-8

DEFAULT_TEMPORAL_FILTERS_PER_KERNEL = 12
DEFAULT_TEMPORAL_POOL_BINS = 4
DEFAULT_TEMPORAL_FEATURE_DIM = 48
DEFAULT_TEMPORAL_DROPOUT = 0.15
DEFAULT_HEAD_DROPOUT = 0.15

DEFAULT_DEPTH_FEATURE_DIM = 16
DEFAULT_DEPTH_HIDDEN_DIM = 32
DEFAULT_NUM_SUPERVISED_TAIL_WINDOWS = 3

DEFAULT_SPECTRAL_SMOOTHING_KERNEL_SIZE = 7

# build_tail_windows/DepthResolvedTailEncoder read these as module-level
# defaults (not per-instance constructor state), matching how
# train_bottom_mua_spectral_smoothing.py defines them, so a checkpoint
# trained with different window settings cannot be replayed just by
# passing different constructor arguments -- see build_model_from_checkpoint.
WINDOW_BINS = 20
WINDOW_STRIDE = 10


# ============================================================
# 3. TEST DATA LOADING
# ============================================================

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
            f"TPSF has shape {x.shape}; one dimension must equal "
            f"{N_WAVELENGTHS}."
        )

    if oriented.shape[1] < N_TIME_GATES:
        raise ValueError(
            f"TPSF contains only {oriented.shape[1]} time gates; "
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


def load_late_start_indices(mat, file_path, key, matlab_one_based):
    if key not in mat:
        available = [name for name in mat if not name.startswith("__")]
        raise KeyError(
            f"'{key}' was not found in:\n{file_path}\n"
            f"Available variables: {available}"
        )

    late_start = np.asarray(mat[key])

    if late_start.ndim == 1:
        late_start = late_start.reshape(-1, 1)
    elif late_start.ndim == 2 and late_start.shape[0] == 1:
        late_start = late_start.T

    if late_start.ndim != 2 or late_start.shape[0] < N_WAVELENGTHS:
        raise ValueError(
            f"'{key}' has shape {late_start.shape}; expected [169,1] or [1,169]."
        )

    late_start = late_start[:N_WAVELENGTHS, :1].astype(np.float64)

    if not np.all(np.isfinite(late_start)):
        raise FloatingPointError("Non-finite late-start indices were found.")

    rounded = np.rint(late_start)
    if np.max(np.abs(late_start - rounded)) > 1.0e-4:
        raise ValueError("Late-start values must be integer indices.")

    late_start = rounded.astype(np.int64)

    if matlab_one_based:
        late_start = late_start - 1

    late_start = late_start - TIME_GATE_START

    if np.any(late_start < 0) or np.any(late_start >= N_TIME_GATES):
        raise ValueError(
            f"Converted late-start indices range from {late_start.min()} "
            f"to {late_start.max()}; valid Python indices are 0 to "
            f"{N_TIME_GATES - 1}."
        )

    return late_start.reshape(N_WAVELENGTHS, 1)


def load_test_data(checkpoint, test_file):
    if not os.path.isfile(test_file):
        raise FileNotFoundError(f"Test file was not found:\n{test_file}")

    mat = loadmat(test_file)

    if TEST_TPSF_KEY not in mat:
        available = {
            key: np.asarray(value).shape
            for key, value in mat.items()
            if not key.startswith("__")
        }
        raise KeyError(
            f"'{TEST_TPSF_KEY}' was not found in:\n{test_file}\n"
            f"Available variables and shapes: {available}"
        )

    late_start_key = TEST_LATE_START_KEY
    if late_start_key is None:
        late_start_key = checkpoint.get("late_start_key", "LATE_START")

    matlab_one_based = bool(checkpoint.get("late_start_is_matlab_one_based", True))

    tpsf = clean_tpsf_array(mat[TEST_TPSF_KEY], test_file)
    late_start = load_late_start_indices(mat, test_file, late_start_key, matlab_one_based)

    print(f"TPSF variable: {TEST_TPSF_KEY}")
    print(f"Late-start variable: {late_start_key}")
    print(f"MATLAB one-based indices: {matlab_one_based}")
    print(f"TPSF shape after cropping: {tpsf.shape}")
    print(f"Late-start range after conversion: {late_start.min()} to {late_start.max()}")

    return tpsf, late_start, late_start_key, matlab_one_based


# ============================================================
# 4. WAVELENGTH AND PER-WAVELENGTH SCALE LOADING
# ============================================================

def normalize_wavelength_vector(array):
    wavelengths = np.asarray(array, dtype=np.float32).reshape(-1)

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

    return (
        wavelengths.reshape(N_WAVELENGTHS, 1).astype(np.float32),
        normalized.reshape(N_WAVELENGTHS, 1).astype(np.float32),
    )


def load_external_wavelengths():
    if not os.path.isfile(WAVELENGTH_FILE):
        raise FileNotFoundError(
            "Wavelengths were not stored in the checkpoint and the "
            f"external file was not found:\n{WAVELENGTH_FILE}"
        )

    mat = loadmat(WAVELENGTH_FILE)

    if WAVELENGTH_KEY not in mat:
        available = [key for key in mat if not key.startswith("__")]
        raise KeyError(f"'{WAVELENGTH_KEY}' was not found. Available variables: {available}")

    raw, normalized = normalize_wavelength_vector(mat[WAVELENGTH_KEY])
    return raw, normalized, f"external:{WAVELENGTH_KEY}"


def get_wavelengths(checkpoint):
    """Returns (raw, normalized, source). 'raw' is what the model actually
    consumes -- this checkpoint format concatenates the true wavelength
    directly, with no learned encoder or normalization.
    """

    if "raw_wavelengths" in checkpoint:
        raw = torch.as_tensor(
            checkpoint["raw_wavelengths"], dtype=torch.float32
        ).cpu().numpy().reshape(-1)

        if raw.size != N_WAVELENGTHS:
            raise ValueError(
                f"Checkpoint contains {raw.size} raw wavelengths; expected {N_WAVELENGTHS}."
            )

        raw = raw.reshape(N_WAVELENGTHS, 1).astype(np.float32)

        if "normalized_wavelengths" in checkpoint:
            normalized = torch.as_tensor(
                checkpoint["normalized_wavelengths"], dtype=torch.float32
            ).cpu().numpy().reshape(N_WAVELENGTHS, 1).astype(np.float32)
        else:
            normalized = np.full((N_WAVELENGTHS, 1), np.nan, dtype=np.float32)

        return raw, normalized, "checkpoint"

    return load_external_wavelengths()


def get_per_wavelength_scale(checkpoint):
    """The [169,1] per-wavelength TPSF input scale used at training time.

    Unlike the earlier single-scalar-wavelength checkpoint format, this
    model normalizes each wavelength's TPSF row by its own scale, so this
    must be a full vector, not one shared float.
    """

    if "per_wavelength_input_scale" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain 'per_wavelength_input_scale'. "
            "This inference script only supports checkpoints produced by "
            "train_bottom_mua_spectral_smoothing.py."
        )

    scale = np.asarray(checkpoint["per_wavelength_input_scale"], dtype=np.float32)
    scale = scale.reshape(N_WAVELENGTHS, 1)

    if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("Checkpoint 'per_wavelength_input_scale' has non-positive or non-finite values.")

    return scale


# ============================================================
# 5. RAW TPSF CHANNEL
# ============================================================

def build_raw_tpsf_channel(tpsf):
    """Convert [N,T] raw TPSFs to [N,1,T]."""

    if tpsf.ndim != 2 or tpsf.shape[1] != N_TIME_GATES:
        raise ValueError(f"Expected [N,{N_TIME_GATES}], received {tuple(tpsf.shape)}.")

    tpsf = torch.nan_to_num(tpsf, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.clamp(tpsf, min=0.0).unsqueeze(1)


# ============================================================
# 6. FULL-TPSF CONVOLUTION ENCODER
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
        filters_per_kernel=DEFAULT_TEMPORAL_FILTERS_PER_KERNEL,
        pool_bins=DEFAULT_TEMPORAL_POOL_BINS,
        feature_dim=DEFAULT_TEMPORAL_FEATURE_DIM,
        dropout=DEFAULT_TEMPORAL_DROPOUT,
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
# 7. DEPTH-RESOLVED TAIL ENCODER
# ============================================================

def build_tail_windows(raw_tail, window_bins=WINDOW_BINS, stride=WINDOW_STRIDE):
    """Slice a raw tail [N,T] into an ordered window sequence [N,K,window_bins]."""

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
    """windows: [N,K,window_bins] -> [N,K,3] (log-amplitude decay slope,
    mean amplitude, peak amplitude)."""

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
        depth_feature_dim=DEFAULT_DEPTH_FEATURE_DIM,
        hidden_dim=DEFAULT_DEPTH_HIDDEN_DIM,
        dropout=DEFAULT_TEMPORAL_DROPOUT,
    ):
        super().__init__()

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


# ============================================================
# 8. SPECTRAL SMOOTHER
# ============================================================

class SpectralSmoother(nn.Module):
    """Residual depthwise-separable smoothing across the wavelength axis.

    Requires the batch to contain one or more complete images (every one
    of the N_WAVELENGTHS rows of an image present together).
    """

    def __init__(
        self,
        feature_dim,
        kernel_size=DEFAULT_SPECTRAL_SMOOTHING_KERNEL_SIZE,
        n_wavelengths=N_WAVELENGTHS,
    ):
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

    def forward(self, fused_features):
        n_total, feature_dim = fused_features.shape

        if feature_dim != self.feature_dim:
            raise ValueError(f"Expected feature dimension {self.feature_dim}, received {feature_dim}.")

        if n_total % self.n_wavelengths != 0:
            raise ValueError(
                f"Batch size {n_total} is not a multiple of {self.n_wavelengths} "
                "wavelengths; spectral mixing requires whole images (all "
                "wavelengths of an image present together in the batch)."
            )

        n_images = n_total // self.n_wavelengths

        sequence = fused_features.view(n_images, self.n_wavelengths, feature_dim)
        sequence = sequence.transpose(1, 2)

        residual = self.pointwise(F.relu(self.norm(self.depthwise(sequence))))
        mixed = sequence + residual

        mixed = mixed.transpose(1, 2).reshape(n_total, feature_dim)
        return mixed


class GlobalSpectralContext(nn.Module):
    """Whole-spectrum (all N_WAVELENGTHS at once) global context conditioning.

    Pools the fused per-wavelength features across the entire spectrum of
    one image into a single per-image context vector, and uses it to
    apply a FiLM-style affine correction to every wavelength's features.
    """

    def __init__(self, feature_dim, n_wavelengths=N_WAVELENGTHS, hidden_dim=None):
        super().__init__()

        self.feature_dim = int(feature_dim)
        self.n_wavelengths = int(n_wavelengths)
        hidden_dim = int(hidden_dim) if hidden_dim is not None else self.feature_dim

        self.context_encoder = nn.Sequential(
            nn.Linear(2 * self.feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.film = nn.Linear(hidden_dim, 2 * self.feature_dim)

    def forward(self, fused_features):
        n_total, feature_dim = fused_features.shape

        if feature_dim != self.feature_dim:
            raise ValueError(
                f"Expected feature dimension {self.feature_dim}, received {feature_dim}."
            )

        if n_total % self.n_wavelengths != 0:
            raise ValueError(
                f"Batch size {n_total} is not a multiple of {self.n_wavelengths} "
                "wavelengths; global context requires whole images (all "
                "wavelengths of an image present together in the batch)."
            )

        n_images = n_total // self.n_wavelengths

        sequence = fused_features.view(n_images, self.n_wavelengths, feature_dim)

        mean_context = sequence.mean(dim=1)
        max_context = sequence.amax(dim=1)
        pooled_context = torch.cat([mean_context, max_context], dim=1)

        context = self.context_encoder(pooled_context)
        film_parameters = self.film(context)
        scale, shift = film_parameters.chunk(2, dim=1)

        modulated = sequence * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        return modulated.reshape(n_total, feature_dim)


# ============================================================
# 9. FULL MODEL
# ============================================================

class BottomMuaSpectralNet(nn.Module):
    """Full-TPSF features + depth-resolved tail sequence + raw wavelength
    scalar, spectrally smoothed across the wavelength axis before the
    regression head.
    """

    def __init__(
        self,
        temporal_feature_dim=DEFAULT_TEMPORAL_FEATURE_DIM,
        filters_per_kernel=DEFAULT_TEMPORAL_FILTERS_PER_KERNEL,
        pool_bins=DEFAULT_TEMPORAL_POOL_BINS,
        temporal_dropout=DEFAULT_TEMPORAL_DROPOUT,
        head_dropout=DEFAULT_HEAD_DROPOUT,
        depth_hidden_dim=DEFAULT_DEPTH_HIDDEN_DIM,
        num_supervised_windows=DEFAULT_NUM_SUPERVISED_TAIL_WINDOWS,
        spectral_kernel_size=DEFAULT_SPECTRAL_SMOOTHING_KERNEL_SIZE,
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

        fused_dim = temporal_feature_dim + depth_hidden_dim + 1

        self.global_context = GlobalSpectralContext(fused_dim)

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

    def encode_raw_tails(self, tpsf, late_start):
        starts = late_start.reshape(-1).to(device=tpsf.device, dtype=torch.long)
        sample_count = tpsf.shape[0]

        if starts.numel() != sample_count:
            raise ValueError("The number of LATE_START values does not match the number of TPSFs.")

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
        fused_features = self.global_context(fused_features)
        fused_features = self.spectral_smoother(fused_features)

        prediction = self.head(fused_features)

        return prediction, supervised_profiles


# ============================================================
# 10. CHECKPOINT LOADING
# ============================================================

def load_checkpoint(device):
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model checkpoint was not found:\n{MODEL_PATH}")

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(MODEL_PATH, map_location=device)

    expected_architecture = "bottom_mua_depth_resolved_spectral_smoothing_global_context"

    if checkpoint.get("architecture") != expected_architecture:
        raise RuntimeError(
            f"Checkpoint architecture is '{checkpoint.get('architecture')}', "
            f"but this script requires '{expected_architecture}'."
        )

    if int(checkpoint.get("n_wavelengths", N_WAVELENGTHS)) != N_WAVELENGTHS:
        raise RuntimeError("Checkpoint wavelength count does not match.")

    if int(checkpoint.get("n_time_gates", N_TIME_GATES)) != N_TIME_GATES:
        raise RuntimeError("Checkpoint time-gate count does not match.")

    if int(checkpoint.get("time_gate_start", TIME_GATE_START)) != TIME_GATE_START:
        raise RuntimeError("Checkpoint time-gate start does not match.")

    if int(checkpoint.get("time_gate_end_exclusive", TIME_GATE_END)) != TIME_GATE_END:
        raise RuntimeError("Checkpoint time-gate end does not match.")

    tail_mode = checkpoint.get("tail_extraction")
    if tail_mode not in (None, "depth_resolved_grouped_by_late_start"):
        raise RuntimeError(f"Unexpected checkpoint tail_extraction: {tail_mode!r}.")

    wavelength_mode = checkpoint.get("wavelength_usage")
    if wavelength_mode not in (None, "raw_scalar_direct"):
        raise RuntimeError(f"Unexpected checkpoint wavelength_usage: {wavelength_mode!r}.")

    scale_mode = checkpoint.get("wavelength_scale_mode")
    if scale_mode not in (None, "per_wavelength"):
        raise RuntimeError(f"Unexpected checkpoint wavelength_scale_mode: {scale_mode!r}.")

    if "per_wavelength_input_scale" not in checkpoint:
        raise KeyError("Checkpoint does not contain 'per_wavelength_input_scale'.")

    stored_window_bins = int(checkpoint.get("window_bins", WINDOW_BINS))
    stored_window_stride = int(checkpoint.get("window_stride", WINDOW_STRIDE))

    if stored_window_bins != WINDOW_BINS or stored_window_stride != WINDOW_STRIDE:
        raise RuntimeError(
            f"Checkpoint was trained with window_bins={stored_window_bins}, "
            f"window_stride={stored_window_stride}, but this script's "
            f"WINDOW_BINS/WINDOW_STRIDE constants are {WINDOW_BINS}/"
            f"{WINDOW_STRIDE}. Edit those constants at the top of this file "
            "to match the checkpoint before running inference."
        )

    return checkpoint


def build_model_from_checkpoint(checkpoint, device):
    model = BottomMuaSpectralNet(
        temporal_feature_dim=int(
            checkpoint.get("temporal_feature_dim", DEFAULT_TEMPORAL_FEATURE_DIM)
        ),
        filters_per_kernel=int(
            checkpoint.get("temporal_filters_per_kernel", DEFAULT_TEMPORAL_FILTERS_PER_KERNEL)
        ),
        pool_bins=int(
            checkpoint.get("temporal_pool_bins", DEFAULT_TEMPORAL_POOL_BINS)
        ),
        temporal_dropout=float(
            checkpoint.get("temporal_dropout", DEFAULT_TEMPORAL_DROPOUT)
        ),
        head_dropout=float(
            checkpoint.get("head_dropout", DEFAULT_HEAD_DROPOUT)
        ),
        depth_hidden_dim=int(
            checkpoint.get("depth_hidden_dim", DEFAULT_DEPTH_HIDDEN_DIM)
        ),
        num_supervised_windows=int(
            checkpoint.get("num_supervised_tail_windows", DEFAULT_NUM_SUPERVISED_TAIL_WINDOWS)
        ),
        spectral_kernel_size=int(
            checkpoint.get("spectral_smoothing_kernel_size", DEFAULT_SPECTRAL_SMOOTHING_KERNEL_SIZE)
        ),
    ).to(device)

    state_dict = checkpoint["model_state_dict"]

    cleaned_state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }

    model.load_state_dict(cleaned_state, strict=True)
    model.eval()

    return model


# ============================================================
# 11. INFERENCE
# ============================================================

def extract_file_index(file_path):
    """Return the numeric index from <TEST_FILE_PREFIX><i>.mat."""

    normalized_path = file_path.replace("\\", os.sep)
    file_name = os.path.basename(normalized_path)

    pattern = r"^" + re.escape(TEST_FILE_PREFIX) + r"(\d+)" + re.escape(TEST_FILE_EXTENSION) + r"$"

    match = re.match(pattern, file_name, flags=re.IGNORECASE)

    if match is None:
        raise ValueError(
            f"Filename '{file_name}' does not match "
            f"'{TEST_FILE_PREFIX}<number>{TEST_FILE_EXTENSION}'."
        )

    return int(match.group(1))


def find_test_files():
    """Find and numerically sort files matching the configured prefix."""

    if not os.path.isdir(TEST_DIR):
        raise FileNotFoundError(f"Test directory was not found:\n{TEST_DIR}")

    indexed_files = []

    for file_name in os.listdir(TEST_DIR):
        file_path = os.path.join(TEST_DIR, file_name)

        if not os.path.isfile(file_path):
            continue

        try:
            file_index = extract_file_index(file_path)
        except ValueError:
            continue

        indexed_files.append((file_index, file_path))

    indexed_files.sort(key=lambda item: item[0])

    if not indexed_files:
        raise FileNotFoundError(
            f"No files matching '{TEST_FILE_PREFIX}<number>{TEST_FILE_EXTENSION}' "
            f"were found in:\n{TEST_DIR}"
        )

    indices = [file_index for file_index, _ in indexed_files]

    if len(indices) != len(set(indices)):
        raise RuntimeError("Two or more input files use the same numeric index.")

    return indexed_files


@torch.no_grad()
def predict_one_file(
    model,
    checkpoint,
    device,
    test_file,
    file_index,
    raw_wavelengths,
    normalized_wavelengths,
    per_wavelength_scale,
    wavelength_source,
):
    """Predict one input MAT file and save pred_bottom_mua_i.mat."""

    tpsf, late_start, late_start_key, matlab_one_based = load_test_data(checkpoint, test_file)

    # Each wavelength's TPSF row is normalized by its own scale, not one
    # shared scale for the whole image.
    scaled_tpsf = (tpsf / per_wavelength_scale).astype(np.float32)

    tpsf_tensor = torch.from_numpy(scaled_tpsf).to(device)
    wavelength_tensor = torch.from_numpy(raw_wavelengths).to(device)
    late_start_tensor = torch.from_numpy(late_start.astype(np.int64)).to(device)

    # RawMuaLoss trains the network directly against bottom_absorption_mul
    # in physical units, so the model output is already the prediction.
    # forward() returns (prediction, depth_profile); depth_profile is the
    # model's own deep-supervised estimate for the deepest few tail
    # windows of each TPSF, saved below for inspection.
    raw_prediction, depth_profile = model(tpsf_tensor, wavelength_tensor, late_start_tensor)

    prediction = raw_prediction.cpu().numpy().astype(np.float32)
    depth_profile = depth_profile.cpu().numpy().astype(np.float32)

    expected_shape = (N_WAVELENGTHS, 1)
    if prediction.shape != expected_shape:
        raise RuntimeError(f"Prediction shape is {prediction.shape}; expected {expected_shape}.")

    late_start_stored = late_start.copy()
    if matlab_one_based:
        late_start_stored = late_start_stored + 1
    late_start_stored = late_start_stored + TIME_GATE_START

    result = {
        "pred_bottom_absorption_mul": prediction,
        "prediction_raw_model_output": prediction.copy(),
        "depth_profile_supervised_windows": depth_profile,
        "wavelengths_raw": raw_wavelengths,
        "wavelengths_normalized": normalized_wavelengths,
        "per_wavelength_tpsf_input_scale": per_wavelength_scale,
        "late_start_python_zero_based": late_start.astype(np.int32),
        "late_start_stored_convention": late_start_stored.astype(np.int32),
        "late_start_normalized": late_start.astype(np.float32) / float(N_TIME_GATES - 1),
        "late_start_is_matlab_one_based": np.array([[int(matlab_one_based)]], dtype=np.int32),
        "source_file_index": np.array([[file_index]], dtype=np.int32),
        "source_file_name": np.array([[os.path.basename(test_file)]], dtype=object),
        "selected_time_gate_start": np.array([[TIME_GATE_START]], dtype=np.int32),
        "selected_time_gate_end_exclusive": np.array([[TIME_GATE_END]], dtype=np.int32),
        "best_validation_bottom_mua_mae": np.array(
            [[float(checkpoint.get("best_validation_bottom_mua_mae", np.nan))]],
            dtype=np.float32,
        ),
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_file = os.path.join(
        OUTPUT_DIR, f"{OUTPUT_FILE_PREFIX}{file_index}{TEST_FILE_EXTENSION}"
    )

    savemat(output_file, result, do_compression=True)

    print(f"Prediction range: {prediction.min():.6e} to {prediction.max():.6e}")
    print(f"Prediction mean/std: {prediction.mean():.6e} / {prediction.std():.6e}")
    print(f"Late-start variable: {late_start_key}")
    print(f"Wavelength source: {wavelength_source}")
    print(f"Saved to:\n{output_file}")

    return output_file


@torch.no_grad()
def run_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Checkpoint: {MODEL_PATH}")
    print(f"Input directory: {TEST_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    checkpoint = load_checkpoint(device)
    model = build_model_from_checkpoint(checkpoint, device)

    raw_wavelengths, normalized_wavelengths, wavelength_source = get_wavelengths(checkpoint)
    per_wavelength_scale = get_per_wavelength_scale(checkpoint)

    test_files = find_test_files()

    print(f"\nFound {len(test_files)} test files.")

    completed_files = []
    failed_files = []

    for position, (file_index, test_file) in enumerate(test_files, start=1):
        print("\n" + "=" * 70)
        print(f"Processing {position}/{len(test_files)}: {os.path.basename(test_file)}")

        try:
            output_file = predict_one_file(
                model=model,
                checkpoint=checkpoint,
                device=device,
                test_file=test_file,
                file_index=file_index,
                raw_wavelengths=raw_wavelengths,
                normalized_wavelengths=normalized_wavelengths,
                per_wavelength_scale=per_wavelength_scale,
                wavelength_source=wavelength_source,
            )

            completed_files.append(output_file)

        except Exception as error:
            failed_files.append((test_file, str(error)))
            print(f"ERROR while processing {os.path.basename(test_file)}:\n{error}")

            if not CONTINUE_ON_ERROR:
                raise

    print("\n" + "=" * 70)
    print("Batch inference completed.")
    print(f"Successful files: {len(completed_files)}")
    print(f"Failed files: {len(failed_files)}")

    if failed_files:
        print("\nFailed-file summary:")
        for failed_file, message in failed_files:
            print(f"- {os.path.basename(failed_file)}: {message}")


if __name__ == "__main__":
    run_inference()
