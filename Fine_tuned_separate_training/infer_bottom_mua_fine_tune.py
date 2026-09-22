# -*- coding: utf-8 -*-
"""
Load a saved BottomMuaSpectralNet checkpoint and predict bottom_absorption_mul
on your test .mat files. Set CHECKPOINT_PATH and TEST_DATA_PATH below, then run.
"""

import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat, savemat

#CHECKPOINT_PATH = r"D:\DTOF_exp_simu_depth_resolved_spectral_bottom_mua_v2_finetuned_exp.pth"
CHECKPOINT_PATH = r"D:\simulated_train_1_2300_finetuned.pth"

TEST_DATA_PATH = (
    r"D:\Simulation_TPSF_Convolution_Animal_IRF"
    r"\Phantom_4_DTOF"
)
 # a .mat file OR a folder of .mat files
#OUTPUT_DIR = r"D:\decon_TPSF\pred_bottom_Phantom_2_DTOF" # one .mat file per input file is saved here

OUTPUT_DIR = r"D:\decon_TPSF\pred_bottom_Phantom_4_DTOF_fine_tuned_only_experimental" # one .mat file per input file is saved here
TPSF_KEY = "dtof_auc"

EPS = 1.0e-8


# ============================================================
# Model architecture (mirrors train_bottom_mua_spectral_smoothing.py)
# ============================================================

def choose_group_count(number_of_channels):
    for candidate in (4, 3, 2):
        if number_of_channels % candidate == 0:
            return candidate
    return 1


class TemporalEncoder(nn.Module):
    def __init__(self, filters_per_kernel, pool_bins, feature_dim, dropout):
        super().__init__()

        groups = choose_group_count(filters_per_kernel)

        def branch(kernel_size):
            return nn.Sequential(
                nn.Conv1d(1, filters_per_kernel, kernel_size=kernel_size,
                           padding=kernel_size // 2, bias=False),
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
        features = torch.cat(
            [self.pool(self.small_filter(x)), self.pool(self.medium_filter(x)), self.pool(self.large_filter(x))],
            dim=1,
        )
        return self.output_projection(features)


def build_tail_windows(raw_tail, window_bins, stride):
    n_samples, tail_length = raw_tail.shape

    if tail_length <= window_bins:
        padded = raw_tail.new_zeros((n_samples, window_bins))
        padded[:, window_bins - tail_length:] = raw_tail
        return padded.unsqueeze(1)

    starts = list(range(0, tail_length - window_bins + 1, stride))
    if starts[-1] != tail_length - window_bins:
        starts.append(tail_length - window_bins)

    return torch.stack([raw_tail[:, start:start + window_bins] for start in starts], dim=1)


def compute_window_slope_features(windows, eps=EPS):
    n_samples, n_windows, window_bins = windows.shape

    log_amplitude = torch.log1p(windows.clamp_min(0.0))

    time_index = torch.arange(window_bins, device=windows.device, dtype=windows.dtype)
    time_centered = time_index - time_index.mean()
    time_variance = time_centered.pow(2).sum().clamp_min(eps)

    mean_log_amplitude = log_amplitude.mean(dim=2, keepdim=True)
    slope = ((log_amplitude - mean_log_amplitude) * time_centered.view(1, 1, -1)).sum(dim=2) / time_variance

    mean_amplitude = windows.mean(dim=2)
    peak_amplitude = windows.amax(dim=2)

    features = torch.stack([slope, mean_amplitude, peak_amplitude], dim=2)
    return torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


class DepthResolvedTailEncoder(nn.Module):
    def __init__(self, window_bins, depth_feature_dim, hidden_dim, dropout):
        super().__init__()

        self.window_bins = int(window_bins)
        window_input_dim = self.window_bins + 3 + 1

        self.window_encoder = nn.Sequential(
            nn.Linear(window_input_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, depth_feature_dim),
            nn.LayerNorm(depth_feature_dim),
            nn.ReLU(),
        )

        self.sequence_model = nn.GRU(input_size=depth_feature_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.depth_readout = nn.Linear(hidden_dim, 1)

    def forward(self, raw_tail, window_stride):
        windows = build_tail_windows(raw_tail, self.window_bins, window_stride)
        n_samples, n_windows, _ = windows.shape

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


class SpectralSmoother(nn.Module):
    def __init__(self, feature_dim, kernel_size, n_wavelengths):
        super().__init__()

        self.feature_dim = int(feature_dim)
        self.n_wavelengths = int(n_wavelengths)
        padding = kernel_size // 2

        self.depthwise = nn.Conv1d(feature_dim, feature_dim, kernel_size=kernel_size, padding=padding, groups=feature_dim, bias=False)
        self.norm = nn.GroupNorm(choose_group_count(feature_dim), feature_dim)
        self.pointwise = nn.Conv1d(feature_dim, feature_dim, kernel_size=1)

    def forward(self, fused_features):
        n_total, feature_dim = fused_features.shape
        n_images = n_total // self.n_wavelengths

        sequence = fused_features.view(n_images, self.n_wavelengths, feature_dim).transpose(1, 2)
        residual = self.pointwise(F.relu(self.norm(self.depthwise(sequence))))
        mixed = sequence + residual

        return mixed.transpose(1, 2).reshape(n_total, feature_dim)


class BottomMuaSpectralNet(nn.Module):
    def __init__(
        self,
        n_wavelengths,
        temporal_feature_dim,
        filters_per_kernel,
        pool_bins,
        temporal_dropout,
        head_dropout,
        depth_feature_dim,
        depth_hidden_dim,
        window_bins,
        window_stride,
        num_supervised_windows,
        spectral_kernel_size,
    ):
        super().__init__()

        self.n_wavelengths = int(n_wavelengths)
        self.window_stride = int(window_stride)
        self.tail_hidden_dim = int(depth_hidden_dim)
        self.num_supervised_windows = int(num_supervised_windows)

        self.full_encoder = TemporalEncoder(filters_per_kernel, pool_bins, temporal_feature_dim, temporal_dropout)
        self.tail_sequence_encoder = DepthResolvedTailEncoder(window_bins, depth_feature_dim, depth_hidden_dim, temporal_dropout)

        fused_dim = temporal_feature_dim + depth_hidden_dim + 1
        self.spectral_smoother = SpectralSmoother(fused_dim, spectral_kernel_size, self.n_wavelengths)

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

        final_features = tpsf.new_zeros((sample_count, self.tail_hidden_dim))
        supervised_profiles = tpsf.new_zeros((sample_count, self.num_supervised_windows))

        for unique_start in torch.unique(starts, sorted=True):
            start = int(unique_start.item())
            sample_indices = torch.nonzero(starts == unique_start, as_tuple=False).reshape(-1)
            raw_tail = tpsf[sample_indices, start:]

            final_feature, depth_profile = self.tail_sequence_encoder(raw_tail, self.window_stride)

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
        full_channels = torch.clamp(torch.nan_to_num(tpsf, nan=0.0, posinf=0.0, neginf=0.0), min=0.0).unsqueeze(1)
        full_features = self.full_encoder(full_channels)

        tail_final_features, supervised_profiles = self.encode_raw_tails(tpsf, late_start)

        wavelength_input = wavelength.to(dtype=full_features.dtype)
        fused_features = torch.cat([full_features, tail_final_features, wavelength_input], dim=1)
        fused_features = self.spectral_smoother(fused_features)

        prediction = self.head(fused_features)
        return prediction, supervised_profiles


# ============================================================
# Load checkpoint, load test data, predict
# ============================================================

def clean_tpsf_array(x, n_wavelengths, n_time_gates):
    x = np.asarray(x)
    oriented = x if x.shape[0] == n_wavelengths else x.T
    oriented = oriented[:, :n_time_gates]
    oriented = np.asarray(oriented, dtype=np.float32)
    oriented = np.nan_to_num(oriented, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(oriented, 0.0, None)


def load_late_start_indices(mat, late_start_key, n_wavelengths, n_time_gates, matlab_one_based):
    late_start = np.asarray(mat[late_start_key])
    if late_start.ndim == 1:
        late_start = late_start.reshape(-1, 1)
    elif late_start.ndim == 2 and late_start.shape[0] == 1:
        late_start = late_start.T

    late_start = late_start[:n_wavelengths, :1].astype(np.float64)
    rounded = np.rint(late_start).astype(np.int64)
    if matlab_one_based:
        rounded = rounded - 1
    rounded = np.clip(rounded, 0, n_time_gates - 1)
    return rounded.reshape(n_wavelengths, 1)


def resolve_input_files(input_path):
    if os.path.isdir(input_path):
        return sorted(glob.glob(os.path.join(input_path, "*.mat")))
    return [input_path]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

model = BottomMuaSpectralNet(
    n_wavelengths=checkpoint["n_wavelengths"],
    temporal_feature_dim=checkpoint["temporal_feature_dim"],
    filters_per_kernel=checkpoint["temporal_filters_per_kernel"],
    pool_bins=checkpoint["temporal_pool_bins"],
    temporal_dropout=checkpoint["temporal_dropout"],
    head_dropout=checkpoint["head_dropout"],
    depth_feature_dim=checkpoint["depth_feature_dim"],
    depth_hidden_dim=checkpoint["depth_hidden_dim"],
    window_bins=checkpoint["window_bins"],
    window_stride=checkpoint["window_stride"],
    num_supervised_windows=checkpoint["num_supervised_tail_windows"],
    spectral_kernel_size=checkpoint["spectral_smoothing_kernel_size"],
)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

n_wavelengths = checkpoint["n_wavelengths"]
n_time_gates = checkpoint["n_time_gates"]
late_start_key = checkpoint["late_start_key"]
matlab_one_based = checkpoint["late_start_is_matlab_one_based"]
raw_wavelengths = checkpoint["raw_wavelengths"].cpu().numpy().reshape(-1)
wavelength_tensor = checkpoint["normalized_wavelengths"].to(device=device, dtype=torch.float32)

os.makedirs(OUTPUT_DIR, exist_ok=True)

for file_path in resolve_input_files(TEST_DATA_PATH):
    mat = loadmat(file_path)

    tpsf = clean_tpsf_array(mat[TPSF_KEY], n_wavelengths, n_time_gates)
    late_start = load_late_start_indices(mat, late_start_key, n_wavelengths, n_time_gates, matlab_one_based)

    tpsf_tensor = torch.from_numpy(tpsf).to(device)
    late_start_tensor = torch.from_numpy(late_start).to(device)

    with torch.no_grad():
        prediction, _ = model(tpsf_tensor, wavelength_tensor, late_start_tensor)

    prediction = prediction.cpu().numpy().reshape(-1, 1).astype(np.float32)

    print(f"{os.path.basename(file_path)}: predicted bottom_mua min={prediction.min():.6e} "
          f"median={np.median(prediction):.6e} max={prediction.max():.6e}")

    output_name = os.path.splitext(os.path.basename(file_path))[0] + "_predicted.mat"
    output_path = os.path.join(OUTPUT_DIR, output_name)
    savemat(output_path, {
        "wavelengths_exp": raw_wavelengths.reshape(-1, 1),
        "predicted_bottom_absorption_mul": prediction,
    })

print(f"\nSaved predictions to {OUTPUT_DIR}")
