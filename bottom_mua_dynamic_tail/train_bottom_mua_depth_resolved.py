# -*- coding: utf-8 -*-
"""
Depth-resolved bottom-mua model.

Extends train_bottom_mua_dynamic_tail.py's full-TPSF + raw-tail design with
an explicit depth-resolved sequence over the tail.

Motivation: in a diffusive medium, photons collected at progressively later
times in a DTOF have, on average, travelled deeper into the tissue before
returning to the detector, and for late times the tail's log-amplitude decay
slope is a direct, low-variance estimator of the local absorption
coefficient (ln I(t) ~= -mua * v * t + const). The tail
[LATE_START:N_TIME_GATES] is therefore not one blob of "late-time
information": it is an ordered sequence of increasingly deep-sampling
segments.

This module:
    1. Slices each tail into an ordered sequence of overlapping windows,
       from the shallowest (starting at LATE_START) to the deepest (ending
       at the last time bin).
    2. Extracts a windowed log-amplitude decay slope plus local amplitude
       statistics for each window.
    3. Runs a GRU over the window sequence (shallow -> deep), producing a
       running, depth-resolved hidden state.
    4. Reads out an explicit depth-resolved absorption profile from every
       step of that sequence.

Only one label (the bottom-layer mua) exists per DTOF, so the profile
itself is not "ground truth" at every depth. Instead the last few
(deepest) points of the profile are supervised directly against that
label with depth-increasing weights, and a plateau penalty encourages
those deep points to stabilize instead of continuing to swing -- both
consistent with the physical expectation that late-enough tail segments
should be dominated by the bottom-layer absorption. The GRU's final
hidden state (the most depth-informed state) also feeds the main
regression head alongside the full-TPSF and wavelength features, so the
depth-resolved supervision also improves gradient flow into the tail
encoder (deep supervision).
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from train_bottom_mua_dynamic_tail import (
    EARLY_STOPPING_PATIENCE,
    EPS,
    GRADIENT_CLIP_NORM,
    HEAD_DROPOUT,
    IMAGE_BATCH_SIZE,
    LEARNING_RATE,
    MIN_DELTA,
    NUM_EPOCHS,
    NUM_WORKERS,
    N_TIME_GATES,
    PIN_MEMORY,
    SCHEDULER_PATIENCE,
    TEMPORAL_DROPOUT,
    TEMPORAL_FEATURE_DIM,
    TEMPORAL_FILTERS_PER_KERNEL,
    TEMPORAL_POOL_BINS,
    TRAIN_DIR,
    TRAIN_FRACTION,
    WAVELENGTH_FILE,
    WEIGHT_DECAY,
    BottomMuaDataset,
    RawMuaLoss,
    SEED,
    TemporalEncoder,
    build_raw_tpsf_channel,
    calculate_constant_baseline,
    estimate_tpsf_input_scale,
    flatten_image_batch,
    load_wavelength_vector,
    summarize_late_start_indices,
    summarize_raw_target,
)
import os
import random
from glob import glob


# ============================================================
# 1. DEPTH-RESOLVED SETTINGS
# ============================================================

DEPTH_MODEL_PATH = r"D:\DTOF_exp_simu_depth_resolved_bottom_mua.pth"

WINDOW_BINS = 20
WINDOW_STRIDE = 10

DEPTH_FEATURE_DIM = 16
DEPTH_HIDDEN_DIM = 32

# How many of the deepest windows in each tail get direct supervision
# against the bottom-mua label.
NUM_SUPERVISED_TAIL_WINDOWS = 3

# Exponent controlling how much more the auxiliary loss weights the
# deepest supervised window relative to the shallowest of the supervised
# set (weights increase with depth).
LATE_WEIGHT_POWER = 2.0

AUX_DEPTH_LOSS_WEIGHT = 0.3
PLATEAU_LOSS_WEIGHT = 0.05


# ============================================================
# 2. TAIL WINDOWING AND PHYSICS-INFORMED WINDOW FEATURES
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

    time_index = torch.arange(
        window_bins,
        device=windows.device,
        dtype=windows.dtype,
    )
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


# ============================================================
# 3. DEPTH-RESOLVED TAIL ENCODER
# ============================================================

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
            0.0,
            1.0,
            n_windows,
            device=raw_tail.device,
            dtype=raw_tail.dtype,
        ).view(1, n_windows, 1).expand(n_samples, -1, -1)

        window_input = torch.cat([windows, slope_features, depth_position], dim=2)
        window_features = self.window_encoder(window_input)

        hidden_sequence, _ = self.sequence_model(window_features)

        depth_profile = F.softplus(
            self.depth_readout(hidden_sequence)
        ).squeeze(-1)

        final_feature = hidden_sequence[:, -1, :]

        return final_feature, depth_profile


# ============================================================
# 4. FULL MODEL
# ============================================================

class BottomMuaDepthResolvedNet(nn.Module):
    """Full TPSF features + depth-resolved tail sequence + raw wavelength
    scalar.

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
                "The number of LATE_START values does not match the "
                "number of TPSFs."
            )

        if torch.any(starts < 0) or torch.any(starts >= N_TIME_GATES):
            raise ValueError("LATE_START values must be within 0..299.")

        final_features = tpsf.new_zeros((sample_count, self.tail_hidden_dim))
        supervised_profiles = tpsf.new_zeros(
            (sample_count, self.num_supervised_windows)
        )

        for unique_start in torch.unique(starts, sorted=True):
            start = int(unique_start.item())

            sample_indices = torch.nonzero(
                starts == unique_start, as_tuple=False
            ).reshape(-1)

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
            supervised_profiles = supervised_profiles.index_copy(
                0, sample_indices, group_supervised
            )

        return final_features, supervised_profiles

    def forward(self, tpsf, wavelength, late_start):
        full_channels = build_raw_tpsf_channel(tpsf)
        full_features = self.full_encoder(full_channels)

        tail_final_features, supervised_profiles = self.encode_raw_tails(
            tpsf, late_start
        )

        wavelength_input = wavelength.to(dtype=full_features.dtype)

        fused_features = torch.cat(
            [full_features, tail_final_features, wavelength_input],
            dim=1,
        )

        prediction = self.head(fused_features)

        return prediction, supervised_profiles


# ============================================================
# 5. DEPTH-PROFILE AUXILIARY LOSSES
# ============================================================

def depth_profile_auxiliary_losses(
    supervised_profiles,
    target,
    late_weight_power=LATE_WEIGHT_POWER,
):
    """Deep-supervision losses on the deepest windows of the depth profile.

    supervised_profiles: [N, L], deepest-first-to-last ordering (column -1
        is the single deepest window).
    target: [N, 1], the bottom-mua label.

    Returns (weighted_l1_loss, plateau_loss). The L1 term pulls every
    supervised window toward the bottom-mua label, with weight increasing
    toward the deepest window. The plateau term penalizes variance across
    those windows, encouraging the profile to have stabilized by the time
    it reaches the label depth.
    """

    n_windows = supervised_profiles.shape[1]

    weights = torch.linspace(
        1.0,
        float(n_windows),
        n_windows,
        device=supervised_profiles.device,
        dtype=supervised_profiles.dtype,
    ).pow(late_weight_power)
    weights = weights / weights.sum()

    error = supervised_profiles - target
    weighted_l1 = (weights.view(1, -1) * error.abs()).sum(dim=1).mean()

    plateau = supervised_profiles.var(dim=1, unbiased=False).mean()

    return weighted_l1, plateau


# ============================================================
# 6. VALIDATION
# ============================================================

@torch.no_grad()
def validate_depth_resolved(model, loader, criterion, device):
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
# 7. TRAINING
# ============================================================

def train_depth_resolved_model():
    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(f"Training directory was not found:\n{TRAIN_DIR}")

    raw_wavelengths, normalized_wavelengths, wavelength_key = load_wavelength_vector()

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

    tpsf_input_scale = estimate_tpsf_input_scale(train_files)
    summarize_raw_target(train_files)
    late_start_statistics = summarize_late_start_indices(train_files)

    constant_prediction, constant_baseline_mae = calculate_constant_baseline(
        train_files, val_files
    )
    print(f"\nConstant bottom-mua prediction: {constant_prediction:.6e}")
    print(f"Constant validation MAE:         {constant_baseline_mae:.6e}")

    train_dataset = BottomMuaDataset(
        file_list=train_files,
        tpsf_input_scale=tpsf_input_scale,
        wavelength_values=raw_wavelengths,
        augment=True,
    )
    val_dataset = BottomMuaDataset(
        file_list=val_files,
        tpsf_input_scale=tpsf_input_scale,
        wavelength_values=raw_wavelengths,
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = BottomMuaDepthResolvedNet().to(device)

    criterion = RawMuaLoss().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
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
            aux_loss, plateau_loss = depth_profile_auxiliary_losses(
                supervised_profiles, target
            )

            loss = (
                primary_loss
                + AUX_DEPTH_LOSS_WEIGHT * aux_loss
                + PLATEAU_LOSS_WEIGHT * plateau_loss
            )

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
        metrics = validate_depth_resolved(model, val_loader, criterion, device)

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
        "architecture": "bottom_mua_depth_resolved_tail_sequence",
        "window_bins": WINDOW_BINS,
        "window_stride": WINDOW_STRIDE,
        "num_supervised_tail_windows": NUM_SUPERVISED_TAIL_WINDOWS,
        "late_start_statistics": late_start_statistics,
        "tpsf_input_scale": tpsf_input_scale,
        "wavelength_file": WAVELENGTH_FILE,
        "wavelength_key": wavelength_key,
        "raw_wavelengths": torch.from_numpy(raw_wavelengths),
        "normalized_wavelengths": torch.from_numpy(normalized_wavelengths),
        "best_epoch": best_epoch,
        "best_validation_bottom_mua_mae": best_val_mae,
        "constant_validation_bottom_mua_mae": constant_baseline_mae,
        "train_bottom_mua_mae_history": train_mae_history,
        "val_bottom_mua_mae_history": val_mae_history,
    }

    torch.save(checkpoint, DEPTH_MODEL_PATH)

    print("\nTraining complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation bottom-mua MAE: {best_val_mae:.6e}")
    print(f"Constant validation baseline MAE: {constant_baseline_mae:.6e}")
    print(f"Checkpoint saved to:\n{DEPTH_MODEL_PATH}")


if __name__ == "__main__":
    train_depth_resolved_model()
