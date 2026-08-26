# -*- coding: utf-8 -*-
"""
Spectral-coherence extension of the depth-resolved bottom-mua model.

train_bottom_mua_dynamic_tail.py and train_bottom_mua_depth_resolved.py
both regress each of the 169 wavelength-specific TPSFs in an image
completely independently: a given TPSF's prediction never sees any
information from the other 168 wavelengths in the same image. Two
symptoms follow directly from that:

    1. High-frequency jitter in the predicted spectrum that isn't present
       in the (physically smooth) ground-truth absorption spectrum, since
       nothing enforces continuity across neighboring wavelengths.
    2. Compressed peak/trough amplitude, since each wavelength's estimate
       has to stand entirely on its own TPSF's evidence instead of being
       able to pool evidence with its neighbors.

A third, related symptom is systematic drift at the spectral edges: a
single dataset-wide TPSF input scale (one 99th-percentile amplitude for
all 169 wavelengths) implicitly assumes uniform signal amplitude/SNR
across wavelength, which is rarely true for a real NIRS/DOT source-detector
system (source power and detector responsivity both vary with
wavelength).

This module keeps the depth-resolved tail encoder from
train_bottom_mua_depth_resolved.py entirely as-is, and adds:

    1. Per-wavelength TPSF input normalization (a length-169 scale vector
       instead of one dataset-wide scalar), so each wavelength's amplitude
       statistics are normalized against its own typical range rather than
       a global one.
    2. A SpectralSmoother: a residual depthwise-separable 1D convolution
       applied across the wavelength axis, over the fused per-wavelength
       feature vectors of one image, right before the final regression
       head. This gives every wavelength's prediction access to a local
       neighborhood of other wavelengths' evidence, which both smooths
       jitter and lets genuinely large local peaks recruit corroborating
       evidence from their neighbors instead of being shrunk toward the
       per-wavelength mean.

The smoother's pointwise-mixing weights are zero-initialized, so at the
start of training it is the identity function (mixed_features ==
fused_features) and can only start contributing once training shows it
reduces the loss -- it cannot make an already-working per-wavelength model
worse by injection of noise before it has learned anything.
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
from torch.utils.data import DataLoader, Dataset
from scipy.io import loadmat

from train_bottom_mua_dynamic_tail import (
    EARLY_STOPPING_PATIENCE,
    EPS,
    GRADIENT_CLIP_NORM,
    LEARNING_RATE,
    MIN_DELTA,
    NUM_EPOCHS,
    NUM_WORKERS,
    N_WAVELENGTHS,
    PIN_MEMORY,
    SCHEDULER_PATIENCE,
    TRAIN_DIR,
    TRAIN_FRACTION,
    WAVELENGTH_FILE,
    WEIGHT_DECAY,
    RawMuaLoss,
    SEED,
    augment_scaled_tpsf,
    calculate_constant_baseline,
    choose_group_count,
    flatten_image_batch,
    load_bottom_mua_target,
    load_late_start_indices,
    load_training_tpsf,
    load_wavelength_vector,
    summarize_late_start_indices,
    summarize_raw_target,
)
from train_bottom_mua_depth_resolved import (
    AUX_DEPTH_LOSS_WEIGHT,
    DEPTH_HIDDEN_DIM,
    HEAD_DROPOUT,
    IMAGE_BATCH_SIZE,
    PLATEAU_LOSS_WEIGHT,
    TEMPORAL_DROPOUT,
    TEMPORAL_FEATURE_DIM,
    TEMPORAL_FILTERS_PER_KERNEL,
    TEMPORAL_POOL_BINS,
    WAVELENGTH_EMBEDDING_DIM,
    BottomMuaDepthResolvedNet,
    build_raw_tpsf_channel,
    depth_profile_auxiliary_losses,
)


# ============================================================
# 1. SPECTRAL-SMOOTHING SETTINGS
# ============================================================

SPECTRAL_MODEL_PATH = r"D:\DTOF_exp_simu_depth_resolved_spectral_bottom_mua.pth"

SPECTRAL_SMOOTHING_KERNEL_SIZE = 7

# Optional light total-variation penalty on the final predicted spectrum,
# on top of the structural smoothing already performed by SpectralSmoother.
# Kept small/off by default: too much of this can flatten genuine peaks
# rather than just removing jitter.
SPECTRAL_TV_LOSS_WEIGHT = 0.0


# ============================================================
# 2. PER-WAVELENGTH INPUT SCALE
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


class PerWavelengthNormalizedDataset(Dataset):
    """Same as BottomMuaDataset, but divides each wavelength's TPSF row
    by its own scale instead of one dataset-wide scalar.
    """

    def __init__(
        self,
        file_list,
        per_wavelength_scale,
        normalized_wavelengths,
        augment=False,
    ):
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

        normalized_wavelengths = np.asarray(normalized_wavelengths, dtype=np.float32)
        expected_wavelength_shape = (N_WAVELENGTHS, 1)
        if normalized_wavelengths.shape != expected_wavelength_shape:
            raise ValueError(
                f"normalized_wavelengths has shape "
                f"{normalized_wavelengths.shape}; expected "
                f"{expected_wavelength_shape}."
            )
        self.wavelengths = torch.from_numpy(normalized_wavelengths)

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


# ============================================================
# 3. SPECTRAL SMOOTHER
# ============================================================

class SpectralSmoother(nn.Module):
    """Residual depthwise-separable smoothing across the wavelength axis.

    Operates on the fused per-wavelength feature vectors of one image at a
    time (shape [N_WAVELENGTHS, feature_dim]), giving every wavelength's
    prediction access to a local neighborhood of the other wavelengths'
    evidence. Zero-initialized so it starts as the identity function.
    """

    def __init__(
        self,
        feature_dim,
        kernel_size=SPECTRAL_SMOOTHING_KERNEL_SIZE,
        n_wavelengths=N_WAVELENGTHS,
    ):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for symmetric padding.")

        self.feature_dim = int(feature_dim)
        self.n_wavelengths = int(n_wavelengths)
        padding = kernel_size // 2

        self.depthwise = nn.Conv1d(
            feature_dim,
            feature_dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=feature_dim,
            bias=False,
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
                f"Expected feature dimension {self.feature_dim}, "
                f"received {feature_dim}."
            )

        if n_total % self.n_wavelengths != 0:
            raise ValueError(
                f"Batch size {n_total} is not a multiple of "
                f"{self.n_wavelengths} wavelengths; spectral mixing "
                "requires whole images (all wavelengths of an image "
                "present together in the batch)."
            )

        n_images = n_total // self.n_wavelengths

        sequence = fused_features.view(n_images, self.n_wavelengths, feature_dim)
        sequence = sequence.transpose(1, 2)  # [n_images, feature_dim, n_wavelengths]

        residual = self.pointwise(F.relu(self.norm(self.depthwise(sequence))))
        mixed = sequence + residual

        mixed = mixed.transpose(1, 2).reshape(n_total, feature_dim)
        return mixed


# ============================================================
# 4. FULL MODEL
# ============================================================

class BottomMuaSpectralNet(BottomMuaDepthResolvedNet):
    """BottomMuaDepthResolvedNet with a spectral-smoothing stage inserted
    between the fused per-wavelength features and the regression head.
    """

    def __init__(
        self,
        temporal_feature_dim=TEMPORAL_FEATURE_DIM,
        filters_per_kernel=TEMPORAL_FILTERS_PER_KERNEL,
        pool_bins=TEMPORAL_POOL_BINS,
        temporal_dropout=TEMPORAL_DROPOUT,
        head_dropout=HEAD_DROPOUT,
        depth_hidden_dim=DEPTH_HIDDEN_DIM,
        wavelength_embedding_dim=WAVELENGTH_EMBEDDING_DIM,
        spectral_kernel_size=SPECTRAL_SMOOTHING_KERNEL_SIZE,
    ):
        super().__init__(
            temporal_feature_dim=temporal_feature_dim,
            filters_per_kernel=filters_per_kernel,
            pool_bins=pool_bins,
            temporal_dropout=temporal_dropout,
            head_dropout=head_dropout,
            depth_hidden_dim=depth_hidden_dim,
            wavelength_embedding_dim=wavelength_embedding_dim,
        )

        fused_dim = temporal_feature_dim + depth_hidden_dim + wavelength_embedding_dim
        self.spectral_smoother = SpectralSmoother(
            fused_dim, kernel_size=spectral_kernel_size
        )

    def forward(self, tpsf, wavelength, late_start):
        full_channels = build_raw_tpsf_channel(tpsf)
        full_features = self.full_encoder(full_channels)

        tail_final_features, supervised_profiles = self.encode_raw_tails(
            tpsf, late_start
        )

        wavelength_embedding = self.wavelength_encoder(
            wavelength.to(dtype=full_features.dtype)
        )

        fused_features = torch.cat(
            [full_features, tail_final_features, wavelength_embedding],
            dim=1,
        )
        fused_features = self.spectral_smoother(fused_features)

        prediction = self.head(fused_features)

        return prediction, supervised_profiles


# ============================================================
# 5. OPTIONAL SPECTRAL SMOOTHNESS LOSS
# ============================================================

def spectral_total_variation_loss(prediction, n_wavelengths=N_WAVELENGTHS):
    """Mean absolute difference between wavelength-adjacent predictions.

    Off by default (SPECTRAL_TV_LOSS_WEIGHT == 0.0): SpectralSmoother
    already gives the network a structural way to produce smooth spectra,
    and a strong explicit penalty here can suppress genuine peaks rather
    than just jitter. Provided for experimentation.
    """

    n_total = prediction.shape[0]
    if n_total % n_wavelengths != 0:
        raise ValueError(
            f"Batch size {n_total} is not a multiple of {n_wavelengths}."
        )

    n_images = n_total // n_wavelengths
    spectrum = prediction.view(n_images, n_wavelengths)
    return (spectrum[:, 1:] - spectrum[:, :-1]).abs().mean()


# ============================================================
# 6. VALIDATION
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
# 7. TRAINING
# ============================================================

def train_spectral_model():
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

    per_wavelength_scale = estimate_per_wavelength_input_scale(train_files)
    summarize_raw_target(train_files)
    late_start_statistics = summarize_late_start_indices(train_files)

    constant_prediction, constant_baseline_mae = calculate_constant_baseline(
        train_files, val_files
    )
    print(f"\nConstant bottom-mua prediction: {constant_prediction:.6e}")
    print(f"Constant validation MAE:         {constant_baseline_mae:.6e}")

    train_dataset = PerWavelengthNormalizedDataset(
        file_list=train_files,
        per_wavelength_scale=per_wavelength_scale,
        normalized_wavelengths=normalized_wavelengths,
        augment=True,
    )
    val_dataset = PerWavelengthNormalizedDataset(
        file_list=val_files,
        per_wavelength_scale=per_wavelength_scale,
        normalized_wavelengths=normalized_wavelengths,
        augment=False,
    )

    # Spectral mixing needs every wavelength of an image present together
    # in the batch, so partial-image batches must never be handed to the
    # model: drop_last protects the (irrelevant here, since every image
    # always contributes exactly N_WAVELENGTHS rows) case of a ragged final
    # batch of *images*, which is unaffected by this; N_WAVELENGTHS rows
    # per image are always contiguous regardless of image batch size.
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

    model = BottomMuaSpectralNet().to(device)
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

            if SPECTRAL_TV_LOSS_WEIGHT > 0.0:
                loss = loss + SPECTRAL_TV_LOSS_WEIGHT * spectral_total_variation_loss(
                    prediction
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
        "spectral_smoothing_kernel_size": SPECTRAL_SMOOTHING_KERNEL_SIZE,
        "spectral_tv_loss_weight": SPECTRAL_TV_LOSS_WEIGHT,
        "late_start_statistics": late_start_statistics,
        "per_wavelength_input_scale": per_wavelength_scale,
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

    torch.save(checkpoint, SPECTRAL_MODEL_PATH)

    print("\nTraining complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation bottom-mua MAE: {best_val_mae:.6e}")
    print(f"Constant validation baseline MAE: {constant_baseline_mae:.6e}")
    print(f"Checkpoint saved to:\n{SPECTRAL_MODEL_PATH}")


if __name__ == "__main__":
    train_spectral_model()
