# -*- coding: utf-8 -*-
"""
Diagnostic: does the training set actually cover the true bottom-mua
values a given test phantom needs, at each wavelength?

Why this matters: a model's cross-wavelength smoother (SpectralSmoother in
train_bottom_mua_spectral_smoothing.py) can only reshape/blend predictions
that the per-wavelength backbone already computed -- it cannot inject
information the backbone never learned. So if a predicted spectrum shows a
systematic, smooth (not jitter-like) mismatch against the true spectrum --
e.g. a compressed peak or an undershooting tail that persists no matter how
much smoother capacity is added -- the likely cause isn't the smoother at
all: it's that the training set never showed the backbone a phantom whose
true bottom-mua reached that value at that wavelength, so the model is
extrapolating rather than interpolating there.

This script builds, for each of the 169 wavelengths, the [min, max] and a
few percentiles of bottom_absorption_mul across every training image, and
compares a specific test phantom's true spectrum against that envelope.
Wavelengths where the true value falls outside the training envelope are
flagged: on plotted results, those flagged wavelengths should line up with
where your predicted-vs-true error is largest. If they do, the fix is more
training-set diversity (more phantoms spanning that value range) rather
than more model capacity. If the true values are well inside the envelope
and the model still misses them, that instead points to a genuine
underfitting/capacity issue in the per-wavelength backbone.
"""

import os
from glob import glob

import numpy as np
from scipy.io import loadmat

try:
    import matplotlib.pyplot as plt
    HAVE_MATPLOTLIB = True
except ImportError:
    HAVE_MATPLOTLIB = False


# ============================================================
# 1. USER SETTINGS
# ============================================================

# Same training folder used by train_bottom_mua_spectral_smoothing.py.
TRAIN_DIR = r"D:\Simulation_TPSF_Convolution_Animal_IRF\DTOF_training_simu_and_exp_resampled"
BOTTOM_MUA_KEY = "bottom_absorption_mul"

# A .mat file containing the TRUE bottom_absorption_mul for the phantom you
# want to check (the one your predicted-vs-true plot is drawn from). This
# does not need to be a training file -- point it at whatever file holds
# the ground truth you plotted as "true".
TEST_TARGET_FILE = r"D:\Simulation_TPSF_Convolution_Animal_IRF\Phantom_2_DTOF\DTOF_2.mat"
TEST_TARGET_KEY = "bottom_absorption_mul"

N_WAVELENGTHS = 169

# A test value is flagged "outside coverage" if it falls below this
# percentile or above the complementary one of the training distribution
# at that wavelength, rather than requiring it to be strictly outside the
# training min/max (a handful of training phantoms at the extreme tails
# would otherwise make almost nothing look uncovered).
LOWER_PERCENTILE = 5.0
UPPER_PERCENTILE = 95.0


# ============================================================
# 2. LOAD TRAINING TARGET ENVELOPE
# ============================================================

def load_bottom_mua_column(mat, file_path, key):
    if key not in mat:
        raise KeyError(f"'{key}' was not found in:\n{file_path}")

    target = np.asarray(mat[key])

    if target.ndim == 1:
        target = target.reshape(-1, 1)
    elif target.ndim == 2 and target.shape[0] == 1:
        target = target.T

    if target.ndim != 2 or target.shape[0] < N_WAVELENGTHS:
        raise ValueError(
            f"'{key}' in '{file_path}' has shape {target.shape}; "
            f"expected at least [{N_WAVELENGTHS},1]."
        )

    return target[:N_WAVELENGTHS, 0].astype(np.float64)


def build_training_target_envelope():
    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(f"Training directory was not found:\n{TRAIN_DIR}")

    file_paths = sorted(glob(os.path.join(TRAIN_DIR, "*.mat")))
    if not file_paths:
        raise FileNotFoundError(f"No .mat files were found in:\n{TRAIN_DIR}")

    columns = []
    skipped = []

    for file_path in file_paths:
        try:
            mat = loadmat(file_path)
            columns.append(load_bottom_mua_column(mat, file_path, BOTTOM_MUA_KEY))
        except (KeyError, ValueError) as error:
            skipped.append((file_path, str(error)))

    if not columns:
        raise RuntimeError("No training file contained a usable bottom-mua target.")

    if skipped:
        print(f"Skipped {len(skipped)} training file(s) without a usable target:")
        for file_path, message in skipped:
            print(f"  - {os.path.basename(file_path)}: {message}")

    stacked = np.stack(columns, axis=0)  # [n_training_images, N_WAVELENGTHS]

    print(f"\nBuilt training-target envelope from {stacked.shape[0]} images.")

    return {
        "n_images": stacked.shape[0],
        "min": stacked.min(axis=0),
        "max": stacked.max(axis=0),
        "lower_percentile": np.percentile(stacked, LOWER_PERCENTILE, axis=0),
        "upper_percentile": np.percentile(stacked, UPPER_PERCENTILE, axis=0),
        "median": np.median(stacked, axis=0),
    }


# ============================================================
# 3. LOAD TEST PHANTOM'S TRUE SPECTRUM
# ============================================================

def load_test_target():
    if not os.path.isfile(TEST_TARGET_FILE):
        raise FileNotFoundError(f"Test target file was not found:\n{TEST_TARGET_FILE}")

    mat = loadmat(TEST_TARGET_FILE)
    return load_bottom_mua_column(mat, TEST_TARGET_FILE, TEST_TARGET_KEY)


# ============================================================
# 4. COMPARE AND REPORT
# ============================================================

def summarize_coverage(envelope, test_target):
    below_range = test_target < envelope["min"]
    above_range = test_target > envelope["max"]
    below_percentile = test_target < envelope["lower_percentile"]
    above_percentile = test_target > envelope["upper_percentile"]

    outside_min_max = below_range | above_range
    outside_percentile_band = below_percentile | above_percentile

    print(f"\nTraining images used to build the envelope: {envelope['n_images']}")
    print(f"Test wavelengths outside the training [min, max] range: "
          f"{int(outside_min_max.sum())} / {N_WAVELENGTHS}")
    print(f"Test wavelengths outside the training "
          f"[{LOWER_PERCENTILE:g}, {UPPER_PERCENTILE:g}] percentile band: "
          f"{int(outside_percentile_band.sum())} / {N_WAVELENGTHS}")

    if outside_percentile_band.any():
        flagged_indices = np.nonzero(outside_percentile_band)[0]
        print("\nFlagged wavelength indices (outside the percentile band), "
              "grouped into contiguous runs:")

        run_start = flagged_indices[0]
        previous = flagged_indices[0]
        for index in flagged_indices[1:]:
            if index != previous + 1:
                print(f"  - indices {run_start}-{previous}")
                run_start = index
            previous = index
        print(f"  - indices {run_start}-{previous}")

        print(
            "\nCompare these index ranges against where your predicted-vs-true "
            "plot shows the largest, smoothest (non-jitter) errors. A match "
            "means the model is extrapolating beyond what training ever "
            "showed it there -- add training phantoms spanning this value "
            "range at these wavelengths rather than tuning model capacity."
        )
    else:
        print(
            "\nNo wavelengths were flagged: the test phantom's true spectrum "
            "falls within the training distribution everywhere. If the "
            "model still mispredicts it, that points to a genuine "
            "underfitting/capacity issue in the per-wavelength backbone "
            "rather than a training-data coverage gap."
        )

    return outside_min_max, outside_percentile_band


def plot_coverage(envelope, test_target, outside_percentile_band):
    if not HAVE_MATPLOTLIB:
        print("\nmatplotlib is not available; skipping the coverage plot.")
        return

    wavelength_index = np.arange(N_WAVELENGTHS)

    figure, axis = plt.subplots(figsize=(9, 6))

    axis.fill_between(
        wavelength_index, envelope["min"], envelope["max"],
        color="tab:gray", alpha=0.25, label="Training min-max",
    )
    axis.fill_between(
        wavelength_index, envelope["lower_percentile"], envelope["upper_percentile"],
        color="tab:gray", alpha=0.45,
        label=f"Training {LOWER_PERCENTILE:g}-{UPPER_PERCENTILE:g} percentile",
    )
    axis.plot(wavelength_index, envelope["median"], color="tab:gray", linestyle="--", label="Training median")
    axis.plot(wavelength_index, test_target, color="tab:red", label="Test phantom true value")

    axis.scatter(
        wavelength_index[outside_percentile_band],
        test_target[outside_percentile_band],
        color="black", marker="x", zorder=5,
        label="Outside training percentile band",
    )

    axis.set_xlabel("Wavelength index")
    axis.set_ylabel("bottom_absorption_mul")
    axis.set_title("Test phantom true spectrum vs. training-target coverage")
    axis.legend()
    axis.grid(True)

    plt.show()


def run_diagnostic():
    envelope = build_training_target_envelope()
    test_target = load_test_target()

    if test_target.shape != envelope["min"].shape:
        raise RuntimeError(
            f"Test target shape {test_target.shape} does not match "
            f"envelope shape {envelope['min'].shape}."
        )

    _, outside_percentile_band = summarize_coverage(envelope, test_target)
    plot_coverage(envelope, test_target, outside_percentile_band)


if __name__ == "__main__":
    run_diagnostic()
