# -*- coding: utf-8 -*-
"""
Simple bottom-mua model with one dynamic raw tail for each TPSF.

Each image contains:
    - 169 wavelength-specific TPSFs
    - 450 selected time bins per TPSF
    - 169 bottom_absorption_mul targets
    - 169 LATE_START indices

For each wavelength-specific TPSF:
    full raw TPSF -> full convolution encoder
    raw TPSF tail -> tail convolution encoder
    normalized wavelength -> one direct scalar, no wavelength encoder

The full and tail feature vectors are concatenated with the wavelength scalar
and used to predict the corresponding bottom-layer absorption coefficient.

The tail is TPSF[LATE_START:450] exactly. There is no interpolation,
resampling, padding, tail mask, or LATE_START feature encoder.
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

#MODEL_PATH = r"D:\bottom_mua_simple_full_tail_best_with_ph_1.pth"

MODEL_PATH = r"D:\DTOF_exp_simu_full_and_tail_cropped_exp_wavelength_resampled_raw_target.pth"




# Folder containing TPSF_1.mat, TPSF_2.mat, TPSF_3.mat, ...
#TEST_DIR = (
#    r"D:\Simulation_TPSF_Convolution_Animal_IRF"
#    r"\Pig_9_for_AI_reconstruction"
#)

#TEST_FILE_PREFIX = "TPST_"
#TEST_FILE_EXTENSION = ".mat"
#TEST_TPSF_KEY = "tpsf_auc"


TEST_DIR = (
    r"D:\Simulation_TPSF_Convolution_Animal_IRF"
    r"\Phantom_4_DTOF"
)

TEST_FILE_PREFIX = "DTOF_"
TEST_FILE_EXTENSION = ".mat"
TEST_TPSF_KEY = "dtof_auc"






# Keep None to use the late-start variable name stored in the checkpoint.
TEST_LATE_START_KEY = "End_start"

# Output files are saved as pred_bottom_mua_1.mat,
# pred_bottom_mua_2.mat, pred_bottom_mua_3.mat, ...
#OUTPUT_DIR = r"D:\decon_TPSF\pred_bottom_pig_9"
OUTPUT_DIR = r"D:\decon_TPSF\pred_bottom_Phantom_4_DTOF"
OUTPUT_FILE_PREFIX = "pred_bottom_mua_"

# When True, one bad file is reported and the remaining files are processed.
CONTINUE_ON_ERROR = True

# Fallback only when wavelength vectors are absent from the checkpoint.
WAVELENGTH_FILE = (
    r"D:\Simulation_TPSF_Convolution_Animal_IRF"
    r"\wavelength_exp_index.mat"
)
WAVELENGTH_KEY = 'wavelengths_exp'


# ============================================================
# 2. DEFAULT SETTINGS
# ============================================================

N_WAVELENGTHS = 169
N_TIME_GATES = 450
TIME_GATE_START = 0
TIME_GATE_END = 450

DEFAULT_TPSF_INPUT_CHANNELS = 1
DEFAULT_TEMPORAL_FILTERS_PER_KERNEL = 12
DEFAULT_TEMPORAL_POOL_BINS = 4
DEFAULT_TEMPORAL_FEATURE_DIM = 48
DEFAULT_TEMPORAL_DROPOUT = 0.15

DEFAULT_HEAD_DROPOUT = 0.15


# Aliases used by the shared model class defaults.
TEMPORAL_FILTERS_PER_KERNEL = DEFAULT_TEMPORAL_FILTERS_PER_KERNEL
TEMPORAL_POOL_BINS = DEFAULT_TEMPORAL_POOL_BINS
TEMPORAL_FEATURE_DIM = DEFAULT_TEMPORAL_FEATURE_DIM
TEMPORAL_DROPOUT = DEFAULT_TEMPORAL_DROPOUT
HEAD_DROPOUT = DEFAULT_HEAD_DROPOUT


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


def load_late_start_indices(
    mat,
    file_path,
    key,
    matlab_one_based,
):
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
            f"'{key}' has shape {late_start.shape}; expected "
            f"[169,1] or [1,169]."
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
        raise FileNotFoundError(
            f"Test file was not found:\n{test_file}"
        )

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

    matlab_one_based = bool(
        checkpoint.get("late_start_is_matlab_one_based", True)
    )

    tpsf = clean_tpsf_array(mat[TEST_TPSF_KEY], test_file)
    late_start = load_late_start_indices(
        mat,
        test_file,
        late_start_key,
        matlab_one_based,
    )

    print(f"TPSF variable: {TEST_TPSF_KEY}")
    print(f"Late-start variable: {late_start_key}")
    print(f"MATLAB one-based indices: {matlab_one_based}")
    print(f"TPSF shape after cropping: {tpsf.shape}")
    print(
        f"Late-start range after conversion: "
        f"{late_start.min()} to {late_start.max()}"
    )

    return tpsf, late_start, late_start_key, matlab_one_based


# ============================================================
# 4. WAVELENGTH LOADING
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

    normalized = (
        wavelengths - wavelength_min
    ) / (wavelength_max - wavelength_min)

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

    if WAVELENGTH_KEY is not None:
        if WAVELENGTH_KEY not in mat:
            available = [key for key in mat if not key.startswith("__")]
            raise KeyError(
                f"'{WAVELENGTH_KEY}' was not found. "
                f"Available variables: {available}"
            )
        detected_key = WAVELENGTH_KEY
        array = mat[WAVELENGTH_KEY]
    else:
        candidates = []
        for key, value in mat.items():
            if key.startswith("__"):
                continue
            candidate = np.asarray(value)
            if (
                np.issubdtype(candidate.dtype, np.number)
                and candidate.size == N_WAVELENGTHS
            ):
                candidates.append((key, candidate))

        if len(candidates) != 1:
            names = [key for key, _ in candidates]
            raise ValueError(
                f"Expected one {N_WAVELENGTHS}-value wavelength variable, "
                f"found: {names}. Set WAVELENGTH_KEY explicitly."
            )

        detected_key, array = candidates[0]

    raw, normalized = normalize_wavelength_vector(array)
    return raw, normalized, f"external:{detected_key}"


def get_wavelengths(checkpoint):
    if "normalized_wavelengths" in checkpoint:
        normalized = torch.as_tensor(
            checkpoint["normalized_wavelengths"],
            dtype=torch.float32,
        ).cpu().numpy().reshape(-1)

        if normalized.size != N_WAVELENGTHS:
            raise ValueError(
                f"Checkpoint contains {normalized.size} normalized "
                f"wavelengths; expected {N_WAVELENGTHS}."
            )

        normalized = normalized.reshape(N_WAVELENGTHS, 1).astype(np.float32)

        if "raw_wavelengths" in checkpoint:
            raw = torch.as_tensor(
                checkpoint["raw_wavelengths"],
                dtype=torch.float32,
            ).cpu().numpy().reshape(-1)

            if raw.size != N_WAVELENGTHS:
                raise ValueError(
                    f"Checkpoint contains {raw.size} raw wavelengths; "
                    f"expected {N_WAVELENGTHS}."
                )

            raw = raw.reshape(N_WAVELENGTHS, 1).astype(np.float32)
        else:
            raw = np.full(
                (N_WAVELENGTHS, 1),
                np.nan,
                dtype=np.float32,
            )

        return raw, normalized, "checkpoint"

    return load_external_wavelengths()


# ============================================================
# 5. RAW TPSF CHANNEL
# ============================================================

def build_raw_tpsf_channel(tpsf):
    """Convert [N,450] raw TPSFs to [N,1,450]."""

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


# ============================================================
# 6. SIMPLE CONVOLUTION MODEL
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
    """Full TPSF features + raw-tail features + wavelength scalar."""

    def __init__(
        self,
        temporal_feature_dim=TEMPORAL_FEATURE_DIM,
        filters_per_kernel=TEMPORAL_FILTERS_PER_KERNEL,
        pool_bins=TEMPORAL_POOL_BINS,
        temporal_dropout=TEMPORAL_DROPOUT,
        head_dropout=HEAD_DROPOUT,
    ):
        super().__init__()

        self.temporal_feature_dim = int(
            temporal_feature_dim
        )

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

        # No wavelength encoder. The normalized wavelength is one scalar.
        fused_dim = (
            2
            * temporal_feature_dim
            + 1
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
            full_channels[group, :, start:450]
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
                "LATE_START values must be within 0..449."
            )

        tail_features = full_channels.new_zeros(
            (
                sample_count,
                self.temporal_feature_dim,
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

        return tail_features

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

        tail_features = self.encode_raw_tails(
            full_channels,
            late_start,
        )

        wavelength_scalar = wavelength.to(
            dtype=full_features.dtype
        )

        fused_features = torch.cat(
            [
                full_features,
                tail_features,
                wavelength_scalar,
            ],
            dim=1,
        )

        return self.head(
            fused_features
        )


# ============================================================
# 7. CHECKPOINT LOADING
# ============================================================

def load_checkpoint(device):
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Model checkpoint was not found:\n{MODEL_PATH}"
        )

    try:
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(MODEL_PATH, map_location=device)

    expected_architecture = "bottom_mua_simple_full_tail"

    if checkpoint.get("architecture") != expected_architecture:
        raise RuntimeError(
            f"Checkpoint architecture is "
            f"'{checkpoint.get('architecture')}', but this script requires "
            f"'{expected_architecture}'."
        )

    if int(checkpoint.get("n_wavelengths", N_WAVELENGTHS)) != N_WAVELENGTHS:
        raise RuntimeError("Checkpoint wavelength count does not match.")

    if int(checkpoint.get("n_time_gates", N_TIME_GATES)) != N_TIME_GATES:
        raise RuntimeError("Checkpoint time-gate count does not match.")

    if int(checkpoint.get("time_gate_start", TIME_GATE_START)) != TIME_GATE_START:
        raise RuntimeError("Checkpoint time-gate start does not match.")

    if int(
        checkpoint.get("time_gate_end_exclusive", TIME_GATE_END)
    ) != TIME_GATE_END:
        raise RuntimeError("Checkpoint time-gate end does not match.")

    tail_mode = checkpoint.get("tail_extraction")
    if tail_mode not in (None, "raw_grouped_by_late_start"):
        raise RuntimeError(
            f"Unexpected checkpoint tail_extraction: {tail_mode!r}."
        )

    wavelength_mode = checkpoint.get("wavelength_usage")
    if wavelength_mode not in (None, "direct_normalized_scalar"):
        raise RuntimeError(
            f"Unexpected checkpoint wavelength_usage: {wavelength_mode!r}."
        )

    if "tpsf_input_scale" not in checkpoint:
        raise KeyError("Checkpoint does not contain 'tpsf_input_scale'.")

    return checkpoint


def build_model_from_checkpoint(
    checkpoint,
    device,
):
    model = BottomMuaSimpleFullTailNet(
        temporal_feature_dim=int(
            checkpoint.get(
                "temporal_feature_dim",
                DEFAULT_TEMPORAL_FEATURE_DIM,
            )
        ),
        filters_per_kernel=int(
            checkpoint.get(
                "temporal_filters_per_kernel",
                DEFAULT_TEMPORAL_FILTERS_PER_KERNEL,
            )
        ),
        pool_bins=int(
            checkpoint.get(
                "temporal_pool_bins",
                DEFAULT_TEMPORAL_POOL_BINS,
            )
        ),
        temporal_dropout=float(
            checkpoint.get(
                "temporal_dropout",
                DEFAULT_TEMPORAL_DROPOUT,
            )
        ),
        head_dropout=float(
            checkpoint.get(
                "head_dropout",
                DEFAULT_HEAD_DROPOUT,
            )
        ),
    ).to(device)

    state_dict = checkpoint[
        "model_state_dict"
    ]

    cleaned_state = {
        (
            key[7:]
            if key.startswith("module.")
            else key
        ): value
        for key, value in state_dict.items()
    }

    model.load_state_dict(
        cleaned_state,
        strict=True,
    )

    model.eval()

    return model


# ============================================================
# 8. INFERENCE
# ============================================================

def extract_file_index(file_path):
    """Return the numeric index from <TEST_FILE_PREFIX><i>.mat."""

    # This replacement also makes Windows paths testable on non-Windows systems.
    normalized_path = file_path.replace("\\", os.sep)
    file_name = os.path.basename(normalized_path)

    pattern = (
        r"^"
        + re.escape(TEST_FILE_PREFIX)
        + r"(\d+)"
        + re.escape(TEST_FILE_EXTENSION)
        + r"$"
    )

    match = re.match(
        pattern,
        file_name,
        flags=re.IGNORECASE,
    )

    if match is None:
        raise ValueError(
            f"Filename '{file_name}' does not match "
            f"'{TEST_FILE_PREFIX}<number>{TEST_FILE_EXTENSION}'."
        )

    return int(match.group(1))


def find_test_files():
    """Find and numerically sort files matching the configured prefix."""

    if not os.path.isdir(TEST_DIR):
        raise FileNotFoundError(
            f"Test directory was not found:\n{TEST_DIR}"
        )

    indexed_files = []

    for file_name in os.listdir(TEST_DIR):
        file_path = os.path.join(
            TEST_DIR,
            file_name,
        )

        if not os.path.isfile(file_path):
            continue

        try:
            file_index = extract_file_index(file_path)
        except ValueError:
            continue

        indexed_files.append(
            (file_index, file_path)
        )

    indexed_files.sort(
        key=lambda item: item[0]
    )

    if not indexed_files:
        raise FileNotFoundError(
            "No files matching "
            f"'{TEST_FILE_PREFIX}<number>{TEST_FILE_EXTENSION}' "
            f"were found in:\n{TEST_DIR}"
        )

    indices = [
        file_index
        for file_index, _ in indexed_files
    ]

    if len(indices) != len(set(indices)):
        raise RuntimeError(
            "Two or more input files use the same numeric index."
        )

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
    wavelength_source,
):
    """Predict one input MAT file and save pred_bottom_mua_i.mat."""

    (
        tpsf,
        late_start,
        late_start_key,
        matlab_one_based,
    ) = load_test_data(
        checkpoint,
        test_file,
    )

    tpsf_input_scale = float(
        checkpoint["tpsf_input_scale"]
    )

    if (
        not np.isfinite(tpsf_input_scale)
        or tpsf_input_scale <= 0.0
    ):
        raise ValueError(
            f"Invalid TPSF input scale: {tpsf_input_scale}"
        )

    # Training divides every raw TPSF by this one fixed training-set scale.
    scaled_tpsf = (
        tpsf / tpsf_input_scale
    ).astype(np.float32)

    tpsf_tensor = torch.from_numpy(
        scaled_tpsf
    ).to(device)

    wavelength_tensor = torch.from_numpy(
        normalized_wavelengths
    ).to(device)

    late_start_tensor = torch.from_numpy(
        late_start.astype(np.int64)
    ).to(device)

    # IMPORTANT: the supplied training code uses RawMuaLoss and trains the
    # network directly against bottom_absorption_mul in physical units.
    # Therefore model output is already the bottom-layer mua prediction.
    raw_prediction = model(
        tpsf_tensor,
        wavelength_tensor,
        late_start_tensor,
    )

    prediction = (
        raw_prediction
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    expected_shape = (
        N_WAVELENGTHS,
        1,
    )

    if prediction.shape != expected_shape:
        raise RuntimeError(
            f"Prediction shape is {prediction.shape}; "
            f"expected {expected_shape}."
        )

    late_start_stored = late_start.copy()

    if matlab_one_based:
        late_start_stored = late_start_stored + 1

    late_start_stored = (
        late_start_stored
        + TIME_GATE_START
    )

    result = {
        "pred_bottom_absorption_mul":
            prediction,

        "prediction_raw_model_output":
            prediction.copy(),

        "wavelengths_raw":
            raw_wavelengths,

        "wavelengths_normalized":
            normalized_wavelengths,

        "late_start_python_zero_based":
            late_start.astype(np.int32),

        "late_start_stored_convention":
            late_start_stored.astype(np.int32),

        "late_start_normalized":
            (
                late_start.astype(np.float32)
                / float(N_TIME_GATES - 1)
            ),

        "late_start_is_matlab_one_based":
            np.array(
                [[int(matlab_one_based)]],
                dtype=np.int32,
            ),

        "source_file_index":
            np.array(
                [[file_index]],
                dtype=np.int32,
            ),

        "source_file_name":
            np.array(
                [[os.path.basename(test_file)]],
                dtype=object,
            ),

        "selected_time_gate_start":
            np.array(
                [[TIME_GATE_START]],
                dtype=np.int32,
            ),

        "selected_time_gate_end_exclusive":
            np.array(
                [[TIME_GATE_END]],
                dtype=np.int32,
            ),

        "tpsf_input_scale":
            np.array(
                [[tpsf_input_scale]],
                dtype=np.float32,
            ),

        "best_validation_bottom_mua_mae":
            np.array(
                [[
                    float(
                        checkpoint.get(
                            "best_validation_bottom_mua_mae",
                            np.nan,
                        )
                    )
                ]],
                dtype=np.float32,
            ),
    }

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        (
            f"{OUTPUT_FILE_PREFIX}"
            f"{file_index}"
            f"{TEST_FILE_EXTENSION}"
        ),
    )

    savemat(
        output_file,
        result,
        do_compression=True,
    )

    print(
        f"Prediction range: "
        f"{prediction.min():.6e} to "
        f"{prediction.max():.6e}"
    )

    print(
        f"Prediction mean/std: "
        f"{prediction.mean():.6e} / "
        f"{prediction.std():.6e}"
    )

    print(
        f"Late-start variable: {late_start_key}"
    )

    print(
        f"Wavelength source: {wavelength_source}"
    )

    print(f"Saved to:\n{output_file}")

    return output_file


@torch.no_grad()
def run_inference():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print(f"Checkpoint: {MODEL_PATH}")
    print(f"Input directory: {TEST_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    checkpoint = load_checkpoint(device)
    model = build_model_from_checkpoint(checkpoint, device)

    (
        raw_wavelengths,
        normalized_wavelengths,
        wavelength_source,
    ) = get_wavelengths(checkpoint)

    test_files = find_test_files()

    print(
        f"\nFound {len(test_files)} test files."
    )

    completed_files = []
    failed_files = []

    for position, (
        file_index,
        test_file,
    ) in enumerate(
        test_files,
        start=1,
    ):
        print("\n" + "=" * 70)
        print(
            f"Processing {position}/{len(test_files)}: "
            f"{os.path.basename(test_file)}"
        )

        try:
            output_file = predict_one_file(
                model=model,
                checkpoint=checkpoint,
                device=device,
                test_file=test_file,
                file_index=file_index,
                raw_wavelengths=raw_wavelengths,
                normalized_wavelengths=normalized_wavelengths,
                wavelength_source=wavelength_source,
            )

            completed_files.append(output_file)

        except Exception as error:
            failed_files.append(
                (test_file, str(error))
            )

            print(
                f"ERROR while processing "
                f"{os.path.basename(test_file)}:\n"
                f"{error}"
            )

            if not CONTINUE_ON_ERROR:
                raise

    print("\n" + "=" * 70)
    print("Batch inference completed.")
    print(f"Successful files: {len(completed_files)}")
    print(f"Failed files: {len(failed_files)}")

    if failed_files:
        print("\nFailed-file summary:")

        for failed_file, message in failed_files:
            print(
                f"- {os.path.basename(failed_file)}: "
                f"{message}"
            )


if __name__ == "__main__":
    run_inference()
