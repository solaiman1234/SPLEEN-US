"""
Plot train vs. validation bottom_mua MAE curves from a saved checkpoint.

train_spectral_model() and fine_tune_on_target_domains() in
train_bottom_mua_spectral_smoothing.py already record per-epoch history
into every checkpoint they save:

    "train_bottom_mua_mae_history": [...]   / "fine_tune_train_bottom_mua_mae_history": [...]
    "val_bottom_mua_mae_history":   [...]   / "fine_tune_val_bottom_mua_mae_history":   [...]

This script just loads that history back out and plots it. It does not
run any training itself, so run it locally against your own checkpoint
file -- there is no real history to plot until you have actually trained
a model.

Usage:
    python plot_training_curves.py --checkpoint "C:\\path\\to\\bottom_mua_spectral_model.pth"
    python plot_training_curves.py --checkpoint "...\\bottom_mua_spectral_model_finetuned.pth"
    python plot_training_curves.py --checkpoint "..." --output "curve.png" --log-scale
"""

import argparse

import matplotlib.pyplot as plt
import torch


def load_history(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "train_bottom_mua_mae_history" in checkpoint:
        train_history = checkpoint["train_bottom_mua_mae_history"]
        val_history = checkpoint["val_bottom_mua_mae_history"]
        is_fine_tune = False
    elif "fine_tune_train_bottom_mua_mae_history" in checkpoint:
        train_history = checkpoint["fine_tune_train_bottom_mua_mae_history"]
        val_history = checkpoint["fine_tune_val_bottom_mua_mae_history"]
        is_fine_tune = True
    else:
        raise KeyError(
            "No recognized MAE history keys found in this checkpoint. "
            "Expected 'train_bottom_mua_mae_history'/'val_bottom_mua_mae_history' "
            "or their fine_tune_ equivalents."
        )

    best_epoch = checkpoint.get("best_epoch")

    return train_history, val_history, is_fine_tune, best_epoch


def plot_curves(train_history, val_history, is_fine_tune, best_epoch, output_path, log_scale):
    epochs = list(range(1, len(train_history) + 1))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(epochs, train_history, label="Train MAE", color="tab:blue")
    ax.plot(epochs, val_history, label="Val MAE", color="tab:orange")

    if best_epoch is not None and 1 <= best_epoch <= len(epochs):
        ax.axvline(best_epoch, color="tab:green", linestyle="--", alpha=0.6,
                    label=f"Best epoch ({best_epoch})")

    if log_scale:
        ax.set_yscale("log")

    title = "Fine-tuning" if is_fine_tune else "Training"
    ax.set_title(f"{title} progress: bottom_mua MAE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"Saved plot to {output_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Path to a saved .pth checkpoint.")
    parser.add_argument("--output", default=None,
                         help="Path to save the plot PNG (defaults to <checkpoint>_curve.png).")
    parser.add_argument("--log-scale", action="store_true", help="Plot MAE on a log y-axis.")
    args = parser.parse_args()

    output_path = args.output
    if output_path is None:
        output_path = args.checkpoint.rsplit(".", 1)[0] + "_curve.png"

    train_history, val_history, is_fine_tune, best_epoch = load_history(args.checkpoint)

    print(f"Loaded {'fine-tune' if is_fine_tune else 'training'} history: "
          f"{len(train_history)} epochs.")
    if best_epoch is not None:
        print(f"Best epoch recorded in checkpoint: {best_epoch}")

    plot_curves(train_history, val_history, is_fine_tune, best_epoch, output_path, args.log_scale)


if __name__ == "__main__":
    main()
