import os
import argparse

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from PIL import Image


def load_image(image_path):
    image = np.asarray(Image.open(image_path))
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]
    return image


def forward_warp_distorted_to_original(distorted, u, v):
    h, w = distorted.shape[:2]
    is_gray = distorted.ndim == 2
    channels = 1 if is_gray else distorted.shape[2]

    accum = np.zeros((h, w, channels), dtype=np.float32)
    weights = np.zeros((h, w), dtype=np.float32)
    source = distorted.astype(np.float32)[:, :, None] if is_gray else distorted.astype(np.float32)

    for yd in range(h):
        for xd in range(w):
            xo = xd + float(u[yd, xd])
            yo = yd + float(v[yd, xd])

            if xo < 0 or xo > w - 1 or yo < 0 or yo > h - 1:
                continue

            x0 = int(np.floor(xo))
            y0 = int(np.floor(yo))
            x1 = min(x0 + 1, w - 1)
            y1 = min(y0 + 1, h - 1)

            wx = xo - x0
            wy = yo - y0

            splat_weights = (
                (x0, y0, (1.0 - wx) * (1.0 - wy)),
                (x1, y0, wx * (1.0 - wy)),
                (x0, y1, (1.0 - wx) * wy),
                (x1, y1, wx * wy),
            )

            pixel = source[yd, xd]
            for xi, yi, weight in splat_weights:
                if weight <= 0:
                    continue
                accum[yi, xi] += pixel * weight
                weights[yi, xi] += weight

    safe_weights = np.where(weights > 1e-8, weights, 1.0)
    restored = accum / safe_weights[:, :, None]
    restored[weights <= 1e-8] = 0
    restored = np.clip(restored, 0, 255).astype(np.uint8)

    if is_gray:
        return restored[:, :, 0], weights
    return restored, weights


def validate_paths(*paths):
    missing = [path for path in paths if not os.path.exists(path)]
    if missing:
        missing_text = "\n".join(missing)
        raise FileNotFoundError(f"Missing required files:\n{missing_text}")


def compute_mse(original, restored):
    if original.shape != restored.shape:
        raise ValueError(
            f"Original and restored images must have the same shape, got {original.shape} and {restored.shape}"
        )
    return float(np.mean((original.astype(np.float32) - restored.astype(np.float32)) ** 2))


def compute_covered_mse(original, restored, weights):
    covered_mask = weights > 1e-8
    if not np.any(covered_mask):
        return float("nan")

    if original.ndim == 2:
        diff = original.astype(np.float32) - restored.astype(np.float32)
        return float(np.mean((diff[covered_mask]) ** 2))

    diff = original.astype(np.float32) - restored.astype(np.float32)
    mask3 = covered_mask[:, :, None]
    return float(np.mean((diff[mask3]) ** 2))


def main():
    parser = argparse.ArgumentParser(
        description="Visualize original image, distorted image, and distorted image after applying the displacement field."
    )
    parser.add_argument("--original_path", type=str, required=True, help="Path to the original image")
    parser.add_argument("--distorted_path", type=str, required=True, help="Path to the distorted image")
    parser.add_argument("--flow_path", type=str, required=True, help="Path to the .mat file with u and v")
    parser.add_argument("--save_path", type=str, default=None, help="Optional path to save the figure")
    parser.add_argument("--no_show", action="store_true", help="Do not open the matplotlib window")
    args = parser.parse_args()

    validate_paths(args.original_path, args.distorted_path, args.flow_path)

    original = load_image(args.original_path)
    distorted = load_image(args.distorted_path)
    flow = scio.loadmat(args.flow_path)
    u = flow["u"].astype(np.float32)
    v = flow["v"].astype(np.float32)

    if original.shape[:2] != distorted.shape[:2]:
        raise ValueError(
            f"Original and distorted images must have matching height and width, got {original.shape[:2]} and {distorted.shape[:2]}"
        )
    if u.shape != original.shape[:2] or v.shape != original.shape[:2]:
        raise ValueError(
            f"Flow shape must match image size, got u={u.shape}, v={v.shape}, image={original.shape[:2]}"
        )

    restored, weights = forward_warp_distorted_to_original(distorted, u, v)
    mse = compute_mse(original, restored)
    covered_mse = compute_covered_mse(original, restored, weights)
    coverage = float(np.mean(weights > 1e-8) * 100.0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    if original.ndim == 2:
        axes[0, 0].imshow(original, cmap="gray")
        axes[0, 1].imshow(distorted, cmap="gray")
        axes[1, 0].imshow(original, cmap="gray")
        axes[1, 1].imshow(restored, cmap="gray")
    else:
        axes[0, 0].imshow(original)
        axes[0, 1].imshow(distorted)
        axes[1, 0].imshow(original)
        axes[1, 1].imshow(restored)

    axes[0, 0].set_title("Original")
    axes[0, 1].set_title("Distorted")
    axes[1, 0].set_title("Original")
    axes[1, 1].set_title(f"Flow applied to distorted\nMSE={mse:.4f}")

    for ax in axes.ravel():
        ax.axis("off")

    fig.suptitle(
        f"Restoration MSE = {mse:.4f} | Covered MSE = {covered_mse:.4f} | Coverage = {coverage:.1f}%",
        fontsize=14,
    )
    plt.tight_layout()

    if args.save_path:
        fig.savefig(args.save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to: {args.save_path}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
