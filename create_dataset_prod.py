import argparse
import glob
import os
import shutil

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from PIL import Image
from tqdm import tqdm

from create_dataset import (
    apply_pitch_distortion_with_map,
    apply_roll_distortion_with_map,
    apply_yaw_distortion_with_map,
    compose_flow,
    load_and_resize_image,
)
from visualize_flow import compute_covered_mse, compute_mse, forward_warp_distorted_to_original


IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")


def discover_images(input_dir):
    image_paths = []
    for pattern in IMAGE_EXTENSIONS:
        image_paths.extend(glob.glob(os.path.join(input_dir, "**", pattern), recursive=True))
        image_paths.extend(glob.glob(os.path.join(input_dir, "**", pattern.upper()), recursive=True))
    return sorted(set(image_paths))


def ensure_output_layout(output_dir):
    required_entries = (
        "train_distorted",
        "train_flow",
        "test_distorted",
        "test_flow",
        "original",
        "dataset_info.txt",
        "example.png",
    )
    if os.path.isdir(output_dir):
        existing = [name for name in required_entries if os.path.exists(os.path.join(output_dir, name))]
        if existing:
            existing_text = ", ".join(existing)
            raise FileExistsError(
                f"Output directory already contains dataset artifacts: {existing_text}. "
                f"Please use an empty directory or a new output path."
            )

    os.makedirs(os.path.join(output_dir, "train_distorted"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "train_flow"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test_distorted"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test_flow"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "original"), exist_ok=True)


def copy_originals(image_paths, destination_dir):
    used_names = set()
    copied_paths = []

    for source_path in image_paths:
        base_name = os.path.basename(source_path)
        stem, ext = os.path.splitext(base_name)
        candidate = base_name
        suffix = 1

        while candidate.lower() in used_names:
            candidate = f"{stem}_{suffix}{ext}"
            suffix += 1

        used_names.add(candidate.lower())
        target_path = os.path.join(destination_dir, candidate)
        shutil.copy2(source_path, target_path)
        copied_paths.append(target_path)

    return copied_paths


def build_generation_tasks(image_paths, samples_per_image, train_ratio, seed):
    tasks = []
    for image_path in image_paths:
        for sample_idx in range(samples_per_image):
            tasks.append(
                {
                    "image_path": image_path,
                    "sample_in_image": sample_idx + 1,
                }
            )

    rng = np.random.default_rng(seed)
    rng.shuffle(tasks)

    total_samples = len(tasks)
    train_count = int(round(total_samples * train_ratio))
    if total_samples > 1:
        train_count = min(max(train_count, 1), total_samples - 1)
    else:
        train_count = total_samples

    train_tasks = tasks[:train_count]
    test_tasks = tasks[train_count:]
    return train_tasks, test_tasks


def generate_sample(original_img, f_mm, pixel_size_mm, sampling_rate_hz):
    pitch_img, src_x1, src_y1, pitch_shifts, pitch_deg_vals, pitch_angle = apply_pitch_distortion_with_map(
        original_img, f_mm, pixel_size_mm
    )
    roll_img, src_x2, src_y2, roll_deg_vals, roll_amp, roll_shifts = apply_roll_distortion_with_map(
        pitch_img, f_mm, pixel_size_mm, sampling_rate_hz
    )
    yaw_img, src_x3, src_y3, yaw_deg_vals, yaw_amp, yaw_freq = apply_yaw_distortion_with_map(
        roll_img, sampling_rate_hz
    )
    u_total, v_total = compose_flow(src_x1, src_y1, src_x2, src_y2, src_x3, src_y3)

    metadata = {
        "pitch_angle_deg": float(pitch_angle),
        "roll_amplitude_deg": float(roll_amp),
        "yaw_amplitude_deg": float(yaw_amp),
        "yaw_frequency_hz": float(yaw_freq),
        "pitch_min_shift_px": float(np.min(pitch_shifts)),
        "pitch_max_shift_px": float(np.max(pitch_shifts)),
        "roll_min_shift_px": float(np.min(roll_shifts)),
        "roll_max_shift_px": float(np.max(roll_shifts)),
        "pitch_profile_min_deg": float(np.min(pitch_deg_vals)),
        "pitch_profile_max_deg": float(np.max(pitch_deg_vals)),
        "roll_profile_min_deg": float(np.min(roll_deg_vals)),
        "roll_profile_max_deg": float(np.max(roll_deg_vals)),
        "yaw_profile_min_deg": float(np.min(yaw_deg_vals)),
        "yaw_profile_max_deg": float(np.max(yaw_deg_vals)),
    }

    return yaw_img, u_total, v_total, metadata


def save_sample(distorted_img, u, v, distorted_dir, flow_dir, index):
    file_number = f"{index:06d}"
    image_path = os.path.join(distorted_dir, f"distorted_{file_number}.png")
    flow_path = os.path.join(flow_dir, f"distorted_{file_number}.mat")

    Image.fromarray(distorted_img[:, :, 0]).save(image_path)
    scio.savemat(flow_path, {"u": u, "v": v})
    return image_path, flow_path


def save_example_figure(original_img, distorted_img, u, v, output_path):
    restored, weights = forward_warp_distorted_to_original(distorted_img[:, :, 0], u, v)
    mse = compute_mse(original_img[:, :, 0], restored)
    covered_mse = compute_covered_mse(original_img[:, :, 0], restored, weights)
    coverage = float(np.mean(weights > 1e-8) * 100.0)
    diff = np.abs(original_img[:, :, 0].astype(np.float32) - restored.astype(np.float32))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(original_img[:, :, 0], cmap="gray")
    axes[0, 0].set_title("Original")
    axes[0, 1].imshow(distorted_img[:, :, 0], cmap="gray")
    axes[0, 1].set_title("Distorted")
    axes[0, 2].imshow(restored, cmap="gray")
    axes[0, 2].set_title("Restored from flow")

    im_u = axes[1, 0].imshow(u, cmap="RdBu")
    axes[1, 0].set_title("u")
    plt.colorbar(im_u, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im_v = axes[1, 1].imshow(v, cmap="RdBu")
    axes[1, 1].set_title("v")
    plt.colorbar(im_v, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im_diff = axes[1, 2].imshow(diff, cmap="hot")
    axes[1, 2].set_title("Absolute difference")
    plt.colorbar(im_diff, ax=axes[1, 2], fraction=0.046, pad=0.04)

    for ax in axes.ravel():
        ax.axis("off")

    fig.suptitle(
        f"Example train sample | MSE={mse:.4f} | Covered MSE={covered_mse:.4f} | Coverage={coverage:.1f}%",
        fontsize=14,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_dataset_info(output_path, input_dir, image_paths, copied_originals, train_count, test_count, args, logs):
    with open(output_path, "w", encoding="utf-8") as info_file:
        info_file.write("DATASET INFORMATION\n")
        info_file.write("=" * 60 + "\n\n")
        info_file.write(f"Input directory: {input_dir}\n")
        info_file.write(f"Unique source images: {len(image_paths)}\n")
        info_file.write(f"Copied originals: {len(copied_originals)}\n")
        info_file.write(f"Samples per image: {args.samples_per_image}\n")
        info_file.write(f"Total samples: {train_count + test_count}\n")
        info_file.write(f"Train samples: {train_count}\n")
        info_file.write(f"Test samples: {test_count}\n")
        info_file.write(f"Train ratio: {args.train_ratio:.4f}\n")
        info_file.write(f"Image size: {args.img_size}x{args.img_size}\n")
        info_file.write(f"f_mm: {args.f_mm}\n")
        info_file.write(f"pixel_size_mm: {args.pixel_size_mm}\n")
        info_file.write(f"sampling_rate_hz: {args.sampling_rate_hz}\n")
        info_file.write(f"seed: {args.seed}\n\n")

        info_file.write("Output structure:\n")
        info_file.write("  original/\n")
        info_file.write("  train_distorted/\n")
        info_file.write("  train_flow/\n")
        info_file.write("  test_distorted/\n")
        info_file.write("  test_flow/\n")
        info_file.write("  dataset_info.txt\n")
        info_file.write("  example.png\n\n")

        info_file.write("Per-sample parameters:\n")
        info_file.write("-" * 60 + "\n")
        for log in logs:
            info_file.write(
                f"{log['split']} #{log['dataset_index']:06d} | "
                f"src={log['source_name']} | "
                f"sample_in_image={log['sample_in_image']} | "
                f"pitch={log['pitch_angle_deg']:.3f} deg | "
                f"roll_amp={log['roll_amplitude_deg']:.4f} deg | "
                f"yaw_amp={log['yaw_amplitude_deg']:.3f} deg | "
                f"yaw_freq={log['yaw_frequency_hz']:.3f} Hz\n"
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Production dataset generator for distorted images and flow fields.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with source images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for the generated dataset")
    parser.add_argument("--samples_per_image", type=int, required=True, help="How many distorted samples to create per source image")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Train split ratio in [0, 1]")
    parser.add_argument("--img_size", type=int, default=256, help="Target square image size")
    parser.add_argument("--f_mm", type=float, default=50.0, help="Focal length in mm")
    parser.add_argument("--pixel_size_mm", type=float, default=0.005, help="Pixel size in mm")
    parser.add_argument("--sampling_rate_hz", type=float, default=512.0, help="Sampling rate in Hz")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible generation")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.samples_per_image < 1:
        raise ValueError("--samples_per_image must be at least 1")
    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train_ratio must be in the open interval (0, 1)")

    image_paths = discover_images(args.input_dir)
    if not image_paths:
        raise FileNotFoundError(f"No source images found in: {args.input_dir}")

    ensure_output_layout(args.output_dir)
    np.random.seed(args.seed)

    copied_originals = copy_originals(image_paths, os.path.join(args.output_dir, "original"))
    train_tasks, test_tasks = build_generation_tasks(
        image_paths=image_paths,
        samples_per_image=args.samples_per_image,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    print("=" * 60)
    print("GENERATING PRODUCTION DATASET")
    print("=" * 60)
    print(f"Input images: {len(image_paths)}")
    print(f"Samples per image: {args.samples_per_image}")
    print(f"Train samples: {len(train_tasks)}")
    print(f"Test samples: {len(test_tasks)}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)

    logs = []
    first_train_example = None

    split_configs = (
        ("train", train_tasks, os.path.join(args.output_dir, "train_distorted"), os.path.join(args.output_dir, "train_flow")),
        ("test", test_tasks, os.path.join(args.output_dir, "test_distorted"), os.path.join(args.output_dir, "test_flow")),
    )

    for split_name, tasks, distorted_dir, flow_dir in split_configs:
        for dataset_index, task in enumerate(tqdm(tasks, desc=f"Generating {split_name}"), start=1):
            original_img = load_and_resize_image(task["image_path"], args.img_size)
            if original_img is None:
                continue

            distorted_img, u, v, metadata = generate_sample(
                original_img=original_img,
                f_mm=args.f_mm,
                pixel_size_mm=args.pixel_size_mm,
                sampling_rate_hz=args.sampling_rate_hz,
            )

            distorted_path, flow_path = save_sample(
                distorted_img=distorted_img,
                u=u,
                v=v,
                distorted_dir=distorted_dir,
                flow_dir=flow_dir,
                index=dataset_index,
            )

            log_row = {
                "split": split_name,
                "dataset_index": dataset_index,
                "source_name": os.path.basename(task["image_path"]),
                "sample_in_image": task["sample_in_image"],
            }
            log_row.update(metadata)
            logs.append(log_row)

            if split_name == "train" and first_train_example is None:
                first_train_example = {
                    "original_img": original_img.copy(),
                    "distorted_img": distorted_img.copy(),
                    "u": u.copy(),
                    "v": v.copy(),
                    "distorted_path": distorted_path,
                    "flow_path": flow_path,
                }

    if first_train_example is None:
        raise RuntimeError("Failed to generate a train sample, so example.png could not be created.")

    save_example_figure(
        original_img=first_train_example["original_img"],
        distorted_img=first_train_example["distorted_img"],
        u=first_train_example["u"],
        v=first_train_example["v"],
        output_path=os.path.join(args.output_dir, "example.png"),
    )

    write_dataset_info(
        output_path=os.path.join(args.output_dir, "dataset_info.txt"),
        input_dir=args.input_dir,
        image_paths=image_paths,
        copied_originals=copied_originals,
        train_count=len(train_tasks),
        test_count=len(test_tasks),
        args=args,
        logs=logs,
    )

    print("\n" + "=" * 60)
    print("DATASET GENERATION COMPLETED")
    print("=" * 60)
    print(f"Copied originals: {len(copied_originals)}")
    print(f"Train samples: {len(train_tasks)}")
    print(f"Test samples: {len(test_tasks)}")
    print(f"Example saved to: {os.path.join(args.output_dir, 'example.png')}")


if __name__ == "__main__":
    main()
