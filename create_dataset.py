import glob
import math
import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from PIL import Image
from scipy.ndimage import gaussian_filter1d, map_coordinates
from tqdm import tqdm


def bilinear_sample_gray(image_2d, y, x):
    h, w = image_2d.shape
    if x < 0 or x > w - 1 or y < 0 or y > h - 1:
        return 0.0

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)

    wx = x - x0
    wy = y - y0

    q11 = float(image_2d[y0, x0])
    q12 = float(image_2d[y0, x1])
    q21 = float(image_2d[y1, x0])
    q22 = float(image_2d[y1, x1])

    return (
        q11 * (1.0 - wx) * (1.0 - wy)
        + q12 * wx * (1.0 - wy)
        + q21 * (1.0 - wx) * wy
        + q22 * wx * wy
    )


def apply_pitch_distortion_with_map(image, f_mm, pixel_size_mm):
    h, w = image.shape[:2]

    max_theta_deg = np.random.uniform(-0.5, 0.0)
    theta_deg = np.linspace(0.0, max_theta_deg, h)

    shifts = np.zeros(h, dtype=np.float32)
    for row in range(h):
        theta_rad = np.radians(theta_deg[row])
        delta_mm = f_mm * np.tan(theta_rad)
        shifts[row] = delta_mm / pixel_size_mm

    distorted = np.zeros_like(image)
    src_x_map = np.zeros((h, w), dtype=np.float32)
    src_y_map = np.zeros((h, w), dtype=np.float32)

    for row in range(h):
        source_row = row - shifts[row]
        if 0.0 <= source_row <= h - 1:
            for col in range(w):
                distorted[row, col, 0] = bilinear_sample_gray(image[:, :, 0], source_row, col)
            src_x_map[row, :] = np.arange(w, dtype=np.float32)
            src_y_map[row, :] = source_row

    return distorted, src_x_map, src_y_map, shifts, theta_deg, max_theta_deg


def apply_roll_distortion_with_map(image, f_mm, pixel_size_mm, sampling_rate_hz):
    h, w = image.shape[:2]

    roll_amplitude_deg = np.random.uniform(0.01, 0.1)
    random_noise = np.random.normal(0.0, 1.0, h)
    smoothing_freq = np.random.uniform(10.0, 50.0)
    sigma = sampling_rate_hz / smoothing_freq / 2.0
    smooth_noise = gaussian_filter1d(random_noise, sigma=sigma)

    max_val = np.max(np.abs(smooth_noise))
    if max_val > 0:
        roll_deg = smooth_noise / max_val * roll_amplitude_deg
    else:
        roll_deg = smooth_noise

    horizontal_shifts = np.zeros(h, dtype=np.float32)
    for row in range(h):
        roll_rad = np.radians(roll_deg[row])
        delta_mm = f_mm * np.tan(roll_rad)
        horizontal_shifts[row] = delta_mm / pixel_size_mm

    distorted = np.zeros_like(image)
    src_x_map = np.zeros((h, w), dtype=np.float32)
    src_y_map = np.zeros((h, w), dtype=np.float32)

    for row in range(h):
        for col in range(w):
            source_col = col - horizontal_shifts[row]
            if 0.0 <= source_col <= w - 1:
                distorted[row, col, 0] = bilinear_sample_gray(image[:, :, 0], row, source_col)
                src_x_map[row, col] = source_col
                src_y_map[row, col] = row

    return distorted, src_x_map, src_y_map, roll_deg, roll_amplitude_deg, horizontal_shifts


def apply_yaw_distortion_with_map(image, sampling_rate_hz):
    h, w = image.shape[:2]

    yaw_amplitude_deg = np.random.uniform(1.0, 10.0)
    yaw_frequency_hz = np.random.uniform(1.0, 10.0)
    t = np.arange(h) / sampling_rate_hz
    yaw_deg = yaw_amplitude_deg * np.sin(2.0 * np.pi * yaw_frequency_hz * t)

    center_i, center_j = h // 2, w // 2

    distorted = np.zeros_like(image)
    src_x_map = np.zeros((h, w), dtype=np.float32)
    src_y_map = np.zeros((h, w), dtype=np.float32)

    for row in range(h):
        psi_rad = np.radians(yaw_deg[row])
        cos_psi = math.cos(psi_rad)
        sin_psi = math.sin(psi_rad)

        for col in range(w):
            di = row - center_i
            dj = col - center_j

            src_i = center_i + di * cos_psi - dj * sin_psi
            src_j = center_j + di * sin_psi + dj * cos_psi

            if 0.0 <= src_i <= h - 1 and 0.0 <= src_j <= w - 1:
                distorted[row, col, 0] = bilinear_sample_gray(image[:, :, 0], src_i, src_j)
                src_x_map[row, col] = src_j
                src_y_map[row, col] = src_i

    return distorted, src_x_map, src_y_map, yaw_deg, yaw_amplitude_deg, yaw_frequency_hz


def load_and_resize_image(image_path, target_size):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        h, w = img.shape[:2]
        if h != target_size or w != target_size:
            img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
        return img[..., np.newaxis]
    except Exception as exc:
        print(f"Error loading {image_path}: {exc}")
        return None


def compose_flow(src_x1, src_y1, src_x2, src_y2, src_x3, src_y3):
    h, w = src_x1.shape
    u_total = np.zeros((h, w), dtype=np.float32)
    v_total = np.zeros((h, w), dtype=np.float32)

    for row in range(h):
        for col in range(w):
            y_before_yaw = src_y3[row, col]
            x_before_yaw = src_x3[row, col]

            if 0 <= int(y_before_yaw) < h and 0 <= int(x_before_yaw) < w:
                y_before_roll = src_y2[int(y_before_yaw), int(x_before_yaw)]
                x_before_roll = src_x2[int(y_before_yaw), int(x_before_yaw)]
            else:
                y_before_roll = y_before_yaw
                x_before_roll = x_before_yaw

            if 0 <= int(y_before_roll) < h and 0 <= int(x_before_roll) < w:
                y_original = src_y1[int(y_before_roll), int(x_before_roll)]
                x_original = src_x1[int(y_before_roll), int(x_before_roll)]
            else:
                y_original = y_before_roll
                x_original = x_before_roll

            u_total[row, col] = x_original - col
            v_total[row, col] = y_original - row

    return u_total, v_total


def restore_from_flow(distorted_image, u, v):
    h, w = distorted_image.shape[:2]
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    y_new = np.clip(y + v, 0, h - 1)
    x_new = np.clip(x + u, 0, w - 1)
    restored = map_coordinates(
        distorted_image[:, :, 0].astype(np.float32),
        [y_new.ravel(), x_new.ravel()],
        order=1,
    ).reshape(h, w)
    return np.clip(restored, 0, 255).astype(np.uint8)


def visualize_result(original, pitch_img, roll_img, yaw_img, distorted_img, restored, u, v, save_path):
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    axes[0, 0].imshow(original[:, :, 0], cmap="gray")
    axes[0, 0].set_title("Original")
    axes[0, 1].imshow(pitch_img[:, :, 0], cmap="gray")
    axes[0, 1].set_title("Pitch")
    axes[0, 2].imshow(roll_img[:, :, 0], cmap="gray")
    axes[0, 2].set_title("Roll")
    axes[0, 3].imshow(yaw_img[:, :, 0], cmap="gray")
    axes[0, 3].set_title("Yaw")

    axes[1, 0].imshow(distorted_img[:, :, 0], cmap="gray")
    axes[1, 0].set_title("Final distorted")
    axes[1, 1].imshow(restored, cmap="gray")
    axes[1, 1].set_title("Restored from flow")
    im_u = axes[1, 2].imshow(u, cmap="RdBu")
    axes[1, 2].set_title("u")
    plt.colorbar(im_u, ax=axes[1, 2], fraction=0.046, pad=0.04)
    im_v = axes[1, 3].imshow(v, cmap="RdBu")
    axes[1, 3].set_title("v")
    plt.colorbar(im_v, ax=axes[1, 3], fraction=0.046, pad=0.04)

    for ax in axes.ravel():
        ax.axis("off")

    mse = np.mean((original[:, :, 0].astype(np.float32) - restored.astype(np.float32)) ** 2)
    fig.suptitle(f"Distortion chain and resulting flow | MSE={mse:.4f}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_dataset(
    sourcedir,
    datasetdir,
    num_samples=4,
    img_size=256,
    f_mm=50.0,
    pixel_size_mm=0.005,
    sampling_rate_hz=512.0,
    visualize=True,
):
    original_path = os.path.join(datasetdir, "original")
    pitch_path = os.path.join(datasetdir, "pitch")
    roll_path = os.path.join(datasetdir, "roll")
    yaw_path = os.path.join(datasetdir, "yaw")
    distorted_path = os.path.join(datasetdir, "distorted")
    flow_path = os.path.join(datasetdir, "flow")

    os.makedirs(original_path, exist_ok=True)
    os.makedirs(pitch_path, exist_ok=True)
    os.makedirs(roll_path, exist_ok=True)
    os.makedirs(yaw_path, exist_ok=True)
    os.makedirs(distorted_path, exist_ok=True)
    os.makedirs(flow_path, exist_ok=True)

    source_images = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]:
        source_images.extend(glob.glob(os.path.join(sourcedir, "**", ext), recursive=True))
        source_images.extend(glob.glob(os.path.join(sourcedir, "**", ext.upper()), recursive=True))
    source_images = sorted(list(set(source_images)))

    if not source_images:
        print("ERROR: No source images found!")
        return

    print("=" * 60)
    print("GENERATING DATASET WITH FLOW FIELD")
    print("=" * 60)
    print(f"Source directory: {sourcedir}")
    print(f"Output directory: {datasetdir}")
    print(f"Number of samples: {num_samples}")
    print(f"Image size: {img_size}x{img_size}")
    print(f"Found source images: {len(source_images)}")
    print("=" * 60)

    successful = 0
    params_log = []

    for i in tqdm(range(num_samples), desc="Generating"):
        src_idx = i % len(source_images)
        src_path = source_images[src_idx]

        original_img = load_and_resize_image(src_path, img_size)
        if original_img is None:
            continue

        pitch_img, src_x1, src_y1, pitch_shifts, pitch_deg_vals, pitch_angle = apply_pitch_distortion_with_map(
            original_img, f_mm, pixel_size_mm
        )
        roll_img, src_x2, src_y2, roll_deg_vals, roll_amp, roll_shifts = apply_roll_distortion_with_map(
            pitch_img, f_mm, pixel_size_mm, sampling_rate_hz
        )
        yaw_img, src_x3, src_y3, yaw_deg_vals, yaw_amp, yaw_freq = apply_yaw_distortion_with_map(
            roll_img, sampling_rate_hz
        )
        distorted_img = yaw_img.copy()

        u_total, v_total = compose_flow(src_x1, src_y1, src_x2, src_y2, src_x3, src_y3)

        number = str(i + 1).zfill(6)
        Image.fromarray(original_img[:, :, 0]).save(os.path.join(original_path, f"original_{number}.png"))
        Image.fromarray(pitch_img[:, :, 0]).save(os.path.join(pitch_path, f"pitch_{number}.png"))
        Image.fromarray(roll_img[:, :, 0]).save(os.path.join(roll_path, f"roll_{number}.png"))
        Image.fromarray(yaw_img[:, :, 0]).save(os.path.join(yaw_path, f"yaw_{number}.png"))
        Image.fromarray(distorted_img[:, :, 0]).save(os.path.join(distorted_path, f"distorted_{number}.png"))
        scio.savemat(os.path.join(flow_path, f"flow_{number}.mat"), {"u": u_total, "v": v_total})

        params_log.append(
            {
                "sample": i + 1,
                "pitch_angle": pitch_angle,
                "roll_amplitude_deg": roll_amp,
                "yaw_amplitude_deg": yaw_amp,
                "yaw_frequency_hz": yaw_freq,
            }
        )

        successful += 1

        if visualize and i == 0:
            restored = restore_from_flow(distorted_img, u_total, v_total)
            visualize_result(
                original_img,
                pitch_img,
                roll_img,
                yaw_img,
                distorted_img,
                restored,
                u_total,
                v_total,
                os.path.join(datasetdir, "verification_flow.png"),
            )

        if (i + 1) % 10 == 0:
            print(
                f"\nSample {i + 1}: pitch={pitch_angle:.3f} deg, "
                f"roll_amp={roll_amp:.4f} deg, "
                f"yaw_amp={yaw_amp:.2f} deg, yaw_freq={yaw_freq:.2f} Hz"
            )

    info_path = os.path.join(datasetdir, "dataset_info.txt")
    with open(info_path, "w", encoding="utf-8") as info_file:
        info_file.write("DATASET INFORMATION\n")
        info_file.write("=" * 60 + "\n\n")
        info_file.write(f"Total samples: {successful}\n")
        info_file.write(f"Image size: {img_size}x{img_size}\n\n")
        info_file.write("Folder structure:\n")
        info_file.write(f"  {datasetdir}/\n")
        info_file.write("    original/\n")
        info_file.write("    pitch/\n")
        info_file.write("    roll/\n")
        info_file.write("    yaw/\n")
        info_file.write("    distorted/\n")
        info_file.write("    flow/\n\n")
        info_file.write("Per-sample parameters:\n")
        info_file.write("-" * 40 + "\n")
        for params in params_log:
            info_file.write(
                f"Sample {params['sample']:04d}: "
                f"pitch={params['pitch_angle']:.3f} deg, "
                f"roll_amp={params['roll_amplitude_deg']:.4f} deg, "
                f"yaw_amp={params['yaw_amplitude_deg']:.2f} deg, "
                f"yaw_freq={params['yaw_frequency_hz']:.2f} Hz\n"
            )

    print("\n" + "=" * 60)
    print("GENERATION COMPLETED")
    print("=" * 60)
    print(f"Generated: {successful} of {num_samples}")
    print("Saved folders: original/, pitch/, roll/, yaw/, distorted/, flow/")


if __name__ == "__main__":
    generate_dataset(
        sourcedir="input_image",
        datasetdir="dataset_real_flow",
        num_samples=10,
        img_size=256,
        f_mm=50.0,
        pixel_size_mm=0.005,
        sampling_rate_hz=512.0,
        visualize=True,
    )
