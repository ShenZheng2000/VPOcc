import os
import os.path as osp
import glob
import pickle
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Simple SemanticKITTI-like color map (fallback-friendly)
# index -> RGB
COLOR_MAP = {
    0:   (255, 255, 255),  # empty -> white
    1:   (245, 150, 100),
    10:  (245, 230, 100),
    11:  (150, 60, 30),
    13:  (180, 30, 80),
    15:  (255, 0, 0),
    16:  (30, 30, 255),
    18:  (200, 40, 255),
    20:  (90, 30, 150),
    30:  (255, 0, 255),
    31:  (255, 150, 255),
    32:  (75, 0, 75),
    40:  (75, 0, 175),
    44:  (0, 200, 255),
    48:  (50, 120, 255),
    49:  (0, 175, 0),
    50:  (0, 60, 135),
    51:  (80, 240, 150),
    52:  (150, 240, 255),
    60:  (0, 0, 255),
    70:  (255, 255, 50),
    71:  (245, 150, 100),
    72:  (255, 0, 0),
    80:  (200, 30, 30),
    81:  (255, 40, 200),
    99:  (0, 0, 0),
    255: (128, 128, 128),  # ignore
}


def colorize(label_2d: np.ndarray) -> np.ndarray:
    h, w = label_2d.shape
    img = np.ones((h, w, 3), dtype=np.uint8) * 255
    unique_vals = np.unique(label_2d)
    for v in unique_vals:
        color = COLOR_MAP.get(int(v), (0, 0, 0))
        img[label_2d == v] = color
    return img


def voxel_to_bev(voxel: np.ndarray, empty_label: int = 0, ignore_label: int = 255) -> np.ndarray:
    """
    voxel: [H, W, Z] or [256,256,32]
    Output: top-down 2D label map [H, W]

    Rule:
    - look from top z -> bottom z
    - first non-empty, non-ignore voxel decides BEV class
    - if none found, keep empty_label
    """
    assert voxel.ndim == 3, f"Expected 3D voxel grid, got shape {voxel.shape}"

    h, w, z = voxel.shape
    bev = np.full((h, w), empty_label, dtype=voxel.dtype)

    valid = (voxel != empty_label) & (voxel != ignore_label)

    # scan from top to bottom
    for k in range(z - 1, -1, -1):
        mask = valid[:, :, k] & (bev == empty_label)
        bev[mask] = voxel[:, :, k][mask]

    return bev


def save_single_panel(img: np.ndarray, save_path: str, title: str = None):
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close()


def save_two_panel(img1: np.ndarray, img2: np.ndarray, save_path: str, title1="pred", title2="target"):
    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.axis("off")
    plt.title(title1)

    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.axis("off")
    plt.title(title2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to pkl dir or single pkl file")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save png files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if osp.isfile(args.path):
        pkl_files = [args.path]
    else:
        pkl_files = sorted(glob.glob(osp.join(args.path, "*.pkl")))

    if len(pkl_files) == 0:
        raise FileNotFoundError(f"No .pkl files found in: {args.path}")

    print(f"Found {len(pkl_files)} pkl files")

    for pkl_path in pkl_files:
        with open(pkl_path, "rb") as f:
            outputs = pickle.load(f)

        print("processing:", pkl_path)

        if "pred" not in outputs:
            raise KeyError(f"'pred' not found in {pkl_path}. Available keys: {list(outputs.keys())}")

        pred = outputs["pred"]
        if pred.shape != (256, 256, 32):
            pred = pred.reshape(256, 256, 32)

        pred_bev = voxel_to_bev(pred)
        pred_img = colorize(pred_bev)

        stem = osp.splitext(osp.basename(pkl_path))[0]

        pred_save = osp.join(args.output_dir, f"{stem}_pred_bev.png")
        save_single_panel(pred_img, pred_save, title="pred")

        if "target" in outputs:
            target = outputs["target"]
            if target.shape != (256, 256, 32):
                target = target.reshape(256, 256, 32)

            target_bev = voxel_to_bev(target)
            target_img = colorize(target_bev)

            target_save = osp.join(args.output_dir, f"{stem}_target_bev.png")
            pair_save = osp.join(args.output_dir, f"{stem}_pred_target_bev.png")

            save_single_panel(target_img, target_save, title="target")
            save_two_panel(pred_img, target_img, pair_save, title1="pred", title2="target")

    print("Done.")


if __name__ == "__main__":
    main()