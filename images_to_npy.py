"""
Convert two folders of paired images into the three .npy files that
Speckle-Image-Reconstruction (repo B) trains on.

  speckle_dir/   0001.png, 0002.png, ...   <- camera pictures of the speckle
  original_dir/  0001.png, 0002.png, ...   <- the image that produced each speckle

Files are paired by file name (without extension), so 0001.tif matches 0001.png.
Pairs are written in sorted file-name order, which matters because the repo
splits train/val/test sequentially (first 70% / next 10% / last 20%).

Output (in --out_dir):
  <name>_speckle.npy   (N, S, S) float32, raw camera units (0-255 or 0-65535)
  <name>_original.npy  (N, S, S) float32, scaled to [0, 1]
  <name>_index.npy     (N,)      int64
"""
import argparse
import os
import sys

import cv2
import numpy as np

EXTS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}


def list_images(folder):
    out = {}
    for f in sorted(os.listdir(folder)):
        stem, ext = os.path.splitext(f)
        if ext.lower() in EXTS:
            out[stem] = os.path.join(folder, f)
    return out


def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Could not read {path}")
    if img.ndim == 3:
        code = cv2.COLOR_BGRA2GRAY if img.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        img = cv2.cvtColor(img, code)
    return img


def dtype_max(img):
    if img.dtype == np.uint8:
        return 255.0
    if img.dtype == np.uint16:
        return 65535.0
    return 1.0  # float images are assumed to be in [0, 1] already


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--speckle_dir", required=True)
    p.add_argument("--original_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--name", default="train", help="prefix for the output files")
    p.add_argument("--size", type=int, default=64,
                   help="side length S of the square network input AND output (multiple of 16)")
    p.add_argument("--roi", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"), default=None,
                   help="crop the speckle images to this box before resizing "
                        "(use the SAME box in your live script)")
    args = p.parse_args()

    if args.size % 16:
        sys.exit("--size must be a multiple of 16 (the U-Nets downsample several times)")

    spk = list_images(args.speckle_dir)
    org = list_images(args.original_dir)
    stems = [s for s in spk if s in org]
    if not stems:
        sys.exit("No matching file names between the two folders.")
    skipped = (len(spk) - len(stems)) + (len(org) - len(stems))
    if skipped:
        print(f"Warning: {skipped} file(s) had no partner and were skipped.")

    S = args.size
    speckle = np.empty((len(stems), S, S), np.float32)
    original = np.empty((len(stems), S, S), np.float32)
    speckle_max = None

    for i, stem in enumerate(stems):
        a = load_gray(spk[stem])
        if speckle_max is None:
            speckle_max = dtype_max(a)
        if args.roi:
            x0, y0, x1, y1 = args.roi
            a = a[y0:y1, x0:x1]
        speckle[i] = cv2.resize(a.astype(np.float32), (S, S), interpolation=cv2.INTER_AREA)

        b = load_gray(org[stem])
        b = b.astype(np.float32) / dtype_max(b)           # labels must be in [0, 1]
        original[i] = cv2.resize(b, (S, S), interpolation=cv2.INTER_AREA)

    os.makedirs(args.out_dir, exist_ok=True)
    paths = [os.path.join(args.out_dir, f"{args.name}_{k}.npy") for k in ("speckle", "original", "index")]
    np.save(paths[0], speckle)
    np.save(paths[1], original)
    np.save(paths[2], np.arange(len(stems), dtype=np.int64))

    nf = int(speckle_max + 1) if speckle_max > 1 else 1
    print(f"Wrote {len(stems)} pairs at {S}x{S}.")
    print(f"Speckle range: {speckle.min():.1f}..{speckle.max():.1f}  | label range: "
          f"{original.min():.3f}..{original.max():.3f}")
    print("\nTrain with:\n")
    print(f"python main.py --data_dir {paths[0]} {paths[1]} {paths[2]} "
          f"--model_type HPM_UNet --code_length {S} --normalize_factor {nf} "
          f"--epochs 90 --batch_size 64")


if __name__ == "__main__":
    main()
