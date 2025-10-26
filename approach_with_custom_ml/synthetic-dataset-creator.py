#!/usr/bin/env python3
"""
Synthesize album-page images by pasting whole rectangular photos on a white canvas.
Each pasted photo is optionally rotated and labeled as a 4-point YOLO-Seg polygon.

Req: pip install pillow numpy tqdm

Examples:
  # standard layout (images/, labels/, dataset yaml)
  python synth_rect_album.py --src /path/to/photos --out ./data --n 500

  # CVAT-ready layout (images/train, labels/train, train.txt, data.yaml)
  python synth_rect_album.py --src /path/to/photos --out ./data --n 500 --cvat-ready
"""

import argparse
import math
import random
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageOps

# ---------------- utils ----------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def list_images(root: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]

def rect_iou(a, b):
    # a,b: (x0,y0,x1,y1)
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def too_overlapping(new_box, boxes, max_iou):
    return any(rect_iou(new_box, b) > max_iou for b in boxes)

def yolo_poly_line(poly_xy, W, H, cls=0):
    flat = []
    for x, y in poly_xy:
        flat += [x / W, y / H]
    return f"{cls} " + " ".join(f"{v:.6f}" for v in flat)

# --- geometry for rotated rectangle corners (with expand=True emulation) ---

def rotate_rect_corners(w, h, angle_deg):
    """
    Return:
      poly_rot: 4 points of the rotated rectangle in the rotated-image coordinates
      rw, rh:   rotated image size that PIL would produce with expand=True
    Corner order: TL, TR, BR, BL after rotation.
    """
    cx, cy = w / 2.0, h / 2.0
    rad = math.radians(angle_deg)
    cosA, sinA = math.cos(rad), math.sin(rad)

    # original corners
    corners = [(0, 0), (w, 0), (w, h), (0, h)]

    # rotate around center
    rot = []
    for x, y in corners:
        dx, dy = x - cx, y - cy
        rx =  cx + dx * cosA - dy * sinA
        ry =  cy + dx * sinA + dy * cosA
        rot.append((rx, ry))

    # translate so min becomes 0 (expand=True offset)
    xs = [p[0] for p in rot]
    ys = [p[1] for p in rot]
    minx, miny = min(xs), min(ys)
    poly_rot = [(x - minx, y - miny) for x, y in rot]

    # rotated canvas size
    rw = int(math.ceil(max(p[0] for p in poly_rot)))
    rh = int(math.ceil(max(p[1] for p in poly_rot)))
    return poly_rot, rw, rh

# ---------------- core ----------------

def synth_one(canvas_W, canvas_H, pool_paths, min_per, max_per,
              max_overlap_iou, scale_min, scale_max, border_px,
              rot_min_deg, rot_max_deg):
    # white canvas
    canvas = Image.new("RGB", (canvas_W, canvas_H), (255, 255, 255))
    placed_aabbs = []      # for simple IoU overlap checks (AABB of rotated image)
    labels = []            # YOLO-Seg 4-pt polygons

    n = random.randint(min_per, max_per)
    tries = 0
    max_tries = n * 50

    while len(placed_aabbs) < n and tries < max_tries:
        tries += 1
        src_path = random.choice(pool_paths)
        img = Image.open(src_path).convert("RGB")

        # optional white border to mimic physical photo
        if border_px > 0:
            img = ImageOps.expand(img, border=border_px, fill=(255, 255, 255))

        # random target width relative to canvas width
        s = random.uniform(scale_min, scale_max)
        tgt_w = int(canvas_W * s)

        w, h = img.size
        if w == 0 or h == 0 or tgt_w < 1:
            continue
        scale = tgt_w / w
        tw = max(1, int(round(w * scale)))
        th = max(1, int(round(h * scale)))
        img_resized = img.resize((tw, th), resample=Image.BICUBIC)

        # random rotation
        angle = random.uniform(rot_min_deg, rot_max_deg)
        # compute rotated rectangle polygon and rotated image size
        poly_rot, rw, rh = rotate_rect_corners(tw, th, -angle)
        # make visual rotated image for pasting
        img_rot = img_resized.rotate(angle, expand=True, fillcolor=(255, 255, 255))

        if rw >= canvas_W or rh >= canvas_H:
            continue

        # random placement on canvas
        x0 = random.randint(0, canvas_W - rw)
        y0 = random.randint(0, canvas_H - rh)
        x1 = x0 + rw
        y1 = y0 + rh
        aabb = (x0, y0, x1, y1)

        if too_overlapping(aabb, placed_aabbs, max_overlap_iou):
            continue

        # paste and record
        canvas.paste(img_rot, (x0, y0))

        # polygon in canvas coordinates (translate by placement)
        poly_canvas = [(x + x0, y + y0) for (x, y) in poly_rot]
        labels.append(yolo_poly_line(poly_canvas, canvas_W, canvas_H, cls=0))
        placed_aabbs.append(aabb)

    return canvas, labels

def write_dataset_yaml_standard(out_dir: Path, name="album_rect", names=None):
    if names is None:
        names = ["photo"]
    (out_dir / f"{name}.yaml").write_text(
        f"path: {out_dir.resolve()}\n"
        f"train: images\n"
        f"val: images\n"
        f"names: {names}\n"
    )

def write_dataset_yaml_cvat(out_dir: Path):
    # exact structure requested
    (out_dir / "data.yaml").write_text(
        "names:\n"
        "  0: Photo\n"
        "path: .\n"
        "train: train.txt\n"
    )

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="Folder with source photos to paste")
    ap.add_argument("--out", type=Path, required=True, help="Output root")
    ap.add_argument("--n", type=int, default=500, help="Number of synthetic canvases")
    ap.add_argument("--canvas", type=int, nargs=2, default=[1600, 1200], metavar=("W", "H"))
    ap.add_argument("--min-per", type=int, default=1)
    ap.add_argument("--max-per", type=int, default=5)
    ap.add_argument("--max-iou", type=float, default=0.05, help="Max IoU overlap between placed AABBs")
    ap.add_argument("--scale-min", type=float, default=0.18, help="Min photo width as fraction of canvas width")
    ap.add_argument("--scale-max", type=float, default=0.45, help="Max photo width as fraction of canvas width")
    ap.add_argument("--border", type=int, default=0, help="White border around each photo in pixels")
    ap.add_argument("--rot-min", type=float, default=-15.0, help="Min rotation degrees")
    ap.add_argument("--rot-max", type=float, default=15.0, help="Max rotation degrees")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cvat-ready", action="store_true",
                    help="Create CVAT-style layout and YAML; write train.txt; use images/train and labels/train")
    args = ap.parse_args()

    random.seed(args.seed)
    srcs = list_images(args.src)
    if not srcs:
        raise SystemExit("No images found in --src")

    # layout
    if args.cvat_ready:
        img_dir = args.out / "images" / "train"
        lbl_dir = args.out / "labels" / "train"
    else:
        img_dir = args.out / "images"
        lbl_dir = args.out / "labels"

    ensure_dir(img_dir)
    ensure_dir(lbl_dir)

    W, H = args.canvas
    image_rel_paths = []  # to populate train.txt when cvat-ready

    for i in tqdm(range(args.n), ncols=80, desc="Synth"):
        img, lbls = synth_one(
            W, H, srcs, args.min_per, args.max_per,
            args.max_iou, args.scale_min, args.scale_max, args.border,
            args.rot_min, args.rot_max
        )
        stem = f"{i:06d}"
        img_path = img_dir / f"{stem}.jpg"
        lbl_path = lbl_dir / f"{stem}.txt"

        img.save(img_path, quality=95)
        with open(lbl_path, "w") as f:
            for line in lbls:
                f.write(line + "\n")

        if args.cvat_ready:
            # paths in train.txt should be relative to the dataset root where data.yaml lives
            image_rel_paths.append(str(img_path.relative_to(args.out)))

    # metadata
    if args.cvat_ready:
        # data.yaml
        write_dataset_yaml_cvat(args.out)
        # train.txt
        (args.out / "train.txt").write_text("\n".join(image_rel_paths) + "\n")

        # zips and removes the original
        import shutil
        zip_path = shutil.make_archive(args.out, "zip", root_dir=args.out)
        shutil.rmtree(args.out)
        print(f"Zipped dataset -> {zip_path}")
    else:
        write_dataset_yaml_standard(args.out, name="album_rect", names=["photo"])

    print("Done.")

if __name__ == "__main__":
    main()