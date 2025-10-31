#!/usr/bin/env python3
"""
Synthesize album-page images by pasting whole rectangular photos on a white canvas.
Each pasted photo is optionally rotated and labeled as a 4-point YOLO-Seg polygon.

Req: pip install pillow numpy tqdm

Examples:
  # standard layout (images/, labels/, dataset yaml)
  python synthetic-dataset-creator.py --src /path/to/photos --out ./data --n 500

  # CVAT-ready layout (images/train, labels/train, train.txt, data.yaml)
  python synthetic-dataset-creator.py --src /path/to/photos --out ./data --n 500 --cvat-ready
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

def rotated_rect_polygon(w, h, angle_deg):
    """4 corners of a w×h rectangle rotated by angle_deg (Pillow CCW), 
    translated so min corner is at (0,0)."""
    cx, cy = w / 2.0, h / 2.0
    rad = math.radians(-angle_deg)  # flip sign to match PIL.rotate
    cosA, sinA = math.cos(rad), math.sin(rad)
    corners = [(0, 0), (w, 0), (w, h), (0, h)]
    rot = []
    for x, y in corners:
        dx, dy = x - cx, y - cy
        rx = cx + dx * cosA - dy * sinA
        ry = cy + dx * sinA + dy * cosA
        rot.append((rx, ry))
    minx = min(p[0] for p in rot); miny = min(p[1] for p in rot)
    return [(x - minx, y - miny) for x, y in rot]

import numpy as np
from PIL import ImageFilter

def apply_vintage(im: Image.Image, strength=0.5, grain_std=4.0, fade=0.08, vignette=0.25):
    """Vintage/yellow tone + slight fade, grain, vignette. Strength in [0,1]."""
    a = np.asarray(im).astype(np.float32)

    # tone curve: slight fade (lift blacks, compress highs)
    if fade > 0:
        a = a*(1.0 - fade) + 255.0*fade*0.1  # lift shadows a bit

    # desaturate slightly
    gray = (0.299*a[...,0] + 0.587*a[...,1] + 0.114*a[...,2])[...,None]
    a = a*(1.0 - 0.25*strength) + gray*(0.25*strength)

    # yellow/sepia cast via channel scaling + matrix
    # base sepia matrix then interpolate with original
    M = np.array([[0.393, 0.769, 0.189],
                  [0.349, 0.686, 0.168],
                  [0.272, 0.534, 0.131]], dtype=np.float32)
    sep = np.tensordot(a, M.T, axes=1)
    # bias toward yellow (boost R,G a little)
    sep[...,0] *= (1.0 + 0.35*strength)
    sep[...,1] *= (1.0 + 0.15*strength)
    sep[...,2] *= (1.0 - 0.10*strength)
    a = a*(1.0 - strength) + sep*strength

    # film grain
    if grain_std > 0:
        noise = np.random.normal(0.0, grain_std, a.shape).astype(np.float32)
        a = a + noise

    # soft vignette
    if vignette > 0:
        h, w = a.shape[:2]
        y, x = np.ogrid[:h, :w]
        cx, cy = (w-1)/2.0, (h-1)/2.0
        rx, ry = w/2.0, h/2.0
        mask = ((x-cx)**2/(rx**2) + (y-cy)**2/(ry**2))
        mask = np.clip(mask, 0, 1)
        vig = 1.0 - vignette*mask.astype(np.float32)
        a = a * vig[...,None]

    a = np.clip(a, 0, 255).astype(np.uint8)
    out = Image.fromarray(a).filter(ImageFilter.GaussianBlur(radius=0.3*strength))
    return out

# ---------------- core ----------------

def synth_one(canvas_W, canvas_H, pool_paths, min_per, max_per,
              max_overlap_iou, scale_min, scale_max, border_px,
              rot_min_deg, rot_max_deg, args):
    # transparent canvas + occupancy mask
    canvas = Image.new("RGBA", (canvas_W, canvas_H), (255, 255, 255, 255))
    occupancy = np.zeros((canvas_H, canvas_W), dtype=bool)
    placed_aabbs = []
    labels = []

    n = random.randint(min_per, max_per)
    tries = 0
    max_tries = n * 80

    while len(placed_aabbs) < n and tries < max_tries:
        tries += 1
        src_path = random.choice(pool_paths)
        img = Image.open(src_path).convert("RGB")

        if border_px > 0:
            img = ImageOps.expand(img, border=border_px, fill=(255, 255, 255))

        # scale
        s = random.uniform(scale_min, scale_max)
        tgt_w = int(canvas_W * s)
        w, h = img.size
        if w == 0 or h == 0 or tgt_w < 1:
            continue
        scale = tgt_w / w
        tw, th = max(1, int(round(w*scale))), max(1, int(round(h*scale)))
        img_resized = img.resize((tw, th), resample=Image.BICUBIC)

        # vintage (optional)
        if random.random() < args.vintage_p:
            strength = random.uniform(args.vintage_min, args.vintage_max)
            img_resized = apply_vintage(img_resized, strength, args.grain, args.fade, args.vignette)

        # convert to RGBA with solid alpha
        inst = img_resized.convert("RGBA")

        # rotate with TRANSPARENT fill (no white corners)
        angle = random.uniform(rot_min_deg, rot_max_deg)
        poly_rot = rotated_rect_polygon(tw, th, angle)            # polygon in rotated-local coords
        img_rot = img_resized.convert("RGBA").rotate(              # keep transparent corners
            angle, expand=True, fillcolor=(0, 0, 0, 0)
        )
        rw, rh = img_rot.size                                      # authoritative size from PIL
        if rw >= canvas_W or rh >= canvas_H:
            continue
        alpha = (np.array(img_rot.split()[-1]) > 0)                # (rh, rw) mask

        # try placements with alpha-overlap check
        placed = False
        for _ in range(40):
            x0 = random.randint(0, canvas_W - rw)
            y0 = random.randint(0, canvas_H - rh)
            x1, y1 = x0 + rw, y0 + rh

            # alpha mask of rotated instance
            alpha = np.array(img_rot.split()[-1]) > 0  # (rh, rw) bool

            # overlap ratio vs existing occupancy
            overlap = (occupancy[y0:y1, x0:x1] & alpha).sum()
            union = alpha.sum() + occupancy[y0:y1, x0:x1].sum() - overlap
            iou = (overlap / union) if union > 0 else 0.0
            if iou > max_overlap_iou or overlap > 0:  # forbid any pixel clash, or relax by iou
                continue

            # paste with alpha (no cutting)
            canvas.alpha_composite(img_rot, (x0, y0))                  # no white cutting
            occupancy[y0:y1, x0:x1] |= alpha
            poly_canvas = [(x + x0, y + y0) for (x, y) in poly_rot]
            labels.append(yolo_poly_line(poly_canvas, canvas_W, canvas_H, cls=0))
            placed_aabbs.append((x0, y0, x1, y1))
            placed = True
            break

        if not placed:
            continue

    # return RGB for saving
    return canvas.convert("RGB"), labels

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
    ap.add_argument("--vintage-p", type=float, default=1.00, help="Chance to tint a photo vintage")
    ap.add_argument("--vintage-min", type=float, default=0.5, help="Min strength")
    ap.add_argument("--vintage-max", type=float, default=1.00, help="Max strength")
    ap.add_argument("--grain", type=float, default=4.0, help="Grain stddev in 0..255 space")
    ap.add_argument("--fade", type=float, default=0.08, help="Fade amount 0..1")
    ap.add_argument("--vignette", type=float, default=0.25, help="Vignette strength 0..1")
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
            args.rot_min, args.rot_max, args
        )

        import uuid 
        stem = uuid.uuid4().hex[:12]
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