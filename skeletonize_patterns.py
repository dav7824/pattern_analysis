#!/usr/bin/env python3
"""Extract one-pixel-wide centerlines from white patterns on a dark background."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import medial_axis, skeletonize


def prune_spurs(skeleton, distances, ratio=1.0):
    """Remove short endpoint-to-junction paths, preserving border endpoints.

    The cutoff is ratio * the radius at the junction, measured along the path.
    Endpoint-to-endpoint paths (whole unbranched patterns) are never removed.
    """
    result = skeleton.copy()
    if ratio <= 0:
        return result
    height, width = result.shape
    offsets = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
               if dy or dx]
    while True:
        degree = ndi.convolve(result.astype(np.uint8), np.ones((3, 3), dtype=np.uint8),
                              mode="constant") - result
        remove = set()
        for y, x in np.argwhere(result & (degree == 1)):
            if y in (0, height - 1) or x in (0, width - 1):
                continue
            path = [(y, x)]
            previous = None
            current = (y, x)
            length = 0.0
            while True:
                cy, cx = current
                neighbors = [(cy + dy, cx + dx) for dy, dx in offsets
                             if 0 <= cy + dy < height and 0 <= cx + dx < width
                             and result[cy + dy, cx + dx]
                             and (cy + dy, cx + dx) != previous]
                if len(neighbors) != 1:
                    break
                following = neighbors[0]
                length += np.hypot(following[0] - cy, following[1] - cx)
                previous, current = current, following
                if degree[current] != 2:
                    break
                path.append(current)
            if degree[current] >= 3 and length < ratio * distances[current]:
                remove.update(path)
        if not remove:
            return result
        yy, xx = zip(*remove)
        result[yy, xx] = False


def extract_skeleton(image, min_area=64, method="thinning", boundary="reflect",
                     prune_ratio=1.0, opening_radius=2):
    """Return cleaned foreground, skeleton, and boundary distance in pixels.

    Thinning preserves topology while reducing regions to centerlines. The
    medial axis follows centers of maximal inscribed disks, but can create
    extra branches in response to small boundary irregularities.
    A small opening removes attached specks before they generate long spurs.
    Set opening_radius=0 to preserve the original foreground outline.
    """
    if not isinstance(opening_radius, (int, np.integer)) or opening_radius < 0:
        raise ValueError("opening_radius must be a nonnegative integer")
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if opening_radius:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * opening_radius + 1, 2 * opening_radius + 1))
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, kernel,
            borderType=cv2.BORDER_REFLECT_101 if boundary == "reflect" else cv2.BORDER_CONSTANT,
            borderValue=0)
    labels, count = ndi.label(binary > 0, structure=np.ones((3, 3)))
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    keep = sizes >= min_area
    keep[0] = False
    foreground = keep[labels]

    # Extend far enough that the artificial outer edge is beyond the local
    # pattern width. Reflection treats clipped patterns as continuing outside
    # the field of view, instead of creating artificial corners at the crop.
    radius = ndi.distance_transform_edt(np.pad(foreground, 1)).max()
    pad = max(2, int(np.ceil(2 * radius))) if boundary == "reflect" else 1
    padded = (np.pad(foreground, pad, mode="reflect") if boundary == "reflect"
              else np.pad(foreground, pad, constant_values=False))
    padded = np.pad(padded, 1, constant_values=False)
    crop = (slice(pad + 1, -pad - 1), slice(pad + 1, -pad - 1))
    distances = ndi.distance_transform_edt(padded)[crop]
    if method == "medial":
        centerlines = medial_axis(padded, rng=0)[crop]
    else:
        centerlines = skeletonize(padded)[crop]
    centerlines = prune_spurs(centerlines, distances, prune_ratio)
    return foreground, centerlines, distances


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=Path("data/binary.jpeg"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--min-area", type=int, default=64,
                        help="Remove white components smaller than this many pixels (default: 64)")
    parser.add_argument("--method", choices=("thinning", "medial"), default="thinning")
    parser.add_argument("--boundary", choices=("reflect", "black"), default="reflect",
                        help="How patterns continue outside the image (default: reflect)")
    parser.add_argument("--prune-ratio", type=float, default=1.0,
                        help="Prune terminal branches shorter than this times local radius; 0 disables")
    parser.add_argument("--opening-radius", type=int, default=2,
                        help="Radius in pixels for removing small attached specks before thinning; 0 disables")
    args = parser.parse_args()
    if args.min_area < 1:
        parser.error("--min-area must be positive")
    if not np.isfinite(args.prune_ratio) or args.prune_ratio < 0:
        parser.error("--prune-ratio must be finite and nonnegative")
    if args.opening_radius < 0:
        parser.error("--opening-radius must be nonnegative")
    gray = cv2.imread(str(args.input), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        parser.error(f"Cannot read image: {args.input}")
    started = perf_counter()
    foreground, skeleton, distances = extract_skeleton(
        gray, args.min_area, args.method, args.boundary, args.prune_ratio, args.opening_radius)
    extraction_seconds = perf_counter() - started
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "timing.json").write_text(json.dumps({
        "input": str(args.input),
        "method": args.method,
        "boundary": args.boundary,
        "prune_ratio": args.prune_ratio,
        "opening_radius": args.opening_radius,
        "extract_skeleton_seconds": extraction_seconds,
    }, indent=2) + "\n")
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay[skeleton] = (0, 0, 255)
    for name, image in (
        ("mask.png", foreground.astype(np.uint8) * 255),
        ("skeleton.png", skeleton.astype(np.uint8) * 255),
        ("overlay.png", overlay),
    ):
        path = args.output_dir / name
        if not cv2.imwrite(str(path), image):
            raise OSError(f"Could not write {path}")
    # Row/column coordinates and local radius are useful for further analysis.
    rows, cols = np.nonzero(skeleton)
    np.savetxt(args.output_dir / "skeleton_points.csv",
               np.column_stack((cols, rows, distances[skeleton])),
               delimiter=",", header="x,y,radius_px", comments="", fmt=("%d", "%d", "%.4f"))
    _, count = ndi.label(foreground, structure=np.ones((3, 3)))
    print(f"Found {count} white patterns; {skeleton.sum()} skeleton pixels.")
    print(f"extract_skeleton: {extraction_seconds * 1000:.3f} ms")
    print(f"Saved mask, skeleton, red overlay, and coordinates to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
