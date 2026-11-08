#!/usr/bin/env python3
"""Generate deterministic binary-pattern examples and extract their skeletons."""

import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import euler_number


ROOT = Path(__file__).resolve().parent
DEST = ROOT / "examples"


def stroke(image, x, y, radius):
    """Rasterize a smooth, round-ended band with possibly varying width."""
    radii = np.broadcast_to(radius, np.shape(x))
    for px, py, r in zip(x, y, radii):
        cv2.circle(image, (int(round(px)), int(round(py))), int(round(r)), 255, -1)


def make_images():
    t = np.linspace(0, 1, 1000)
    image = np.zeros((600, 800), np.uint8)
    for i, (start, end) in enumerate(((-60, 730), (80, 860), (-60, 860))):
        stroke(image, start + (end - start) * t,
               95 + i * 195 + 35 * np.sin(2 * np.pi * t + i), 23 + 4 * i)
    yield "curved_clipped_bands", image

    image = np.zeros((600, 800), np.uint8)
    for i in range(3):
        stroke(image, -70 + 940 * t,
               95 + 200 * i + 24 * np.sin(2 * np.pi * t),
               16 + 34 * (0.5 + 0.5 * np.sin(2 * np.pi * t + i)))
    yield "variable_width_bands", image

    image = np.zeros((600, 800), np.uint8)
    # A real three-way junction, a closed ring, and a bent band.
    for end in ((80, 60), (350, 60), (210, 310)):
        stroke(image, 210 + (end[0] - 210) * t,
               185 + (end[1] - 185) * t, 24)
    angle = np.linspace(0, 2 * np.pi, 1400)
    stroke(image, 610 + 110 * np.cos(angle), 180 + 100 * np.sin(angle), 25)
    stroke(image, -40 + 900 * t, 465 + 45 * np.sin(2 * np.pi * t), 29)
    yield "junction_and_loop", image

    image = np.zeros((600, 800), np.uint8)
    # Slanted crossings and small outline bumps challenge reflection/pruning.
    for i in range(3):
        stroke(image, -80 + 960 * t,
               80 + i * 200 + 50 * t + 24 * np.sin(3 * np.pi * t),
               26 + 3 * np.sin(32 * np.pi * t + i))
    rng = np.random.default_rng(42)
    for x, y in rng.integers((0, 0), (800, 600), size=(50, 2)):
        cv2.circle(image, (int(x), int(y)), 1, 255, -1)
    yield "irregular_edges_and_specks", image


def main():
    DEST.mkdir(exist_ok=True)
    panels = []
    report = []
    for name, image in make_images():
        directory = DEST / name
        directory.mkdir(exist_ok=True)
        source = directory / "input.jpeg"
        if not cv2.imwrite(str(source), image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise OSError(source)
        subprocess.run([sys.executable, str(ROOT / "skeletonize_patterns.py"),
                        str(source), "--output-dir", str(directory)], check=True)
        mask = cv2.imread(str(directory / "mask.png"), 0) > 0
        skeleton = cv2.imread(str(directory / "skeleton.png"), 0) > 0
        components = int(ndi.label(mask, np.ones((3, 3)))[1])
        skeleton_components = int(ndi.label(skeleton, np.ones((3, 3)))[1])
        assert np.all(mask[skeleton]), f"Skeleton outside mask: {name}"
        assert components == skeleton_components, f"Split or lost component: {name}"
        assert components == 3, f"Unexpected pattern count: {name}"
        assert euler_number(mask, connectivity=2) == euler_number(skeleton, connectivity=2), name
        report.append(dict(name=name, foreground_components=components,
                           skeleton_components=skeleton_components,
                           foreground_holes=components - int(euler_number(mask, connectivity=2)),
                           skeleton_holes=skeleton_components - int(euler_number(skeleton, connectivity=2)),
                           skeleton_pixels=int(skeleton.sum())))
        original = cv2.imread(str(source))
        # Thicken only the preview so centerlines survive half-size display.
        overlay = original.copy()
        overlay[ndi.binary_dilation(skeleton, iterations=1)] = (0, 0, 255)
        row = np.hstack([cv2.resize(original, (400, 300)),
                         cv2.resize(overlay, (400, 300), interpolation=cv2.INTER_NEAREST)])
        title = np.full((35, 800, 3), 35, np.uint8)
        cv2.putText(title, name + "   |   input / skeleton", (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(np.vstack([title, row]))
    cv2.imwrite(str(DEST / "comparison.png"), np.vstack(panels))
    (DEST / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
