# Pattern centerlines

Extract the centerlines of the white regions in `data/binary.jpeg`:

```sh
pyenv virtualenv 3.11.9 pattern_analysis
pyenv local pattern_analysis
python -m pip install -r requirements.txt
python skeletonize_patterns.py
```

Results are saved in `output/`: a cleaned binary mask, a one-pixel-wide
`skeleton.png`, a red centerline `overlay.png`, and `skeleton_points.csv`
containing zero-based x/y coordinates and distance to the background in pixels.
The CLI prints the elapsed `extract_skeleton` time in milliseconds and saves it
as `extract_skeleton_seconds` in `timing.json`. It uses `time.perf_counter()`
around the function call, excluding imports, image loading, and output writing.

OpenCV reads the image and applies Otsu thresholding. SciPy removes small
connected components and calculates boundary distances. Scikit-image thins
each retained region. By default the mask is reflected beyond the image edges
before thinning, then cropped back. This avoids artificial forks where a wide
pattern is cut off by the image. Padding is twice the largest foreground radius.
This assumes the pattern continues symmetrically outside the field of view.

Short terminal spurs are then removed iteratively when their path length is
less than the local radius at their junction. Image-edge endpoints and whole
unbranched components are protected. Genuine short branches can also be removed:
lower `--prune-ratio` to retain more, or use `--prune-ratio 0` to disable pruning.
Reported radii use the extended mask, so crop edges are not treated as background.

To reproduce the original closed-boundary behavior:

```sh
python skeletonize_patterns.py --boundary black --prune-ratio 0 --output-dir output_original
```

For centers of maximal inscribed disks, use `--method medial`; this can produce
more branches near irregular boundaries. Use `--min-area` to adjust the minimum
retained component size and `--output-dir` to choose a destination:

```sh
python skeletonize_patterns.py data/binary.jpeg --method medial --output-dir output_medial
```

Generate four reproducible synthetic JPEG examples and extract their skeletons:

```sh
pyenv exec python generate_examples.py
```

`examples/comparison.png` shows inputs beside centerline overlays (lines are
thickened only in this preview). Each example folder contains the input JPEG,
mask, one-pixel skeleton, overlay, and coordinate CSV. Cases cover clipped curved
bands, varying width, a real junction and loop, and irregular outlines with tiny
noise components. `examples/report.json` records component and hole counts;
the generator checks containment, connected-component preservation, and hole
preservation. These are synthetic sanity checks rather than a general accuracy
benchmark.
Each report entry also records `extract_skeleton_seconds` and `script_seconds`.
The latter measures the complete CLI subprocess, including Python startup,
imports, image loading, and writing results; it excludes image generation and
the generator's validation/preview work. Timings are single-run wall-clock
measurements and will vary between runs.
