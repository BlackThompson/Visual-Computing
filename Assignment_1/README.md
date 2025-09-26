## Visual Computing - Assignment 1: Panorama Stitching (C++)

### Overview
This project implements a classic image stitching pipeline with OpenCV and several self-implemented components:

- Module 1 (Self): Image I/O & Preprocessing — `toGray`, `normalizeImage`, optional `gaussianBlur`
- Module 2 (Lib): Feature detection/description — SIFT, ORB, AKAZE
- Module 3 (Self): Feature matching — Euclidean/Hamming distance, brute-force 1-NN/KNN, Lowe ratio test
- Module 4 (Self): Homography estimation — RANSAC + normalized DLT (SVD)
- Module 5 (Self): Image warping — inverse mapping + bilinear interpolation
- Module 6 (Self): Blending — overlay, robust feathering (distance-transform based)
- Module 7 (Self): Multi-image panorama stitching
- Module 8: Experiments & evaluation — debug visualizations, per-run outputs, timing counters

The binary produces per-run output folders that include the final panorama and diagnostics.

### Dependencies
- C++17 toolchain (g++ or clang++)
- CMake ≥ 3.15
- OpenCV ≥ 4.x
  - Ubuntu/WSL: `sudo apt-get install -y libopencv-dev`

If CMake cannot find OpenCV automatically, pass `-DOpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4` (path may vary by system).

### Build
```bash
cmake -S "/mnt/d/Code/Course/Visual Computing/Assignment_1" \
      -B "/mnt/d/Code/Course/Visual Computing/Assignment_1/build" \
      -DOpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4
cmake --build "/mnt/d/Code/Course/Visual Computing/Assignment_1/build" -j
```

### Run
Basic usage:
```bash
"/mnt/d/Code/Course/Visual Computing/Assignment_1/build/panorama" <img1> <img2> [img3 ...]
```

Optional CLI flags:
- `--det [sift|orb|akaze]` — feature detector/descriptor
- `--blend [overlay|feather]` — blending mode
- `--ratio <0.5-0.95>` — Lowe ratio (KNN=2)
- `--ransac <iters>` — RANSAC iterations
- `--th <px>` — RANSAC reprojection threshold (pixels)
- `--debug` — save keypoints, matches, and inlier matches

Examples:
```bash
# SIFT + Feather, stricter ratio, more RANSAC, slightly looser reprojection threshold
"/mnt/d/Code/Course/Visual Computing/Assignment_1/build/panorama" \
  --det sift --blend feather --ratio 0.8 --ransac 2000 --th 4 \
  "/mnt/d/Code/Course/Visual Computing/Assignment_1/data/1.jpg" \
  "/mnt/d/Code/Course/Visual Computing/Assignment_1/data/2.jpg"

# ORB + Feather, default parameters with three images
"/mnt/d/Code/Course/Visual Computing/Assignment_1/build/panorama" \
  "/mnt/d/Code/Course/Visual Computing/Assignment_1/data/1.jpg" \
  "/mnt/d/Code/Course/Visual Computing/Assignment_1/data/2.jpg" \
  "/mnt/d/Code/Course/Visual Computing/Assignment_1/data/3.jpg"
```

### Outputs
Each run writes to a unique folder:
```
results/run_YYYYmmdd_HHMMSS/
  params.txt        # recorded CLI parameters for the run
  kps_1_a.jpg       # keypoints on current panorama
  kps_1_b.jpg       # keypoints on next input image
  matches_1.jpg     # raw good matches (after ratio test)
  inliers_1.jpg     # RANSAC inlier matches
  panorama.jpg      # final cropped panorama for the run
```
For multi-image input, indices increase per pair added to the panorama (2nd image => "1", 3rd image => "2", etc.).

### Implementation Notes
- Homography uses normalized DLT (Hartley normalization) and RANSAC with unique 4-point sampling.
- Stitching direction auto-selection: compute pano→new and new→pano homographies and pick the direction with more inliers; internally use new→pano for warping.
- Dynamic canvas: transform corners, compute translation to avoid negative coordinates, and allocate sufficient output size.
- Robust feather blending: weights from distance transforms of both masks; blend only in overlap while preserving non-overlap regions.
- Auto-cropping removes outer black borders in the final output.

### Tips & Troubleshooting
- Image order matters. Keep a consistent order (e.g., left→right).
- Low overlap or repeated textures can hurt matching. Try SIFT/AKAZE, increase `--ransac`, relax `--th`, or adjust `--ratio`.
- If OpenCV is not found during configure, set `-DOpenCV_DIR=...` to your OpenCV cmake config path.
- If you see very large black regions, increase RANSAC iterations and ensure inputs have enough overlap and texture.

### Project Structure
```
include/
  preprocess.hpp  features.hpp  matching.hpp  homography.hpp  warp.hpp  blend.hpp  stitch.hpp
src/
  main.cpp  preprocess.cpp  features.cpp  matching.cpp  homography.cpp  warp.cpp  blend.cpp  stitch.cpp
results/              # per-run outputs are created under this directory
data/                 # place your input images here (not tracked by default)
CMakeLists.txt
```

## Dataset

### Inroom

**Indoor Scene (Office)**

- **Lighting:** Mixed natural daylight and warm ceiling light; moderate brightness.
- **Motion Blur:** Slight, from handheld capture.
- **Texture Richness:** Good; mix of smooth (desk, screens) and varied (clothes, curtains, papers).

### Outroom

**Outdoor Scene (Courtyard)**

- **Lighting:** Bright natural daylight, evenly lit with minimal shadows.
- **Motion Blur:** None noticeable; images appear sharp.
- **Texture Richness:** High; variety of textures from brick pavement, wooden bench, glass windows, concrete posts, foliage.

### Laptop

**Lighting:** Low ambient light with strong screen illumination.

**Motion Blur:** Minimal; images mostly sharp.

**Texture Richness:** Moderate; smooth laptop surface contrasted with rough paper note, wooden desk, and other small objects.

