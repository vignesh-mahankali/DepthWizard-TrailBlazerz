# DepthWizard

Software pipeline that reconstructs elevation/height data from a single 2D optical image (satellite/aerial/drone photo) and renders it as an interactive 3D terrain flythrough in the browser.

**SIH Problem Statement:** SIH26175, posted by ISRO, Disaster Management theme, Software category.

## The Problem

Getting real elevation data (building heights, terrain slope, hills/valleys) normally requires LiDAR scanners, stereo camera pairs, or satellite radar (InSAR) — all expensive and slow. This project extracts the same kind of information from a single ordinary photograph instead.

## Two Branches (by input type)

| Input | Branch | Output | Status |
|---|---|---|---|
| Plain PNG/JPG (no location metadata) | **Branch A** | Relative DSM — shows what's taller/shorter than what, NOT in real-world units | ✅ **Working end-to-end** |
| GeoTIFF (has embedded coordinates) | **Branch B** | Absolute DSM — real metric elevation (e.g. "42m above sea level") | ❌ **Not yet built** |

## Current Pipeline (Branch A) — how it works

1. **Depth estimation** — Depth Anything V2 (pre-trained monocular depth model) takes a single RGB image and predicts per-pixel relative depth/height. Output: a grayscale depth map, arbitrary units (not meters).
2. **3D reconstruction** (`make_mesh.py`) — converts the RGB image + depth map into a colored point cloud, estimates normals, then runs Poisson surface reconstruction (Open3D) to build a continuous 3D mesh. Low-density/noisy vertices (bottom 5% by density) are trimmed — this can leave small missing chunks on noisier source images, which is expected and tunable via the `np.quantile(densities, 0.05)` threshold.
3. **Format conversion** (`convertN.py`) — converts the `.obj` mesh to `.glb` using `trimesh`, for web rendering.
4. **Rendering** (`viewer.html`) — Three.js (WebGL) loads the `.glb`, applies vertex-color materials, and runs an automatic circular flythrough camera path. Manual mouse drag (OrbitControls) pauses the flythrough; a "Restart Flythrough" button resets it. A dropdown lets you switch between pre-processed scenes (`terrain.glb`, `terrain2.glb`, `terrain3.glb`, ...).

**Known limitation:** the depth-to-height scale factor in `make_mesh.py` (`z = float(depth[v, u]) / 255.0 * 50.0`) is an arbitrary placeholder, not calibrated against any real-world reference. This means Branch A's *shape* (relative height ranking) is meaningful, but the actual height numbers are not real metric values. No RMSE/MAE/correlation validation exists yet.

## What's NOT built yet (Branch B + validation)

- **GeoTIFF handling** — reading embedded coordinate/projection metadata via GDAL/Rasterio, to detect and route georeferenced inputs.
- **Scale calibration** — cross-referencing the relative depth map against SRTM 30m Global DEM (or manual Ground Control Points) at known-elevation points, then fitting a linear/regression conversion to produce real metric Absolute DSM values.
- **Validation module** — comparing output DSM against reference elevation data (LiDAR/DEM ground truth) using RMSE, MAE, and correlation (R), across terrain types (urban, hilly, forested, sparse). This is 50% of ISRO's grading criteria and is currently unimplemented.
- **GAMUS dataset** (https://github.com/EarthNets/RSI-MMSegmentation) — NOT a depth-estimation model. It's a semantic segmentation benchmark (RGB + nDSM → land cover class: ground/vegetation/building/water/road/tree). Its likely relevance to this project: GAMUS tiles come with real LiDAR-derived nDSM ground truth paired with RGB, at 0.33m resolution — could be used as a validation dataset (run our pipeline on GAMUS RGB tiles, compare against their nDSM) rather than as a pipeline component itself.
- **Web upload UI** — currently the pipeline is run manually per-scene via command line (see below). A Flask backend + upload form (to fully automate image → 3D result with zero command-line use) has not been built yet.

## Repo Structure

```
depthwizard/
├── make_mesh.py          # point cloud + mesh generation (currently hardcoded per-scene, edit paths manually)
├── convert2.py            # .obj -> .glb conversion for scene 2 (convert3.py, etc. per scene)
├── convert_to_glb.py      # original/first conversion script
├── viewer.html            # Three.js viewer with flythrough + scene dropdown
├── requirements.txt       # Python dependencies (add if missing)
├── .gitignore
└── Depth-Anything-V2/     # NOT tracked in this repo — clone separately, see Setup below
```

Generated files (NOT tracked in git — regenerate locally or share via Drive/USB):
`output*/`, `*.ply`, `*.obj`, `*.glb`, `test*.jpg`

## Setup

```bash
git clone https://github.com/vignesh-mahankali/DepthWizard-TrailBlazerz.git
cd DepthWizard-TrailBlazerz

# Clone Depth Anything V2 separately (not included in this repo)
git clone https://github.com/DepthAnything/Depth-Anything-V2.git

# Create and activate venv
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt --break-system-packages
```

## Running the current pipeline (manual, per scene)

```bash
cd Depth-Anything-V2
python run.py --encoder vits --img-path testN.jpg --outdir outputN --pred-only --grayscale
cd ..

# Edit make_mesh.py: update image path, depth map path, and output filenames to match scene N
python make_mesh.py

# Edit/create convertN.py to point at output_meshN.obj -> terrainN.glb
python convertN.py
```

Then open `viewer.html` (via local server, e.g. `python -m http.server 8000`) and add `<option value="terrainN.glb">Scene N</option>` to the dropdown in the HTML.

⚠️ **Common bug to watch for:** in `make_mesh.py`, make sure there is only ONE `o3d.io.write_triangle_mesh(...)` call, placed AFTER the `mesh` variable is created by Poisson reconstruction (near the bottom of the script). Placing it earlier, or leaving a duplicate call with the old filename, will either crash with `NameError: name 'mesh' is not defined` or silently overwrite a previous scene's file.

## Task Split (in progress)

- **Branch A web UI** — Flask server wrapping the existing pipeline into callable functions + upload form in `viewer.html`, removing manual command-line steps.
- **Branch B: GDAL/Rasterio** — read GeoTIFF coordinate/projection metadata, standalone script, no dependency on existing pipeline yet.
- **Branch B: SRTM calibration** — download SRTM 30m tiles, match points against relative depth map, fit regression to produce absolute elevation.
- **GAMUS validation** (lower priority) — run pipeline on GAMUS RGB tiles, compute RMSE/MAE/correlation against GAMUS nDSM ground truth.

Each task should be built on its own branch (`branch-b-gdal`, `branch-b-srtm`, `flask-ui`, `gamus-validation`) and merged into `main` once working standalone, to avoid merge conflicts.

## Tech Stack Rationale

| Component | Choice | Why |
|---|---|---|
| Depth estimation | Depth Anything V2 (MiDaS/DPT as backup) | Strongest open-source pre-trained monocular depth model; no training data or GPU cluster needed |
| Geospatial metadata | GDAL, Rasterio | Standard Python libraries for reading GeoTIFF coordinate systems — required for Branch B |
| Elevation reference | SRTM 30m Global DEM | Free, global, zero-permission elevation ground truth for scale calibration |
| Mesh generation | Open3D, Poisson surface reconstruction | Converts height grid + coordinates into a continuous navigable 3D surface |
| Rendering | Three.js (WebGL) | Browser-based, zero install for judges, real-time flythrough rendering |
| Validation metrics | RMSE, MAE, Correlation (R) | Exact metrics specified in ISRO's evaluation criteria |

**Honest accuracy statement (current state):** Branch A's relative height ranking is qualitatively correct (taller structures render taller), but there is no calibration or quantitative validation yet — do not claim metric accuracy numbers until Branch B + validation are implemented. When they are, the ceiling is bounded by SRTM's 30m resolution and the domain gap between the depth model's training data (ground-level photos) and satellite/aerial imagery — mitigated via calibration and cross-terrain validation, not eliminated.
