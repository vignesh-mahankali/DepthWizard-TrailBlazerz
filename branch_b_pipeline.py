"""
Branch B: Georeferenced GeoTIFF to Absolute Metric Digital Surface Model (DSM) Pipeline
========================================================================================
Processes georeferenced satellite GeoTIFF rasters with embedded coordinate metadata,
queries matching genuine SRTM 30m / Copernicus GLO-30 elevation models, fits linear
regression photogrammetric scale calibration, computes authentic validation metrics
(RMSE, MAE, Pearson Correlation R), and exports 32-bit Float GeoTIFF DSMs.
"""

import os
import sys
import argparse
import numpy as np

# Ensure parent directory is in sys.path
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

TASK2_DIR = os.path.join(PARENT_DIR, 'task2')
if TASK2_DIR not in sys.path:
    sys.path.insert(0, TASK2_DIR)

try:
    from geotiff_reader import get_metadata as read_gtif_meta, read_raster_data
except ImportError:
    read_gtif_meta = None

from depth_engine import (
    DepthEngine,
    GeoSpatialManager,
    ScaleCalibrator,
    DSMValidator,
    SRTMDataProvider,
    MeshBuilder
)
from sample_data import ensure_sample_datasets


def run_branch_b(
    input_geotiff_path: str,
    output_dsm_path: str = None,
    encoder: str = 'vits'
):
    print("=" * 70)
    print("   DEPTHWIZARD BRANCH B: GEOREFERENCED METRIC DSM PIPELINE")
    print("=" * 70)

    if not os.path.exists(input_geotiff_path):
        raise FileNotFoundError(f"Input GeoTIFF not found: {input_geotiff_path}")

    # 1. Read Spatial Coordinates and Metadata
    print(f"\n[Step 1/5] Extracting Geospatial Metadata via rasterio...")
    geo_meta = GeoSpatialManager.extract_geometadata(input_geotiff_path)
    print(f" - Input File: {os.path.basename(input_geotiff_path)}")
    print(f" - CRS: {geo_meta.get('crs')}")
    print(f" - Bounds (W, S, E, N): {geo_meta.get('bounds')}")
    print(f" - Dimensions: {geo_meta.get('width')} x {geo_meta.get('height')} ({geo_meta.get('bands')} bands)")
    print(f" - Ground Resolution: {geo_meta.get('resolution_m')} m/pixel")
    print(f" - Georeferenced Status: {'YES (Valid WGS-84 / Projected CRS)' if geo_meta.get('has_georeference') else 'NO'}")

    # Cross-verify with task2/geotiff_reader if available
    if read_gtif_meta is not None:
        raw_meta = read_gtif_meta(input_geotiff_path)
        print(f" - task2/geotiff_reader verification: CRS={raw_meta.get('crs')}, Bounds={raw_meta.get('bounds')}")

    # 2. Query / Retrieve Matching Genuine Reference Topography (SRTM / Copernicus 30m)
    print(f"\n[Step 2/5] Retrieving Genuine Topographic Ground Truth...")
    bounds = geo_meta.get("bounds")
    filename = os.path.basename(input_geotiff_path)
    ref_dem, prov = ScaleCalibrator.get_reference_dem(
        shape=(geo_meta.get("height", 512), geo_meta.get("width", 512)),
        bounds=bounds,
        sample_key=filename
    )

    if ref_dem is not None:
        print(f" - DEM Provider Tier: {prov.source_tier}")
        print(f" - Dataset Name: {prov.dataset_name}")
        print(f" - Attribution: {prov.attribution}")
        print(f" - Reference Elevation Range: {ref_dem.min():.1f}m to {ref_dem.max():.1f}m (mean {ref_dem.mean():.1f}m)")
    else:
        print(" - Warning: No reference DEM available. Will fall back to prior calibration.")

    # 3. Monocular Depth Estimation with Depth-Anything-V2
    print(f"\n[Step 3/5] Inferring Monocular Depth Map (Depth-Anything-V2 '{encoder}')...")
    image_bgr = GeoSpatialManager.read_geotiff_image(input_geotiff_path)
    engine = DepthEngine(encoder=encoder)
    rel_depth = engine.predict_relative_depth(image_bgr)
    print(f" - Relative Depth Inferred. Shape: {rel_depth.shape}, Range: [{rel_depth.min():.3f}, {rel_depth.max():.3f}]")

    # 4. Photogrammetric Scale Calibration
    print(f"\n[Step 4/5] Calibrating Scale to Absolute Metric Elevation (Z in meters)...")
    abs_dsm, scale_info = ScaleCalibrator.calibrate_relative_to_absolute(
        rel_depth,
        reference_dem=ref_dem,
        provenance=prov,
        base_elev=530.0,
        height_range=1200.0
    )
    print(f" - Method: {scale_info.get('method')}")
    print(f" - Scale Factor (s): {scale_info.get('scale')} m")
    print(f" - Offset (b): {scale_info.get('offset')} m")
    print(f" - Disparity Inversion Rectified: {scale_info.get('disparity_inverted')}")
    print(f" - Calibrated Absolute Elevation: Min={abs_dsm.min():.1f}m, Max={abs_dsm.max():.1f}m, Mean={abs_dsm.mean():.1f}m")

    # 5. Photogrammetric Accuracy Validation
    val_report = None
    if ref_dem is not None:
        print(f"\n[Step 5/5] Quantitative Photogrammetric Validation against Reference DEM...")
        val_report = DSMValidator.validate(
            estimated_dsm=abs_dsm,
            reference_dsm=ref_dem,
            pixel_res_m=geo_meta.get("resolution_m"),
            provenance=prov
        )
        print(f" ---------------------------------------------------")
        print(f"  Root Mean Square Error (RMSE) : {val_report['rmse']} m")
        print(f"  Mean Absolute Error (MAE)     : {val_report['mae']} m")
        print(f"  Pearson Correlation (R)       : {val_report['correlation']}")
        print(f"  Max Absolute Error            : {val_report['max_error']} m")
        print(f" ---------------------------------------------------")

    # 6. Save Calibrated GeoTIFF DSM
    if output_dsm_path is None:
        exports_dir = os.path.join(PARENT_DIR, 'exports')
        os.makedirs(exports_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(input_geotiff_path))[0]
        output_dsm_path = os.path.join(exports_dir, f"branch_b_dsm_{base_name}.tif")

    print(f"\nSaving Output Absolute DSM GeoTIFF to: {output_dsm_path}...")
    GeoSpatialManager.save_dsm_geotiff(output_dsm_path, abs_dsm, geo_meta)

    # Verification of written file
    if read_gtif_meta is not None:
        saved_meta = read_gtif_meta(output_dsm_path)
        print(f" - Verification of exported file: CRS={saved_meta.get('crs')}, Bounds={saved_meta.get('bounds')}")

    print("\n[SUCCESS] Branch B execution complete!")
    return {
        "output_dsm_path": output_dsm_path,
        "geo_metadata": geo_meta,
        "scale_info": scale_info,
        "validation": val_report
    }


def main():
    parser = argparse.ArgumentParser(description="DepthWizard Branch B: Georeferenced GeoTIFF -> Metric DSM")
    parser.add_argument("--input", "-i", type=str, help="Path to input georeferenced GeoTIFF (.tif/.tiff)")
    parser.add_argument("--sample", "-s", type=str, default="wayanad", choices=["wayanad", "wayanad_post", "kolkata"], help="Use built-in georeferenced sample")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output path for calibrated GeoTIFF DSM")
    parser.add_argument("--encoder", "-e", type=str, default="vits", choices=["vits", "vitb", "vitl"], help="Depth Anything encoder")

    args = parser.parse_args()

    # Ensure samples are created
    samples = ensure_sample_datasets()

    if args.input:
        input_path = args.input
    else:
        sample_map = {
            "wayanad": "wayanad_real_optical.tif",
            "wayanad_post": "wayanad_post_real_optical.tif",
            "kolkata": "kolkata_real_optical.tif"
        }
        target_name = sample_map.get(args.sample, "wayanad_real_optical.tif")
        input_path = samples.get(target_name)
        if not input_path or not os.path.exists(input_path):
            sample_dir = os.path.join(PARENT_DIR, 'samples')
            input_path = os.path.join(sample_dir, target_name)

    run_branch_b(input_path, args.output, args.encoder)


if __name__ == "__main__":
    main()
