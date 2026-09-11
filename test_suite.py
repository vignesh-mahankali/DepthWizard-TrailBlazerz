"""
Comprehensive Scientific Test Suite for DepthWizard
Verifies:
1. Cryptographic DEM provenance tracking (no silent fallbacks or unverified arrays)
2. Georeference extraction & prevention of location hallucination
3. Monocular disparity inversion detection and mathematical rectification
4. Dynamic Ground Sampling Distance (GSD) & Minimum Detectable Change (MDC)
5. End-to-end FastAPI endpoint integration (/api/status, /api/process, /api/process_disaster, /api/validate)
"""

import os
import sys
import math
import hashlib
import numpy as np
from fastapi.testclient import TestClient

# Ensure root directory is on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from depth_engine import (
    DepthEngine, 
    SRTMDataProvider, 
    GeoSpatialManager, 
    ScaleCalibrator, 
    DisasterChangeDetector, 
    DSMValidator,
    DEMProvenance
)
from server import app

client = TestClient(app)


def test_provenance_tracking():
    """Verify cryptographic SHA-256 verification and metadata provenance tracking."""
    # 1. Wayanad pre-disaster verified benchmark
    dem, prov = SRTMDataProvider.get_reference_dem(shape=(512, 512), sample_key="wayanad_pre_disaster.jpg")
    assert dem is not None
    assert isinstance(prov, DEMProvenance)
    assert prov.source_tier == "VERIFIED_BENCHMARK"
    assert prov.is_synthetic is False
    assert prov.verification_hash == "0bf0c5432c2398f10e5a17b5a7ac7a322c1c8629283208c23b65e3165d28253b"
    assert "NASA SRTM" in prov.dataset_name
    assert 8.0 < prov.pixel_resolution_m < 9.0  # ~8.58m GSD

    # 2. Unknown un-georeferenced sample must NOT silently fallback to fake ramps or Wayanad
    dem_none, prov_none = SRTMDataProvider.get_reference_dem(shape=(512, 512), sample_key="unknown_photo.jpg")
    assert dem_none is None
    assert prov_none.source_tier == "NONE"
    assert prov_none.bounds is None


def test_georeference_and_bounds_extraction():
    """Verify non-GeoTIFFs do not hallucinate coordinates, and calculate_gsd is mathematically exact."""
    # 1. Plain image should not claim georeferencing
    meta = GeoSpatialManager.extract_geometadata("non_existent_image.jpg")
    assert meta["has_georeference"] is False
    assert meta["bounds"] is None

    # 2. GSD Calculation verification
    # Bounds: [76.11, 11.43, 76.15, 11.47] -> 0.04 deg span in lat and lon
    bounds = [76.11, 11.43, 76.15, 11.47]
    gsd_512 = SRTMDataProvider.calculate_gsd(bounds, 512, 512)
    assert 8.5 < gsd_512 < 8.7
    # If pixel dimensions double to 1024x1024, GSD must halve
    gsd_1024 = SRTMDataProvider.calculate_gsd(bounds, 1024, 1024)
    assert math.isclose(gsd_1024, gsd_512 / 2.0, rel_tol=1e-2)


def test_disparity_inversion_rectification():
    """Verify negative regression slopes (monocular disparity inversion) are detected and rectified."""
    # Synthetic reference DEM: terrain elevation ascends from 400m to 900m
    h, w = 256, 256
    ref_dem = np.linspace(400.0, 900.0, h)[:, None].repeat(w, axis=1).astype(np.float32)

    # Inverted monocular depth: sensor predicts 1.0 for distant high mountains and 0.0 for low valley
    # This causes a raw least-squares slope s < 0
    rel_inverted = np.linspace(1.0, 0.0, h)[:, None].repeat(w, axis=1).astype(np.float32)

    prov = DEMProvenance(
        source_tier="VERIFIED_BENCHMARK",
        dataset_name="Synthetic Test DEM",
        bounds=[76.11, 11.43, 76.15, 11.47],
        pixel_resolution_m=8.58,
        is_synthetic=False,
        verification_hash="dummy",
        attribution="Unit Test"
    )

    calibrated_dsm, scale_info = ScaleCalibrator.calibrate_relative_to_absolute(
        rel_inverted, reference_dem=ref_dem, provenance=prov
    )

    assert scale_info["disparity_inverted"] is True
    assert scale_info["scale"] > 0
    # The resulting DSM must positively correlate with the reference DEM (> 0.99)
    corr = np.corrcoef(calibrated_dsm.flatten(), ref_dem.flatten())[0, 1]
    assert corr > 0.999
    # RMSE should be near 0
    rmse = np.sqrt(np.mean((calibrated_dsm - ref_dem)**2))
    assert rmse < 1.0


def test_disaster_change_detection_geometrics():
    """Verify volumetric and areal geomorphic change metrics with dynamic GSD and MDC threshold."""
    gsd = 8.58  # meters per pixel
    pre_dsm = np.ones((100, 100), dtype=np.float32) * 500.0
    post_dsm = pre_dsm.copy()

    # Create a 20x20 pixel landslide pit (depth loss of 10 meters)
    post_dsm[40:60, 40:60] -= 10.0

    res = DisasterChangeDetector.analyze_change(pre_dsm, post_dsm, pixel_res_m=gsd)
    metrics = res["metrics"]

    expected_pixels = 20 * 20  # 400 pixels
    expected_area_m2 = expected_pixels * (gsd * gsd)
    expected_volume_m3 = expected_area_m2 * 10.0

    assert math.isclose(metrics["area_loss_m2"], expected_area_m2, rel_tol=1e-3)
    assert math.isclose(metrics["volume_loss_m3"], expected_volume_m3, rel_tol=1e-3)
    assert metrics["max_elevation_loss_m"] == 10.0
    assert metrics["net_volume_change_m3"] == -metrics["volume_loss_m3"]
    assert "MDC" in metrics["threshold_method"]


def test_dsm_validation_metrics():
    """Verify RMSE, MAE, Pearson r, and Euclidean transect profile calculations."""
    h, w = 128, 128
    ref_dsm = np.random.uniform(500, 800, (h, w)).astype(np.float32)
    # Estimated DSM is reference with added Gaussian noise (sigma = 5m)
    est_dsm = ref_dsm + np.random.normal(0, 5.0, (h, w)).astype(np.float32)

    val = DSMValidator.validate(est_dsm, ref_dsm, pixel_res_m=8.58)

    assert 3.5 < val["rmse"] < 6.5
    assert 2.5 < val["mae"] < 5.5
    assert val["correlation"] > 0.98
    assert val["transect_total_length_m"] > 1000.0
    assert len(val["transect"]) == 50
    assert len(val["scatter"]) <= 200


def test_api_status_endpoint():
    """Verify /api/status returns healthy system status and verified benchmark registry."""
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["model_loaded"] is True
    assert "wayanad_pre" in data["verified_benchmarks"]


def test_api_process_single_image():
    """Verify /api/process on benchmark dataset returns verified provenance and proper 3D grids."""
    response = client.post("/api/process", data={"sample_key": "wayanad_pre_disaster.jpg"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["dem_provenance"]["source_tier"] == "VERIFIED_BENCHMARK"
    assert data["stats"]["pixel_resolution_m"] > 8.0
    assert "elevation_grid" in data
    assert len(data["elevation_grid"]) == 128


def test_api_validate_endpoint():
    """Verify /api/validate executes monocular estimation and benchmarks against genuine ground truth."""
    response = client.post("/api/validate", data={"sample_key": "wayanad_pre_disaster.jpg"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "validation" in data
    val = data["validation"]
    assert "rmse" in val
    assert "mae" in val
    assert "correlation" in val
    assert val["correlation"] > 0.6  # Genuine model correlation on complex terrain
    assert data["dem_provenance"]["source_tier"] == "VERIFIED_BENCHMARK"


def test_api_process_disaster_endpoint():
    """Verify /api/process_disaster computes geomorphic displacement on pre/post disaster pair."""
    response = client.post("/api/process_disaster", data={"sample_disaster_key": "wayanad"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    metrics = data["metrics"]
    assert metrics["area_loss_m2"] > 0
    assert metrics["volume_loss_m3"] > 0
    assert data["dem_provenance"]["source_tier"] == "VERIFIED_BENCHMARK"


if __name__ == "__main__":
    tests = [name for name in globals() if name.startswith("test_") and callable(globals()[name])]
    passed = 0
    print(f"Discovered {len(tests)} scientific validation tests.\n")
    for t_name in tests:
        print(f"Running {t_name:40s} ... ", end="", flush=True)
        try:
            globals()[t_name]()
            print("[PASS]")
            passed += 1
        except Exception as e:
            print("[FAIL]")
            import traceback
            traceback.print_exc()
    print(f"\nTest Suite Results: {passed}/{len(tests)} passed.")
    if passed != len(tests):
        sys.exit(1)
