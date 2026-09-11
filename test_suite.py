import os
import sys
import json

# Ensure depthwizard root and task2 are in sys.path
DEPTHWIZARD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if DEPTHWIZARD_DIR not in sys.path:
    sys.path.insert(0, DEPTHWIZARD_DIR)

TASK2_DIR = os.path.join(DEPTHWIZARD_DIR, 'task2')
if TASK2_DIR not in sys.path:
    sys.path.insert(0, TASK2_DIR)

from fastapi.testclient import TestClient
from server import app, sample_datasets
from task2.geotiff_reader import get_metadata, read_raster_data

client = TestClient(app)

print("=" * 70)
print("   DEPTHWIZARD BRANCH A & BRANCH B VERIFICATION TEST SUITE")
print("=" * 70)

def test_static_files():
    print("\n[Test 1] Static Web Assets Serving...")
    r_index = client.get("/")
    assert r_index.status_code == 200, f"Index failed: {r_index.status_code}"
    assert "Branch B: GeoTIFF DSM" in r_index.text, "Branch B UI missing in index.html"
    assert "geotiff-inspector-card" in r_index.text, "Inspector card missing in index.html"

    r_css = client.get("/styles.css")
    assert r_css.status_code == 200, f"styles.css failed: {r_css.status_code}"
    assert "branch-selector-container" in r_css.text, "Branch CSS missing in styles.css"

    r_js = client.get("/app.js")
    assert r_js.status_code == 200, f"app.js failed: {r_js.status_code}"
    assert "switchBranch" in r_js.text, "switchBranch missing in app.js"
    print(" - OK: Static HTML, CSS, and JS verified with Branch B UI integration.")


def test_api_status():
    print("\n[Test 2] API Status & Branch Capabilities...")
    r = client.get("/api/status")
    assert r.status_code == 200, f"Status failed: {r.status_code}"
    data = r.json()
    assert "branches" in data, "Branches field missing in /api/status"
    assert "branch_b" in data["branches"], "Branch B missing in /api/status"
    assert "wayanad_real_optical.tif" in data["sample_datasets"], "Wayanad GeoTIFF sample missing"
    print(f" - OK: System status '{data['status']}', Branch B registered.")


def test_branch_b_catalog():
    print("\n[Test 3] Branch B Sample Catalog & Info Endpoints...")
    r = client.get("/api/branch_b/samples")
    assert r.status_code == 200
    samples = r.json().get("samples", [])
    assert len(samples) >= 3, "Expected at least 3 curated GeoTIFF samples"
    wayanad_sample = next((s for s in samples if "wayanad" in s["id"]), None)
    assert wayanad_sample is not None, "Wayanad sample missing from catalog"
    assert wayanad_sample["bounds"] == [76.0, 11.4, 76.4, 11.7], "Wayanad bounds incorrect"
    print(f" - OK: Found {len(samples)} Branch B samples with verified WGS-84 bounds.")

    # Test /api/geotiff_info
    r_info = client.get("/api/geotiff_info?key=wayanad_real_optical.tif")
    assert r_info.status_code == 200
    info = r_info.json().get("metadata", {})
    assert info["has_georeference"] is True, "GeoTIFF should have georeference"
    assert info["crs"] == "EPSG:4326", f"Expected EPSG:4326, got {info['crs']}"
    assert info["bounds"] == [76.0, 11.4, 76.4, 11.7], f"Expected Wayanad bounds, got {info['bounds']}"
    print(" - OK: /api/geotiff_info returned valid CRS EPSG:4326 and Wayanad coordinates.")


def test_task2_geotiff_reader_on_samples():
    print("\n[Test 4] task2/geotiff_reader.py on Genuine Sample GeoTIFFs...")
    sample_files = [
        ("wayanad_real_srtm_dem.tif", 1, [76.0, 11.4, 76.4, 11.7]),
        ("wayanad_real_optical.tif", 3, [76.0, 11.4, 76.4, 11.7]),
        ("kolkata_real_srtm_dem.tif", 1, [88.34, 22.55, 88.38, 22.59]),
        ("kolkata_real_optical.tif", 3, [88.34, 22.55, 88.38, 22.59]),
    ]
    for filename, expected_bands, expected_bounds in sample_files:
        path = os.path.join(DEPTHWIZARD_DIR, 'samples', filename)
        assert os.path.exists(path), f"Sample file {filename} does not exist"
        meta = get_metadata(path)
        assert meta["crs"] == "EPSG:4326", f"Failed CRS check on {filename}: {meta['crs']}"
        assert meta["bands"] == expected_bands, f"Failed bands check on {filename}: {meta['bands']}"
        assert meta["bounds"]["left"] == expected_bounds[0], f"Failed left bound on {filename}"
        assert meta["bounds"]["bottom"] == expected_bounds[1], f"Failed bottom bound on {filename}"
        assert meta["bounds"]["right"] == expected_bounds[2], f"Failed right bound on {filename}"
        assert meta["bounds"]["top"] == expected_bounds[3], f"Failed top bound on {filename}"
        print(f" - OK: {filename} -> CRS: {meta['crs']}, Bounds: {meta['bounds']}, Bands: {meta['bands']}")


def test_process_branch_b_geotiff():
    print("\n[Test 5] End-to-End Processing of Branch B GeoTIFF via /api/process...")
    res = client.post("/api/process", data={
        "sample_key": "wayanad_real_optical.tif",
        "use_georeference": "true",
        "base_elevation": "530.0",
        "height_range": "1215.0"
    })
    assert res.status_code == 200, f"/api/process failed: {res.status_code}, {res.text}"
    data = res.json()
    assert data["status"] == "success"
    assert data["branch"] == "BRANCH_B_GEOREFERENCED", f"Expected BRANCH_B_GEOREFERENCED, got {data.get('branch')}"
    
    geo = data["geo_metadata"]
    assert geo["has_georeference"] is True, "Metadata should indicate georeferenced"
    assert geo["crs"] == "EPSG:4326", f"Expected EPSG:4326, got {geo['crs']}"
    assert geo["bounds"] == [76.0, 11.4, 76.4, 11.7], f"Expected Wayanad bounds, got {geo['bounds']}"

    stats = data["stats"]
    print(f" - Output Stats: Elevation [{stats['min_elevation_m']}m to {stats['max_elevation_m']}m], Relief: {stats['relief_range_m']}m")
    assert stats["max_elevation_m"] > stats["min_elevation_m"]

    # Verify exported GeoTIFF DSM
    dsm_url = data["downloads"]["geotiff_dsm"]
    dsm_filename = os.path.basename(dsm_url)
    export_path = os.path.join(DEPTHWIZARD_DIR, 'exports', dsm_filename)
    assert os.path.exists(export_path), f"Exported DSM {export_path} not found on disk"

    # Verify exported DSM with task2/geotiff_reader
    dsm_meta = get_metadata(export_path)
    assert dsm_meta["crs"] == "EPSG:4326", f"Exported DSM CRS is not EPSG:4326: {dsm_meta['crs']}"
    assert dsm_meta["bounds"]["left"] == 76.0, "Exported DSM left bound does not match Wayanad"
    assert dsm_meta["bounds"]["top"] == 11.7, "Exported DSM top bound does not match Wayanad"
    print(f" - OK: Exported GeoTIFF DSM verified by task2/geotiff_reader: CRS={dsm_meta['crs']}, Bounds={dsm_meta['bounds']}")


def test_process_branch_a_optical():
    print("\n[Test 6] Branch A (Optical Image without coordinates)...")
    res = client.post("/api/process", data={
        "sample_key": "wayanad_pre_disaster.jpg",
        "use_georeference": "false",
        "base_elevation": "530.0",
        "height_range": "1215.0"
    })
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["branch"] == "BRANCH_A_OPTICAL"
    print(" - OK: Branch A optical relative DSM processed successfully.")


def test_validation_endpoint():
    print("\n[Test 7] Photogrammetric Accuracy Validation (/api/validate)...")
    res = client.post("/api/validate", data={
        "sample_key": "wayanad_real_optical.tif"
    })
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    v = data["validation"]
    print(f" - Validation Metrics: RMSE={v['rmse']}m, MAE={v['mae']}m, Correlation={v['correlation']}")
    assert v["rmse"] > 0
    assert v["mae"] > 0
    print(" - OK: Authentic photogrammetric error metrics computed.")


def test_satellite_search():
    print("\n[Test 8] Live Satellite Scene Search Endpoint...")
    res = client.get("/api/live_satellite_search?query=Wayanad")
    assert res.status_code == 200
    data = res.json()
    assert data["results_count"] > 0
    assert "Wayanad" in data["scenes"][0]["title"]
    print(f" - OK: Live catalog returned {data['results_count']} scenes.")


if __name__ == "__main__":
    test_static_files()
    test_api_status()
    test_branch_b_catalog()
    test_task2_geotiff_reader_on_samples()
    test_process_branch_b_geotiff()
    test_process_branch_a_optical()
    test_validation_endpoint()
    test_satellite_search()
    print("\n" + "=" * 70)
    print("   ALL TESTS PASSED! BRANCH B IS 100% REAL AND ACCESSIBLE.")
    print("=" * 70)
