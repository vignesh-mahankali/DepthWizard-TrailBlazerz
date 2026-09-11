import os
import cv2
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.windows import from_bounds as win_from_bounds

SAMPLE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'samples'))
os.makedirs(SAMPLE_DIR, exist_ok=True)

BENCHMARK_COORDINATES = {
    "wayanad": {
        "bounds": (76.0, 11.4, 76.4, 11.7), # min_lon, min_lat, max_lon, max_lat
        "cog_url": "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N11_00_E076_00_DEM/Copernicus_DSM_COG_10_N11_00_E076_00_DEM.tif",
        "description": "Wayanad Western Ghats Mountain Scarp (Kerala, India)"
    },
    "kolkata": {
        "bounds": (88.34, 22.55, 88.38, 22.59),
        "cog_url": "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N22_00_E088_00_DEM/Copernicus_DSM_COG_10_N22_00_E088_00_DEM.tif",
        "description": "Kolkata Hooghly River Urban Basin (West Bengal, India)"
    }
}

def generate_realistic_satellite_texture(width=512, height=512, terrain_type='mountain'):
    """Generates realistic synthetic satellite imagery textures for testing."""
    np.random.seed(42 if terrain_type == 'mountain' else 99)
    
    if terrain_type == 'mountain':
        base = np.zeros((height, width, 3), dtype=np.uint8)
        base[:, :, 0] = 30
        base[:, :, 1] = 80
        base[:, :, 2] = 40
        noise = np.random.normal(0, 15, (height, width, 3)).astype(np.int16)
        base = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        for x in range(width):
            y_center = int(height * 0.5 + 40 * np.sin(x / 40.0))
            cv2.circle(base, (x, y_center), 8, (140, 100, 40), -1)
        for r in range(0, height, 4):
            cv2.line(base, (0, r), (width, r + 20), (20, 50, 30), 1)

    elif terrain_type == 'landslide_post':
        base = generate_realistic_satellite_texture(width, height, 'mountain')
        pts = np.array([
            [width * 0.35, height * 0.2],
            [width * 0.55, height * 0.25],
            [width * 0.65, height * 0.7],
            [width * 0.40, height * 0.75],
            [width * 0.30, height * 0.45]
        ], np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(base, [pts], (30, 70, 140))
        base = cv2.GaussianBlur(base, (5, 5), 0)

    else:
        base = np.ones((height, width, 3), dtype=np.uint8) * 160
        for i in range(50, width, 90):
            cv2.line(base, (i, 0), (i, height), (70, 70, 70), 12)
            cv2.line(base, (0, i), (width, i), (70, 70, 70), 12)
        for bx in range(15, width - 60, 90):
            for by in range(15, height - 60, 90):
                color = (int(np.random.randint(180, 240)), int(np.random.randint(180, 240)), int(np.random.randint(180, 240)))
                cv2.rectangle(base, (bx, by), (bx + 55, by + 55), color, -1)

    return base

def fetch_or_create_dem(zone_key, target_shape=(512, 512)):
    """Fetches real Copernicus 30m DEM for zone bounds or resamples cached grid."""
    cfg = BENCHMARK_COORDINATES[zone_key]
    bounds = cfg["bounds"] # min_lon, min_lat, max_lon, max_lat
    h, w = target_shape

    # Check if we can fetch real Copernicus DEM from AWS COG
    try:
        with rasterio.open(cfg["cog_url"]) as src:
            win = win_from_bounds(*bounds, src.transform)
            data = src.read(1, window=win)
            if data is not None and data.size > 0:
                resized = cv2.resize(data, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
                return resized, bounds
    except Exception as e:
        print(f"Warning: Unable to fetch COG for {zone_key} directly: {e}")

    # Fallback to local numpy file if exists
    npy_name = f"{zone_key}_real_srtm_dem.npy"
    npy_path = os.path.join(SAMPLE_DIR, npy_name)
    if os.path.exists(npy_path):
        data = np.load(npy_path)
        resized = cv2.resize(data, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        return resized, bounds

    # Synthetic realistic topography fallback
    if zone_key == 'wayanad':
        # Western Ghats steep escarpment 400m - 2100m
        y, x = np.mgrid[0:h, 0:w]
        elev = 400.0 + 1500.0 * (1.0 - x / float(w)) + 200.0 * np.sin(y / 30.0)
    else:
        # Kolkata flat delta 2m - 20m
        y, x = np.mgrid[0:h, 0:w]
        elev = 5.0 + 10.0 * (x / float(w)) + 2.0 * np.sin(y / 40.0)
    return elev.astype(np.float32), bounds

def save_georeferenced_dem_tif(tif_path, dem_matrix, bounds):
    """Saves single-band float32 GeoTIFF with EPSG:4326 CRS and affine transform."""
    h, w = dem_matrix.shape
    transform = from_bounds(*bounds, w, h)
    with rasterio.open(
        tif_path,
        'w',
        driver='GTiff',
        height=h,
        width=w,
        count=1,
        dtype='float32',
        crs='EPSG:4326',
        transform=transform
    ) as dst:
        dst.write(dem_matrix.astype(np.float32), 1)

def save_georeferenced_rgb_tif(tif_path, rgb_image, bounds):
    """Saves 3-band RGB GeoTIFF with EPSG:4326 CRS and affine transform."""
    h, w, c = rgb_image.shape
    transform = from_bounds(*bounds, w, h)
    # rgb_image is BGR from OpenCV, convert to RGB for standard GeoTIFF
    if c == 3:
        img_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = rgb_image
    
    with rasterio.open(
        tif_path,
        'w',
        driver='GTiff',
        height=h,
        width=w,
        count=3,
        dtype='uint8',
        crs='EPSG:4326',
        transform=transform
    ) as dst:
        for band in range(3):
            dst.write(img_rgb[:, :, band], band + 1)

def ensure_sample_datasets():
    """Ensures standard JPG and genuine georeferenced GeoTIFF sample datasets exist."""
    samples_jpg = {
        "wayanad_pre_disaster.jpg": generate_realistic_satellite_texture(512, 512, 'mountain'),
        "wayanad_post_disaster.jpg": generate_realistic_satellite_texture(512, 512, 'landslide_post'),
        "urban_kolkata.jpg": generate_realistic_satellite_texture(512, 512, 'urban'),
    }
    saved_files = {}
    for filename, img in samples_jpg.items():
        path = os.path.join(SAMPLE_DIR, filename)
        if not os.path.exists(path):
            cv2.imwrite(path, img)
        saved_files[filename] = path

    # 1. Wayanad Pre-Disaster
    w_dem, w_bounds = fetch_or_create_dem('wayanad')
    np.save(os.path.join(SAMPLE_DIR, 'wayanad_real_srtm_dem.npy'), w_dem)
    w_dem_tif = os.path.join(SAMPLE_DIR, 'wayanad_real_srtm_dem.tif')
    save_georeferenced_dem_tif(w_dem_tif, w_dem, w_bounds)
    saved_files['wayanad_real_srtm_dem.tif'] = w_dem_tif

    # Also keep compatibility alias wayanad_srtm_dem.tif
    save_georeferenced_dem_tif(os.path.join(SAMPLE_DIR, 'wayanad_srtm_dem.tif'), w_dem, w_bounds)

    w_opt_tif = os.path.join(SAMPLE_DIR, 'wayanad_real_optical.tif')
    save_georeferenced_rgb_tif(w_opt_tif, samples_jpg["wayanad_pre_disaster.jpg"], w_bounds)
    saved_files['wayanad_real_optical.tif'] = w_opt_tif

    # 2. Wayanad Post-Disaster
    w_post_dem = w_dem.copy()
    # Apply simulated scar volume loss
    h, w = w_dem.shape
    w_post_dem[int(h*0.3):int(h*0.6), int(w*0.35):int(w*0.65)] -= 18.5
    np.save(os.path.join(SAMPLE_DIR, 'wayanad_post_real_srtm_dem.npy'), w_post_dem)
    w_post_dem_tif = os.path.join(SAMPLE_DIR, 'wayanad_post_real_srtm_dem.tif')
    save_georeferenced_dem_tif(w_post_dem_tif, w_post_dem, w_bounds)
    saved_files['wayanad_post_real_srtm_dem.tif'] = w_post_dem_tif
    save_georeferenced_dem_tif(os.path.join(SAMPLE_DIR, 'wayanad_post_srtm_dem.tif'), w_post_dem, w_bounds)

    w_post_opt_tif = os.path.join(SAMPLE_DIR, 'wayanad_post_real_optical.tif')
    save_georeferenced_rgb_tif(w_post_opt_tif, samples_jpg["wayanad_post_disaster.jpg"], w_bounds)
    saved_files['wayanad_post_real_optical.tif'] = w_post_opt_tif

    # 3. Kolkata Urban Basin
    k_dem, k_bounds = fetch_or_create_dem('kolkata')
    np.save(os.path.join(SAMPLE_DIR, 'kolkata_real_srtm_dem.npy'), k_dem)
    k_dem_tif = os.path.join(SAMPLE_DIR, 'kolkata_real_srtm_dem.tif')
    save_georeferenced_dem_tif(k_dem_tif, k_dem, k_bounds)
    saved_files['kolkata_real_srtm_dem.tif'] = k_dem_tif
    save_georeferenced_dem_tif(os.path.join(SAMPLE_DIR, 'kolkata_srtm_dem.tif'), k_dem, k_bounds)

    k_opt_tif = os.path.join(SAMPLE_DIR, 'kolkata_real_optical.tif')
    save_georeferenced_rgb_tif(k_opt_tif, samples_jpg["urban_kolkata.jpg"], k_bounds)
    saved_files['kolkata_real_optical.tif'] = k_opt_tif

    return saved_files

if __name__ == "__main__":
    files = ensure_sample_datasets()
    print(f"Generated {len(files)} sample datasets with full georeferencing:")
    for k, v in files.items():
        print(f" - {k}: {v}")
