import os
import cv2
import numpy as np

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'samples')
os.makedirs(SAMPLE_DIR, exist_ok=True)

def generate_realistic_satellite_texture(width=512, height=512, terrain_type='mountain'):
    """Generates realistic synthetic satellite imagery textures for testing."""
    np.random.seed(42 if terrain_type == 'mountain' else 99)
    
    # Base canvas
    if terrain_type == 'mountain':
        # Forested mountainous terrain with river valley
        base = np.zeros((height, width, 3), dtype=np.uint8)
        base[:, :, 0] = 30  # Blue channel
        base[:, :, 1] = 80  # Green channel
        base[:, :, 2] = 40  # Red channel
        
        # Add texture noise
        noise = np.random.normal(0, 15, (height, width, 3)).astype(np.int16)
        base = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Add river feature
        for x in range(width):
            y_center = int(height * 0.5 + 40 * np.sin(x / 40.0))
            cv2.circle(base, (x, y_center), 8, (140, 100, 40), -1)  # Muddy river BGR
            
        # Add mountain ridge shadows
        for r in range(0, height, 4):
            cv2.line(base, (0, r), (width, r + 20), (20, 50, 30), 1)

    elif terrain_type == 'landslide_post':
        # Post-disaster landslide (exposed mud/earth red-brown scar)
        base = generate_realistic_satellite_texture(width, height, 'mountain')
        # Scar polygon
        pts = np.array([
            [width * 0.35, height * 0.2],
            [width * 0.55, height * 0.25],
            [width * 0.65, height * 0.7],
            [width * 0.40, height * 0.75],
            [width * 0.30, height * 0.45]
        ], np.int32).reshape((-1, 1, 2))
        
        # Fill scar with exposed brownish soil
        cv2.fillPoly(base, [pts], (30, 70, 140))  # BGR for red-brown soil
        base = cv2.GaussianBlur(base, (5, 5), 0)

    else:
        # Urban scene with buildings and grid streets
        base = np.ones((height, width, 3), dtype=np.uint8) * 160
        # Draw road grid
        for i in range(50, width, 90):
            cv2.line(base, (i, 0), (i, height), (70, 70, 70), 12)
            cv2.line(base, (0, i), (width, i), (70, 70, 70), 12)
        # Draw building blocks
        for bx in range(15, width - 60, 90):
            for by in range(15, height - 60, 90):
                color = (int(np.random.randint(180, 240)), int(np.random.randint(180, 240)), int(np.random.randint(180, 240)))
                cv2.rectangle(base, (bx, by), (bx + 55, by + 55), color, -1)

    return base

def ensure_sample_datasets():
    """Ensures sample remote sensing dataset images exist in samples/ directory."""
    samples = {
        "wayanad_pre_disaster.jpg": generate_realistic_satellite_texture(512, 512, 'mountain'),
        "wayanad_post_disaster.jpg": generate_realistic_satellite_texture(512, 512, 'landslide_post'),
        "urban_kolkata.jpg": generate_realistic_satellite_texture(512, 512, 'urban'),
    }
    
    saved_files = {}
    for filename, img in samples.items():
        path = os.path.join(SAMPLE_DIR, filename)
        if not os.path.exists(path):
            cv2.imwrite(path, img)
        saved_files[filename] = path
        
    return saved_files

if __name__ == "__main__":
    files = ensure_sample_datasets()
    print(f"Sample datasets ready: {list(files.keys())}")
