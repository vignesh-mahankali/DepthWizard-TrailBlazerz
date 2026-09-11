# DepthWizard Production Hosting & Deployment Guide

This guide outlines battle-tested deployment pathways to host DepthWizard in production environments ranging from zero-cost cloud hosting to GPU-accelerated hyperscalers.

---

## Architecture Overview

- **Core Engine:** FastAPI backend (`server.py`) serving inference, photogrammetric calibration, and geospatial export endpoints.
- **Inference Pipeline:** Depth-Anything-V2 (ViT-Small/Base/Large) with monocular disparity inversion rectification.
- **Data Integration:** Genuine NASA SRTM 30m / Copernicus DEM validation and caching.
- **Frontend Client:** High-performance WebGL 3D Flythrough, HUD telemetry, and Chart.js analytics.

---

## Option 1: Hugging Face Spaces (Recommended for Quick Free Hosting)

Hugging Face Spaces provides a free CPU tier and on-demand GPU instances (T4, A10G) with automatic SSL and zero server management.

1. **Create a Space:**
   - Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **Create new Space**.
   - Set **SDK** to **Docker** (Blank).
   - Select Hardware: **CPU Basic (Free)** or **T4 Small ($0.60/hr)** for instant sub-second GPU inference.

2. **Configure Space:**
   - Upload the repository files.
   - Hugging Face automatically reads the `Dockerfile` and builds the container.
   - Set `PORT=7860` in Space settings or change `EXPOSE 7860` in `Dockerfile`.

---

## Option 2: Docker / Docker Compose on Linux VM (Render, Railway, Fly.io, or DigitalOcean)

### Quick Start with Docker Compose:

```bash
# Clone the repository
git clone https://github.com/your-org/depthwizard.git
cd depthwizard

# Launch container with persistent volume mounts
docker-compose up -d --build

# Verify deployment health
curl http://localhost:8000/api/status
```

### Deploying to Render / Railway:
1. Connect your GitHub repository to [Render.com](https://render.com) or [Railway.app](https://railway.app).
2. Select **Docker** environment.
3. Set the Health Check path to `/api/status`.
4. Add a persistent disk mounted to `/app/dem_cache` and `/app/exports`.

---

## Option 3: Dedicated Cloud VM (AWS EC2 / GCP Compute Engine / Azure)

For production enterprise deployments with high throughput:

### 1. Recommended Hardware:
- **AWS:** `g4dn.xlarge` (1x NVIDIA T4 GPU, 4 vCPUs, 16 GB RAM) or `c6i.xlarge` (CPU).
- **OS:** Ubuntu 22.04 LTS.

### 2. Systemd Service Daemon:
Create `/etc/systemd/system/depthwizard.service`:

```ini
[Unit]
Description=DepthWizard Photogrammetric AI Server
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/depthwizard
ExecStart=/home/ubuntu/depthwizard/venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000 --workers 2
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable depthwizard
sudo systemctl start depthwizard
```

### 3. Nginx Reverse Proxy with SSL (Let's Encrypt):
Create `/etc/nginx/sites-available/depthwizard`:

```nginx
server {
    server_name depthwizard.yourdomain.com;

    client_max_body_size 50M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
    }
}
```

Enable SSL certificate with Certbot:
```bash
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d depthwizard.yourdomain.com
```

---

## Environment Variables Reference

| Variable | Description | Default |
| :--- | :--- | :--- |
| `PORT` | Listening network port | `8000` |
| `MODEL_ENCODER` | Model variant (`vits`, `vitb`, `vitl`) | `vits` |
| `DEVICE` | Computation device (`cpu`, `cuda`, `mps`) | Auto-detected |

---

## Verification & Health Check

After deployment, verify that the API is fully functional:

```bash
curl -s http://localhost:8000/api/status | jq .
```

Expected output:
```json
{
  "status": "ready",
  "system": "DepthWizard Engine v2.2 (Scientific Photogrammetry)",
  "device": "cpu",
  "model_loaded": true,
  "verified_benchmarks": ["wayanad_pre", "wayanad_post", "kolkata"]
}
```
