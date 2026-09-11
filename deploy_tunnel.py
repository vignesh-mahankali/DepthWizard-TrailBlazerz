#!/usr/bin/env python3
"""
DepthWizard Instant Cloudflare Tunnel Launcher
Provides a free, zero-config, secure public HTTPS URL (e.g. https://xxxx.trycloudflare.com)
for showcasing DepthWizard without opening ports or configuring routers.
"""
import os
import sys
import subprocess
import time
import re
import platform

def find_or_download_cloudflared():
    # Check if cloudflared is already in PATH
    try:
        res = subprocess.run(["cloudflared", "--version"], capture_output=True, text=True)
        if res.returncode == 0:
            return "cloudflared"
    except FileNotFoundError:
        pass

    system = platform.system().lower()
    machine = platform.machine().lower()
    
    bin_name = "cloudflared.exe" if system == "windows" else "cloudflared"
    bin_path = os.path.abspath(bin_name)
    
    if os.path.exists(bin_path):
        return bin_path

    print("Cloudflared binary not found in PATH. Downloading standalone binary...")
    import urllib.request
    
    if system == "windows":
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
    elif system == "darwin":
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-amd64"
    else:
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"

    print(f"Downloading from: {url}")
    urllib.request.urlretrieve(url, bin_path)
    
    if system != "windows":
        os.chmod(bin_path, 0o755)
        
    print(f"Downloaded to {bin_path}")
    return bin_path

def main():
    print("==================================================")
    print("  DepthWizard Enterprise - Instant Public Tunnel  ")
    print("==================================================")
    
    cf_bin = find_or_download_cloudflared()
    print("Starting Cloudflare Tunnel on http://127.0.0.1:8000...")
    
    cmd = [cf_bin, "tunnel", "--url", "http://127.0.0.1:8000"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    url_found = False
    for line in iter(proc.stdout.readline, ''):
        sys.stdout.write(line)
        match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line)
        if match and not url_found:
            url_found = True
            public_url = match.group(0)
            print("\n" + "=" * 60)
            print(f"  DEPTHWIZARD IS LIVE PUBLICLY AT:")
            print(f"  >>> {public_url} <<<")
            print("=" * 60 + "\n")
            
    proc.wait()

if __name__ == "__main__":
    main()
