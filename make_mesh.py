import cv2
import numpy as np
import open3d as o3d

# Load the original color image and the depth map

color = cv2.imread("Depth-Anything-V2/test3.jpg")
depth = cv2.imread("Depth-Anything-V2/output3/test3.png", cv2.IMREAD_GRAYSCALE)

# Resize color to match depth map size if needed

color = cv2.resize(color, (depth.shape[1], depth.shape[0]))
color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

h, w = depth.shape
fx = fy = w  # rough placeholder focal length
cx, cy = w / 2, h / 2

points = []
colors = []

# Downsample for speed (every 2nd pixel) since we're CPU-only

step = 2
for v in range(0, h, step):
    for u in range(0, w, step):
        z = float(depth[v, u]) / 255.0 * 50.0  # scale depth arbitrarily for now
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        points.append([x, -y, -z])
        colors.append(color_rgb[v, u] / 255.0)

points = np.array(points)
colors = np.array(colors)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

o3d.io.write_point_cloud("output_pointcloud3.ply", pcd)
print("Point cloud saved:", len(points), "points")

# Estimate normals (required for mesh reconstruction)

pcd.estimate_normals()

# Poisson surface reconstruction

mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd,
    depth=9
)

# Remove low-density (noisy/unreliable) parts of the mesh

densities = np.asarray(densities)
vertices_to_remove = densities < np.quantile(densities, 0.05)
mesh.remove_vertices_by_mask(vertices_to_remove)

o3d.io.write_triangle_mesh("output_mesh3.obj", mesh)
print("Mesh saved with", len(mesh.vertices), "vertices")