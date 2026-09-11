import trimesh
mesh = trimesh.load("output_mesh3.obj")
mesh.export("terrain3.glb")
print("Exported terrain3.glb")