import trimesh
mesh = trimesh.load("output_mesh2.obj")
mesh.export("terrain2.glb")
print("Exported terrain2.glb")