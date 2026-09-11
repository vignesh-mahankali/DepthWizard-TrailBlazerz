import trimesh

mesh = trimesh.load("output_mesh.obj")
mesh.export("terrain.glb")
print("Exported terrain.glb")