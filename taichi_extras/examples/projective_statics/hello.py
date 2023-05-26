import meshtaichi_patcher
import numpy as np
import taichi as ti

ti.init()


# mesh = ti.Mesh(ti.f32)
# ti.MeshInstance
# mesh_builder = mesh.RelationVisitor(mesh)
# mesh_builder.begin()
# mesh_builder.Vertex(0.0, 0.0)  # vertex 0
# mesh_builder.Vertex(1.0, 0.0)  # vertex 1
# mesh_builder.Vertex(0.5, 0.5)  # vertex 2
# mesh_builder.Cell(3)  # cell with three vertices
# mesh_builder.CellVertex(0)  # cell vertex 0
# mesh_builder.CellVertex(1)  # cell vertex 1
# mesh_builder.CellVertex(2)  # cell vertex 2
# mesh_builder.end()
# ti.TriMesh()

# # create a mesh instance with vertices and cells from the mesh builder
# vertices = np.array(mesh.get_vertices())
# cells = np.array(mesh.get_cells())
# mesh_instance = ti.MeshInstance(mesh)
# mesh_instance.set_vertices(vertices)
# mesh_instance.set_cells(cells)

# # set material and mass properties of the mesh instance
# material = ti.Material()
# material.set_lame_parameters(1e5, 1e5)
# material.set_density(1e3)
# mesh_instance.set_material(material)
# mesh_instance.set_mass_from_density()
# mesh = ti.Mesh()
# mesh._create_instance(metadata=)


print(dir(meshtaichi_patcher))
