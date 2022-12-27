import numpy as np
import open3d as o3d
from io import StringIO

pts = np.random.rand(100, 3)
faces = np.random.randint(0, 100, (100, 3))

trimesh = o3d.geometry.TriangleMesh()
trimesh.vertices = o3d.utility.Vector3dVector(pts)
trimesh.triangles = o3d.utility.Vector3iVector(faces)


class inMemoryMesh(StringIO):
    
    def __init__(self, name):
        self.name = name
        
        
        super().__init__()


def createInMemoryMesh(Data):

    inMesh = StringIO()
    inMesh.write("# Created by Mattia's code")
    inMesh.write("# object name: trimesh\n")

    nVertices = Data.pos.shape[0]
    nFaces = Data.faces.shape[0]

    inMesh.write("# number of vertices: {}\n".format(nVertices))
    inMesh.write("# number of triangles: {}\n".format(nFaces))

    for p in Data.pos:
        inMesh.write("v {} {} {}\n".format(p[0], p[1], p[2]))
    
    for f in Data.faces:
        inMesh.write("f {} {} {}\n".format(f[0], f[1], f[2]))
    
    inMesh.seek(0)

    return inMesh


class Data():

    def __init__(self) -> None:
        pass




data = Data()
data.pos = np.random.rand(100, 3)
data.faces = np.random.randint(0, 100, (100, 3))


inMemMesh = createInMemoryMesh(data)

print(inMemMesh.read())

print("Finished!")
inMemMesh.close()



"""
o3d.io.write_triangle_mesh(f,
                            trimesh, write_vertex_normals=False,
                            write_vertex_colors=False,
                            write_triangle_uvs=False)

"""




