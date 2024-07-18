import traceback
import data_processing.utils as utils
import trimesh
import numpy as np
import os

def val_in_arr(val, array):
    if val in array:
        return 1
    return 0

class MeshArray ():
    
    def __init__(self, R = None, G = None, B = None, S = None,colors = None, pcd = None, bbox = None, res = None, id = None, grid_points = None, kdtree = None, num_points = None, meshPath=None):
        self.R = R
        self.G = G
        self.B = B
        self.S = S
        self.colors = colors
        self.pcd = pcd
        self.bbox = bbox
        self.res = res
        self.id = id
        self.grid_points = grid_points
        self.kdtree = kdtree
        self.num_points = num_points
        self.meshPath = meshPath

        self.fullPcd = None
        
    
    def from_trimesh(self, meshPath, colorMode:str = "single"):
        
        self.meshPath = meshPath
        mesh = trimesh.load(meshPath)
        scale = 1 / np.max(mesh.extents)
        center = mesh.centroid
        for3matx = np.hstack((np.identity(3) * scale, center.reshape((3,1)) ))
        transformMtx = np.vstack((for3matx, [0,0,0,1]))
        mesh.apply_transform(transformMtx)
        if(colorMode == "rgb"):
            pointArray = self.sample_colors(mesh)
            colored_point_cloud = pointArray[np.random.choice(len(pointArray), size=self.num_points, replace=False)]
            #print(colored_point_cloud.shape)
        else:
            # check if the mesh multiple materials (sub meshes)
            if isinstance(mesh, trimesh.Scene):
                pointArray = []
                i = 1
                subMeshes = mesh.dump()
                for sub in subMeshes:
                    colors = np.ones((self.num_points,3)) * i
                    if(colorMode == "extrapolate"):
                        colors = colors * 255/len(subMeshes)
                    elif(colorMode == "singleChannel"):
                        ColorArray = 255 * np.array([[val_in_arr(i % 8, [1, 4, 6, 7]), val_in_arr(i % 8, [2,4,5,7]), val_in_arr(i % 8, [3,5,6,7])]])
                        colors = np.repeat(ColorArray, self.num_points, axis = 0)
                    colored_addition = np.hstack((sub.sample(self.num_points), colors))
                    #print(colored_addition)
                    pointArray.append(colored_addition)
                    i+=1

                pointArray = np.asarray(pointArray).reshape((-1,6))
                colored_point_cloud = pointArray[np.random.choice(len(pointArray), size=self.num_points, replace=False)]
            else:
                colored_point_cloud = np.hstack((mesh.sample(self.num_points), np.zeros((self.num_points,3)))) 

        # encode uncolorized, complete shape of object (at inference time obtained from IF-Nets surface reconstruction)
        # encoding is done by sampling a pointcloud and voxelizing it (into discrete grid for 3D CNN usage)
        full_shape = utils.as_mesh(mesh)
        shape_point_cloud = full_shape.sample(self.num_points)
        S = np.zeros(len(self.grid_points), dtype=np.int8)
        self.fullPcd = shape_point_cloud

        _, idx = self.kdtree.query(shape_point_cloud)
        S[idx] = 1

        self.S = S
        self.R = colored_point_cloud[:,3]
        self.G = colored_point_cloud[:,4]
        self.B = colored_point_cloud[:,5]
        self.colors = np.hstack((self.R.reshape((-1,1)), self.G.reshape((-1,1)), self.B.reshape((-1,1))))
        self.pcd = colored_point_cloud[:,:3]

        return self

    def filter_points(self, outPath, nrOfVariants, nrOfHoles, dropout):
        
        # Create the ground truth data
        
        
        # step 1: remove a number of points from the lists
        # voxelise the partial pointcloud
        # relink the colors to the voxels
        
        for nr in range(nrOfVariants):
            f = open(outPath / (self.id + "_normalized-partial-" + str(nr) + ".txt"), "w")
            f.write(str(outPath / (self.id + "_normalized-partial-" + str(nr) + ".txt")))
            f.close()
            filteredIndexes = utils.shoot_holes(self.pcd, nrOfHoles, dropout)
            filteredPcd = np.delete(self.pcd,filteredIndexes,0)
            filteredR = np.delete(self.R,filteredIndexes,0)
            filteredG = np.delete(self.G,filteredIndexes,0)
            filteredB = np.delete(self.B,filteredIndexes,0)

            R = - 1 * np.ones(len(self.grid_points), dtype=np.int16)
            G = - 1 * np.ones(len(self.grid_points), dtype=np.int16)
            B = - 1 * np.ones(len(self.grid_points), dtype=np.int16)

            _, idx = self.kdtree.query(filteredPcd)
            R[idx] = filteredR
            G[idx] = filteredG
            B[idx] = filteredB

            if(not os.path.exists(outPath / (self.id + "-partial-" + str(nr) + ".npz"))):
                np.savez(outPath / (self.id + "_normalized-partial-" + str(nr) + "_voxelized_colored_point_cloud_res128_points100000_bbox-1,1,-1,1,-1,1.npz"), R=R, G=G,B=B, S=self.S,  colored_point_cloud=filteredPcd, bbox = self.bbox, res = self.res)


    def sample_colors(self,gt_mesh):
        try:
            pointArray = []
            if isinstance(gt_mesh, trimesh.Scene):
                subMeshes = gt_mesh.dump()
                for sub in subMeshes:
                    pointArray.append(self.sample_mesh_colors(sub))
            else: 
                pointArray.append(self.sample_mesh_colors(gt_mesh))
            
            pointArray = np.asarray(pointArray).reshape((-1,6))

            return pointArray

        except Exception as err:
            print('Error with: {}'.format(traceback.format_exc()))

    def sample_mesh_colors(self, sub, ):
        sample_points, face_idxs = sub.sample(self.num_points, return_index = True)
        self.pcd = sample_points
        triangles = sub.triangles[face_idxs]
        face_vertices = sub.faces[face_idxs]
        #print(sub.visual.kind)
        if(type(sub.visual) == trimesh.visual.texture.TextureVisuals):
            texture = sub.visual.material.image
            if(sub.visual.uv is not None and texture is not None):
                faces_uvs = sub.visual.uv[face_vertices]

                q = triangles[:, 0]
                u = triangles[:, 1]
                v = triangles[:, 2]
                uvs = []

                for i, p in enumerate(sample_points):
                    barycentric_weights = utils.barycentric_coordinates(p, q[i], u[i], v[i])
                    uv = np.average(faces_uvs[i], 0, barycentric_weights)
                    uvs.append(uv)

                colors = trimesh.visual.color.uv_to_color(np.array(uvs), texture)[:,:3]
            else:
                #print("No texture of uvs found")
                colors = np.tile(np.array((0,0,0,0)),self.num_points).reshape(-1,4)[:,:3]
        elif(type(sub.visual) == trimesh.visual.ColorVisuals):
            main_color = sub.visual.main_color()
            if(main_color is None):
                #print("No main color")
                main_color = np.array((0,0,0,0))
            colors = np.tile(main_color,self.num_points).reshape(-1,4)[:,:3]
        else:
            colors = np.tile(np.array((0,0,0,0)),self.num_points).reshape(-1,4)[:,:3]
        #print(colors)
        return np.hstack((sample_points, colors))


    def savez(self, out_file):
        
        np.savez(out_file, points = self.pcd, grid_coords = utils.to_grid_sample_coords(self.pcd, self.bbox), colors = self.colors)
        #np.savez(out_file, R=self.R, G=self.G,B=self.B, S=self.S,  colored_point_cloud=self.pcd, bbox = self.bbox, res = self.res)