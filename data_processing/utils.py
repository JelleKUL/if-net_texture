import numpy as np
import trimesh
from scipy.spatial import cKDTree as KDTree
import numbers
import os

def make_dir_if_not_exist(path):
    if(not os.path.exists(path)):
        print("Folder does not exist, creating the folder " + str(path))
        os.mkdir(path)

def create_grid_points_from_xyz_bounds(min_x, max_x, min_y, max_y ,min_z, max_z, res):
    x = np.linspace(min_x, max_x, res)
    y = np.linspace(min_y, max_y, res)
    z = np.linspace(min_z, max_z, res)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij', sparse=False)
    X = X.reshape((np.prod(X.shape),))
    Y = Y.reshape((np.prod(Y.shape),))
    Z = Z.reshape((np.prod(Z.shape),))

    points_list = np.column_stack((X, Y, Z))
    del X, Y, Z, x
    return points_list


# conversion of points into coordinates understood by pytorch's grid_sample function
# bbox = [min_x,max_x,min_y,max_y,min_z,max_z]
def to_grid_sample_coords(points, bbox):
    min_values, max_values = bbox[::2], bbox[1::2]
    points = 2 * 1 / (max_values - min_values) * (points - min_values) - 1
    grid_coords = points.copy()
    grid_coords[:, 0], grid_coords[:, 2] = points[:, 2], points[:, 0]
    return grid_coords

def barycentric_coordinates(p, q, u, v):
    """
    Calculate barycentric coordinates of the given point
    :param p: a given point
    :param q: triangle vertex
    :param u: triangle vertex
    :param v: triangle vertex
    :return: 1X3 ndarray with the barycentric coordinates of p
    """
    v0 = u - q
    v1 = v - q
    v2 = p - q
    d00 = v0.dot(v0)
    d01 = v0.dot(v1)
    d11 = v1.dot(v1)
    d20 = v2.dot(v0)
    d21 = v2.dot(v1)
    denom = d00 * d11 - d01 * d01
    y = (d11 * d20 - d01 * d21) / denom
    z = (d00 * d21 - d01 * d20) / denom
    x = 1.0 - z - y
    return np.array([x, y, z])

def as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([
            trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
            for m in scene_or_mesh.geometry.values()])
    else:
        mesh = scene_or_mesh
    return mesh

def shoot_holes(vertices, n_holes, dropout, mask_faces=None, faces=None,
                rng=None):
    """Generate a partial shape by cutting holes of random location and size.

    Each hole is created by selecting a random point as the center and removing
    the k nearest-neighboring points around it.

    Args:
        vertices: The array of vertices of the mesh.
        n_holes (int or (int, int)): Number of holes to create, or bounds from
            which to randomly draw the number of holes.
        dropout (float or (float, float)): Proportion of points (with respect
            to the total number of points) in each hole, or bounds from which
            to randomly draw the proportions (a different proportion is drawn
            for each hole).
        mask_faces: A boolean mask on the faces. 1 to keep, 0 to ignore. If
                    set, the centers of the holes are sampled only on the
                    non-masked regions.
        faces: The array of faces of the mesh. Required only when `mask_faces`
               is set.
        rng: (optional) An initialised np.random.Generator object. If None, a
             default Generator is created.

    Returns:
        array: Indices of the points defining the holes.
    """
    if rng is None:
        rng = np.random.default_rng()

    if not isinstance(n_holes, numbers.Integral):
        n_holes_min, n_holes_max = n_holes
        n_holes = rng.integers(n_holes_min, n_holes_max)

    if mask_faces is not None:
        valid_vertex_indices = np.unique(faces[mask_faces > 0])
        valid_vertices = vertices[valid_vertex_indices]
    else:
        valid_vertices = vertices

    # Select random hole centers.
    center_indices = rng.choice(len(valid_vertices), size=n_holes)
    centers = valid_vertices[center_indices]

    n_vertices = len(valid_vertices)
    if isinstance(dropout, numbers.Number):
        hole_size = n_vertices * dropout
        hole_sizes = [hole_size] * n_holes
    else:
        hole_size_bounds = n_vertices * np.asarray(dropout)
        hole_sizes = rng.integers(*hole_size_bounds, size=n_holes)

    # Identify the points indices making up the holes.
    kdtree = KDTree(vertices, leafsize=200)
    to_crop = []
    for center, size in zip(centers, hole_sizes):
        _, indices = kdtree.query(center, k=size)
        to_crop.append(indices)
    to_crop = np.unique(np.concatenate(to_crop))
    return to_crop