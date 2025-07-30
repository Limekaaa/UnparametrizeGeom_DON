import numpy as np
import gmsh

import os

import trimesh
from tqdm import tqdm

from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from scipy.spatial import cKDTree
import meshio

#import deep_sdf
from typing import List, Literal

def create_square_with_hole_dataset(num_samples = 100, min_hole=1, max_hole=5, min_radius=0.05, path:str="", save_format: List[Literal["msh", "stl", "obj"]] = ["msh"]):
    """
    Create a dataset of square meshes with circular holes.
    
    Parameters:
    - num_samples: Number of samples to generate.
    - min_hole: Minimum radius of the hole.
    - max_hole: Maximum radius of the hole.
    - path: Directory to save the generated meshes.
    """
    if min_radius > 0.5 * (1/max_hole) :
        raise ValueError(f"Minimum radius must be smaller than {0.5 * (1/max_hole)} for max_hole = {max_hole}.")
    
    if path[-1] != "/":
        path += "/"
    if not os.path.exists(path):
        os.makedirs(path)

    for i in tqdm(range(num_samples), desc="Generating square meshes with holes", unit="sample"):
        n_holes = np.random.randint(min_hole, max_hole + 1)
        bounds = np.linspace(0, 1, n_holes + 1)
        max_R = 0.5 * (1 / n_holes)
        hole_radii = np.random.uniform(min_radius, max_R, n_holes)

        hole_centers_bound_x = [(bounds[k] + hole_radii[k], bounds[k + 1] - hole_radii[k]) for k in range(0, n_holes, 1)]
        hole_centers_bound_y = [(hole_radii[k], 1 - hole_radii[k]) for k in range(0, n_holes, 1)]

        hole_centers = [(np.random.uniform(*hole_centers_bound_x[k]), np.random.uniform(*hole_centers_bound_y[k])) for k in range(n_holes)]

        # mesh creation ________________________________________________________________________________________________________________________________________________

        gmsh.initialize()
        gmsh.model.add("multi_circle_quad")

        # Parameters
        #lc = 0.05  # mesh size
        square_size = 1.0
        #n_circle_segments = 40  # resolution of each circular hole (more = better approximation)

        square = gmsh.model.occ.addRectangle(0, 0, 0, square_size, square_size)

        # --- 2. Circular holes
        circles = []
        for center, radius in zip(hole_centers, hole_radii):
            cx, cy = center
            circle = gmsh.model.occ.addDisk(cx, cy, 0, radius, radius)

            circles.append(circle)

        # --- 3. Plane surface with holes
        
        surface, _ = gmsh.model.occ.cut([(2, square)], [(2, c) for c in circles])

        gmsh.model.occ.synchronize()

        surf_tag = surface[0][1]
        gmsh.model.addPhysicalGroup(2, [surf_tag], tag=1)
        gmsh.model.setPhysicalName(2, 1, "Domain")

        boundary = gmsh.model.getBoundary([(2, surf_tag)], oriented=False)
        curve_tags = [e[1] for e in boundary if e[0] == 1]
        gmsh.model.addPhysicalGroup(1, curve_tags, tag=2)
        gmsh.model.setPhysicalName(1, 2, "Boundary")

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.05)
        gmsh.model.mesh.generate(2)


        filename = f"square_with_holes_{n_holes}_"
        for c in hole_centers:
            filename += f"{c[0]:.3f}_"
        filename = filename[:-1]

        if "stl" in save_format:
            if not os.path.exists(f"{path}stl/"):
                os.makedirs(f"{path}stl/")
            c = 0
            new_filename = filename
            while os.path.exists(f"{path}stl/{new_filename}.stl"):
                c += 1
                new_filename = f"{filename}_{c}"
            gmsh.write(f"{path}stl/{new_filename}.stl")
            
        if "msh" in save_format:
            if not os.path.exists(f"{path}msh/"):
                os.makedirs(f"{path}msh/")
            c = 0
            new_filename = filename
            while os.path.exists(f"{path}msh/{new_filename}.msh"):
                c += 1
                new_filename = f"{filename}_{c}"
            gmsh.write(f"{path}msh/{new_filename}.msh")

        if "obj" in save_format:
            if "stl" not in save_format:
                print("Warning: 'obj' format requires 'stl' format to be saved as well. Saving as 'stl'.")
                if not os.path.exists(f"{path}stl/"):
                    os.makedirs(f"{path}stl/")
                c = 0
                new_filename = filename
                while os.path.exists(f"{path}stl/{new_filename}.stl"):
                    c += 1
                    new_filename = f"{filename}_{c}"
                gmsh.write(f"{path}stl/{new_filename}.stl")

            if not os.path.exists(f"{path}obj/"):
                os.makedirs(f"{path}obj/")
            
            tri_mesh = trimesh.load(f"{path}stl/{new_filename}.stl")

            trimesh.exchange.export.export_mesh(tri_mesh, f"{path}obj/{new_filename}.obj", file_type='obj')

        gmsh.finalize()

def compute_sdf_2d(mesh_path, query_range=np.array([[0, 1], [0, 1], [0, 0]]), resolution=0.01):

    mesh = meshio.read(mesh_path)
    points = mesh.points[:, :2]  # Use only the first two dimensions for 2D SDF

    boundary_edges = []
    for cell_block in mesh.cells:
        if cell_block.type in ["line", "line2"]:
            for edge in cell_block.data:
                boundary_edges.append(points[edge[0]])
                boundary_edges.append(points[edge[1]])
    if not boundary_edges:
        raise ValueError(f"No boundary edges found in {mesh_path}")
    boundary_points = np.unique(np.vstack(boundary_edges), axis=0)
    tree = cKDTree(boundary_points)

    # Sample grid
    # x = np.linspace(-0.1, 1.1, grid_size)
    # y = np.linspace(-0.1, 1.1, grid_size)
    
    # The bounding box
    x = np.arange(query_range[0][0], query_range[0][1]+resolution, resolution)
    y = np.arange(query_range[1][0], query_range[1][1]+resolution, resolution)
    #z = np.arange(query_range[2][0], query_range[2][1], resolution)

    #X, Y, Z = np.meshgrid(x, y, z)
    X, Y = np.meshgrid(x, y)
    grid_points = np.c_[X.ravel(), Y.ravel()]
    distances, _ = tree.query(grid_points)

    # Create polygon union of all triangles to test for inside
    polygons = []
    for cell_block in mesh.cells:
        if cell_block.type.startswith("triangle"):
            for tri in cell_block.data:
                polygons.append(Polygon(points[tri]))
    domain = unary_union(polygons)

    signs = np.array([1 if domain.contains(Point(p)) else -1 for p in grid_points])
    sdf = distances * signs

    Z = np.zeros_like(X)  # Z is zero for 2D case
    grid_points = np.c_[grid_points, Z.ravel()]  # Add Z coordinate
    return grid_points, sdf, X, Y, Z

def save_data_to_npz_file(data:np.ndarray, filename:str):
    """
    Save data to npz files

    Args:
        data (np.ndarray): data of shape (N, 4) where N is the number of points and 4 represents (x, y, z, sdf_value).
        filename (str): Path to the output npz file.
    """

    pos = data[data[:, -1] >= 0]
    neg = data[data[:, -1] < 0]

    np.savez(filename, pos=pos, neg=neg)

def sdf_data_generator(specs, specs_data):
   
    full_path = os.path.join(specs["DataSource"], specs_data["dataset_name"])

    if not os.path.exists(full_path):
        os.makedirs(full_path)

    if "npz" in specs_data["SDFData"]["save_format"] and "msh" not in specs_data["SDFData"]["save_format"]:
        print("Warning: 'npz' format requires 'msh' format to be saved as well. Saving as 'msh'.")
        specs_data["SDFData"]["save_format"].append("msh")
        
    create_square_with_hole_dataset(
        num_samples=specs_data["SDFData"]["num_samples"],
        min_hole=specs_data["SDFData"]["min_hole"],
        max_hole=specs_data["SDFData"]["max_hole"],
        min_radius=specs_data["SDFData"]["min_radius"],
        path=full_path,
    )
    print(f"Generated {specs_data['SDFData']['num_samples']} samples of square meshes with circular holes in '{full_path}' directory.")

    if "npz" in specs_data["SDFData"]["save_format"]:
        if not os.path.exists(os.path.join(full_path, "npz/")):
            os.makedirs(os.path.join(full_path, "npz/"))

        samples_per_scene = specs_data["SDFData"]["SamplesPerScene"]
        resolution = 1 / (np.sqrt(samples_per_scene) - 1)
        query_range = np.array([specs_data["SDFData"]["query_range"]["x"], specs_data["SDFData"]["query_range"]["y"], specs_data["SDFData"]["query_range"]["z"]])
        print(f"Computing SDF for {len(os.listdir(os.path.join(full_path, 'msh')))} meshes with samples {samples_per_scene} per scene between this range {query_range}")

        for file in os.listdir(os.path.join(full_path, "msh")):
            grid_points, sdf, X, Y, Z = compute_sdf_2d(
                os.path.join(full_path, "msh", file),
                query_range=query_range,
                resolution=resolution
            )

            data = np.concatenate(
                [grid_points, sdf.reshape(-1, 1)], axis=1
            )

            npz_path = os.path.join(full_path, "npz", file.replace('.msh', '') + ".npz")
            save_data_to_npz_file(data, filename=npz_path)
