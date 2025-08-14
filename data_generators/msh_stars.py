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

def create_stars_dataset(num_samples = 100, min_branches=3, max_branches=8, irregularity=0.5, min_inner_circle_prop = 0.15, max_inner_circle_prop = 0.6, mesh_size=0.02, path:str="", save_format: List[Literal["msh", "stl", "obj"]] = ["msh"]):
    """
    Create a dataset of star shapes with a specified number of points.
    
    Args:
        num_samples (int): Number of star shapes to generate.
        min_branches (int): Minimum number of branches for the star shapes.
        max_branches (int): Maximum number of branches for the star shapes.
        irregularity (float): Irregularity factor for the star shapes. close to 0 for regular stars, higher values for more irregular shapes.
        min_inner_circle_prop (float): Minimum proportion of the inner circle radius.
        max_inner_circle_prop (float): Maximum proportion of the inner circle radius.
        mesh_size (float): Mesh size for the generated shapes.
        path (str): Path to save the data.
        save_format (List[Literal["msh", "stl", "obj"]]): File formats to save the generated shapes.
    """
    max_branches += 1  # to include max_branches in the range

    if not os.path.exists(path):
        os.makedirs(path)

    if min_branches < 3 or max_branches < min_branches:
        raise ValueError("Invalid number of branches specified. Minimum branches must be at least 3 and max branches must be greater than or equal to min branches.")

    shapes = []

    rand_inner_circle = np.random.uniform(min_inner_circle_prop, max_inner_circle_prop, num_samples)

    for i in range(num_samples):
        #n_branches = np.random.randint(3, 8)
        n_branches = min_branches+i//(num_samples//(max_branches-min_branches))
        angles = np.array([np.random.normal(2*np.pi*j / n_branches, irregularity/n_branches) for j in range(n_branches)])

        for j in range(len(angles)):
            while angles[j] > 2 * np.pi or angles[j] < 0:
                angles[j] = 2*np.pi + angles[j] if angles[j] < 0 else angles[j] - 2 * np.pi
                
                
            
        angles = np.sort(angles)
        angles2 = np.array([angles[i] + (angles[i+1] - angles[i]) / 2 for i in range(len(angles)-1)] + [angles[-1] + (angles[-1] - angles[-2]) / 2])
        angles = np.sort(np.concatenate((angles2, angles)))

        shapes.append(angles)

    #shapes = [np.append(s, s[0]) for s in shapes]

    for idx, shape in tqdm(enumerate(shapes), desc="Generating polygon meshes", unit="shape"):
        """
        gmsh.initialize()
        gmsh.model.add("polygon")

        square_size = 1.0
        square = gmsh.model.occ.addRectangle(0, 0, 0, square_size, square_size)
        """
        #To complete________________________________________________________________
        n_sides = len(shape) #- 1

        #print(n_sides)
        
        gmsh.initialize()
        gmsh.model.add("star")

        Xs = np.cos(shapes[idx])
        Xs = np.array([Xs[i] if i%2 == 0 else rand_inner_circle[idx]*Xs[i] for i in range(len(Xs))])/2+0.5

        Ys = np.sin(shapes[idx])
        Ys = np.array([Ys[i] if i%2 == 0 else rand_inner_circle[idx]*Ys[i] for i in range(len(Ys))])/2+0.5

        pts = [(Xs[i], Ys[i], 0) for i in range(len(Xs))] 

        # Add points to model
        point_tags = [gmsh.model.occ.addPoint(x, y, z) for x, y, z in pts]
        # Close the polygon
        point_tags.append(point_tags[0])

        # Create lines between consecutive points
        line_tags = [gmsh.model.occ.addLine(point_tags[i], point_tags[i + 1]) for i in range(len(point_tags)-1)]

        # Create curve loop and surface
        loop = gmsh.model.occ.addCurveLoop(line_tags)
        surf = gmsh.model.occ.addPlaneSurface([loop])
        gmsh.model.occ.synchronize()

        # Set mesh size and generate
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
        gmsh.model.mesh.generate(2)


        #______________________________________________________________________________

        filename = f"{int((n_sides)/2)}_branches_star_{idx%(num_samples//(max_branches-min_branches))}"

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

        #gmsh.fltk.run()
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

    create_stars_dataset(
        num_samples=specs_data["SDFData"]["num_samples"],
        min_branches=specs_data["SDFData"]["min_branches"],
        max_branches=specs_data["SDFData"]["max_branches"],
        irregularity=specs_data["SDFData"]["irregularity"],
        mesh_size=specs_data["SDFData"]["mesh_size"],
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