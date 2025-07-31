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

def create_polygons_dataset(num_samples = 100, min_sides=3, max_sides=8, irregularity=1, mesh_size=0.05, path:str="", save_format: List[Literal["msh", "stl", "obj"]] = ["msh"]):
    """
    Create a dataset of polygons with a specified number of sides.
    
    Parameters:
    - num_samples: Number of samples to generate.
    - min_sides: Minimum number of sides for the polygon.
    - max_sides: Maximum number of sides for the polygon.
    - irregularity: Irregularity factor for the polygon shapes. close to 0 for regular polygons, higher values for more irregular shapes.
    """
    max_sides += 1  # to include max_sides in the range

    if not os.path.exists(path):
        os.makedirs(path)

    if min_sides < 3 or max_sides < min_sides:
        raise ValueError("Invalid number of sides specified. Minimum sides must be at least 3 and max sides must be greater than or equal to min sides.")
    
    shapes = []

    for i in range(num_samples):#, desc="Generating polygon characteristics", unit="sample"):
        n_sides = min_sides + i//(num_samples // (max_sides - min_sides - 1))
        #print(n_sides)
        angles = np.array([np.random.normal(2*np.pi*j / n_sides, irregularity/n_sides) for j in range(n_sides)])
        
        for j in range(len(angles)):
            while angles[j] > 2 * np.pi or angles[j] < 0:
                angles[j] = 2*np.pi + angles[j] if angles[j] < 0 else angles[j] - 2 * np.pi
                
        angles = np.sort(angles)
        
        shapes.append(angles)

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
        gmsh.model.add("polygon")

        # Compute polygon points centered in unit square
        radius = 0.5  # ensure stays inside [0,1]
        cx, cy = 0.5, 0.5
        pts = []
        for theta in shape:#[:-1]:
            x = cx + radius * np.cos(theta)
            y = cy + radius * np.sin(theta)
            pts.append((x, y, 0))

        # Add points to model
        point_tags = [gmsh.model.occ.addPoint(x, y, z) for x, y, z in pts]
        # Close the polygon
        point_tags.append(point_tags[0])

        # Create lines between consecutive points
        line_tags = [gmsh.model.occ.addLine(point_tags[i], point_tags[i + 1]) for i in range(n_sides)]

        # Create curve loop and surface
        loop = gmsh.model.occ.addCurveLoop(line_tags)
        surf = gmsh.model.occ.addPlaneSurface([loop])
        gmsh.model.occ.synchronize()

        # Set mesh size and generate
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
        gmsh.model.mesh.generate(2)


        #______________________________________________________________________________

        filename = f"{n_sides}_sides_polygon_{idx%(num_samples//(max_sides-min_sides))}"

        if "stl" in save_format:
            if not os.path.exists(os.path.join(path, "stl")):
                os.makedirs(os.path.join(path, "stl"))
            c = 0
            new_filename = filename
            while os.path.exists(os.path.join(path, "stl", f"{new_filename}.stl")):
                c += 1
                new_filename = f"{filename}_{c}"
            gmsh.write(os.path.join(path, "stl", f"{new_filename}.stl"))

        if "msh" in save_format:
            if not os.path.exists(os.path.join(path, "msh")):
                os.makedirs(os.path.join(path, "msh"))
            c = 0
            new_filename = filename
            while os.path.exists(os.path.join(path, "msh", f"{new_filename}.msh")):
                c += 1
                new_filename = f"{filename}_{c}"
            gmsh.write(os.path.join(path, "msh", f"{new_filename}.msh"))

        if "obj" in save_format:
            if "stl" not in save_format:
                print("Warning: 'obj' format requires 'stl' format to be saved as well. Saving as 'stl'.")
                if not os.path.exists(os.path.join(path, "stl")):
                    os.makedirs(os.path.join(path, "stl"))
                c = 0
                new_filename = filename
                while os.path.exists(os.path.join(path, "stl", f"{new_filename}.stl")):
                    c += 1
                    new_filename = f"{filename}_{c}"
                gmsh.write(os.path.join(path, "stl", f"{new_filename}.stl"))

            if not os.path.exists(os.path.join(path, "obj")):
                os.makedirs(os.path.join(path, "obj"))

            tri_mesh = trimesh.load(os.path.join(path, "stl", f"{new_filename}.stl"))

            trimesh.exchange.export.export_mesh(tri_mesh, os.path.join(path, "obj", f"{new_filename}.obj"), file_type='obj')

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

    create_polygons_dataset(
        num_samples=specs_data["SDFData"]["num_samples"],
        min_sides=specs_data["SDFData"]["min_sides"],
        max_sides=specs_data["SDFData"]["max_sides"],
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