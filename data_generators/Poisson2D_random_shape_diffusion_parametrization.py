# In this script, we generate data for a 2D Poisson problem with a coefficient equals to 1 and a parametrized right-hand side.
# RHS takes the coordinates and a coeff as input. ex: rhs = lambda x: sin(x[2] * x[0]) * sin(x[2] * x[1]) with x = (x[0], x[1], coeff)
# The script uses Firedrake for finite element methods.

try:
    from firedrake import *
except Exception:
    print("firedrake not imported")
    
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except Exception:
    print("torch not imported")

class Poisson2D_random_shape:
    def __init__(self, mesh , coeffs:list[float]=[1.0], rhs:callable=lambda x,y: sin(y * x[0]) * sin(y * x[1]), bc_val:float=0.0, save_sol = False):
        """
        Args:
            mesh (Mesh): Firedrake mesh
            coeffs (list[float], optional): Coefficients for the rhs. Defaults to [1.0].
            rhs (callable, optional): Right-hand side function. Defaults to lambda x,y: sin(y * x[0]) * sin(y * x[1]).
            bc_val (float, optional): Boundary condition value. Defaults to 0.0.
            save_sol (bool, optional): Save solution to file. Defaults to False.
        """

        self.mesh = mesh
        self.coeffs = coeffs
        self.rhs = rhs
        self.coords = mesh.coordinates.dat.data
        self.save_sol = save_sol
        self.bc_val = bc_val

    def create_data(self) -> list[tuple[float, np.ndarray]]:
        """Create data for the Poisson problem.
        This method solves the Poisson equation for each coefficient in self.coeffs and returns the solution.
        Returns:
            list[tuple[float, np.ndarray]]: List of tuples containing the coefficient and the solution, the coordinates are stored in self.coords.
        """
        to_ret = []
        for coeff in tqdm(self.coeffs):
            V = FunctionSpace(self.mesh, "CG", 1)

            # 3) Trial and test functions
            u = TrialFunction(V)
            v = TestFunction(V)

            # 4) Forcing f(x,y) = sin(pi*x)*sin(pi*y)
            #    Note: mesh is 2D embedded in R^3, so we ignore z (x[2])

            x = SpatialCoordinate(self.mesh)
            f = self.rhs(x, 1.0)
            fh = Function(V, name="RHS")
            fh.interpolate(f)
            # 5) Variational forms for the Poisson problem
            
            a = dot(grad(u), coeff * grad(v)) * dx
            L = f * v * dx

            # 6) Homogeneous Dirichlet BC on the entire outer boundary
            bc = DirichletBC(V, self.bc_val, "on_boundary")

            # 7) Solve for u
            uh = Function(V, name="Solution")
            solve(a == L, uh, bcs=bc)

            # 8) (Optional) Save to VTK for Paraview
            if self.save_sol:
                File(f"outputs/poisson_solution_{coeff}_{self.bc_val}.pvd").write(uh)
                File(f"outputs/rhs_{coeff}_{self.bc_val}.pvd").write(fh)


            uh_matrix = uh.dat.data
            #uh_matrix = np.zeros(len(self.coords))
            #for i, coord in enumerate(self.coords):
            #    uh_matrix[i] = uh.at(coord)

            to_ret.append((coeff, uh_matrix))
        return to_ret
    
def PDEDataGenerator(specs_data, args):
    """
    Generate data for the Poisson problem based on the specifications provided in specs_data_filename.
    
    Args:
        specs_data_filename (str): Path to the JSON file containing specifications.
        args (argparse.Namespace): Command line arguments including batch size.
    """
    import os
    import json
    import logging 
    import numpy as np
    import random
    from firedrake import *


    msh_filenames = os.listdir(os.path.join(specs_data["root_dir"], specs_data["dataset_name"],"msh"))
    msh_filenames = [f for f in msh_filenames if f.endswith(".msh")]

    for msh_filename in msh_filenames:
        path_to_save = os.path.join(specs_data["root_dir"], specs_data["dataset_name"], "PDEData", msh_filename[:-4])
        if os.path.exists(path_to_save):
            logging.warning(f"Folder {msh_filename[:-4]} already exists. Skipping generation for {msh_filename}.")
            continue
        mesh = Mesh(os.path.join(specs_data["root_dir"], specs_data["dataset_name"], "msh", msh_filename))

        coords = mesh.coordinates.dat.data
        rhs = eval(specs_data["PDEData"]["rhs"])

        coeffs = [1.0] if specs_data["PDEData"]["n_coeffs"] == 1 else [random.uniform(0.0, 2.0) for _ in range(specs_data["PDEData"]["n_coeffs"])]
        if args.batch_size == -1:
            batch_coeffs = [coeffs]
        else:
            if len(coeffs) < args.batch_size:
                raise ValueError("Batch size is larger than the number of coefficients.")
            batch_coeffs = [coeffs[i : i + args.batch_size] for i in range(0, len(coeffs), args.batch_size)]

        for coeff in batch_coeffs:
            data_gen = Poisson2D_random_shape(
                mesh,
                coeffs=coeff,
                rhs=rhs,
                bc_val=specs_data["PDEData"]["bc_val"],
            )
            data = data_gen.create_data()

            # Save the data to a file
            for i, (coeff, sol) in enumerate(data):
                path_to_save = os.path.join(specs_data["root_dir"], specs_data["dataset_name"], "PDEData", msh_filename[:-4])
                #c = 0
                #while os.path.exists(path_to_save):
                #    c += 1
                #    path_to_save = os.path.join(specs_data["root_dir"], specs_data["dataset_name"], "PDEData", f"{msh_filename[:-4]}_{c}")
                os.makedirs(path_to_save, exist_ok=True)
                files_path_to_save = os.listdir(path_to_save)
                f_name = f"coeff_{coeff:.4f}.npz"
                count = sum([s.count(f_name) for s in files_path_to_save])
                if count > 0:
                    f_name = f"coeff_{coeff:.4f}_{count}.npz"
                np.savez(os.path.join(path_to_save, f_name), rhs=np.array(coeff), sol=sol, coords=coords)
        
        logging.info(f"Data for mesh {msh_filename} generated and saved.")
                    

    train_msh_filenames = random.sample(msh_filenames, int(len(msh_filenames) * specs_data["Split"]["train_proportion"]))
    test_msh_filenames = [f for f in msh_filenames if f not in train_msh_filenames]

    return train_msh_filenames, test_msh_filenames

"""
class DeepONetDataset(Dataset):
    '''
    Custom PyTorch Dataset for DeepONet training.

    Assumes data is structured as triplets:
    (branch_input, trunk_input, output_value)
    corresponding to (u evaluated at sensors, y location, G(u)(y)).
    '''
    def __init__(self, branch_input_data, trunk_input_data, output_data, device='cpu'):
        '''
        Args:
            branch_input_data (np.ndarray): Data for the branch net input.
                                            Shape: (N, num_sensors)
            trunk_input_data (np.ndarray): Data for the trunk net input (y locations).
                                           Shape: (N, y_dim)
            output_data (np.ndarray): Target output data G(u)(y).
                                      Shape: (N, 1) or (N,)
        '''
        # Ensure data are numpy arrays
        if not isinstance(branch_input_data, np.ndarray):
            raise TypeError("branch_input_data must be a NumPy array.")
        if not isinstance(trunk_input_data, np.ndarray):
            raise TypeError("trunk_input_data must be a NumPy array.")
        if not isinstance(output_data, np.ndarray):
            raise TypeError("output_data must be a NumPy array.")

        # Basic dimension check
        if not (branch_input_data.shape[0] == trunk_input_data.shape[0] == output_data.shape[0]):
             raise ValueError("All input arrays must have the same number of samples (first dimension).")

        # Convert data to PyTorch tensors
        # It's often recommended to use float32 for neural network training
        self.branch_inputs = torch.tensor(branch_input_data, dtype=torch.float32, device=device)
        self.trunk_inputs = torch.tensor(trunk_input_data, dtype=torch.float32, device=device)
        self.outputs = torch.tensor(output_data, dtype=torch.float32, device=device)

        # Reshape output to be (N, 1) if it's (N,)
        if self.outputs.ndim == 1:
            self.outputs = self.outputs.unsqueeze(1)

        self.num_samples = branch_input_data.shape[0]

    def __len__(self):
        '''Returns the total number of samples in the dataset.'''
        return self.num_samples

    def __getitem__(self, idx):
        '''
        Fetches the sample (triplet) at the given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing (branch_input, trunk_input, output)
                   for the given index.
        '''
        if not 0 <= idx < self.num_samples:
            raise IndexError(f"Index {idx} is out of bounds for dataset with size {self.num_samples}")

        branch_input = self.branch_inputs[idx]
        trunk_input = self.trunk_inputs[idx]
        output = self.outputs[idx]

        return branch_input, trunk_input, output
"""

