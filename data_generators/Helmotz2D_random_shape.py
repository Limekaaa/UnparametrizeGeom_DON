try:
    from firedrake import *
except Exception:
    print("firedrake not imported")

import numpy as np
import os
import logging
import random
from tqdm import tqdm

class Helmholtz2D_random_shape:
    def __init__(
        self,
        mesh,
        n_source_term:int = 1,
        ks: list[float] = [10.0],
        rhs: callable = lambda x, theta: exp(-50*((x[0]-theta[0])**2 + (x[1]-theta[1])**2)),
        bc_val: float = 0.0,
        save_sol: bool = False,
        solver_parameters: dict | None = None,
    ):
        """
        Helmholtz dataset generator for arbitrary 2D meshes.

        Args:
            mesh (Mesh): Firedrake mesh
            ks (list[float], optional): list of wave numbers k. Defaults to [10.0].
            rhs (callable, optional): RHS constructor: rhs(x, k) must return a UFL expression
                                      given SpatialCoordinate x and parameter k.
                                      Defaults to a centered Gaussian.
            bc_val (float, optional): Dirichlet BC value on boundary. Defaults to 0.0.
            save_sol (bool, optional): if True write .pvd files of solution and rhs. Defaults False.
            solver_parameters (dict | None): Firedrake solver parameters for linear solve.
                                             If None, a reasonable default is used.
        """
        self.mesh = mesh
        self.ks = ks if isinstance(ks, (list, tuple)) else [ks]
        self.rhs = rhs
        self.coords = mesh.coordinates.dat.data
        self.save_sol = save_sol
        self.bc_val = bc_val
        self.n_source_term = n_source_term

        # default solver params (Helmholtz is indefinite -> GMRES preconditioned by LU is a safe start)
        if solver_parameters is None:
            self.solver_parameters = {
                "ksp_type": "gmres",
                "pc_type": "lu",
                "ksp_rtol": 1e-8,
                "ksp_atol": 1e-12,
                "ksp_max_it": 10000,
            }
        else:
            self.solver_parameters = solver_parameters

    def create_data(self) -> list[tuple[float, np.ndarray]]:
        """
        Solve Helmholtz for each k in self.ks and return list of tuples (k, solution_values).
        Coordinates are available in self.coords (matching solution vector indexing).
        """
        results = []
        source_terms_idx = np.random.randint(0, len(self.coords), size=self.n_source_term)

        for idx in tqdm(source_terms_idx, desc="Helmholtz source terms"):
            theta = self.coords[idx]
            for k in self.ks:
                V = FunctionSpace(self.mesh, "CG", 1)

                # Trial/test
                u = TrialFunction(V)
                v = TestFunction(V)

                x = SpatialCoordinate(self.mesh)
                # rhs must return a UFL expression given x and theta
                f_ufl = self.rhs(x, theta)

                fh = Function(V, name="RHS")
                fh.interpolate(f_ufl)

                # Helmholtz variational form: (grad u, grad v) - k^2 (u, v) = (f, v)
                a = (dot(grad(u), grad(v)) - (Constant(k) ** 2) * inner(u, v)) * dx
                L = inner(fh, v) * dx

                # BC - homogeneous Dirichlet on outer boundary (customize if needed)
                bc = DirichletBC(V, Constant(self.bc_val), "on_boundary")

                uh = Function(V, name="Solution")

                # Use Firedrake's linear solver with given parameters.
                # solve(a == L, uh, bcs=bc, solver_parameters=...) works for linear systems.
                try:
                    solve(a == L, uh, bcs=bc, solver_parameters=self.solver_parameters)
                except Exception as e:
                    logging.warning(f"Direct solve failed for k={k}: {e}. Trying fallback solve without solver params.")
                    # fallback (let Firedrake pick defaults)
                    solve(a == L, uh, bcs=bc)

                if self.save_sol:
                    # optional output for inspection
                    outdir = "outputs"
                    os.makedirs(outdir, exist_ok=True)
                    File(os.path.join(outdir, f"helmholtz_solution_k_{k:.4f}.pvd")).write(uh)
                    File(os.path.join(outdir, f"helmholtz_rhs_k_{k:.4f}.pvd")).write(fh)

                # Extract nodal values (Function.dat.data matches mesh.coordinates ordering)
                uh_vals = uh.dat.data.copy()
                results.append((k, uh_vals))

        return results


def PDEDataGenerator(specs_data, args):
    """
    Generate data for Helmholtz problems on a folder of meshes.

    specs_data expected keys (example):
      "root_dir": "...",
      "dataset_name": "...",
      "PDEData": {
          "ks": "list or expression or 'random'",    # optional
          "n_ks": 10,                                # used when generating random ks
          "rhs": "lambda x,k: ...",                  # string that evals to a callable
          "bc_val": 0.0
      },
      "Split": {"train_proportion": 0.8}

    args: must include `batch_size` (same semantics as your Poisson generator).
    """
    msh_dir = os.path.join(specs_data["root_dir"], specs_data["dataset_name"], "msh")
    msh_filenames = [f for f in os.listdir(msh_dir) if f.endswith(".msh")]

    for msh_filename in msh_filenames:
        mesh_name = msh_filename[:-4]
        path_to_save = os.path.join(specs_data["root_dir"], specs_data["dataset_name"], "PDEData", mesh_name)
        if os.path.exists(path_to_save):
            logging.warning(f"Folder {mesh_name} already exists. Skipping generation for {msh_filename}.")
            continue

        mesh = Mesh(os.path.join(msh_dir, msh_filename))
        coords = mesh.coordinates.dat.data
        
        # build rhs callable
        rhs_callable = eval(specs_data["PDEData"]["rhs"]) if "rhs" in specs_data["PDEData"] else (lambda x, k: exp(-50*((x[0]-0.5)**2 + (x[1]-0.5)**2)))

        # build ks
        if "ks" in specs_data["PDEData"]:
            ks = specs_data["PDEData"]["ks"]
            # allow stringified list, single float, or 'random'
            if isinstance(ks, str):
                if ks.strip().lower() == "random":
                    n_ks = specs_data["PDEData"].get("n_ks", 10)
                    ks_list = [random.uniform(0.0, specs_data["PDEData"].get("k_max", 20.0)) for _ in range(n_ks)]
                else:
                    ks_list = eval(ks)
            else:
                ks_list = ks if isinstance(ks, (list, tuple)) else [ks]
        else:
            # default single k = 10
            ks_list = [10.0]

        # batching like your Poisson generator
        if args.batch_size == -1:
            batch_ks = [ks_list]
        else:
            if len(ks_list) < args.batch_size:
                raise ValueError("Batch size is larger than the number of ks.")
            batch_ks = [ks_list[i : i + args.batch_size] for i in range(0, len(ks_list), args.batch_size)]

        for batch in batch_ks:
            data_gen = Helmholtz2D_random_shape(
                mesh,
                ks=batch,
                rhs=rhs_callable,
                bc_val=specs_data["PDEData"].get("bc_val", 0.0),
                save_sol=specs_data["PDEData"].get("save_sol", False),
            )
            data = data_gen.create_data()

            # Save the data to a file
            os.makedirs(path_to_save, exist_ok=True)
            files_path_to_save = os.listdir(path_to_save)

            for i, (k_val, sol) in enumerate(data):
                f_name = f"k_{k_val:.4f}.npz"
                count = sum([s.count(f_name) for s in files_path_to_save])
                if count > 0:
                    f_name = f"k_{k_val:.4f}_{count}.npz"
                np.savez(os.path.join(path_to_save, f_name), rhs=np.array(k_val), sol=sol, coords=coords)

        logging.info(f"Helmholtz Data for mesh {msh_filename} generated and saved.")

    # create train/test mesh splits
    train_msh_filenames = random.sample(msh_filenames, int(len(msh_filenames) * specs_data["Split"]["train_proportion"]))
    test_msh_filenames = [f for f in msh_filenames if f not in train_msh_filenames]

    return train_msh_filenames, test_msh_filenames
