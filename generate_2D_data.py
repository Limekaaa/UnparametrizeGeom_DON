import json 
import numpy as np
import random
import argparse
import os 
from data_generators.Poisson2D_random_shape import Poisson2D_random_shape

from firedrake import *

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Generate 2D Poisson data with random shapes.")
    parser.add_argument(
        "--experiment_directory",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include specifications file 'specs.json' and 'specs_data.json'."
    )

    parser.add_argument(
        "--batch_size",
        "-b",
        dest="batch_size",
        type=int,
        default=10,
        help="The batch size for data generation."
    )

    args = parser.parse_args()

    #specs_filename = f"{args.experiment_directory}/specs.json"
    specs_data_filename = f"{args.experiment_directory}/specs_data.json"

    """
    with open(specs_filename, "r") as f:
        specs = json.load(f)
    """
    with open(specs_data_filename, "r") as f:
        specs_data = json.load(f)


    msh_filenames = os.listdir(os.path.join(specs_data["root_dir"], specs_data["dataset_name"],"msh"))
    msh_filenames = [f for f in msh_filenames if f.endswith(".msh")]

    for msh_filename in msh_filenames:
        mesh = Mesh(os.path.join(specs_data["root_dir"], specs_data["dataset_name"], "msh", msh_filename))

        coords = mesh.coordinates.dat.data
        rhs = eval(specs_data["PDEData"]["rhs"])

        coeffs = [1.0] if specs_data["PDEData"]["n_coeffs"] == 1 else [random.uniform(0.0, 2.0) for _ in range(specs_data["PDEData"]["n_coeffs"])]
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
                c = 0
                while os.path.exists(path_to_save):
                    c += 1
                    path_to_save = os.path.join(specs_data["root_dir"], specs_data["dataset_name"], "PDEData", f"{msh_filename[:-4]}_{c}")
                os.makedirs(path_to_save, exist_ok=True)
                np.savez(os.path.join(path_to_save, f"coeff_{coeff:.4f}.npz"), rhs=np.array(coeff), sol=sol, coords=coords)

    train_msh_filenames = random.sample(msh_filenames, int(len(msh_filenames) * specs_data["Split"]["train_proportion"]))
    test_msh_filenames = [f for f in msh_filenames if f not in train_msh_filenames]

    os.makedirs(specs_data["Split"]["split_path"], exist_ok=True)

    train_dict = {specs_data["dataset_name"]: {"PDEData": [train_msh_filenames[i][:-4] for i in range(len(train_msh_filenames))]}}
    test_dict = {specs_data["dataset_name"]: {"PDEData": [test_msh_filenames[i][:-4] for i in range(len(test_msh_filenames))]}}

    json.dump(train_dict, open(os.path.join(specs_data["Split"]["split_path"], f"{specs_data["dataset_name"]}_train.json"), "w"), indent=4)
    json.dump(test_dict, open(os.path.join(specs_data["Split"]["split_path"], f"{specs_data["dataset_name"]}_test.json"), "w"), indent=4)
