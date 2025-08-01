import json
import logging 
import numpy as np
import random
import argparse
import os 


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
        default=-1,
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

    data_gen = __import__("data_generators." + specs_data["PDEDataGenerator"], fromlist=["PDEDataGenerator"])

    train_msh_filenames, test_msh_filenames = data_gen.PDEDataGenerator(specs_data, args)


    os.makedirs(specs_data["Split"]["split_path"], exist_ok=True)

    train_dict = {specs_data["dataset_name"]: {"PDEData": [train_msh_filenames[i][:-4] for i in range(len(train_msh_filenames))]}}
    test_dict = {specs_data["dataset_name"]: {"PDEData": [test_msh_filenames[i][:-4] for i in range(len(test_msh_filenames))]}}

    json.dump(train_dict, open(os.path.join(specs_data["Split"]["split_path"], f"{specs_data['dataset_name']}_train.json"), "w"), indent=4)
    json.dump(test_dict, open(os.path.join(specs_data["Split"]["split_path"], f"{specs_data['dataset_name']}_test.json"), "w"), indent=4)

    train_dict = {specs_data["dataset_name"]: {"npz": [train_msh_filenames[i][:-4] for i in range(len(train_msh_filenames))]}}
    test_dict = {specs_data["dataset_name"]: {"npz": [test_msh_filenames[i][:-4] for i in range(len(test_msh_filenames))]}}

    json.dump(train_dict, open(os.path.join(specs_data["Split"]["split_path"], f"{specs_data['dataset_name']}_train_npz.json"), "w"), indent=4)
    json.dump(test_dict, open(os.path.join(specs_data["Split"]["split_path"], f"{specs_data['dataset_name']}_test_npz.json"), "w"), indent=4)