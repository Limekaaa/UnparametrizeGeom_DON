import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.tri as tri

import json
import os

from networks.MetaLearningDON import DeepONet
import UDON
import UDON.workspace as ws


def get_sol(pde_sample, idx):
    sol = pde_sample[idx][0][:, -1]
    return sol

def get_rhs(pde_sample, idx):
    coords = pde_sample[idx][0][:, -2]
    return coords

def get_trunk_inputs(pde_sample, idx):
    trunk_inputs = pde_sample[idx][0][:, :-2]
    return trunk_inputs

def get_coords(pde_samples, idx):
    coords = get_trunk_inputs(pde_samples, idx)[:, -2:]
    return coords

def get_lat_vec(pde_samples, idx):
    trunk_inputs = get_trunk_inputs(pde_samples, idx)
    lat_vec = trunk_inputs[:, :-2]
    return lat_vec

def get_preds(pde_samples, deeponet,idx):
    pde_rhs = get_rhs(pde_samples, idx).unsqueeze(1).cuda()
    pde_trunk_inputs = get_trunk_inputs(pde_samples, idx).cuda()
    pde_data = get_coords(pde_samples, idx).cuda()
    pde_gt = get_sol(pde_samples, idx).unsqueeze(1).cuda()
    
    pde_trunk_inputs = pde_trunk_inputs.unsqueeze(0)  # Add batch dimension
    deeponet_out = deeponet(pde_rhs, pde_trunk_inputs)

    return deeponet_out, pde_gt, pde_data


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Train a DeepONet model for PDEs")

    parser.add_argument(
        "--experiment",
        "-e", 
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include "
        + "experiment specifications in 'specs.json', and logging will be "
        + "done in this directory as well.",
    )

    parser.add_argument(
        "--checkpoint_sdf",
        "-cs",
        dest="checkpoint_sdf",
        default="latest",
        help="The checkpoint of DeepSDF model to load. If 'latest', the most recent checkpoint will be used."
    )

    parser.add_argument(
        "--checkpoint_don",
        "-cd",
        dest="checkpoint_don",
        default="latest",
        help="The checkpoint of DeepONet model to load. If 'latest', the most recent checkpoint will be used."
    )

    parser.add_argument(
        "--n_reconstructions",
        type=int,
        default=1,
        help="The number of reconstructions to perform."
    )

    parser.add_argument(
        "--split",
        "-s",
        dest="split",
        default="test",
        help="The data split to use (train, test)."
    )

    args = parser.parse_args()
    experiment_directory = args.experiment_directory

    specs = json.load(open(experiment_directory + "/specs.json"))
    specs_data = json.load(open(experiment_directory + "/specs_data.json"))

    ws.specs = specs
    ws.specs_data = specs_data

    ws.split = args.split
    ws.deeponet_model = args.checkpoint_don
    ws.deepsdf_model = args.checkpoint_sdf

    data_source = specs["DataSource"]
    train_split_file = specs["PDETrainSplit"]
    test_split_file = specs["PDETestSplit"]

    arch = __import__(
        "networks." + specs["DeepONet"]["NetworkArch"], fromlist=["DeepONet"]
    )

    deeponet = arch.DeepONet(**specs["DeepONet"]["NetworkSpecs"])
    model_path = os.path.join(experiment_directory, ws.deep_o_net_folder, ws.parameters_folder, ws.deeponet_model + ".pth")
    deeponet.load_state_dict(torch.load(model_path)["model_state_dict"])
    deeponet.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deeponet.to(device)

    if args.split == "train":
        with open(train_split_file, "r") as f:
            split = json.load(f)
    elif args.split == "test":
        with open(test_split_file, "r") as f:
            split = json.load(f)
    else:
        raise ValueError("Split must be either 'train' or 'test'.")

    pde_samples = UDON.data.PDESamples(
        data_source,
        split,
    )

    idxs = np.random.choice(len(pde_samples), args.n_reconstructions, replace=False)

    for idx in idxs:
        print(f"Processing sample {idx}...")

        # Get predictions, ground truth, and coordinates
        preds, gt, coords = get_preds(pde_samples, deeponet, idx)

        # Convert tensors to numpy arrays and flatten values
        coords_np = coords.detach().cpu().numpy()
        preds_np = preds.detach().cpu().numpy().flatten()
        gt_np = gt.detach().cpu().numpy().flatten()

        # Create a triangulation from the coordinates
        triang = tri.Triangulation(coords_np[:, 0], coords_np[:, 1])

        # Compute common color scale limits
        vmin = min(preds_np.min(), gt_np.min())
        vmax = max(preds_np.max(), gt_np.max())

        # Plot predictions
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        contour1 = plt.tricontourf(triang, preds_np, levels=100, cmap='viridis', vmin=vmin, vmax=vmax)
        plt.colorbar(contour1, label='Prediction')
        plt.title("Predictions")
        plt.xlabel("x")
        plt.ylabel("y")

        # Plot ground truth
        plt.subplot(1, 2, 2)
        contour2 = plt.tricontourf(triang, gt_np, levels=100, cmap='viridis', vmin=vmin, vmax=vmax)
        plt.colorbar(contour2, label='Ground Truth')
        plt.title("Ground Truth")
        plt.xlabel("x")
        plt.ylabel("y")

        plt.tight_layout()
        os.makedirs(os.path.join(experiment_directory, ws.deep_o_net_folder, ws.reconstruction_folder, args.split), exist_ok=True)
        plt.savefig(os.path.join(experiment_directory, ws.deep_o_net_folder, ws.reconstruction_folder, args.split, f"reconstruction_{pde_samples.shapes_names[idx]}.png"))
        plt.close()

        contour1 = plt.tricontourf(triang, abs(preds_np - gt_np), levels=100, cmap='viridis')#, vmin=vmin, vmax=vmax)
        plt.colorbar(contour1, label='absolute error')
        plt.title("Absolute error")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.savefig(os.path.join(experiment_directory, ws.deep_o_net_folder, ws.reconstruction_folder, args.split, f"reconstruction_error_{pde_samples.shapes_names[idx]}.png"))
        plt.close()

