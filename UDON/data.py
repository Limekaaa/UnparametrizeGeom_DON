

import glob
import logging
import numpy as np
import os
import random
import torch
import torch.utils.data

import UDON.workspace as ws

# From DeepSDF repo ______________________________________________________________________________________

def unpack_sdf_samples_from_ram(data, subsample=None):
    if subsample is None:
        return data
    pos_tensor = data[0]
    neg_tensor = data[1]

    if subsample == pos_tensor.shape[0] + neg_tensor.shape[0]:
        # If subsample is equal to the total number of samples, return all samples
        return torch.cat([pos_tensor, neg_tensor], 0).float()
    # split the sample into half
    half = int(subsample / 2)

    if pos_tensor.shape[0] < half:
        sample_pos = pos_tensor
        to_complete = half - pos_tensor.shape[0]
        random_neg = (torch.rand(to_complete+half) * neg_tensor.shape[0]).long()

        sample_neg = torch.index_select(neg_tensor, 0, random_neg)#[:to_complete+half]

    elif neg_tensor.shape[0] < half:
        sample_neg = neg_tensor
        to_complete = half - neg_tensor.shape[0]
        random_pos = (torch.rand(to_complete+half) * pos_tensor.shape[0]).long()

        sample_pos = torch.index_select(pos_tensor, 0, random_pos)

    else:
        random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
        random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

        sample_pos = torch.index_select(pos_tensor, 0, random_pos)
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)

    samples = torch.cat([sample_pos, sample_neg], 0).float()

    return samples

def get_instance_filenames(data_source, split):
    npzfiles = []
    for dataset in split:
        for class_name in split[dataset]:
            for instance_name in split[dataset][class_name]:
                instance_filename = os.path.join(
                    dataset, class_name, instance_name + ".npz"
                )
                if not os.path.isfile(
                    os.path.join(data_source, instance_filename)
                ):
                    # raise RuntimeError(
                    #     'Requested non-existent file "' + instance_filename + "'"
                    # )
                    logging.warning(
                        "Requested non-existent file '{}'".format(instance_filename)
                    )
                npzfiles += [instance_filename]
    return npzfiles

def read_sdf_samples_into_ram(filename):
    npz = np.load(filename)
    pos_tensor = torch.from_numpy(npz["pos"])
    neg_tensor = torch.from_numpy(npz["neg"])

    return [pos_tensor, neg_tensor]


# From DeepSDF repo ______________________________________________________________________________________

def get_PDE_instance_filenames(data_source, split):
    npzfiles = []
    for dataset in split:
        for class_name in split[dataset]:
            for instance_name in split[dataset][class_name]:
                npy_files = os.listdir(os.path.join(data_source, dataset, class_name, instance_name))
                for file in npy_files:
                    instance_filename = os.path.join(
                        dataset, class_name, instance_name, file
                    )
                    if not os.path.isfile(
                        os.path.join(data_source, instance_filename)
                    ):
                        # raise RuntimeError(
                        #     'Requested non-existent file "' + instance_filename + "'"
                        # )
                        logging.warning(
                            "Requested non-existent file '{}'".format(instance_filename)
                        )
                    npzfiles += [instance_filename]
    return npzfiles


class PDESamples(torch.utils.data.Dataset):
    def __init__(self, data_source, split, subsample, load_ram=False):
        self.subsample = subsample

        self.data_source = data_source
        self.npyfiles = get_PDE_instance_filenames(data_source, split)
        self.shapes_names = [os.path.split(os.path.split(path)[0])[1] for path in self.npyfiles]


        logging.debug(
            "using "
            + str(len(self.npyfiles))
            + " PDE instances from data source "
            + data_source
        )

        self.load_ram = load_ram
        if self.load_ram:
            self.loaded_data = []
            for i, npyfile in enumerate(self.npyfiles):
                data = np.load(os.path.join(data_source, npyfile))
                rhs = torch.from_numpy(data["rhs"])
                coords = torch.from_numpy(data["coords"])
                sol = torch.from_numpy(data["sol"])
                lat_vectors = torch.load(os.path.join(ws.experiment_folder, ws.deep_sdf_folder, ws.latent_vectors_folder, self.shapes_names[i] + ".pth"))
                rhs = rhs.repeat(sol.shape[0], 1)

                if self.subsample is not None:
                    rand_idxs = torch.randperm(rhs.shape[0])[:self.subsample]
                    rhs = rhs[rand_idxs]
                    coords = coords[rand_idxs]
                    sol = sol[rand_idxs]

                self.loaded_data.append((lat_vectors, coords, rhs, sol))

    def __len__(self):
        return len(self.npyfiles)
    
    def __getitem__(self, idx):
        if self.load_ram:
            if self.subsample is not None:
                rhs, coords, sol = self.loaded_data[idx]
                rand_idxs = torch.randperm(rhs.shape[0])[:self.subsample]
                rhs = rhs[rand_idxs]
                coords = coords[rand_idxs]
                sol = sol[rand_idxs]
                lat_vectors = lat_vectors[rand_idxs]

            else:
                rhs, coords, sol = self.loaded_data[idx]
            return torch.cat((lat_vectors, coords, rhs, sol), dim=1).float(), idx
        else:
            npyfile = self.npyfiles[idx]
            data = np.load(os.path.join(self.data_source, npyfile))
            rhs = torch.from_numpy(data["rhs"])
            coords = torch.from_numpy(data["coords"])
            sol = torch.from_numpy(data["sol"]).reshape(-1, 1)
            rhs = rhs.repeat(sol.shape[0], 1)
            lat_vectors = torch.load(os.path.join(ws.experiment_folder, ws.specs["experiment_name"],ws.deep_sdf_folder, ws.latent_vectors_folder, ws.deepsdf_model, ws.split, self.shapes_names[idx] + ".pth"), weights_only=False)
            lat_vectors = lat_vectors.repeat(sol.shape[0], 1).to("cpu")

            if self.subsample is not None:
                rand_idxs = torch.randperm(rhs.shape[0])[:self.subsample]
                rhs = rhs[rand_idxs]
                coords = coords[rand_idxs]
                sol = sol[rand_idxs]
                lat_vectors = lat_vectors[rand_idxs]

            #print(lat_vectors.get_device(), coords.get_device(), rhs.get_device(), sol.get_device())

            #print(torch.cat((lat_vectors, coords, rhs, sol), dim=1).float().shape)
                
            return torch.cat((lat_vectors, coords, rhs, sol), dim=1).float(), idx