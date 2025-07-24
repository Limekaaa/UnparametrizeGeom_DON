

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
        
        try:
            n_shapes = ws.specs["DeepONet"]["ShapesToLoad"]
        except:
            n_shapes = -1
        
        try:
            n_eq_per_shape = ws.specs["DeepONet"]["EqPerShape"]
        except:
            n_eq_per_shape = -1

        if n_shapes == 0 or n_eq_per_shape == 0:
            raise ValueError("Number of shapes or equations per shape cannot be zero. At least one shape and one equation per shape is required.")
        
        if n_shapes != -1:
            unique_shapes = list(set(self.shapes_names))
            if n_shapes > len(unique_shapes):
                logging.warning(
                    "Requested number of shapes ({}) is greater than available shapes ({})".format(
                        n_shapes, len(unique_shapes)
                    )
                )
                n_shapes = len(unique_shapes)
            
            selected_shapes = random.sample(unique_shapes, n_shapes)
            self.npyfiles = [self.npyfiles[i] for i in range(len(self.npyfiles)) if self.shapes_names[i] in selected_shapes]
            self.shapes_names = [os.path.split(os.path.split(path)[0])[1] for path in self.npyfiles]

        if n_eq_per_shape != -1:
            new_npyfiles = []
            new_shapes_names = []
            unique_shapes = list(set(self.shapes_names))
            for shape_name in unique_shapes:
                shape_files = [self.npyfiles[i] for i in range(len(self.npyfiles)) if self.shapes_names[i] == shape_name]
                if len(shape_files) < n_eq_per_shape:
                    logging.warning(
                        "Requested number of equations per shape ({}) is greater than available equations ({}) for shape {}".format(
                            n_eq_per_shape, len(shape_files), shape_name
                        )
                    )
                    n_eq_per_shape = len(shape_files)
                
                selected_files = random.sample(shape_files, n_eq_per_shape)
                new_npyfiles.extend(selected_files)
                new_shapes_names.extend([shape_name] * n_eq_per_shape)

            self.npyfiles = new_npyfiles
            self.shapes_names = new_shapes_names

        try:
            latent_vectors_subfolder = ws.specs_data["LatentVectors"]["folder_name"]
            if latent_vectors_subfolder is None or latent_vectors_subfolder == "":
                latent_vectors_subfolder = ws.specs["experiment_name"]
        except:
            latent_vectors_subfolder = ws.specs["experiment_name"]
        
        self.lat_vec_dir = os.path.join(
            ws.specs["DataSource"],
            ws.specs_data["dataset_name"],
            ws.latent_vectors_folder,
            latent_vectors_subfolder,
            ws.specs["DeepSDFDecoder"]["LatentVectors"]["checkpoint"],
        )

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
                sol = torch.from_numpy(data["sol"]).reshape(-1, 1)

                lat_vectors = torch.load(os.path.join(self.lat_vec_dir, self.shapes_names[i] + ".pth") , weights_only=False)
                #lat_vectors = torch.load(os.path.join(os.path.join(ws.experiment_folder, ws.specs["experiment_name"],ws.deep_sdf_folder, ws.latent_vectors_folder, ws.deepsdf_model, ws.split, self.shapes_names[i] + ".pth")))
                lat_vectors = lat_vectors.repeat(sol.shape[0], 1)
                rhs = rhs.repeat(sol.shape[0], 1)
                """
                if self.subsample is not None:
                    rand_idxs = torch.randperm(rhs.shape[0])[:self.subsample]
                    rhs = rhs[rand_idxs]
                    coords = coords[rand_idxs]
                    sol = sol[rand_idxs].reshape(-1, 1)
                    lat_vectors = lat_vectors[rand_idxs]
                """
                self.loaded_data.append((lat_vectors.cuda(), coords.cuda(), rhs.cuda(), sol.cuda()))

    def __len__(self):
        return len(self.npyfiles)
    
    def __getitem__(self, idx):
        if self.load_ram:
            
            if self.subsample is not None:
                lat_vectors, coords, rhs, sol = self.loaded_data[idx]
                rand_idxs = torch.randperm(rhs.shape[0])[:self.subsample]
                rhs = rhs[rand_idxs]
                coords = coords[rand_idxs]
                sol = sol[rand_idxs]
                lat_vectors = lat_vectors[rand_idxs].clone().detach()

            else:
                lat_vectors, coords, rhs, sol = self.loaded_data[idx]
            return torch.cat((lat_vectors, coords, rhs, sol), dim=1).float(), idx
        else:
            npyfile = self.npyfiles[idx]
            data = np.load(os.path.join(self.data_source, npyfile))
            rhs = torch.from_numpy(data["rhs"])
            coords = torch.from_numpy(data["coords"])
            sol = torch.from_numpy(data["sol"]).reshape(-1, 1)
            rhs = rhs.repeat(sol.shape[0], 1)

            lat_vectors = torch.load(os.path.join(self.lat_vec_dir, self.shapes_names[idx] + ".pth") , weights_only=False)
            #lat_vectors = torch.load(os.path.join(ws.experiment_folder, ws.specs["experiment_name"],ws.deep_sdf_folder, ws.latent_vectors_folder, ws.deepsdf_model, ws.split, self.shapes_names[idx] + ".pth"), weights_only=False)
            
            lat_vectors = lat_vectors.repeat(sol.shape[0], 1)#.to("cpu")

            if self.subsample is not None:
                rand_idxs = torch.randperm(rhs.shape[0])[:self.subsample]
                rhs = rhs[rand_idxs]
                coords = coords[rand_idxs]
                sol = sol[rand_idxs]
                lat_vectors = lat_vectors[rand_idxs]

            #print(lat_vectors.get_device(), coords.get_device(), rhs.get_device(), sol.get_device())

            #print(torch.cat((lat_vectors, coords, rhs, sol), dim=1).float().shape)

            return torch.cat((lat_vectors.cuda(), coords.cuda(), rhs.cuda(), sol.cuda()), dim=1).float(), idx