

import glob
import logging
import numpy as np
import os
import random
import torch
import torch.utils.data



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
