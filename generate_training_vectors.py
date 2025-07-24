# Generate training vectors for DeepSDF from SDF samples.
# How to use:
# python generate_training_vectors.py -e path/to/experiment_dir -s train|test
# Save latent vectors in the specified experiment directory under DeepSDFDecoder/LatentVectors/<checkpoint>/<split>.

import math
import torch
import UDON
import logging
import json 
import numpy as np

import os
import random
import time

import UDON.workspace as ws


if __name__ == "__main__":
    import argparse

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    arg_parser = argparse.ArgumentParser(
        description="Generate training vectors for DeepONet from SDF samples."
    )
    
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )

    arg_parser.add_argument(
        "--split",
        "-s",
        dest="split",
        default="train",
        help="The data split to use (train, test)"
    )

    arg_parser.add_argument(
        "--log-level",
        "-l",
        dest="log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )

    args = arg_parser.parse_args()

    if args.log_level == "DEBUG":
        logging.basicConfig(level=logging.DEBUG)
    elif args.log_level == "INFO":
        logging.basicConfig(level=logging.INFO)
    elif args.log_level == "WARNING":
        logging.basicConfig(level=logging.WARNING)
    elif args.log_level == "ERROR":
        logging.basicConfig(level=logging.ERROR)
    elif args.log_level == "CRITICAL":
        logging.basicConfig(level=logging.CRITICAL)
    else:
        raise ValueError("Unknown log level: {}".format(args.log_level))

    specs = json.load(open(os.path.join(args.experiment_directory, "specs.json"), "r"))
    specs_data = json.load(open(os.path.join(args.experiment_directory, "specs_data.json"), "r"))

    try:
        latent_vectors_subfolder = specs_data["LatentVectors"]["folder_name"]
        if latent_vectors_subfolder is None or latent_vectors_subfolder == "":
            latent_vectors_subfolder = specs["experiment_name"]
    except:
        latent_vectors_subfolder = specs["experiment_name"]

    

    arch = __import__("networks." + specs["DeepSDFDecoder"]["NetworkArch"], fromlist=["Decoder"])

    latent_size = specs["DeepSDFDecoder"]["CodeLength"]
    decoder = arch.Decoder(
        latent_size=latent_size,
        **specs["DeepSDFDecoder"]["NetworkSpecs"]
    )

    decoder = torch.nn.DataParallel(decoder)

    saved_model_state = torch.load(
        os.path.join(
            args.experiment_directory,
            ws.deep_sdf_folder,
            ws.parameters_folder,
            f"{specs['DeepSDFDecoder']['LatentVectors']['checkpoint']}.pth",
        ),
        map_location=device
    )

    decoder.load_state_dict(saved_model_state["model_state_dict"])
    #decoder = decoder.module.cuda()
    decoder = decoder.module.to(device)

    if args.split == "train":
        split = specs["DeepSDFTrainSplit"]
    elif args.split == "test":
        split = specs["DeepSDFTestSplit"]
    else:
        raise ValueError("Unknown split: {}".format(args.split))
    
    with open(split, "r") as f:
        split = json.load(f)

    npz_filenames = UDON.data.get_instance_filenames(specs["DataSource"], split)

    random.shuffle(npz_filenames)

    logging.debug(decoder)
    """
    save_vec_dir = os.path.join(
        args.experiment_directory,
        ws.deep_sdf_folder,
        ws.latent_vectors_folder,
        specs["DeepSDFDecoder"]["LatentVectors"]["checkpoint"],
        args.split,
    )
    """
    save_vec_dir = os.path.join(
        specs["DataSource"],
        specs_data["dataset_name"],
        ws.latent_vectors_folder,
        latent_vectors_subfolder,
        specs["DeepSDFDecoder"]["LatentVectors"]["checkpoint"],
    )

    os.makedirs(save_vec_dir, exist_ok=True)

    logging.info("Generating latent vectors for {} samples".format(len(npz_filenames)))

    for ii, npz_filename in enumerate(npz_filenames):
        logging.info("Processing file {}/{}".format(ii + 1, len(npz_filenames)))

        npz_path = os.path.join(specs["DataSource"], npz_filename)

        data_sdf = UDON.data.read_sdf_samples_into_ram(npz_path)

        data_sdf[0] = data_sdf[0][torch.randperm(data_sdf[0].shape[0])]
        data_sdf[1] = data_sdf[1][torch.randperm(data_sdf[1].shape[0])]

        start = time.time()
        err, latent = UDON.inference.reconstruct(
            decoder=decoder,
            num_iterations=specs["DeepSDFDecoder"]["LatentVectors"]["num_iterations"],
            latent_size=latent_size,
            test_sdf=data_sdf,
            stat=specs["DeepSDFDecoder"]["CodeInitStdDev"],
            clamp_dist=UDON.inference.get_spec_with_default(specs["DeepSDFDecoder"], "ClampingDistance", 1.0) / math.sqrt(latent_size),
            lr=specs["DeepSDFDecoder"]["LatentVectors"]["lr"],
            l2reg=specs["DeepSDFDecoder"]["LatentVectors"]["CodeRegularization"],
        )
        end = time.time()

        logging.debug("reconstruct time: {}".format(time.time() - start))
        logging.debug("reconstruct error: {}".format(err))
        logging.debug(ii)

        logging.debug("latent: {}".format(latent.detach().cpu().numpy()))

        npz_name = os.path.basename(npz_filename)
        # save latent vector as pth file
        latent_vector_filename = os.path.join(
            save_vec_dir, npz_name.replace(".npz", ".pth")
        )

        torch.save(latent, latent_vector_filename)
