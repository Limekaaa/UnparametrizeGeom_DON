import torch
import torch.utils.data as data_utils
import signal
import sys
import os
import logging
import math
import json
import time

import numpy as np

import UDON
import UDON.workspace as ws

logging.basicConfig(level=logging.DEBUG)


def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except:
        return default

class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass


class ConstantLearningRateSchedule(LearningRateSchedule):
    def __init__(self, value):
        self.value = value

    def get_learning_rate(self, epoch):
        return self.value


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):

        return self.initial * (self.factor ** (epoch // self.interval))


class WarmupLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, warmed_up, length):
        self.initial = initial
        self.warmed_up = warmed_up
        self.length = length

    def get_learning_rate(self, epoch):
        if epoch > self.length:
            return self.warmed_up
        return self.initial + (self.warmed_up - self.initial) * epoch / self.length


def get_learning_rate_schedules(specs):

    schedule_specs = specs["LearningRateSchedule"]

    schedules = []

    for schedule_specs in schedule_specs:

        if schedule_specs["Type"] == "Step":
            schedules.append(
                StepLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Interval"],
                    schedule_specs["Factor"],
                )
            )
        elif schedule_specs["Type"] == "Warmup":
            schedules.append(
                WarmupLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Final"],
                    schedule_specs["Length"],
                )
            )
        elif schedule_specs["Type"] == "Constant":
            schedules.append(ConstantLearningRateSchedule(schedule_specs["Value"]))

        else:
            raise Exception(
                'no known learning rate schedule of type "{}"'.format(
                    schedule_specs["Type"]
                )
            )

    return schedules
"""

def get_learning_rate_schedules(specs):

    schedule_specs = specs["LearningRateSchedule"]

    schedules = []

    #for schedule_specs in schedule_specs:

    if schedule_specs["Type"] == "Step":
        schedules.append(
            StepLearningRateSchedule(
                schedule_specs["Initial"],
                schedule_specs["Interval"],
                schedule_specs["Factor"],
            )
        )
    elif schedule_specs["Type"] == "Warmup":
        schedules.append(
            WarmupLearningRateSchedule(
                schedule_specs["Initial"],
                schedule_specs["Final"],
                schedule_specs["Length"],
            )
        )
    elif schedule_specs["Type"] == "Constant":
        schedules.append(ConstantLearningRateSchedule(schedule_specs["Value"]))

    else:
        raise Exception(
            'no known learning rate schedule of type "{}"'.format(
                schedule_specs["Type"]
            )
        )

    return schedules
"""
def save_model(experiment_directory, filename, decoder, epoch):

    model_params_dir = os.path.join(
        experiment_directory, ws.deep_o_net_folder, ws.parameters_folder
    )

    if not os.path.exists(model_params_dir):
        os.makedirs(model_params_dir)

    torch.save(
        {"epoch": epoch, "model_state_dict": decoder.state_dict()},
        os.path.join(model_params_dir, filename),
    )


def save_optimizer(experiment_directory, filename, optimizer, epoch):

    optimizer_params_dir = os.path.join(
        experiment_directory, ws.deep_o_net_folder, ws.optimizer_parameters_folder
    )

    if not os.path.exists(optimizer_params_dir):
        os.makedirs(optimizer_params_dir)

    torch.save(
        {"epoch": epoch, "optimizer_state_dict": optimizer.state_dict()},
        os.path.join(optimizer_params_dir, filename),
    )


def load_optimizer(experiment_directory, filename, optimizer):

    full_filename = os.path.join(
        experiment_directory, ws.deep_o_net_folder, ws.optimizer_parameters_folder, filename
    )

    if not os.path.isfile(full_filename):
        raise Exception(
            'optimizer state dict "{}" does not exist'.format(full_filename)
        )

    data = torch.load(full_filename)

    optimizer.load_state_dict(data["optimizer_state_dict"])

    return data["epoch"]



def main_function(experiment_directory):
    logging.debug("running " + experiment_directory)
    specs = json.load(open(experiment_directory + "/specs.json"))
    specs_data = json.load(open(experiment_directory + "/specs_data.json"))

    ws.specs = specs
    ws.specs_data = specs_data
    ws.split = "train"
    ws.deeponet_model = "latest"
    ws.deepsdf_model = "latest"

    logging.info("Experiment description: " + specs["Description"])

    data_source = specs["DataSource"]
    train_split_file = specs["PDETrainSplit"]
    test_split_file = specs["PDETestSplit"]

    arch = __import__(
        "networks." + specs["DeepONet"]["NetworkArch"], fromlist=["DeepONet"]
    )

    logging.debug(specs["DeepONet"]["NetworkSpecs"])

    checkpoints = list(
        range(
            specs["DeepONet"]["SnapshotFrequency"],
            specs["DeepONet"]["NumEpochs"] + 1,
            specs["DeepONet"]["SnapshotFrequency"],
        )
    )

    for checkpoint in specs["DeepONet"]["AdditionalSnapshots"]:
        checkpoints.append(checkpoint)
    checkpoints.sort()

    lr_schedules = get_learning_rate_schedules(specs["DeepONet"])

    grad_clip = get_spec_with_default(specs["DeepONet"], "GradientClipNorm", None)
    if grad_clip is not None:
        logging.debug("clipping gradients to max norm {}".format(grad_clip))

    def save_latest(epoch):

        save_model(experiment_directory, "latest.pth", deeponet, epoch)
        save_optimizer(experiment_directory, "latest.pth", optimizer_all, epoch)

    def save_checkpoints(epoch):

        save_model(experiment_directory, str(epoch) + ".pth", deeponet, epoch)
        save_optimizer(experiment_directory, str(epoch) + ".pth", optimizer_all, epoch)

    def adjust_learning_rate(lr_schedules, optimizer, epoch):

        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)

    num_samp_per_scene = get_spec_with_default(specs["DeepONet"], "SamplesPerScene", None)
    scene_per_batch = specs["DeepONet"]["ScenesPerBatch"]

    deeponet = arch.DeepONet(**specs["DeepONet"]["NetworkSpecs"]).cuda()

    logging.info("training with {} GPU(s)".format(torch.cuda.device_count()))

    if torch.cuda.device_count() > 1:
        deeponet = torch.nn.DataParallel(deeponet)

    num_epochs = specs["DeepONet"]["NumEpochs"]
    log_frequency = get_spec_with_default(specs, "LogFrequency", 10)

    with open(train_split_file, "r") as f:
        train_split = json.load(f)

    with open(test_split_file, "r") as f:
        test_split = json.load(f)

    train_dataset = UDON.data.PDESamples(data_source, train_split, subsample=num_samp_per_scene, load_ram=get_spec_with_default(specs["DeepONet"], "LoadRam", False))

    if get_spec_with_default(specs["DeepONet"], "LoadRam", False):
        ws.split = "test"
        test_dataset = UDON.data.PDESamples(
            data_source, test_split, subsample=num_samp_per_scene, load_ram=True
        )
        ws.split = "train"
    else:
        test_dataset = UDON.data.PDESamples(
            data_source, test_split, subsample=num_samp_per_scene, load_ram=False
        )

    num_data_loader_threads = get_spec_with_default(specs["DeepONet"], "DataLoaderThreads", 1)
    logging.debug("loading data with {} threads".format(num_data_loader_threads))

    pde_loader = data_utils.DataLoader(
        train_dataset,
        batch_size=scene_per_batch,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=True,
    )

    logging.debug("torch num_threads: {}".format(torch.get_num_threads()))

    num_scenes = len(train_dataset)

    logging.info("There are {} scenes".format(num_scenes))

    logging.debug(deeponet)

    loss_l1 = torch.nn.L1Loss(reduction="sum")

    optimizer_all = torch.optim.Adam(
        [
            {
                "params": deeponet.parameters(),
                "lr": lr_schedules[0].get_learning_rate(0),
                "weight_decay": get_spec_with_default(specs["DeepONet"], "L2_regularization", 0.0),
            }
        ]
    )

    loss_log = []
    lr_log = []
    timing_log = []
    normalized_err_log = []
    normalized_test_err_log = []
    parameter_magnitude_log = None
    gradient_norm_log = []

    def normalized_error(pred, gt):
        return torch.norm(pred - gt) / torch.norm(gt)


    test_loss_log = []

    start_epoch = 1

    logging.info(
        "Number of deeponet parameters: {}".format(
            sum(p.data.nelement() for p in deeponet.parameters())
        )
    )

    for epoch in range(start_epoch, num_epochs + 1):

        start = time.time()
        logging.info("epoch {}...".format(epoch))
        deeponet.train()

        adjust_learning_rate(lr_schedules, optimizer_all, epoch)

        for pde_data, indices in pde_loader:

            #print("pde_data shape: ", pde_data.shape)
            
            pde_data = pde_data.cuda()

            pde_gt = pde_data[:, :,-1].unsqueeze(1)
            pde_rhs = pde_data[:, :, -2]
            pde_rhs = pde_rhs.reshape(pde_rhs.shape[0], pde_rhs.shape[1], 1)

            pde_trunk_inputs = pde_data[:, :, :-2]

            batch_loss = 0.0
            optimizer_all.zero_grad()       

            deeponet_out = deeponet(pde_rhs.cuda(), pde_trunk_inputs.cuda())

            """
            for pde_d, indice in zip(pde_data, indices):
                modulation = deeponet.Modulator(

            """
            #print(pde_rhs.shape, pde_trunk_inputs.shape, pde_data.shape, deeponet_out.shape, pde_gt.shape)

            #raise Exception("Debugging shapes")

            loss = loss_l1(deeponet_out, pde_gt.cuda()) / pde_data.shape[0]

            batch_loss += loss.item()
            loss.backward()
            
            loss_log.append(batch_loss)
            lr_log.append(optimizer_all.param_groups[0]["lr"])
            normalized_err = normalized_error(deeponet_out, pde_gt.cuda())
            normalized_err_log.append(normalized_err.item())

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(deeponet.parameters(), grad_clip)

            optimizer_all.step()

        end = time.time()

        total_norm = 0
        for p in deeponet.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        gradient_norm_log.append(total_norm)

        timing_log.append(end - start)

        if parameter_magnitude_log is None:
            parameter_magnitude_log = {
                name: [torch.norm(param).item()]
                for name, param in deeponet.named_parameters()
            }
        else:
            for name, param in deeponet.named_parameters():
                parameter_magnitude_log[name].append(torch.norm(param).item())


        if epoch % log_frequency == 0:
            logging.info(
                "epoch {}: loss {:.6f}, time {:.2f} sec, lr {:.6f}".format(
                    epoch,
                    np.mean(loss_log[-len(pde_loader) :]),
                    np.mean(timing_log[-len(pde_loader) :]),
                    optimizer_all.param_groups[0]["lr"],
                )
            )

            test_loss = []
            deeponet.eval()
            ws.split = "test"
            with torch.no_grad():
                err_test = []
                for pde_data, indices in data_utils.DataLoader(
                    test_dataset,
                    batch_size=scene_per_batch,
                    shuffle=False,
                    num_workers=num_data_loader_threads,
                    drop_last=False,
                ):
                    pde_data = pde_data.cuda()

                    pde_gt = pde_data[:, :,-1].unsqueeze(1)
                    pde_rhs = pde_data[:, :, -2]
                    pde_rhs = pde_rhs.reshape(pde_rhs.shape[0], pde_rhs.shape[1], 1)

                    pde_trunk_inputs = pde_data[:, :, :-2]

                    deeponet_out = deeponet(pde_rhs.cuda(), pde_trunk_inputs.cuda())

                    loss = loss_l1(deeponet_out, pde_gt.cuda()) / pde_data.shape[0]

                    test_loss.append(loss.item())
                    normalized_test_err = normalized_error(deeponet_out, pde_gt.cuda())
                    err_test.append(normalized_test_err.item())
            normalized_test_err_log.append(np.mean(err_test))


            logging.info(
                "epoch {}: test loss: {:.6f}".format(
                    epoch, np.mean(test_loss)
                )
            )
            test_loss_log.append(np.mean(test_loss))
            ws.split = "train"
            save_latest(epoch)

            

            np.savez(
                os.path.join(experiment_directory, ws.deep_o_net_folder, "logs.npz"),
                loss=loss_log,
                lr=lr_log,
                timing=timing_log,
                test_loss=test_loss_log,
                normalized_err=normalized_err_log,
                normalized_test_err=normalized_test_err_log,
                gradient_norm=gradient_norm_log,
                **parameter_magnitude_log
            )
        
        if epoch in checkpoints:
            save_checkpoints(epoch)


    logging.info("training finished")



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

    args = parser.parse_args()

    main_function(args.experiment_directory)



