import UDON.workspace as ws
import matplotlib.pyplot as plt
import numpy as np
import os
import json 

def plot_log(experiment_dir, model, type):
    if model == "DeepONet":
        log_file = os.path.join(experiment_dir, ws.deep_o_net_folder, "logs.npz")
        save_path = os.path.join(experiment_dir,ws.deep_o_net_folder, ws.log_folder)
        os.makedirs(save_path, exist_ok=True)
        
    elif model == "DeepSDF":
        log_file = os.path.join(experiment_dir, ws.deep_sdf_folder, "logs.npz")
        save_path = os.path.join(experiment_dir, ws.deep_sdf_folder, ws.log_folder)
        os.makedirs(save_path, exist_ok=True)
    else:
        raise ValueError("Model must be either 'DeepONet' or 'DeepSDF'.")

    if not os.path.exists(log_file):
        raise FileNotFoundError(f"Log file {log_file} does not exist.")

    logs = np.load(log_file)

    try:
        log_freq = ws.specs["LogFrequency"]
    except :
        log_freq = 10

    if type == "loss":
        freq_loss = int(len(logs["loss"]) / len(logs["test_loss"]))
        to_plot = [np.mean(logs["loss"][i:i+freq_loss]) for i in range(0, len(logs["loss"]), freq_loss)]
        plt.plot(to_plot, label="Train Loss")
        plt.plot(logs["test_loss"], label="Test Loss")
        plt.xlabel("Iteration x{}".format(log_freq))
        plt.ylabel("Loss")
        plt.title("Loss Comparison")
        plt.legend()
        plt.savefig(os.path.join(save_path,"loss_plot.png"))
        plt.clf()
    elif type == "learning_rate":
        plt.plot(logs["lr"], label="Learning Rate")
        plt.xlabel("Iteration")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.legend()
        plt.savefig(os.path.join(save_path,"learning_rate_plot.png"))
        plt.clf()
    elif type == "time":
        plt.plot(logs["timing"], label="Time per Iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Time (seconds)")
        plt.title("Time per Iteration")
        plt.legend()
        plt.savefig(os.path.join(save_path,"time_plot.png"))
        plt.clf()
    elif type == "error":
        freq_err = int(len(logs["normalized_err"]) / len(logs["normalized_test_err"]))
        to_plot = [np.mean(logs["normalized_err"][i:i+freq_err]) for i in range(0, len(logs["normalized_err"]), freq_err)]
        plt.plot(to_plot, label="Error")
        plt.plot(logs["normalized_test_err"], label="Test Error")
        plt.xlabel("Iteration x{}".format(log_freq))
        plt.ylabel("Error")
        plt.title("Error per Iteration")
        plt.legend()
        plt.savefig(os.path.join(save_path,"error_plot.png"))
        plt.clf()
    elif type == "param_mag":
        param_mags = [key for key in logs if key.startswith("branch") or key.startswith("trunk")]
        for name in param_mags:
            plt.plot(logs[name], label=name)
        plt.xlabel("Epoch")
        plt.ylabel("Magnitude")
        plt.title("Parameter Magnitude")
        plt.legend(
            bbox_to_anchor=(1.05, 1),  # x=1.05 puts it slightly to the right
            loc='upper left',          # align the top of the legend to the top left corner of its box
            borderaxespad=0.           # optional, reduces spacing between plot and legend
        )
        plt.savefig(os.path.join(save_path,"param_mag_plot.png"), bbox_inches='tight')
        plt.clf()
    elif type == "gradient_norm":
        if "gradient_norm" in logs.keys():
            plt.plot(logs["gradient_norm"], label="Gradient Norm")
            plt.xlabel("Iteration")
            plt.ylabel("Gradient Norm")
            plt.title("Gradient Norm per Iteration")
            plt.legend()
            plt.savefig(os.path.join(save_path,"gradient_norm_plot.png"))
            plt.clf()
        else:
            raise ValueError("Gradient norm log not found in the logs.")
    elif type == "all":
        types = ["loss", "learning_rate", "time", "error", "param_mag", "gradient_norm"]
        for t in types:
            plot_log(experiment_dir, model, t)
    else:
        raise ValueError("Type must be one of 'loss', 'learning_rate', 'time', 'error', or 'all'.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot training logs for DeepONet.")
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
        "--model",
        "-m",
        dest="model",
        default="DeepONet",
        help="The model to plot logs from. If 'DeepONet', training logs from DeepONet will be used, if 'DeepSDF', logs from DeepSDF will be used.",
    )

    parser.add_argument(
        "--type",
        "-t",
        dest="type",
        default="all",
        help="Type of log to plot. Options are 'loss', 'learning_rate', 'time', 'lat_mag', 'param_mag', or 'all'. Default is 'all'.",
    )

    args = parser.parse_args()

    ws.specs = json.load(open(os.path.join(args.experiment_directory, "specs.json")))

    plot_log(args.experiment_directory, args.model, args.type)