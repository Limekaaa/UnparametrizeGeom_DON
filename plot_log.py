import UDON.workspace as ws
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import json 

# --- IMPROVED PLOT STYLING ---
def set_plot_style():
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'lines.linewidth': 2.5,
        'figure.figsize': (10, 6),
        'axes.grid': True,
        'grid.alpha': 0.3
    })

# Helper to format axis to 3 significant digits
def format_axis_func(x, pos):
    return f'{x:.3g}'

formatter = ticker.FuncFormatter(format_axis_func)

def plot_log(experiment_dir, model, type):
    set_plot_style() # Apply style

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

    # Create figure/ax explicitly to apply formatters easier
    fig, ax = plt.subplots()

    if type == "loss":
        freq_loss = int(len(logs["loss"]) / len(logs["test_loss"]))
        to_plot = [np.mean(logs["loss"][i:i+freq_loss]) for i in range(0, len(logs["loss"]), freq_loss)]
        ax.plot(to_plot, label="Train Loss")
        ax.plot(logs["test_loss"], label="Test Loss")
        ax.set_xlabel("Iteration x{}".format(log_freq))
        ax.set_ylabel("Loss")
        ax.set_title("Loss Comparison")
        ax.legend(frameon=True, fancybox=True, framealpha=0.9)
        ax.yaxis.set_major_formatter(formatter) # 3 sig figs
        plt.savefig(os.path.join(save_path,"loss_plot.png"), dpi=300, bbox_inches='tight')
        plt.close()

    elif type == "learning_rate":
        ax.plot(logs["lr"], label="Learning Rate")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.legend(frameon=True, fancybox=True, framealpha=0.9)
        ax.yaxis.set_major_formatter(formatter)
        plt.savefig(os.path.join(save_path,"learning_rate_plot.png"), dpi=300, bbox_inches='tight')
        plt.close()

    elif type == "time":
        ax.plot(logs["timing"], label="Time per Iteration")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Time (seconds)")
        ax.set_title("Time per Iteration")
        ax.legend(frameon=True, fancybox=True, framealpha=0.9)
        ax.yaxis.set_major_formatter(formatter)
        plt.savefig(os.path.join(save_path,"time_plot.png"), dpi=300, bbox_inches='tight')
        plt.close()

    elif type == "error":
        freq_err = int(len(logs["normalized_err"]) / len(logs["normalized_test_err"]))
        to_plot = [np.mean(logs["normalized_err"][i:i+freq_err]) for i in range(0, len(logs["normalized_err"]), freq_err)]
        ax.plot(to_plot, label="Error")
        ax.plot(logs["normalized_test_err"], label="Test Error")
        ax.set_xlabel("Iteration x{}".format(log_freq))
        ax.set_ylabel("Error")
        ax.set_title("Error per Iteration")
        ax.legend(frameon=True, fancybox=True, framealpha=0.9)
        ax.yaxis.set_major_formatter(formatter)
        plt.savefig(os.path.join(save_path,"error_plot.png"), dpi=300, bbox_inches='tight')
        plt.close()

    elif type == "param_mag":
        param_mags = [key for key in logs if key.startswith("branch") or key.startswith("trunk")]
        for name in param_mags:
            ax.plot(logs[name], label=name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Magnitude")
        ax.set_title("Parameter Magnitude")
        ax.yaxis.set_major_formatter(formatter)
        ax.legend(
            bbox_to_anchor=(1.05, 1),  
            loc='upper left',          
            borderaxespad=0.,
            frameon=True, 
            fancybox=True
        )
        plt.savefig(os.path.join(save_path,"param_mag_plot.png"), dpi=300, bbox_inches='tight')
        plt.close()

    elif type == "gradient_norm":
        if "gradient_norm" in logs.keys():
            ax.plot(logs["gradient_norm"], label="Gradient Norm")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Gradient Norm")
            ax.set_title("Gradient Norm per Iteration")
            ax.legend(frameon=True, fancybox=True, framealpha=0.9)
            ax.yaxis.set_major_formatter(formatter)
            plt.savefig(os.path.join(save_path,"gradient_norm_plot.png"), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            raise ValueError("Gradient norm log not found in the logs.")
            
    elif type == "all":
        types = ["loss", "learning_rate", "time", "error", "param_mag", "gradient_norm"]
        # Note: We don't need to loop recursively here creating figures because the functions 
        # above now create their own figures/axes. We can just call plot_log recursively.
        # However, to prevent open figure warning, we ensure closes happen.
        plt.close('all') 
        for t in types:
            try:
                plot_log(experiment_dir, model, t)
            except Exception as e:
                print(f"Could not plot {t}: {e}")
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
        help="The experiment directory.",
    )

    parser.add_argument(
        "--model",
        "-m",
        dest="model",
        default="DeepONet",
        help="The model to plot logs from.",
    )

    parser.add_argument(
        "--type",
        "-t",
        dest="type",
        default="all",
        help="Type of log to plot.",
    )

    args = parser.parse_args()

    ws.specs = json.load(open(os.path.join(args.experiment_directory, "specs.json")))

    plot_log(args.experiment_directory, args.model, args.type)