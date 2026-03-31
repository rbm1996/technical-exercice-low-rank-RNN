"""-----------------------------"""
"""fix path stuff"""
"""-----------------------------"""
import numpy as np
import sys
import os
from pathlib import Path

# 1. Get the absolute path to the root of the 'smc_rnns' folder
# This goes up 2 levels from where train_EEG.py is located
script_path = Path(__file__).resolve()
vi_rnn_dir = script_path.parents[1] 

# 2. Add the root to sys.path so we can import 'vi_rnn'
sys.path.append(str(vi_rnn_dir))

from vi_rnn.vae import VAE
from vi_rnn.train import train_VAE
from vi_rnn.datasets import Basic_dataset

# 3. Define data and output paths relative to the root
data_path = vi_rnn_dir / "data" / "eeg"
out_dir = vi_rnn_dir / "models" / "sweep_eeg"

# Ensure directories exist
out_dir.mkdir(parents=True, exist_ok=True)

# Convert back to strings for the rest of the script (which expects strings)
data_path = str(data_path) + "/"
out_dir = str(out_dir) + "/"
vi_rnn_dir = str(vi_rnn_dir)

print(f"Project Root: {vi_rnn_dir}")
print(f"Looking for data in: {data_path}")
"""-----------------------------"""
"""-----------------------------"""


# We used openly accessible electroencephalogram (EEG) data from Schalk et al. 2004
# available from https://www.physionet.org/content/eegmmidb/1.0.0/ (Moody et al. 2000; ODC-BY licence).
# This repo includes preprocessed data from session S001R01

# Set key parameters
# ------------------
dim_z = 3  # latent dimensionality
dim_N = 512  # number of neurons
n_runs = 1  # number of runs


data_eval_name = "EEG_data_smoothed.npy"  # Use smooth on data
data_name = "EEG_data_zscored.npy"  # Use raw (but zcored) data for training


wandb = True  # Sync with wandb
n_epochs = 1500  # number of epochs
bs = 10  # batch size
cuda = True  # use cuda
out_dir = vi_rnn_dir + "/models/sweep_eeg/"  # output directory
data_path = vi_rnn_dir + "/data/eeg/"  # data directory


"""-----------------------------"""
"""Change here for my Laptop """
"""-----------------------------"""

# Updated environment logic for your Mac
import platform

# Determine if we should use CUDA (NVIDIA) or CPU/MPS (Mac)
# Note: Currently, the repo is built for CUDA. On Mac, we set cuda=False to use CPU.
if platform.system() == "Darwin": # Darwin is the system name for macOS
    cuda = False
    print("Running on macOS: CUDA disabled.")
else:
    cuda = True

# Set paths relative to the script location so it works anywhere
out_dir = os.path.join(vi_rnn_dir, "models", "sweep_eeg")

# data_path = os.path.join(vi_rnn_dir, "data", "eeg")


""" fit model on Run 3 and 5"""
data_path = os.path.join(vi_rnn_dir, "rbm_stuff")
data_name = "S001R03_concat_S001R05_zscored.npy" 
data_eval_name = "S001R03_concat_S001R05_smoothed.npy"


# Ensure the output directory exists
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

print(f"Data path: {data_path}")
print(f"Output path: {out_dir}")

"""-----------------------------"""

"""-----------------------------"""
""" Other path fix """
"""-----------------------------"""
# Updated data loading using robust path joining
# ------------------
data_file_path = os.path.join(data_path, data_name)
data_eval_file_path = os.path.join(data_path, data_eval_name)

print(f"Loading training data from: {data_file_path}")
print(f"Loading eval data from: {data_eval_file_path}")

# Load and Transpose
data = np.float32(np.load(data_file_path)).T
data_eval = np.float32(np.load(data_eval_file_path)).T
"""-----------------------------"""

# initialise dataset
# ------------------
task_params = {"name": "EEG", "dur": 50, "n_trials": 50 * bs}
task = Basic_dataset(task_params, data, data_eval)
dim_x = task.data.shape[0]


# Train the VAE
# ------------------
for _ in range(n_runs):
    # initialise encoder

    # initialise prior
    rnn_params = {
        "train_noise_x": True,
        "train_noise_z": True,
        "train_noise_z_t0": True,
        "init_noise_z": 0.1,
        "init_noise_z_t0": 1,
        "init_noise_x": 0.1,
        "noise_z": "full",
        "noise_x": "diag",
        "noise_z_t0": "full",
        "transition": "low_rank",
        "activation": "clipped_relu",
        "decay": 0.9,
        "train_neuron_bias": True,
        "weight_dist": "uniform",
        "initial_state": "trainable",
        "simulate_input": False,
        "observation": "affine",
        "readout_from": "z",
        "train_obs_bias": True,
        "train_obs_weights": True,
        "obs_nonlinearity": "identity",
        "obs_likelihood": "Gauss",
    }

    training_params = {
        "lr": 1e-3,
        "lr_end": 1e-6,
        "n_epochs": n_epochs,
        "grad_norm": 0,
        "eval_epochs": 25,
        "batch_size": bs,
        "cuda": cuda,
        "smoothing": 20,
        "freq_cut_off": -1,
        "k": 10,
        "resample": "systematic",
        "loss_f": "opt_smc",
        "run_eval": True,
        "smooth_at_eval": True,
        "init_state_eval": "posterior_sample",
    }

    VAE_params = {
        "dim_x": dim_x,
        "dim_z": dim_z,
        "dim_N": dim_N,
        "rnn_params": rnn_params,
    }

    vae = VAE(VAE_params)

    train_VAE(vae, training_params, task, sync_wandb=wandb, out_dir=out_dir, fname=None)
