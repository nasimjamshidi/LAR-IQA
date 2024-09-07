import torch
import sys
import os
import wandb
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from utils.train import train

def main():
    parser = argparse.ArgumentParser(description="Training and evaluation script")
    parser.add_argument("--csv_files", type=str, nargs='+', required=True, help="Paths to the input CSV files")
    parser.add_argument("--root_dirs", type=str, nargs='+', required=True, help="Root directories of the datasets")
    parser.add_argument("--val_csv_file", type=str, required=True, help="Path to the validation CSV file")
    parser.add_argument("--val_root_dir", type=str, required=True, help="Root directory of the validation dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output CSV file")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--use_kan", action="store_true", help="Use MobileNetMergedWithKAN model")
    parser.add_argument("--color_space", type=str, choices=["RGB", "HSV", "LAB", "YUV"], default="RGB", help="Color space to use for training")
    parser.add_argument("--use_sweep", action="store_true", help="Use W&B sweep for hyperparameter optimization")

    args = parser.parse_args()

    def train_with_config(config=None):
        with wandb.init(config=config):
            config = wandb.config
            config.csv_files = args.csv_files
            config.root_dirs = args.root_dirs
            config.val_csv_file = args.val_csv_file
            config.val_root_dir = args.val_root_dir
            config.output_path = args.output_path
            config.batch_size = args.batch_size
            config.use_kan = args.use_kan
            config.color_space = args.color_space
            train(config)

    if args.use_sweep:
        sweep_configuration = {
            'method': 'random',
            'metric': {'name': 'val_loss', 'goal': 'minimize'},
            'parameters': {
                'learning_rate': {'values': [5e-5]},
                'weight_decay': {'values': [1e-4]},
                'warmup_epochs': {'values': [5]},
                'num_epochs': {'values': [10]},
                'l_num_epochs': {'values': [10]},
                'optimizer': {'values': ['adamw']},
                'NR_msel_weight': {'values': [1.0]},
                'NR_crl_weight': {'values': [1.0]}
            }
        }

        sweep_id = wandb.sweep(sweep=sweep_configuration, project='MyProject')
        wandb.agent(sweep_id, train_with_config, count=5)
    else:
        wandb.init(project="my_project")
        config = wandb.config
        config.csv_files = args.csv_files
        config.root_dirs = args.root_dirs
        config.val_csv_file = args.val_csv_file
        config.val_root_dir = args.val_root_dir
        config.output_path = args.output_path
        config.batch_size = args.batch_size
        config.num_epochs = args.num_epochs
        config.use_kan = args.use_kan
        config.color_space = args.color_space

        train(config)

if __name__ == "__main__":
    main()
