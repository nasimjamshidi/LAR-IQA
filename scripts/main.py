import torch
from models.mobilenet_merged_with_kan import MobileNetMergedWithKAN
from utils.train import train
import argparse
import wandb

def main():
    parser = argparse.ArgumentParser(description="Training and evaluation script")
    parser.add_argument("--csv_files", type=str, nargs='+', required=True, help="Paths to the input CSV files")
    parser.add_argument("--root_dirs", type=str, nargs='+', required=True, help="Root directories of the datasets")
    parser.add_argument("--val_csv_file", type=str, required=True, help="Path to the validation CSV file")
    parser.add_argument("--val_root_dir", type=str, required=True, help="Root directory of the validation dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output CSV file")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size for training")
    parser.add_argument("--use_kan", action="store_true", help="Use MobileNetMergedWithKAN model")
    parser.add_argument("--loss_type", type=str, choices=["l2", "plcc"], default="l2", help="Loss function type")
    parser.add_argument("--color_space", type=str, choices=["RGB", "HSV", "LAB", "YUV"], default="RGB", help="Color space to use for training")

    args = parser.parse_args()

    wandb.init(project="my_project")
    config = wandb.config
    config.csv_files = args.csv_files
    config.root_dirs = args.root_dirs
    config.val_csv_file = args.val_csv_file
    config.val_root_dir = args.val_root_dir
    config.output_path = args.output_path
    config.batch_size = args.batch_size
    config.use_kan = args.use_kan
    config.loss_type = args.loss_type
    config.color_space = args.color_space

    # Call the train function
    train(config)

if __name__ == "__main__":
    main()
