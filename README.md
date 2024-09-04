# LAR-IQA: A Lightweight, Accurate, and Robust No-Reference Image Quality Assessment Model

## Description
This project involves evaluating and training image quality assessment models using MobileNet architectures with optional KAN layers.


## Installation
1. Clone the repository.
2. Install the required dependencies using:

```
    git clone https://github.com/nasimjamshidi/LAR-IQA
    pip install -r requirements.txt
```

## Usage
### Training
Run the training script:

```bash
python scripts/main.py --csv_files path/to/your/train_csv1.csv path/to/your/train_csv2.csv --root_dirs /path/to/train_dataset1 /path/to/train_dataset2 --val_csv_file path/to/your/val_csv.csv --val_root_dir /path/to/val_dataset --output_path /path/to/output.csv --batch_size 32 [--use_kan] [--loss_type l2|plcc] [--color_space RGB|HSV|LAB|YUV]
```

- csv_files: Required. Paths to the input CSV files for training.
- root_dirs: Required. Root directories of the datasets for training.
- val_csv_file: Required. Path to the validation CSV file.
- val_root_dir: Required. Root directory of the validation dataset.
- output_path: Required. Path to save the output CSV file.
- batch_size: Optional. Batch size for training.
- use_kan: Optional. Use the MobileNetMergedWithKAN model.
- loss_type: Optional. Choose between L2 loss (l2) or Pearson Correlation Loss (plcc).
- color_space: Optional. Select the color space to train on.

### Inference 
Run the inference script on a single image:
```bash
python scripts/inference.py --image_path path/to/your/image.jpg --model_path path/to/trained_model.pt [--use_kan] [--color_space RGB|HSV|LAB|YUV]
```

Options

- use_kan: Optional. Use the MobileNetMergedWithKAN model.
- color_space: Optional. Select the color space to train on (e.g., RGB, HSV, LAB, YUV).

Directory Structure

- models/: Contains model definitions.
- data/: Contains dataset classes.
- utils/: Contains utility functions for image processing, training, and loss definitions.
- scripts/: Contains the main script to run the training and evaluation.
- logs/: Output directory for results.

### License

This project is licensed under the MIT License.
