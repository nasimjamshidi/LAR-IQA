# Image Quality Assessment Project

## Description
This project involves evaluating and training image quality assessment models using MobileNet architectures with optional KAN layers.

## Installation
1. Clone the repository.
2. Install the required dependencies using:



## Usage
### Training
Run the training script:

pip install -r requirements.txt


```bash
python scripts/main.py --csv_path path/to/your/csv_file.csv --root_dir /path/to/dataset --output_path /path/to/output.csv [--use_kan] [--loss_type l2|plcc] [--color_space RGB|HSV|LAB|YUV]


--use_kan: Optional. Use the MobileNetMergedWithKAN model.
--loss_type: Optional. Choose between L2 loss (l2) or Pearson Correlation Loss (plcc).
--color_space: Optional. Select the color space to train on.
Directory Structure
models/: Contains model definitions.
data/: Contains dataset classes.
utils/: Contains utility functions for image processing, training, and loss definitions.
scripts/: Contains the main script to run the training and evaluation.
logs/: Output directory for results.


License
MIT License

### Summary

The updated project now includes a comprehensive training pipeline with options to choose the loss function, color space, and the model variant (with or without KAN). The project is structured for ease of use and flexibility, and the `README.md` provides clear instructions for running the training process.
