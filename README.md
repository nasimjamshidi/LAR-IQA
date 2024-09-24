# LAR-IQA @ ECCV 2024 AIM Workshop

### [LAR-IQA: A Lightweight, Accurate, and Robust No-Reference Image Quality Assessment Model](https://arxiv.org/abs/2408.17057)
Nasim Jamshidi Avanaki, Abhijay Ghildyal, Nabajeet Barman, Saman Zadtootaghaj

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2408.17057)
[ <a href="https://colab.research.google.com/drive/1g0hm-S25oYOd5OSFT91uMgZA2UANGSOb?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>](https://colab.research.google.com/drive/1g0hm-S25oYOd5OSFT91uMgZA2UANGSOb?usp=sharing)
<br>

## Description
This model is a lightweight No-Reference Image Quality Assessment (NR-IQA) model designed for efficient deployment on mobile devices. It uses a dual-branch architecture, with one branch trained on synthetically distorted images and the other on authentically distorted images, improving generalizability across distortion types. Each branch includes a mobile-based image encoder (MobileNet V3) and a Kolmogorov-Arnold Network (KAN) for quality prediction, offering better accuracy than traditional MLPs. The model also incorporates multiple color spaces during training for enhanced robustness. 

![Fig](images/Model_Architecture.png)
## Installation
1. Clone the repository.
2. Install the required dependencies using:

```
    git clone https://github.com/nasimjamshidi/LAR-IQA
    pip install -r requirements.txt
```

## Usage
### Training

The training is currently available only for the dual-branch setup, where training begins from the pretrained synthetic and authentic branches. Please note that these branches were not trained on the UHD-IQA dataset, but rather on other synthetic and authentic datasets (refer to section 4.5 in the paper). They are used as pretrained weights to initialize the dual-branch training. We believe that fine-tuning the model from this starting point yields solid results for a new No-Reference Image Quality Assessment (NR-IQA) dataset.

Run the training script:

```bash
python scripts/main.py --csv_files path/to/your/train.csv path/to/your/train.csv --root_dirs /path/to/train_dataset /path/to/train_dataset --val_csv_file path/to/your/validation.csv --val_root_dir /path/to/validation_dataset --output_path /path/to/output.csv --batch_size 32 [--use_kan] [--loss_type l2|plcc] [--color_space RGB|HSV|LAB|YUV]```
```
Arguments:

-- csv_files: Required. Paths to the input CSV files for training. 

-- root_dirs: Required. Root directories of the datasets for training.

-- val_csv_file: Required. Path to the validation CSV file.

-- val_root_dir: Required. Root directory of the validation dataset.

-- output_path: Required. Path to save the output CSV file.

-- batch_size: Optional. Batch size for training (default: 20).

-- num_epochs: Optional. Number of epochs for training (default: 10).

-- use_kan: Optional. Use the MobileNetMergedWithKAN model.

-- color_space: Optional. Select the color space for training. Choose from RGB, HSV, LAB, or YUV (default: RGB).

-- use_sweep: Optional. Use W&B sweep for hyperparameter optimization.

Note: If --use_sweep is provided, the script will initiate a W&B sweep for hyperparameter optimization using random search. The sweep optimizes metrics such as val_loss and tunes hyperparameters like learning rate, weight decay, optimizer type, etc. 

For hyperparameter sweeps, you can customize the configuration in the code. 
Please note that we added an example of how the csv files for training and validation must look like, using a very small portion of UHD-IQA dataset; located in the dataset folder. This is only for the demo of training; please refer to the main dataset. 

Please note that the training and validation CSV files must follow this format: each file should contain two columns— the first column for the image name and the second for the corresponding subjective score of the image.

### Inference 
If you would like to run a demo of the inference, you can easily do so through Google Colab. Click [ <a href="https://colab.research.google.com/drive/1g0hm-S25oYOd5OSFT91uMgZA2UANGSOb?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>](https://colab.research.google.com/drive/1g0hm-S25oYOd5OSFT91uMgZA2UANGSOb?usp=sharing) to get started.

Alternatively, to run the inference script manually on a single image, use the code below:

```bash
python scripts/inference.py --image_path path/to/your/image.jpg --model_path path/to/trained_model.pt [--use_kan] [--color_space RGB|HSV|LAB|YUV]
```

-- use_kan: Optional. Use the MobileNetMergedWithKAN model.

-- color_space: Optional. Select the color space to train on (e.g., RGB, LAB, YUV).

### Directory Structure

- data/: Contains dataset classes.
- images/: Contains a sample image used for the demo and an image of the model architecture.
- models/: Contains model definitions.
- scripts/: Contains the main script to run the training and evaluation.
- utils/: Contains utility functions for image processing, training, and loss definitions.
- log/: Directory where new training models will be saved.
- pretrained/: Contains pretrained models used for ECCV paper. 
- Dataset/: Contains examples of how the training and validation datasets should be structured. Please note that this is a small example of the UHD-IQA dataset.

### Pre-trained models

Please note that we have provided two groups of models: the 2-branch models trained and tested for the ECCV paper, and the 1-branch pretrained models. The 2-branch setup is specifically designed to meet the requirements of the UHD-IQA challenge. For instance, we used 1280x1280 center cropping for the synthetic branch to maintain low computational complexity.

We also offer 1-branch pretrained models, which we recommend using for training the model on new datasets. For the synthetic branch, ensure you adjust the cropping based on your image resolution. However, for the authentic branch, we recommend resizing the images to 384x384.

### Citation and Acknowledgement

```
@inproceedings{Avanaki2024LAR-IQA,
  title={LAR-IQA: A Lightweight, Accurate, and Robust No-Reference Image Quality Assessment Model},
  author={Jamshidi Avanaki, Nasim and Ghildiyal, Abhijay and Barman, Nabajeet and Zadtootaghaj, Saman},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV) Workshops},
  year={2024}
}
```

### License

This project is licensed under the MIT License.
