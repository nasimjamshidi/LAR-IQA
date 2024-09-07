import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class MultiColorSpaceDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, transform2=None):
        df = pd.read_csv(csv_file)
        self.annotations = pd.concat([df, df], ignore_index=True)
        self.root_dir = root_dir
        self.transform = transform
        self.transform2 = transform2

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.annotations.iloc[idx, 0]))
        image = Image.open(img_name).convert('RGB')
        rgb_image = image.convert('RGB')
        hsv_image = Image.fromarray(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV))
        lab_image = Image.fromarray(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2LAB))
        yuv_image = Image.fromarray(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2YUV))

        if self.transform:
            rgb_image_authentic = self.transform(rgb_image)
            hsv_image_authentic = self.transform(hsv_image)
            lab_image_authentic = self.transform(lab_image)
            yuv_image_authentic = self.transform(yuv_image)

        if self.transform2:
            rgb_image_synthetic = self.transform2(rgb_image)
            hsv_image_synthetic = self.transform2(hsv_image)
            lab_image_synthetic = self.transform2(lab_image)
            yuv_image_synthetic = self.transform2(yuv_image)

        annotations = (self.annotations.iloc[idx, 1:]).to_numpy()
        annotations = annotations.astype('float').reshape(-1, 1)

        sample = {
            'img_id': img_name,
            'RGB_authentic': rgb_image_authentic,
            'HSV_authentic': hsv_image_authentic,
            'LAB_authentic': lab_image_authentic,
            'YUV_authentic': yuv_image_authentic,
            'RGB_synthetic': rgb_image_synthetic,
            'HSV_synthetic': hsv_image_synthetic,
            'LAB_synthetic': lab_image_synthetic,
            'YUV_synthetic': yuv_image_synthetic,
            'annotations': annotations
        }

        return sample
