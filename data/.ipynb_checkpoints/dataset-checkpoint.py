import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import cv2
import numpy as np
from torchvision import transforms

class MultiColorSpaceDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        df = pd.read_csv(csv_file)
        self.annotations = pd.concat([df, df], ignore_index=True)
        self.root_dir = root_dir
        self.transform = transform

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
            rgb_image = self.transform(rgb_image)
            hsv_image = self.transform(hsv_image)
            lab_image = self.transform(lab_image)
            yuv_image = self.transform(yuv_image)

        annotations = (self.annotations.iloc[idx, 1:]).to_numpy()
        annotations = annotations.astype('float').reshape(-1, 1)

        sample = {
            'img_id': img_name,
            'RGB': rgb_image,
            'HSV': hsv_image,
            'LAB': lab_image,
            'YUV': yuv_image,
            'annotations': annotations
        }

        return sample
