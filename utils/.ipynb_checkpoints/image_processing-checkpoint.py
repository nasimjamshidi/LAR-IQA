from PIL import Image
import torch
from torchvision import transforms

def read_image(img_name, transform):
    image = Image.open(img_name).convert('RGB')
    return transform(image)
