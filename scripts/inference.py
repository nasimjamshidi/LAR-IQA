import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import torch
from PIL import Image
from torchvision import transforms
import argparse
import timm
from models.mobilenet_merged import MobileNetMerged
from models.mobilenet_merged_with_kan import MobileNetMergedWithKAN

def load_model(model_path, use_kan, device):
    if use_kan:
        model = MobileNetMergedWithKAN()
    else:
        model = MobileNetMerged()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path, color_space, device):
    image = Image.open(image_path).convert('RGB')
    if color_space == "HSV":
        image = Image.fromarray(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV))
    elif color_space == "LAB":
        image = Image.fromarray(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2LAB))
    elif color_space == "YUV":
        image = Image.fromarray(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2YUV))
    
    transform_authentic = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_synthetic = transforms.Compose([
        transforms.Resize((1280, 1280)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_authentic = transform_authentic(image).unsqueeze(0).to(device)
    image_synthetic = transform_synthetic(image).unsqueeze(0).to(device)

    return image_authentic, image_synthetic

def infer(model, image_authentic, image_synthetic):
    with torch.no_grad():
        output = model(image_authentic, image_synthetic)
        return output.item()

def main():
    parser = argparse.ArgumentParser(description="Inference on a single image")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--use_kan", action="store_true", help="Use MobileNetMergedWithKAN model")
    parser.add_argument("--color_space", type=str, choices=["RGB", "HSV", "LAB", "YUV"], default="RGB", help="Color space to use for inference")
    
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model_path, args.use_kan, device)
    image_authentic, image_synthetic = preprocess_image(args.image_path, args.color_space, device)
    score = infer(model, image_authentic, image_synthetic)
    
    print(f"Predicted quality score: {score}")

if __name__ == "__main__":
    main()
