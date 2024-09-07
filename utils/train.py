import torch
import wandb
from .loss import get_loss_function
from data.dataset import MultiColorSpaceDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from models.mobilenet_merged_with_kan import MobileNetMergedWithKAN
from models.mobilenet_merged import MobileNetMerged

class MultiDatasetLoader:
    def __init__(self, datasets, batch_size, shuffle=True, num_workers=8):
        self.datasets = datasets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.loaders = [DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers) for dataset in datasets]
        self.iterators = [iter(loader) for loader in self.loaders]

    def __iter__(self):
        self.iterators = [iter(loader) for loader in self.loaders]
        return self

    def __next__(self):
        if not self.iterators:
            raise StopIteration
        current_loader = np.random.choice(self.iterators)
        try:
            return next(current_loader)
        except StopIteration:
            self.iterators.remove(current_loader)
            if not self.iterators:
                raise StopIteration
            return next(self)

    def __len__(self):
        return sum(len(loader) for loader in self.loaders)

def build_optimizer_and_scheduler(network, config, train_loaders):
    optimizer = torch.optim.AdamW(
        lr= config.learning_rate,
        params= network.parameters(),
        weight_decay= config.weight_decay
    )


    warmup_epochs = config.warmup_epochs
    warmup_iter = 0
    for train_loader in train_loaders.values():
        warmup_iter += int(config.warmup_epochs * len(train_loader))
    max_iter = int((config.num_epochs + config.l_num_epochs) * len(train_loader))
    
    lr_lambda = (
        lambda cur_iter: cur_iter / warmup_iter
        if cur_iter <= warmup_iter
        else 0.5 * (1 + torch.cos(torch.pi * (cur_iter - warmup_iter) / (max_iter - warmup_iter)))
    )
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    return optimizer, scheduler

def train_epoch(network, loader, optimizer, scheduler, l2loss, plccloss, weights, color_space):
    cumu_loss = 0
    network.train()
    for _, data in enumerate(loader):
        images_authentic = data[color_space + '_authentic'].cuda()
        images_synthetic = data[color_space + '_synthetic'].cuda()
        labels = data['annotations'].cuda().float()

        outputs = network(images_authentic, images_synthetic)

        outputs = outputs.view(outputs.size()[0], 1, 1)

        optimizer.zero_grad()
        
        NR_msel = l2loss(labels.flatten(), outputs.flatten())
        
        if torch.isnan(outputs).any() or torch.isnan(labels).any():
            print(f"NaNs found in outputs or labels for task {task_id}")
            NR_crl = torch.tensor(0.0, device=outputs.device)
        else:
            NR_crl = plccloss(outputs.flatten()[None, :], labels.flatten()[None, :])
        
        
        loss = weights['NR_crl'] * NR_crl + weights['NR_msel'] * NR_msel

        loss.backward()
        optimizer.step()
        scheduler.step()  

        cumu_loss += loss.item()
        wandb.log({"batch_loss": loss.item(), "NR_msel": NR_msel.item(), "NR_crl": NR_crl.item()})

    return cumu_loss / len(loader)

def validate_epoch(network, loader, l2loss, plccloss, weights, color_space):
    cumu_loss = 0
    network.eval()
    with torch.no_grad():
        for _, data in enumerate(loader):
            images_authentic = data[color_space + '_authentic'].cuda()
            images_synthetic = data[color_space + '_synthetic'].cuda()

            labels = data['annotations'].cuda().float()

            outputs = network(images_authentic, images_synthetic)
            outputs = outputs.view(outputs.size()[0], 1, 1)

            loss = weights['NR_msel'] * l2loss(outputs.flatten(), labels.flatten()) + \
                   weights['NR_crl'] * plccloss(outputs.flatten()[None, :], labels.flatten()[None, :])

            cumu_loss += loss.item()

    return cumu_loss / len(loader)

def train(config=None):
    with wandb.init(config=config):
        config = wandb.config

        epochs = config.num_epochs

        train_loader = build_dataset(config.batch_size, config.csv_files, config.root_dirs)
        val_loader = build_dataset(config.batch_size, [config.val_csv_file], [config.val_root_dir])

        device = "cuda" if torch.cuda.is_available() else "cpu"

        if config.use_kan:
            model = MobileNetMergedWithKAN()
        else:
            model = MobileNetMerged(
                        authentic_weights_path="./pretrained/single_branch_pretrained/Authentic.pth", 
                        synthetic_weights_path="./pretrained/single_branch_pretrained/Synthetic.pth"
                    )
            
            
        model.to(device)

        optimizer, scheduler = build_optimizer_and_scheduler(model, config, {"train_loader": train_loader})
        l2loss = get_loss_function('l2')
        plccloss = get_loss_function('plcc')

        weights = {
            'NR_msel': config.NR_msel_weight,
            'NR_crl': config.NR_crl_weight
        }

        for epoch in range(epochs):
            avg_train_loss = train_epoch(model, train_loader, optimizer, scheduler, l2loss, plccloss, weights, config.color_space)
            avg_val_loss = validate_epoch(model, val_loader, l2loss, plccloss, weights, config.color_space)
            wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "epoch": epoch})

            # Save model checkpoint
            model_name = f'./log/checkpoint_epoch_{epoch}.pt'
            torch.save(model.state_dict(), model_name)

def build_dataset(batch_size, csv_files, root_dirs):
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform2 = transforms.Compose([
        transforms.CenterCrop((1280, 1280)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    datasets = [MultiColorSpaceDataset(csv_file=csv_file, root_dir=root_dir, transform=transform, transform2=transform2) for csv_file, root_dir in zip(csv_files, root_dirs)]
    loader = MultiDatasetLoader(datasets, batch_size)
    return loader

