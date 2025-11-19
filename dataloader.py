import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from typing import List


class ISICDataset(Dataset):
     def __init__(self, csv_file, img_dir, transform=None):

        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        img_name = os.path.join(
            self.img_dir, 
            self.data_frame.iloc[idx, 1] + '.jpg'
        )
        
        try:
            image = Image.open(img_name).convert("RGB")
        except:
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # 获取标签
        label = int(self.data_frame.iloc[idx, 3])
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def create_non_iid_split(dataset, num_clients, alpha=0.5):

    labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        labels.append(label)
    labels = np.array(labels)
    
    num_classes = len(np.unique(labels))

    client_indices = [[] for _ in range(num_clients)]
    
    for k in range(num_classes):

        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)
        
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

        idx_k_split = np.split(idx_k, proportions)

        for i in range(num_clients):
            client_indices[i].extend(idx_k_split[i].tolist())

    for i in range(num_clients):
        np.random.shuffle(client_indices[i])
    
    return client_indices


def create_federated_dataloaders(csv_file=None, img_dir=None, 
                                 num_clients=10, batch_size=16,
                                 alpha=0.5, num_workers=4,
                                 use_real_data=False):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    if use_real_data and csv_file and img_dir:
        # ISIC 2020
        dataset = ISICDataset(
            csv_file=csv_file,
            img_dir=img_dir,
            transform=transform
        )

        client_indices = create_non_iid_split(dataset, num_clients, alpha)

        client_dataloaders = []
        for i in range(num_clients):
            client_dataset = Subset(dataset, client_indices[i])
            loader = DataLoader(
                client_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True
            )
            client_dataloaders.append(loader)

    else:

        from torch.utils.data import TensorDataset
        
        client_dataloaders = []
        for i in range(num_clients):
            num_samples = 100
            images = torch.randn(num_samples, 3, 224, 224)
            labels = torch.randint(0, 2, (num_samples,))
            
            dataset = TensorDataset(images, labels)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True
            )
            client_dataloaders.append(loader)

    return client_dataloaders


def create_test_dataloader(csv_file, img_dir, batch_size=32, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    dataset = ISICDataset(
        csv_file=csv_file,
        img_dir=img_dir,
        transform=transform
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader


