#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset Loader for Stage 2: Aesthetic Quality Assessment.
Handles AVA image loading, data augmentation, and normalized aesthetic scores.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from pathlib import Path
import config_stage2 as config

class AestheticDataset(Dataset):
    def __init__(self, annotation_file, image_dir, mode='train'):
        self.df = pd.read_csv(annotation_file)
        self.image_dir = Path(image_dir)
        self.mode = mode
        
        # Standard preprocessing for ViT architectures (Bicubic interpolation)
        if mode == 'train':
            self.transform = T.Compose([
                T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomCrop((224, 224), padding=4), 
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = T.Compose([
                T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        image_path = self.image_dir / str(row['image_path'])
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image)
        except Exception:
            # Fallback for corrupted images to prevent training interruption
            image_tensor = torch.zeros(3, 224, 224)
        
        # y_norm: Normalized aesthetic score in [0, 1]
        y_tensor = torch.tensor(np.float32(row['y_norm']))
        
        return image_tensor, y_tensor