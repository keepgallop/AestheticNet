#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
from pathlib import Path
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
import config_stage1 as config

class GazeDataset(Dataset):
    """
    Dataset for Contrastive Gaze Alignment.
    Pairs raw images with human eye-tracking sequences (x, y, duration).
    """
    def __init__(self, image_paths, gaze_seq_paths, mode='train'):
        self.image_paths = image_paths
        self.gaze_seq_paths = gaze_seq_paths
        self.mode = mode
        self.target_size = config.MODEL_INPUT_SIZE
        self.target_seq_len = config.GAZE_SEQ_LEN

        if mode == 'train':
            self.image_transform = T.Compose([
                T.Resize(self.target_size, interpolation=TF.InterpolationMode.BICUBIC),
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.image_transform = T.Compose([
                T.Resize(self.target_size, interpolation=TF.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        gaze_seq_path = self.gaze_seq_paths[idx]
        
        image = Image.open(image_path).convert("RGB")
        gaze_data_full = np.load(gaze_seq_path) 
        
        do_hflip = (self.mode == 'train') and (random.random() > 0.5)
        if do_hflip:
            image = TF.hflip(image)
        
        image_tensor = self.image_transform(image)
            
        # Resampling gaze points to fixed length
        num_points_total = gaze_data_full.shape[0]
        if num_points_total < self.target_seq_len:
            indices = np.random.choice(num_points_total, self.target_seq_len, replace=True)
        else:
            indices = np.random.choice(num_points_total, self.target_seq_len, replace=False)
        
        gaze_data_sampled = gaze_data_full[indices, :]
        
        if do_hflip:
            gaze_data_sampled[:, 0] = 1.0 - gaze_data_sampled[:, 0]
            
        gaze_tensor = torch.from_numpy(gaze_data_sampled.astype(np.float32))
        return image_tensor, gaze_tensor