#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Training Script for Stage 1: Contrastive Gaze Alignment (CGA)
Aligns the DINOv2 visual backbone with human oculomotor dynamics.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import time

import config_stage1 as config
from dataset_stage1 import get_data_loaders
from model_stage1 import AlignmentModelDINO

class ContrastiveAlignmentLoss(nn.Module):
    """InfoNCE-style symmetric contrastive loss."""
    def __init__(self, temperature=config.TEMPERATURE):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, image_features, gaze_features):
        logits = torch.matmul(image_features, gaze_features.T) / self.temperature
        labels = torch.arange(logits.shape[0]).to(logits.device)
        loss_i = self.cross_entropy(logits, labels)
        loss_g = self.cross_entropy(logits.T, labels)
        return (loss_i + loss_g) / 2

def save_checkpoint(model_state, optimizer_state, epoch, loss, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'epoch': epoch, 'model_state_dict': model_state, 'optimizer_state_dict': optimizer_state, 'loss': loss}, path)

def load_checkpoint(model, optimizer, path, device):
    if path.exists():
        try:
            checkpoint = torch.load(path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"INFO: Resumed training from checkpoint: {path}")
            return checkpoint['epoch'], checkpoint['loss']
        except Exception as e:
            print(f"WARNING: Failed to load checkpoint - {e}")
    return 0, float('inf')

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="Training", leave=False)
    for img, gaze in pbar:
        img, gaze = img.to(device), gaze.to(device)
        optimizer.zero_grad()
        img_feat, gaze_feat = model(img, gaze)
        loss = criterion(img_feat, gaze_feat)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(loader)

def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for img, gaze in tqdm(loader, desc="Validation", leave=False):
            img, gaze = img.to(device), gaze.to(device)
            img_feat, gaze_feat = model(img, gaze)
            loss = criterion(img_feat, gaze_feat)
            total_loss += loss.item()
    return total_loss / len(loader)

def main():
    print("--- Stage 1: Contrastive Gaze Alignment ---")
    device = torch.device(config.DEVICE)
    config.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = get_data_loaders(config.VALIDATION_SPLIT, config.RANDOM_SEED)
    if not train_loader: return

    model = AlignmentModelDINO(projection_dim=config.PROJECTION_DIM).to(device)
    criterion = ContrastiveAlignmentLoss(temperature=config.TEMPERATURE).to(device)
    
    optimizer = optim.AdamW([
        {'params': model.gaze_encoder.parameters(), 'lr': config.LEARNING_RATE_GAZE},
        {'params': model.image_encoder.parameters(), 'lr': config.LEARNING_RATE_IMG}
    ], weight_decay=config.WEIGHT_DECAY)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    start_epoch, best_loss = load_checkpoint(model, optimizer, config.GAZENET_LATEST_PATH, device)
    if best_loss < 0.0001: best_loss = float('inf')

    patience = 0
    
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        start_t = time.time()
        t_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        v_loss = validate_one_epoch(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Epoch {epoch+1:03d}/{config.NUM_EPOCHS} | Time: {time.time()-start_t:.1f}s | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f}")
        
        save_checkpoint(model.state_dict(), optimizer.state_dict(), epoch+1, v_loss, config.GAZENET_LATEST_PATH)
        
        if v_loss < best_loss:
            best_loss = v_loss
            patience = 0
            # Export only the ImageEncoderDINO (GAVE) weights for Stage 2
            torch.save(model.image_encoder.state_dict(), config.GAZENET_BEST_PATH)
            print(f"[*] Optimal GAVE parameters exported. Loss: {best_loss:.4f}")
        else:
            patience += 1
            if patience >= config.EARLY_STOPPING_PATIENCE:
                print("INFO: Early stopping triggered. Convergence reached.")
                break

if __name__ == "__main__":
    main()