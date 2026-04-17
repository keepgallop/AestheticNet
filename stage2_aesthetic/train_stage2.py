#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Training Script for Stage 2: AestheticNet
Optimizes the cross-attention fusion network using a hybrid loss (MSE + PLCC penalty).
Features automatic checkpoint resumption, cosine annealing, and AMP scaling.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import logging
import sys
import time
from PIL import ImageFile

import config_stage2 as config
from dataset_stage2 import AestheticDataset
from model_stage2 import AestheticNet

# Ensure robust loading of truncated image files
ImageFile.LOAD_TRUNCATED_IMAGES = True

def setup_logger(log_file):
    """Configures a professional dual-stream logger (Console + File)."""
    logger = logging.getLogger("AestheticNetTrain")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers(): 
        logger.handlers.clear()
    
    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

class PLCCLoss(nn.Module):
    """Differentiable Pearson Linear Correlation Coefficient penalty."""
    def __init__(self):
        super().__init__()

    def forward(self, preds, labels):
        if preds.size(0) < 2: 
            return torch.tensor(0.0, device=preds.device)
        
        preds_mean = torch.mean(preds)
        labels_mean = torch.mean(labels)
        
        preds_centered = preds - preds_mean
        labels_centered = labels - labels_mean
        
        covariance = torch.sum(preds_centered * labels_centered)
        preds_std = torch.sqrt(torch.sum(preds_centered ** 2) + 1e-8)
        labels_std = torch.sqrt(torch.sum(labels_centered ** 2) + 1e-8)
        
        plcc = covariance / (preds_std * labels_std + 1e-8)
        return 1.0 - plcc 

def make_balanced_sampler(dataset):
    """Constructs a weighted sampler to mitigate aesthetic score central tendency bias."""
    targets = dataset.df['y_norm'].values
    bins = np.linspace(0, 1, 11)
    bin_indices = np.digitize(targets, bins) - 1
    bin_counts = np.bincount(bin_indices, minlength=10)
    bin_counts[bin_counts == 0] = 1 
    
    bin_weights = 1.0 / bin_counts
    sample_weights = bin_weights[bin_indices]
    
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(dataset),
        replacement=True
    )

def train_epoch(model, loader, optimizer, criterion_mse, criterion_plcc, scaler, device):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc="Training", leave=False)
    
    alpha_mse, beta_plcc = 1.0, 0.5
    
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        with autocast():
            preds = model(imgs)
            loss_mse = criterion_mse(preds, labels)
            loss_plcc = criterion_plcc(preds, labels)
            loss = alpha_mse * loss_mse + beta_plcc * loss_plcc
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'MSE': f"{loss_mse.item():.4f}"})
        
    return running_loss / len(loader)

def validate(model, loader, criterion_mse, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Validation", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            loss = criterion_mse(preds, labels)
            running_loss += loss.item()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    val_loss = running_loss / len(loader)
    
    # Reverse normalization for accurate metric computation (1-10 scale)
    p_np = np.array(all_preds) * 9.0 + 1.0
    l_np = np.array(all_labels) * 9.0 + 1.0
    
    plcc = np.corrcoef(p_np, l_np)[0, 1] if len(p_np) > 1 else 0.0
    return val_loss, plcc

def save_checkpoint(state, is_best, filename_latest, filename_best):
    torch.save(state, filename_latest)
    if is_best:
        torch.save(state, filename_best)

def main():
    config.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    log_file = config.OUTPUT_ROOT / "train_stage2.log"
    logger = setup_logger(log_file)
    
    logger.info("="*60)
    logger.info("Initializing AestheticNet Stage 2 Training (Dual-Pathway Fusion)")
    logger.info("="*60)
    
    device = torch.device(config.DEVICE)
    
    # 1. Dataset & Loaders
    train_ds = AestheticDataset(config.TRAIN_LIST_FILE, config.AVA_IMAGE_DIR, mode='train')
    val_ds = AestheticDataset(config.TEST_LIST_FILE, config.AVA_IMAGE_DIR, mode='val')
    
    train_sampler = make_balanced_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, sampler=train_sampler, num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)

    # 2. Model Initialization
    logger.info("Constructing dual-pathway architecture...")
    model = AestheticNet(gaze_model_path=config.GAZE_MODEL_PATH).to(device)
    
    # 3. Optimization Strategy
    param_groups = [
        {'params': model.branch_F.parameters(), 'lr': config.BASE_LEARNING_RATE * config.FINE_TUNE_SCALE},
        {'params': model.c_projector.parameters(), 'lr': config.BASE_LEARNING_RATE},
        {'params': model.fusion.parameters(), 'lr': config.BASE_LEARNING_RATE}, 
        {'params': model.regressor.parameters(), 'lr': config.BASE_LEARNING_RATE}
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=1e-7)
    
    criterion_mse = nn.MSELoss()
    criterion_plcc = PLCCLoss()
    scaler = GradScaler()

    # 4. Auto-Resume Logic
    start_epoch, patience_counter = 0, 0
    best_plcc = -1.0
    latest_ckpt = config.CHECKPOINT_DIR / "aesthetic_net_latest.pth"

    if latest_ckpt.exists():
        logger.info(f"Discovered existing checkpoint. Resuming from {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        scaler.load_state_dict(ckpt['scaler_state_dict'])
        
        start_epoch = ckpt['epoch'] + 1
        best_plcc = ckpt.get('best_plcc', -1.0)
        patience_counter = ckpt.get('patience_counter', 0)
        logger.info(f"Resumed successfully. Best PLCC history: {best_plcc:.4f}")

    # 5. Main Training Loop
    logger.info("Commencing hybrid optimization (MSE + PLCC penalty)...")
    
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        logger.info(f"\n--- Epoch {epoch+1:03d}/{config.NUM_EPOCHS} ---")
        start_t = time.time()
        
        t_loss = train_epoch(model, train_loader, optimizer, criterion_mse, criterion_plcc, scaler, device)
        v_loss, v_plcc = validate(model, val_loader, criterion_mse, device)
        
        scheduler.step()
        
        logger.info(f"Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | LR: {optimizer.param_groups[-1]['lr']:.2e}")
        logger.info(f"Val PLCC: {v_plcc:.4f} (Best: {best_plcc:.4f}) | Elapsed: {time.time()-start_t:.0f}s")
        
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_plcc': best_plcc,
            'patience_counter': patience_counter
        }
        
        if v_plcc > best_plcc:
            best_plcc = v_plcc
            patience_counter = 0
            logger.info(">>> New optimal convergence achieved. Saving checkpoint.")
            save_checkpoint(state, True, latest_ckpt, config.AESTHETIC_NET_BEST_PATH)
        else:
            patience_counter += 1
            logger.info(f"Patience counter: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}")
            save_checkpoint(state, False, latest_ckpt, config.AESTHETIC_NET_BEST_PATH)

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            logger.info("Early stopping criteria met. Terminating training.")
            break
            
    logger.info(f"Training finalized. Maximum Validation PLCC: {best_plcc:.4f}")

if __name__ == "__main__":
    main()