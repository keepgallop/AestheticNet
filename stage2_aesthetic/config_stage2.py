#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration for Stage 2: AestheticNet (Dual-Pathway Fusion)
Defines hyperparameters, paths, and training settings.
"""

from pathlib import Path

# --- Project Paths ---
# Navigate to the root directory (AestheticNet/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

# Dataset paths
AVA_IMAGE_DIR = DATA_ROOT / "images" 
PREPROCESSED_DIR = DATA_ROOT / "preprocessed"
TRAIN_LIST_FILE = PREPROCESSED_DIR / "train_list.csv"
TEST_LIST_FILE = PREPROCESSED_DIR / "test_list.csv"

# --- Stage 1 Dependencies ---
# Load the pre-trained Gaze-Aligned Visual Encoder (GAVE) weights from Stage 1
GAZE_MODEL_PATH = (
    PROJECT_ROOT / 
    "stage1_alignment" / 
    "outputs" / 
    "stage1_cga" / 
    "checkpoints" / 
    "gazenet_dinov2_best.pth"
)

# --- Stage 2 Outputs ---
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "stage2_aesthetic"
CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
AESTHETIC_NET_BEST_PATH = CHECKPOINT_DIR / "aesthetic_net_best.pth"

# --- Training Hyperparameters ---
MODEL_INPUT_SIZE = (224, 224) 
DEVICE = "cuda" 

BASE_LEARNING_RATE = 1e-4  
FINE_TUNE_SCALE = 0.1      

BATCH_SIZE = 32            
NUM_EPOCHS = 200           
EARLY_STOPPING_PATIENCE = 50
NUM_WORKERS = 4
RANDOM_SEED = 42