#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration for Stage 1: Contrastive Gaze Alignment (CGA)
Optimized for DINOv2-based Gaze-Aligned Visual Encoder (GAVE).
"""

from pathlib import Path

# --- 1. Project Directory Structure ---
# Set the root to the current directory of the repo
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_ROOT = PROJECT_ROOT / "data"

# Input paths
IMAGE_DIR = DATA_ROOT / "images_gaze"
RAW_GAZE_DATA_DIR = DATA_ROOT / "raw_gaze_data"

# Gaze data mapping
GAZE_DATA_MAPPING = {
    'source_baseline': {
        'filename': 'Baseline_1d_data.csv',
        'categories': [7, 8],
        'source_w': 1280,
        'source_h': 960
    },
}

TARGET_CANVAS_WIDTH = 1280
TARGET_CANVAS_HEIGHT = 960
GAZE_INPUT_DIM = 3

# --- 2. Output and Checkpoint Paths ---
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "stage1_cga"
PROCESSED_GAZE_DIR = OUTPUT_ROOT / "processed_sequences"
CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"

GAZENET_BEST_PATH = CHECKPOINT_DIR / "gazenet_dinov2_best.pth"
GAZENET_LATEST_PATH = CHECKPOINT_DIR / "gazenet_dinov2_latest.pth"

# --- 3. Hyperparameters ---
# Gaze Transformer (Encoder Eg)
GAZE_SEQ_LEN = 256
GAZE_EMBED_DIM = 128
TRANSFORMER_HEADS = 8
TRANSFORMER_LAYERS = 6
TRANSFORMER_DROPOUT = 0.1

# Image Encoder (Ei - DINOv2 Small)
IMAGE_EMBED_DIM = 384 
MODEL_INPUT_SIZE = (224, 224) # Optimized for DINOv2 patch size 14

# Training
PROJECTION_DIM = 256 
DEVICE = "cuda"
LEARNING_RATE_IMG = 5e-6   # Small LR for ViT backbone fine-tuning
LEARNING_RATE_GAZE = 1e-4  
WEIGHT_DECAY = 1e-4

BATCH_SIZE = 64 
NUM_EPOCHS = 500
VALIDATION_SPLIT = 0.15 
RANDOM_SEED = 42
NUM_WORKERS = 4 

TEMPERATURE = 0.05
EARLY_STOPPING_PATIENCE = 30