#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AestheticNet Architecture
Integrates the Gaze-Aligned Visual Encoder (GAVE) with a Semantic Branch (CLIP)
via Cross-Attention Fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel
import sys
from pathlib import Path

# Dynamically add the project root to import Stage 1 modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from stage1_alignment.model_stage1 import ImageEncoderDINO
except ImportError:
    raise ImportError("Failed to import ImageEncoderDINO. Ensure 'stage1_alignment' exists in the project root.")

class SemanticGazeAttention(nn.Module):
    """
    Cross-Attention mechanism: Semantic intents actively query perceptual features.
    """
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, content_query, gaze_kv):
        # Query: Semantic context (B, 1, 256) | Key/Value: Perceptual structure (B, 1, 256)
        attn_out, _ = self.multihead_attn(query=content_query, key=gaze_kv, value=gaze_kv)
        
        # Residual connection: crucial for preventing OOD collapse during single-branch ablation
        x = self.norm1(content_query + self.dropout(attn_out))
        x = self.norm2(x + self.ffn(x))
        return x.squeeze(1)

class AestheticNet(nn.Module):
    def __init__(self, gaze_model_path=None, local_clip_path=None):
        super().__init__()
        
        # --- Visual Pathway (GAVE) ---
        self.branch_F = ImageEncoderDINO(projection_dim=256)
        
        if gaze_model_path and Path(gaze_model_path).exists():
            print(f"INFO: Loading pretrained GAVE weights from {gaze_model_path}")
            ckpt = torch.load(gaze_model_path, map_location='cpu')
            state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            clean = {k.replace('image_encoder.', ''): v for k, v in state.items() if 'image_encoder' in k}
            if not clean: clean = state 
            self.branch_F.load_state_dict(clean, strict=False)
        else:
            print("WARNING: Pretrained GAVE weights not found. Using default initialization.")
            
        # --- Semantic Pathway (CLIP) ---
        if local_clip_path and Path(local_clip_path).exists():
            print(f"INFO: Loading local CLIP from {local_clip_path}")
            self.branch_C = CLIPModel.from_pretrained(str(local_clip_path), local_files_only=True)
        else:
            print("INFO: Initializing CLIP from HuggingFace Hub (openai/clip-vit-large-patch14)...")
            self.branch_C = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")

        # Freeze the Semantic Encoder to act as a stable semantic prior
        for p in self.branch_C.parameters(): 
            p.requires_grad = False 
        
        clip_dim = self.branch_C.config.projection_dim
        self.c_projector = nn.Sequential(
            nn.Linear(clip_dim, 512), 
            nn.LayerNorm(512), 
            nn.GELU(), 
            nn.Dropout(0.2), 
            nn.Linear(512, 256)
        )
        
        # --- Fusion & Regression ---
        self.fusion = SemanticGazeAttention(embed_dim=256)
        
        self.regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        # 1. Perceptual form extraction
        f_feat = self.branch_F(img)
        f_kv = f_feat.unsqueeze(1)
        
        # 2. Semantic content extraction
        with torch.no_grad():
            c_out = self.branch_C.get_image_features(pixel_values=img)
            c_out = c_out / c_out.norm(p=2, dim=-1, keepdim=True)
            
        c_feat = self.c_projector(c_out.float())
        c_query = c_feat.unsqueeze(1)
        
        # 3. Cross-Attention Fusion
        final_feat = self.fusion(content_query=c_query, gaze_kv=f_kv)
        
        # 4. Final aesthetic score prediction
        return self.regressor(final_feat).squeeze()