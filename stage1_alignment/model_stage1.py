#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import config_stage1 as config

class ImageEncoderDINO(nn.Module):
    """
    Image Encoder (Ei) based on DINOv2 ViT-Small.
    Acts as a computational proxy for the primary visual cortex (V1/V2).
    """
    def __init__(self, projection_dim=256):
        super(ImageEncoderDINO, self).__init__()
        # Load DINOv2 from TorchHub. 
        # Note: In a production repo, we recommend users to have internet access 
        # or pre-download the hub cache.
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        
        # Projection head to align with gaze latent space
        self.projection_head = nn.Sequential(
            nn.Linear(384, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
        
    def forward(self, x):
        # Extract [CLS] token features
        features_dict = self.backbone.forward_features(x)
        cls_token = features_dict['x_norm_clstoken'] 
        
        projected = self.projection_head(cls_token)
        return F.normalize(projected, p=2, dim=1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class GazeTransformer(nn.Module):
    """
    Gaze Sequence Encoder (Eg) to process human oculomotor traces (x, y, duration).
    """
    def __init__(self, input_dim=3, embed_dim=128, nhead=8, num_layers=6, seq_len=256, projection_dim=256):
        super(GazeTransformer, self).__init__()
        self.input_projector = nn.Linear(input_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=seq_len + 1)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim * 4, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, projection_dim)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.input_projector(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        cls_output = x[:, 0, :] 
        return F.normalize(self.projection_head(cls_output), p=2, dim=1)

class AlignmentModelDINO(nn.Module):
    """
    Complete CGA Architecture for Stage 1.
    """
    def __init__(self, projection_dim=256):
        super(AlignmentModelDINO, self).__init__()
        self.image_encoder = ImageEncoderDINO(projection_dim)
        self.gaze_encoder = GazeTransformer(
            input_dim=config.GAZE_INPUT_DIM,
            embed_dim=config.GAZE_EMBED_DIM,
            nhead=config.TRANSFORMER_HEADS,
            num_layers=config.TRANSFORMER_LAYERS,
            seq_len=config.GAZE_SEQ_LEN,
            projection_dim=projection_dim
        )

    def forward(self, image_tensor, gaze_tensor):
        img_feat = self.image_encoder(image_tensor)
        gaze_feat = self.gaze_encoder(gaze_tensor)
        return img_feat, gaze_feat