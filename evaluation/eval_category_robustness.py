#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluation: Category-wise Robustness
Decomposes performance across 8 distinct compositional domains to validate
the universal necessity of the oculomotor prior (S-Only vs. AestheticNet).
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import logging
import sys
from pathlib import Path
from sklearn.utils import resample

# --- Dynamic Imports ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "stage2_aesthetic"))
import config_stage2 as config
from dataset_stage2 import AestheticDataset
from model_stage2 import AestheticNet

# --- Configuration ---
MODEL_PATH = config.AESTHETIC_NET_BEST_PATH
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BOOTSTRAP_ROUNDS = 1000 

TARGET_CATEGORIES = {
    15: 'Nature', 14: 'Landscape', 2: 'Cityscape', 20: 'Architecture',  
    18: 'Still Life', 28: 'Water', 7: 'Sky', 27: 'Rural'          
}

def setup_logger():
    logger = logging.getLogger("CategoryRobustness")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter('%(message)s'))
    if logger.hasHandlers(): logger.handlers.clear()
    logger.addHandler(ch)
    return logger

logger = setup_logger()

def run_dual_inference(model, loader, device):
    """Executes single-pass dual inference using masking (Semantic-Only vs. Dual-Branch)."""
    model.eval()
    preds_c, preds_fc, gts = [], [], []
    
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Dual Inference", leave=False):
            imgs = imgs.to(device)
            
            f_feat = model.branch_F(imgs)
            f_kv = f_feat.unsqueeze(1)
            
            c_out = model.branch_C.get_image_features(pixel_values=imgs)
            c_out = c_out / c_out.norm(p=2, dim=-1, keepdim=True)
            c_query = model.c_projector(c_out.float()).unsqueeze(1)
            
            # Semantic-Only (Graceful degradation via residual connection)
            feat_c = model.fusion(content_query=c_query, gaze_kv=torch.zeros_like(f_kv))
            preds_c.extend(model.regressor(feat_c).squeeze().cpu().numpy())
            
            # Full AestheticNet
            feat_fc = model.fusion(content_query=c_query, gaze_kv=f_kv)
            preds_fc.extend(model.regressor(feat_fc).squeeze().cpu().numpy())
            
            gts.extend(labels.cpu().numpy())
            
    return np.array(preds_c) * 9.0 + 1.0, np.array(preds_fc) * 9.0 + 1.0, np.array(gts) * 9.0 + 1.0

def bootstrap_stats(p_c, p_fc, gt):
    n = len(gt)
    if n < 10: return None
    data = pd.DataFrame({'c': p_c, 'fc': p_fc, 'g': gt})
    stats_c, stats_fc, diffs = {'plcc': [], 'mse': []}, {'plcc': [], 'mse': []}, {'plcc': [], 'mse': []}
    
    for _ in range(BOOTSTRAP_ROUNDS):
        s = resample(data, n_samples=n)
        sc, sfc, sg = s['c'].values, s['fc'].values, s['g'].values
        
        plcc_c, mse_c = pearsonr(sc, sg)[0], np.mean((sc - sg)**2)
        plcc_fc, mse_fc = pearsonr(sfc, sg)[0], np.mean((sfc - sg)**2)
        
        stats_c['plcc'].append(plcc_c); stats_c['mse'].append(mse_c)
        stats_fc['plcc'].append(plcc_fc); stats_fc['mse'].append(mse_fc)
        diffs['plcc'].append(plcc_fc - plcc_c); diffs['mse'].append(mse_fc - mse_c)

    def summarize(vals): return np.mean(vals), (np.percentile(vals, 2.5), np.percentile(vals, 97.5))
    
    return {
        'N': n,
        'C': {'plcc': summarize(stats_c['plcc']), 'mse': summarize(stats_c['mse'])},
        'FC': {'plcc': summarize(stats_fc['plcc']), 'mse': summarize(stats_fc['mse'])},
        'P': {'plcc': np.sum(np.array(diffs['plcc']) <= 0) / BOOTSTRAP_ROUNDS}
    }

def main():
    logger.info("="*100)
    logger.info("TABLE: CATEGORY-WISE ROBUSTNESS")
    logger.info("="*100)
    
    df_test = pd.read_csv(config.TEST_LIST_FILE)
    test_ds = AestheticDataset(config.TEST_LIST_FILE, config.AVA_IMAGE_DIR, mode='val')
    loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    
    model = AestheticNet().to(DEVICE)
    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE)['model_state_dict'])
    else:
        logger.error(f"Weights not found: {MODEL_PATH}")
        return

    all_c, all_fc, all_gt = run_dual_inference(model, loader, DEVICE)
    
    logger.info(f"{'Category':<14} | {'Model':<6} | {'PLCC [95% CI]':<26} | {'MSE [95% CI]':<26} | {'P-Val'}")
    logger.info("-" * 100)
    
    for tag_id, tag_name in TARGET_CATEGORIES.items():
        indices = df_test.index[df_test['tag_id'] == tag_id].tolist()
        if not indices: continue
            
        res = bootstrap_stats(all_c[indices], all_fc[indices], all_gt[indices])
        if not res: continue
        
        fmt = lambda m, ci: f"{m:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
        logger.info(f"{tag_name:<14} | {'S-Only':<6} | {fmt(*res['C']['plcc']):<26} | {fmt(*res['C']['mse']):<26} | -")
        
        p_str = "<.001*" if res['P']['plcc'] < 0.001 else f"{res['P']['plcc']:.3f}*"
        logger.info(f"{'N='+str(res['N']):<14} | {'Dual':<6} | {fmt(*res['FC']['plcc']):<26} | {fmt(*res['FC']['mse']):<26} | {p_str}")
        logger.info("-" * 100)

if __name__ == "__main__":
    main()