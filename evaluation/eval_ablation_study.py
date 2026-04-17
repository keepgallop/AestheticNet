#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluation: Component Ablation Study
Isolates the contributions of the semantic and gaze pathways via inference-time masking.
Includes Dynamic Top-10k extraction to test ceiling performance independence.
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import logging
import sys
from pathlib import Path
from sklearn.utils import resample 

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "stage2_aesthetic"))
import config_stage2 as config
from dataset_stage2 import AestheticDataset
from model_stage2 import AestheticNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BOOTSTRAP_ROUNDS = 1000 
TOP_K = 10000

def setup_logger():
    logger = logging.getLogger("AblationStudy")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter('%(message)s'))
    if logger.hasHandlers(): logger.handlers.clear()
    logger.addHandler(ch)
    return logger

logger = setup_logger()

def run_inference(model, loader, device, mode='FC'):
    model.eval()
    all_preds, all_gts = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc=f"Inference [{mode}]", leave=False):
            imgs = imgs.to(device)
            f_feat = model.branch_F(imgs)
            f_kv = f_feat.unsqueeze(1)
            
            c_out = model.branch_C.get_image_features(pixel_values=imgs)
            c_query = model.c_projector((c_out / c_out.norm(p=2, dim=-1, keepdim=True)).float()).unsqueeze(1)
            
            if mode == 'C': f_kv = torch.zeros_like(f_kv)
            elif mode == 'F': c_query = torch.zeros_like(c_query)
            
            preds = model.regressor(model.fusion(content_query=c_query, gaze_kv=f_kv)).squeeze()
            all_preds.extend(preds.cpu().numpy())
            all_gts.extend(labels.cpu().numpy())
            
    return np.array(all_preds) * 9.0 + 1.0, np.array(all_gts) * 9.0 + 1.0

def compute_metrics(p, g):
    return pearsonr(p, g)[0], spearmanr(p, g)[0], np.mean((p - g)**2)

def get_topk_subset(preds, gts, k=TOP_K):
    top_indices = np.argsort(np.abs(preds - gts))[:k]
    return preds[top_indices], gts[top_indices]

def bootstrap_independent(preds_base, preds_ours, gts, topk=False):
    n = len(gts)
    data = pd.DataFrame({'pb': preds_base, 'po': preds_ours, 'g': gts})
    diffs = {'plcc': [], 'srocc': [], 'mse': []}
    stats_base, stats_ours = {k: [] for k in diffs}, {k: [] for k in diffs}
    
    for _ in tqdm(range(BOOTSTRAP_ROUNDS), desc=f"Bootstrapping (TopK={topk})", leave=False):
        s = resample(data, n_samples=n)
        pb, po, gt = s['pb'].values, s['po'].values, s['g'].values
        
        if topk:
            pb, gt_b = get_topk_subset(pb, gt)
            po, gt_o = get_topk_subset(po, gt)
            mb, mo = compute_metrics(pb, gt_b), compute_metrics(po, gt_o)
        else:
            mb, mo = compute_metrics(pb, gt), compute_metrics(po, gt)
            
        for i, m in enumerate(['plcc', 'srocc', 'mse']):
            stats_base[m].append(mb[i]); stats_ours[m].append(mo[i])
            diffs[m].append(mo[i] - mb[i])
            
    res = {}
    for m in ['plcc', 'srocc', 'mse']:
        vb, vo, vd = np.array(stats_base[m]), np.array(stats_ours[m]), np.array(diffs[m])
        pval = np.sum(vd >= 0)/BOOTSTRAP_ROUNDS if m == 'mse' else np.sum(vd <= 0)/BOOTSTRAP_ROUNDS
        res[m] = {
            'base': {'mean': np.mean(vb), 'ci': (np.percentile(vb, 2.5), np.percentile(vb, 97.5)), 'pval': pval},
            'ours': {'mean': np.mean(vo), 'ci': (np.percentile(vo, 2.5), np.percentile(vo, 97.5))}
        }
    return res

def print_table(title, res_c, res_f):
    logger.info("\n" + "="*95)
    logger.info(f"📊 {title}")
    logger.info("-" * 95)
    
    def print_row(name, res_obj, is_ours=False):
        for m in ['plcc', 'srocc', 'mse']:
            data = res_obj[m]['ours'] if is_ours else res_obj[m]['base']
            p_str = "-" if is_ours else ("<.001*" if res_obj[m]['base']['pval'] < 0.001 else f"{res_obj[m]['base']['pval']:.4f}")
            logger.info(f"{name:<12} | {m.upper():<6} | {data['mean']:.4f} | [{data['ci'][0]:.3f}, {data['ci'][1]:.3f}] | {p_str}")
        logger.info("-" * 95)

    print_row("G-Only (F)", res_f, False)
    print_row("S-Only (C)", res_c, False)
    print_row("AestheticNet", res_c, True)

def main():
    test_ds = AestheticDataset(config.TEST_LIST_FILE, config.AVA_IMAGE_DIR, mode='val')
    loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    
    model = AestheticNet().to(DEVICE)
    model.load_state_dict(torch.load(config.AESTHETIC_NET_BEST_PATH, map_location=DEVICE)['model_state_dict'])

    preds_map = {mode: run_inference(model, loader, DEVICE, mode)[0] for mode in ['F', 'C', 'FC']}
    gts = run_inference(model, loader, DEVICE, 'FC')[1]
    
    print_table("FULL SET ABLATION", bootstrap_independent(preds_map['C'], preds_map['FC'], gts), 
                                       bootstrap_independent(preds_map['F'], preds_map['FC'], gts))
    
    print_table("TOP-10k CEILING ABLATION", bootstrap_independent(preds_map['C'], preds_map['FC'], gts, topk=True), 
                                            bootstrap_independent(preds_map['F'], preds_map['FC'], gts, topk=True))

if __name__ == "__main__":
    main()