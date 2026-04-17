#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sequence Aggregation for Stage 1
Compiles isolated fixation points into continuous sequence arrays (x, y, duration).
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import config_stage1 as config
from build_gaze_heatmaps import load_all_gaze_data, get_image_path_and_stem

def process_and_save_gaze_sequence(gaze_points_df, output_path):
    """Processes and normalizes the full sequence of fixations for a single image."""
    gaze_points_df = gaze_points_df.dropna(subset=['x', 'y', 'duration'])
    if gaze_points_df.empty:
        return 0

    gaze_data = np.zeros((len(gaze_points_df), config.GAZE_INPUT_DIM))
    
    # Normalize spatial coordinates to [0, 1]
    gaze_data[:, 0] = gaze_points_df['x'].values / config.TARGET_CANVAS_WIDTH
    gaze_data[:, 1] = gaze_points_df['y'].values / config.TARGET_CANVAS_HEIGHT
    
    # Normalize duration (clipped at 1000ms)
    durations = np.clip(gaze_points_df['duration'].values, 0, 1000)
    d_min, d_max = durations.min(), durations.max()
    if d_max > d_min:
        gaze_data[:, 2] = (durations - d_min) / (d_max - d_min)
    else:
        gaze_data[:, 2] = 0.5 
        
    gaze_data = np.clip(gaze_data, 0.0, 1.0)

    try:
        np.save(output_path, gaze_data.astype(np.float32))
        return len(gaze_data)
    except Exception as e:
        print(f"ERROR: Failed to save sequence array: {e}")
        return 0

def main():
    print("--- Stage 1: Sequence Aggregation ---")
    config.PROCESSED_GAZE_DIR.mkdir(parents=True, exist_ok=True)
    
    df_all_gaze = load_all_gaze_data()
    if df_all_gaze.empty: return

    unique_images = df_all_gaze[['category', 'filenumber']].drop_duplicates()
    process_count = 0
    total_points = 0
    
    for _, row in tqdm(unique_images.iterrows(), total=len(unique_images), desc="Compiling Sequences"):
        cat = int(row['category'])
        f_num_original = row['filenumber'] 
        
        img_gaze_df = df_all_gaze[(df_all_gaze['category'] == cat) & (df_all_gaze['filenumber'] == f_num_original)]
        
        cat_folder_path = config.IMAGE_DIR / f"Category_{cat}"
        original_img_path, file_stem = get_image_path_and_stem(cat_folder_path, f_num_original)
        
        if original_img_path is None:
            continue

        output_dir = config.PROCESSED_GAZE_DIR / f"Category_{cat}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{file_stem}.npy"
        num_points = process_and_save_gaze_sequence(img_gaze_df, output_path)
        
        if num_points > 0:
            process_count += 1
            total_points += num_points

    print(f"INFO: Compilation complete. Saved {process_count} sequence arrays ({total_points} total points).")

if __name__ == "__main__":
    main()