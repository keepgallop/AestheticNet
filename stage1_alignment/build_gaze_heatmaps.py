#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data Preparation for Stage 1
Handles raw eye-tracking data parsing, rescaling, and heatmap generation.
"""

import pandas as pd
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import config_stage1 as config

def create_heatmap_for_image(gaze_points_df, width, height, kernel_size=33, sigma=7):
    """Generates a Gaussian-blurred heatmap from fixation points."""
    heatmap = np.zeros((height, width), dtype=np.float32)

    if gaze_points_df.empty:
        return heatmap.astype(np.uint8)

    for _, row in gaze_points_df.iterrows():
        if pd.isna(row.get('x')) or pd.isna(row.get('y')) or pd.isna(row.get('duration')):
            continue
        
        x, y = int(row['x']), int(row['y'])
        duration = row['duration']
        
        if 0 <= x < width and 0 <= y < height:
            heatmap[y, x] += duration

    if np.sum(heatmap) > 0:
        heatmap = cv2.GaussianBlur(heatmap, (kernel_size, kernel_size), sigma)

    max_val = np.max(heatmap)
    if max_val > 0:
        heatmap = (heatmap / max_val) * 255
    
    return heatmap.astype(np.uint8)

def load_all_gaze_data():
    """
    Loads, filters, and standardizes raw eye-tracking data based on config mapping.
    Handles dynamic coordinate rescaling and duration computation.
    """
    print("INFO: Initiating gaze data aggregation...")
    all_filtered_dfs = []
    mapping = config.GAZE_DATA_MAPPING
    
    target_w = config.TARGET_CANVAS_WIDTH
    target_h = config.TARGET_CANVAS_HEIGHT

    for source_name, info in mapping.items():
        filename = info['filename']
        categories_to_keep = info['categories']
        csv_path = config.RAW_GAZE_DATA_DIR / filename
        
        source_w, source_h = info.get('source_w'), info.get('source_h')
        if not source_w or not source_h:
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"WARNING: Failed to load {csv_path} - {e}")
            continue

        if 'duration' not in df.columns and 'start' in df.columns and 'end' in df.columns:
            df['duration'] = df['end'] - df['start']
            
        if 'x' in df.columns and 'y' in df.columns:
            df = df.dropna(subset=['x', 'y'])
            if df.empty:
                continue
                
            max_x, max_y = df['x'].max(), df['y'].max()
            
            # Rescale normalized coordinates
            if max_x <= 1.5 and max_y <= 1.5:
                df['x'] = df['x'] * target_w
                df['y'] = df['y'] * target_h
            # Rescale pixel coordinates proportionally if needed
            elif source_w != target_w or source_h != target_h:
                df['x'] = (df['x'] / source_w) * target_w
                df['y'] = (df['y'] / source_h) * target_h
        
        required_cols = ['category', 'filenumber', 'x', 'y', 'duration']
        if not all(col in df.columns for col in required_cols):
            continue
            
        df_filtered = df[df['category'].isin(categories_to_keep)]
        if not df_filtered.empty:
            all_filtered_dfs.append(df_filtered)

    if not all_filtered_dfs:
        print("ERROR: No valid gaze data loaded.")
        return pd.DataFrame()
        
    df_all_gaze = pd.concat(all_filtered_dfs, ignore_index=True)
    print(f"INFO: Successfully aggregated {len(df_all_gaze)} fixation records.")
    return df_all_gaze

def get_image_path_and_stem(cat_folder_path, f_num_original):
    """Intelligently resolves image extensions (.png, .jpg, .jpeg) given an ID."""
    f_stem = str(int(f_num_original)) if isinstance(f_num_original, float) and f_num_original.is_integer() else str(f_num_original)
    
    for ext in ['.png', '.jpg', '.jpeg']:
        if f_stem.endswith(ext):
            f_stem = f_stem[:-len(ext)]
            
    for ext in ['.png', '.jpg', '.jpeg']:
        potential_path = cat_folder_path / f"{f_stem}{ext}"
        if potential_path.exists():
            return potential_path, f_stem
            
    return None, f_stem

def main():
    print("--- Stage 1: Heatmap Generation ---")
    
    heatmap_dir = getattr(config, 'LABEL_HEATMAP_DIR', config.OUTPUT_ROOT / "heatmaps")
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    
    df_all_gaze = load_all_gaze_data()
    if df_all_gaze.empty: return

    unique_images = df_all_gaze[['category', 'filenumber']].drop_duplicates()
    process_count = 0
    
    for _, row in tqdm(unique_images.iterrows(), total=len(unique_images), desc="Rendering Heatmaps"):
        cat = int(row['category'])
        f_num_original = row['filenumber']
        
        img_gaze_df = df_all_gaze[(df_all_gaze['category'] == cat) & (df_all_gaze['filenumber'] == f_num_original)]
        cat_folder_path = config.IMAGE_DIR / f"Category_{cat}"
        original_img_path, file_stem = get_image_path_and_stem(cat_folder_path, f_num_original)
        
        if original_img_path is None:
            continue

        heatmap_np = create_heatmap_for_image(
            img_gaze_df,
            width=config.TARGET_CANVAS_WIDTH,
            height=config.TARGET_CANVAS_HEIGHT
        )
        
        output_dir = heatmap_dir / f"Category_{cat}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        Image.fromarray(heatmap_np, mode='L').save(output_dir / f"{file_stem}.png")
        process_count += 1

    print(f"INFO: Generated {process_count} heatmaps in {heatmap_dir}")

if __name__ == "__main__":
    main()