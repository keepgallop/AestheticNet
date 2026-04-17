#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data Preparation Utility: AVA Dataset Filter
Filters the massive raw AVA dataset down to the 8 targeted architectural/compositional categories 
used in the Cognitive Visual Pathway experiments.
"""

import pandas as pd
from pathlib import Path
import shutil
import argparse
from tqdm import tqdm
import sys

# Target compositional categories defined in the AestheticNet study
TARGET_TAG_NAMES = [
    "Nature",
    "Landscape",
    "Cityscape",
    "Architecture",
    "Still Life", 
    "Water",
    "Sky", 
    "Rural"
]

def load_tag_mapping(tags_file_path):
    """Reads tags.txt to map category names to tag IDs."""
    tag_map = {}
    if not tags_file_path.exists():
        print(f"ERROR: Cannot find tags definition file at {tags_file_path}")
        sys.exit(1)
        
    with open(tags_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                tag_id = int(parts[0])
                tag_name = " ".join(parts[1:])
                tag_map[tag_name] = tag_id
    return tag_map

def main():
    parser = argparse.ArgumentParser(description="Filter AVA dataset for AestheticNet.")
    parser.add_argument("--source_dir", type=str, required=True, help="Path to the downloaded raw AVA dataset root.")
    parser.add_argument("--output_dir", type=str, default="./ava_filtered", help="Output directory for filtered dataset.")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    image_source_dir = source_dir / "images"
    ava_txt_file = source_dir / "AVA.txt"
    tags_file = source_dir / "tags.txt"

    output_base = Path(args.output_dir)
    output_image_dir = output_base / "images"
    output_annotations = output_base / "filtered_annotations.csv"

    print("--- AestheticNet Data Preparation ---")
    
    if output_base.exists():
        print(f"WARNING: Output directory {output_base} already exists. Cleaning...")
        shutil.rmtree(output_base)
    output_image_dir.mkdir(parents=True, exist_ok=True)

    full_tag_map = load_tag_mapping(tags_file)
    target_tag_ids = []
    
    print("Target Cognitive Categories:")
    for name in TARGET_TAG_NAMES:
        if name in full_tag_map:
            target_tag_ids.append(full_tag_map[name])
            print(f"  [ID: {full_tag_map[name]}] {name}")
        else:
            print(f"  [!] Missing from tags definition: {name}")

    if not target_tag_ids:
        print("ERROR: No valid target tags resolved.")
        sys.exit(1)

    print(f"\nScanning master annotation file: {ava_txt_file}")
    filtered_data = []
    total_rows = 0
    
    try:
        # Stream processing to handle massive TXT files
        for chunk in pd.read_csv(ava_txt_file, sep=r'\s+', header=None, chunksize=10000, engine='python', on_bad_lines='skip'):
            mask = chunk.iloc[:, 12].isin(target_tag_ids) | chunk.iloc[:, 13].isin(target_tag_ids)
            valid_rows = chunk[mask]
            
            if not valid_rows.empty:
                for _, row in valid_rows.iterrows():
                    img_id = int(row.iloc[1])
                    votes = row.iloc[2:12].values.astype(int)
                    
                    t1, t2 = row.iloc[12], row.iloc[13]
                    final_tag = t1 if t1 in target_tag_ids else t2
                    if final_tag == 0: final_tag = t1
                    
                    if (image_source_dir / f"{img_id}.jpg").exists():
                        record = {
                            'image_id': img_id,
                            'tag_id': final_tag,
                            'tag_name': [k for k, v in full_tag_map.items() if v == final_tag][0]
                        }
                        for i, v in enumerate(votes):
                            record[f'votes_{i+1}'] = v
                        filtered_data.append(record)
            
            total_rows += len(chunk)
            print(f"\rScanned {total_rows} records. Found {len(filtered_data)} matching images.", end="")
            
    except Exception as e:
        print(f"\nERROR: Failed to process AVA.txt: {e}")
        sys.exit(1)

    print(f"\n\nScan complete. Extracting {len(filtered_data)} images to {output_image_dir}")
    
    if not filtered_data:
        print("No matches found. Exiting.")
        sys.exit(1)
        
    df_out = pd.DataFrame(filtered_data)
    df_out.to_csv(output_annotations, index=False)
    
    for _, row in tqdm(df_out.iterrows(), total=len(df_out), desc="Copying Image Files"):
        img_id = row['image_id']
        src_file = image_source_dir / f"{img_id}.jpg"
        dst_file = output_image_dir / f"{img_id}.jpg"
        try:
            shutil.copy2(src_file, dst_file)
        except Exception as e:
            print(f"Failed to copy {img_id}: {e}")

    print("\n[Data Preparation Successful]")
    print(f"Subset Annotations: {output_annotations}")
    print(f"Dataset ready for model training pipeline.")

if __name__ == "__main__":
    main()