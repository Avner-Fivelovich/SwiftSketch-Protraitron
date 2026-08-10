#!/usr/bin/env python3
import os
import glob
import re
import numpy as np
from collections import defaultdict
import warnings

# Suppress numpy warnings about pickle
warnings.filterwarnings("ignore")

def get_file_list(input_dir, max_files=None, allowed_categories=None):
    category_files = defaultdict(list)
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".npz"):
                rel_dir = os.path.relpath(root, input_dir)
                if rel_dir == ".": rel_dir = "" # Handle root level
                if allowed_categories is not None and rel_dir not in allowed_categories:
                    continue
                category_files[rel_dir].append(os.path.join(root, file))
                
    npz_files = []
    if max_files is not None:
        num_categories = len(category_files)
        if num_categories == 0: return []
        files_per_category = max_files // num_categories
        remainder = max_files % num_categories
        
        for i, cat in enumerate(sorted(category_files.keys())):
            files = category_files[cat]
            take_count = files_per_category + (1 if i < remainder else 0)
            files.sort()
            npz_files.extend(files[:take_count])
    else:
        for cat in sorted(category_files.keys()):
            files = category_files[cat]
            files.sort()
            npz_files.extend(files)
            
    return npz_files

def check_progress():
    print("Scanning actual output .npz files for 100% accurate ground truth...")
    
    # 1. Gather expected files (Exact same logic as job generator)
    orig_inputs = get_file_list("ControlSketch/data/train", max_files=5000, allowed_categories=["woman", "angel", "astronaut", "sculpture", "robot"])
    ffhq_inputs = get_file_list("data/ffhq_raw_npz")
    
    datasets = {
        "orig": {
            "inputs": orig_inputs,
            "input_base": "ControlSketch/data/train",
            "output_base": "data/original/controlsketch_96/train"
        },
        "ffhq": {
            "inputs": ffhq_inputs,
            "input_base": "data/ffhq_raw_npz",
            "output_base": "data/ffhq/controlsketch_96/train"
        }
    }
    
    block_counts = defaultdict(lambda: defaultdict(int))
    total_finished = 0
    images_per_job = 100
    num_strokes = 96
    svg_key = f"svg_{num_strokes}s"
    
    # 2. Check physical files
    for prefix, ds in datasets.items():
        for i, src_file in enumerate(ds["inputs"]):
            batch_idx = i // images_per_job
            
            # Reconstruct destination path
            rel_path = os.path.relpath(src_file, ds["input_base"])
            dest_file = os.path.join(ds["output_base"], rel_path)
            
            if os.path.exists(dest_file):
                try:
                    with np.load(dest_file, allow_pickle=True) as loader:
                        if svg_key in loader:
                            block_counts[prefix][batch_idx] += 1
                            total_finished += 1
                except:
                    pass

    # 3. Extract speed from logs (We still need logs for average speed calculation)
    log_dir = "outputs/logs/strokes_96"
    log_files = glob.glob(os.path.join(log_dir, "*.out"))
    total_speed = 0.0
    speed_count = 0
    speed_pattern = re.compile(r"Average speed:\s+([\d\.]+)\s+seconds")
    
    if log_files:
        for log_file in log_files:
            try:
                with open(log_file, "r") as f:
                    for line in f:
                        speed_match = speed_pattern.search(line)
                        if speed_match:
                            total_speed += float(speed_match.group(1))
                            speed_count += 1
            except:
                pass

    avg_speed = total_speed / speed_count if speed_count > 0 else 0.0
    images_per_gpu_per_hour = 3600 / avg_speed if avg_speed > 0 else 0
    cluster_output_per_hour = images_per_gpu_per_hour * 24

    print("=" * 60)
    print("📊 DATASET GENERATION PROGRESS (NPZ GROUND TRUTH)")
    print("=" * 60)
    print(f"✅ Total Images Finished:  {total_finished}")
    print(f"⏱️  Average Speed:         {avg_speed:.2f} seconds/image ({(avg_speed/60):.2f} min)")
    print("-" * 60)
    print("⚡ CLUSTER THROUGHPUT (At 24 GPUs)")
    print(f"🚀 Images per hour:      ~{int(cluster_output_per_hour)}")
    print(f"📅 Images per day:       ~{int(cluster_output_per_hour * 24)}")
    print("-" * 60)
    print("🔍 INCOMPLETE OR MISSING BLOCKS (Based on physical .npz output files)")
    print("-" * 60)
    
    for prefix, ds in datasets.items():
        total_files = len(ds["inputs"])
        num_batches = (total_files + images_per_job - 1) // images_per_job
        
        missing_or_incomplete = []
        for i in range(num_batches):
            count = block_counts[prefix].get(i, 0)
            
            # Calculate expected count for this block (last block might have < 100)
            expected = images_per_job
            if i == num_batches - 1:
                expected = total_files - (i * images_per_job)
                
            if count < expected:
                missing_or_incomplete.append((i, count, expected))
        
        if not missing_or_incomplete:
            print(f"✅ {prefix.upper()}: All {num_batches} blocks appear to be 100% complete!")
        else:
            print(f"⚠️  {prefix.upper()} has {len(missing_or_incomplete)} incomplete blocks:")
            zeros = [str(b[0]) for b in missing_or_incomplete if b[1] == 0]
            partials = [f"{b[0]} ({b[1]}/{b[2]})" for b in missing_or_incomplete if b[1] > 0]
            
            if zeros:
                def collapse_ranges(lst):
                    lst = sorted(int(x) for x in lst)
                    ranges = []
                    start = lst[0]
                    prev = lst[0]
                    for x in lst[1:]:
                        if x == prev + 1:
                            prev = x
                        else:
                            if start == prev: ranges.append(str(start))
                            else: ranges.append(f"{start}-{prev}")
                            start = x
                            prev = x
                    if start == prev: ranges.append(str(start))
                    else: ranges.append(f"{start}-{prev}")
                    return ", ".join(ranges)
                
                print(f"   Missing entirely (0 files): Blocks {collapse_ranges(zeros)}")
            
            if partials:
                display_partials = partials[:15]
                suffix = f" ... and {len(partials)-15} more" if len(partials) > 15 else ""
                print(f"   Partially finished: Blocks {', '.join(display_partials)}{suffix}")
                
    print("=" * 60)

if __name__ == "__main__":
    check_progress()
