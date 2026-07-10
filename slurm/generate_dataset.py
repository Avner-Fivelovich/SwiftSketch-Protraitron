import argparse
import os
import shutil
import subprocess
import sys
import torch
import numpy as np
from PIL import Image
import io
import time

# Add SwiftSketch to path to import features extractor
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'SwiftSketch'))
from model.image_features_models import CLIPMidlleFeutures

def extract_clip_features(image_pil, features_model, device):
    """
    Extract CLIP middle layer features (layer 4) for SwiftSketch training.
    """
    with torch.no_grad():
        features = features_model(image_pil)
    return features.cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description="Multi-Scale Dataset Generation Script")
    parser.add_argument("--num_strokes", type=int, required=True, help="Target stroke count (e.g., 16, 48, 64, 96, 128)")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing source .npz files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the customized dataset")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process (for debugging or testing)")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index for processing files (for splitting/parallelizing jobs)")
    parser.add_argument("--limit", type=int, default=None, help="Number of files to process in this job (for splitting/parallelizing jobs)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Find all .npz files in the input directory (recursive)
    npz_files = []
    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith(".npz"):
                npz_files.append(os.path.join(root, file))
    
    npz_files = sorted(npz_files)
    if args.max_samples is not None:
        npz_files = npz_files[:args.max_samples]

    # Slice list of files based on start_idx and limit
    start_idx = args.start_idx
    if args.limit is not None:
        end_idx = start_idx + args.limit
        npz_files = npz_files[start_idx:end_idx]
    else:
        npz_files = npz_files[start_idx:]

    if not npz_files:
        print(f"No .npz files to process in range starting at index {start_idx} (limit {args.limit}) under {args.input_dir}")
        sys.exit(0)

    print(f"Found {len(npz_files)} files to process in this batch (range start: {start_idx}, limit: {args.limit}) for {args.num_strokes} strokes.", flush=True)

    # Initialize CLIP model for feature extraction
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Initializing CLIP features model on {device}...", flush=True)
    features_model = CLIPMidlleFeutures(device, 3).to(device)
    features_model.eval()

    # Create temporary output directory for ControlSketch logs (append start_idx to avoid race conditions in parallel executions)
    temp_sketch_logs = os.path.join(args.output_dir, f"_temp_sketch_logs_{args.start_idx}_{args.num_strokes}")
    os.makedirs(temp_sketch_logs, exist_ok=True)

    durations = []
    processed_count = 0
    consecutive_failures = 0
    total_start_time = time.time()

    for idx, src_file in enumerate(npz_files):
        print(f"\n[{idx+1}/{len(npz_files)}] Processing: {os.path.basename(src_file)}", flush=True)
        img_start_time = time.time()
        
        # Determine relative path to maintain folder structure
        rel_path = os.path.relpath(src_file, args.input_dir)
        dest_file = os.path.join(args.output_dir, rel_path)
        os.makedirs(os.path.dirname(dest_file), exist_ok=True)

        # Copy the original file to destination
        shutil.copy2(src_file, dest_file)

        # Load file data to extract PIL Image for CLIP features
        data = dict(np.load(dest_file, allow_pickle=True))
        
        # Reconstruct PIL image
        img_bytes = data["image"].tobytes()
        image_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Extract and add CLIP features if missing
        features_key = "CLIPMiddle_layer4_features"
        if features_key not in data:
            print("  Extracting CLIP features...", flush=True)
            clip_feat = extract_clip_features(image_pil, features_model, device)
            data[features_key] = clip_feat
            np.savez_compressed(dest_file, **data)

        # Run ControlSketch optimization to generate svg_<num_strokes>s key
        print(f"  Optimizing {args.num_strokes} strokes using ControlSketch...", flush=True)
        
        # Copy environment and enforce unbuffered output so child logs stream immediately
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        cmd = [
            sys.executable, "ControlSketch/object_sketching.py",
            "--target", dest_file,
            "--num_strokes", str(args.num_strokes),
            "--save_svg_in_dict", "1",
            "--output_dir", temp_sketch_logs,
            "--use_cpu", "0" if device.type in ["cuda", "mps"] else "1"
        ]
        
        # Execute the optimization
        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            print(f"  Warning: Optimization failed for {src_file}. Error code: {result.returncode}", flush=True)
            consecutive_failures += 1
            if consecutive_failures >= 3:
                print("  FATAL ERROR: 3 consecutive failures encountered. Aborting dataset generation to save compute resources.", flush=True)
                # Cleanup temp logs directory before exiting
                try:
                    shutil.rmtree(temp_sketch_logs)
                except:
                    pass
                sys.exit(1)
        else:
            consecutive_failures = 0
            duration = time.time() - img_start_time
            durations.append(duration)
            processed_count += 1
            
            avg_duration = sum(durations) / len(durations)
            remaining_files = len(npz_files) - processed_count
            etc_seconds = remaining_files * avg_duration
            
            etc_hours = int(etc_seconds // 3600)
            etc_mins = int((etc_seconds % 3600) // 60)
            
            print(f"  Successfully optimized and updated: {dest_file}", flush=True)
            print(f"  Image processed in {duration:.2f} seconds ({duration/60:.2f} minutes).", flush=True)
            print(f"  Average speed: {avg_duration:.2f} seconds/image.", flush=True)
            print(f"  Estimated Time of Completion (ETC): {etc_hours}h {etc_mins}m remaining.", flush=True)

    # Cleanup temp logs directory
    try:
        shutil.rmtree(temp_sketch_logs)
    except Exception as e:
        pass

    total_duration = time.time() - total_start_time
    print(f"\nDataset generation completed successfully! Total elapsed time: {total_duration/3600:.2f} hours.", flush=True)

if __name__ == "__main__":
    main()
