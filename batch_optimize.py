import os
import argparse
import subprocess
import glob

def main():
    parser = argparse.ArgumentParser(description="Batch optimize images using SwiftSketch.")
    parser.add_argument("--input_dir", type=str, default="Pictures/ffhq", help="Directory containing target images")
    parser.add_argument("--output_base_dir", type=str, default="outputs/ffhq_runs", help="Base directory for outputs")
    parser.add_argument("--num_strokes", type=int, default=98, help="Number of strokes")
    parser.add_argument("--num_iter", type=int, default=1200, help="Number of optimization iterations")
    parser.add_argument("--feather_face_mask", type=int, default=3, help="Feather face mask mode")
    parser.add_argument("--condition", type=str, default="depth", help="ControlNet conditioning mode")
    parser.add_argument("--object_name", type=str, default="face", help="Object name for attention mapping")
    
    args, unknown = parser.parse_known_args()
    
    # Find all images
    extensions = ("*.png", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.PNG")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(args.input_dir, ext)))
        
    image_paths = sorted(list(set(image_paths)))
    
    if not image_paths:
        print(f"No images found in {args.input_dir}")
        return
        
    print(f"Found {len(image_paths)} images to optimize.")
    
    for idx, img_path in enumerate(image_paths):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        print(f"\n==================================================")
        print(f"Processing image {idx+1}/{len(image_paths)}: {basename}")
        print(f"==================================================")
        
        output_dir = os.path.join(args.output_base_dir, basename)
        wandb_name = f"{basename}_98_strokes_edges"
        
        # Build command list
        cmd = [
            "python", "ControlSketch/object_sketching.py",
            "--target", img_path,
            "--num_strokes", str(args.num_strokes),
            "--num_iter", str(args.num_iter),
            "--feather_face_mask", str(args.feather_face_mask),
            "--output_dir", output_dir,
            "--wandb_name", wandb_name,
            "--condition", args.condition,
            "--object_name", args.object_name,
        ]
        
        # Forward any additional command line arguments
        cmd.extend(unknown)
        
        print(f"Executing command: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            print(f"Successfully processed {basename}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {basename}: {e}")
            print("Skipping and moving to next image...")

if __name__ == "__main__":
    main()
