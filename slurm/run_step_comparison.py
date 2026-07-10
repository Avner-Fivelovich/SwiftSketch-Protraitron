import os
import shutil
import subprocess
import sys
import numpy as np
from PIL import Image
import io

def main():
    # 6 target files from the dataset
    input_dir = "ControlSketch/data/train"
    output_comparison_dir = "data/step_comparison"
    os.makedirs(output_comparison_dir, exist_ok=True)

    # Walk to find first 6 .npz files
    npz_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".npz"):
                npz_files.append(os.path.join(root, file))
                if len(npz_files) >= 6:
                    break
        if len(npz_files) >= 6:
            break

    print(f"Selected 6 images for step comparison experiment:")
    for f in npz_files:
        print(f" - {f}")

    device = "mps" # Enforce Apple Silicon GPU

    for idx, src_file in enumerate(npz_files):
        img_name = os.path.basename(src_file).replace(".npz", "")
        img_dir = os.path.join(output_comparison_dir, f"image_{idx+1}_{img_name}")
        os.makedirs(img_dir, exist_ok=True)
        
        print(f"\n==========================================")
        print(f"Processing Image {idx+1}/6: {img_name}")
        print(f"==========================================")

        # Copy original npz to destination
        dest_npz = os.path.join(img_dir, f"{img_name}.npz")
        shutil.copy2(src_file, dest_npz)

        # Extract and save original PNG image for reference comparison
        data = dict(np.load(src_file, allow_pickle=True))
        img_bytes = data["image"].tobytes()
        image_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        image_pil.save(os.path.join(img_dir, "original.png"))

        # Run object_sketching optimization (2000 iterations, saving every 100 steps)
        cmd = [
            sys.executable, "ControlSketch/object_sketching.py",
            "--target", dest_npz,
            "--num_strokes", "16",
            "--save_svg_in_dict", "1",
            "--output_dir", img_dir,
            "--use_cpu", "0",
            "--save_interval", "100"
        ]
        
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        print(f"Running ControlSketch optimization for {img_name}...")
        result = subprocess.run(cmd, env=env)
        
        if result.returncode == 0:
            print(f"Successfully finished optimization for {img_name}!")
        else:
            print(f"Error occurred during optimization for {img_name}. Code: {result.returncode}")

    print("\nStep comparison experiment completed! Check data/step_comparison/ for the results.")

if __name__ == "__main__":
    main()
