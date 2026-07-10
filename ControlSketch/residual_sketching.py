import os
import sys
import argparse
import subprocess
import shutil
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image

def mask_image_with_sketch(original_image_path, sketch_image_path, output_masked_path):
    print(f"Masking original image: {original_image_path} with sketch: {sketch_image_path}...")
    orig_img = Image.open(original_image_path).convert("RGB")
    sketch_img = Image.open(sketch_image_path).convert("RGB")
    
    # Resize sketch to match original image size
    sketch_img = sketch_img.resize(orig_img.size, Image.BILINEAR)
    
    orig_arr = np.array(orig_img) / 255.0
    sketch_arr = np.array(sketch_img) / 255.0
    
    # Multiply element-wise to mask out the lines that have already been sketched
    masked_arr = orig_arr * sketch_arr
    
    masked_img = Image.fromarray((masked_arr * 255.0).astype(np.uint8))
    masked_img.save(output_masked_path)
    print(f"Masked residual image saved to: {output_masked_path}")

def merge_svgs(svg_path1, svg_path2, output_svg_path):
    print(f"Merging SVGs: {svg_path1} + {svg_path2}...")
    ET.register_namespace('', 'http://www.w3.org/2000/svg')
    
    tree1 = ET.parse(svg_path1)
    tree2 = ET.parse(svg_path2)
    
    root1 = tree1.getroot()
    root2 = tree2.getroot()
    
    # Locate the <g> group wrapper in both SVGs
    g1 = root1.find('.//{http://www.w3.org/2000/svg}g')
    g2 = root2.find('.//{http://www.w3.org/2000/svg}g')
    
    if g1 is None:
        g1 = root1.find('.//g')
    if g2 is None:
        g2 = root2.find('.//g')
        
    if g1 is not None and g2 is not None:
        for child in list(g2):
            g1.append(child)
        print("Merged group paths successfully.")
    else:
        # Fallback: append directly to root1 if no <g> is found
        for child in list(root2):
            if child.tag.endswith('path') or child.tag == 'path':
                root1.append(child)
        print("Merged root paths successfully (fallback).")
        
    tree1.write(output_svg_path)
    print(f"Combined SVG written to: {output_svg_path}")

def main():
    parser = argparse.ArgumentParser(description="Residual Sketching Subprocess Orchestrator")
    parser.add_argument("--target", type=str, required=True, help="Path to target image")
    parser.add_argument("--object_name", type=str, required=True, help="Name of the object (e.g. 'cat')")
    parser.add_argument("--num_strokes", type=int, default=64, help="Total stroke count across both passes (default 64)")
    parser.add_argument("--output_dir", type=str, required=True, help="Final output directory")
    parser.add_argument("--num_iter", type=int, default=1000, help="Number of steps per pass (default 1000)")
    parser.add_argument("--use_cpu", type=int, default=0, help="Enforce CPU rendering (default 0)")
    
    # Allow passing any additional unknown args directly to object_sketching
    args, unknown_args = parser.parse_known_args()
    
    # Setup directories
    os.makedirs(args.output_dir, exist_ok=True)
    temp_dir_1 = os.path.join(args.output_dir, "pass1_temp")
    temp_dir_2 = os.path.join(args.output_dir, "pass2_temp")
    os.makedirs(temp_dir_1, exist_ok=True)
    os.makedirs(temp_dir_2, exist_ok=True)
    
    pass_strokes = args.num_strokes // 2
    print(f"=== RUNNING RESIDUAL SKETCHING: {args.num_strokes} strokes total ({pass_strokes} + {pass_strokes}) ===")
    
    # ----------------------------------------------------
    # PASS 1: First half of strokes
    # ----------------------------------------------------
    print("\n>>> Launching Pass 1...")
    cmd1 = [
        sys.executable, "ControlSketch/object_sketching.py",
        "--target", args.target,
        "--object_name", args.object_name,
        "--num_strokes", str(pass_strokes),
        "--num_iter", str(args.num_iter),
        "--output_dir", temp_dir_1,
        "--use_cpu", str(args.use_cpu)
    ] + unknown_args
    
    subprocess.check_call(cmd1)
    
    # Save intermediate Pass 1 outputs
    shutil.copy2(os.path.join(temp_dir_1, "final_svg.svg"), os.path.join(args.output_dir, "pass1_svg.svg"))
    shutil.copy2(os.path.join(temp_dir_1, "final_sketch.png"), os.path.join(args.output_dir, "pass1_sketch.png"))
    
    # ----------------------------------------------------
    # MASKING / SUBTRACTION
    # ----------------------------------------------------
    # Input png resized in pass 1 is saved as input.png in output dir
    input_png = os.path.join(temp_dir_1, "input.png")
    if not os.path.exists(input_png):
        input_png = args.target  # Fallback to original target
        
    sketch_png = os.path.join(temp_dir_1, "final_sketch.png")
    masked_png = os.path.join(temp_dir_2, "masked_input.png")
    
    mask_image_with_sketch(input_png, sketch_png, masked_png)
    
    # Copy the masked image to the main output directory for analysis
    shutil.copy2(masked_png, os.path.join(args.output_dir, "masked_input.png"))
    
    # ----------------------------------------------------
    # PASS 2: Second half of strokes on residual image
    # ----------------------------------------------------
    print("\n>>> Launching Pass 2 (on residual)...")
    cmd2 = [
        sys.executable, "ControlSketch/object_sketching.py",
        "--target", masked_png,
        "--object_name", args.object_name,
        "--num_strokes", str(pass_strokes),
        "--num_iter", str(args.num_iter),
        "--output_dir", temp_dir_2,
        "--use_cpu", str(args.use_cpu)
    ] + unknown_args
    
    subprocess.check_call(cmd2)
    
    # Save intermediate Pass 2 outputs
    shutil.copy2(os.path.join(temp_dir_2, "final_svg.svg"), os.path.join(args.output_dir, "pass2_svg.svg"))
    shutil.copy2(os.path.join(temp_dir_2, "final_sketch.png"), os.path.join(args.output_dir, "pass2_sketch.png"))
    
    # ----------------------------------------------------
    # MERGE & RENDER FINAL
    # ----------------------------------------------------
    svg_1 = os.path.join(temp_dir_1, "final_svg.svg")
    svg_2 = os.path.join(temp_dir_2, "final_svg.svg")
    final_svg = os.path.join(args.output_dir, "final_svg.svg")
    
    merge_svgs(svg_1, svg_2, final_svg)
    
    # Render final PNG from merged SVG using pydiffvg or simple PIL multiplication
    # We can just multiply final_sketch_1.png and final_sketch_2.png to get the visual representation!
    print("\nGenerating final combined sketch PNG...")
    sketch_png1 = os.path.join(temp_dir_1, "final_sketch.png")
    sketch_png2 = os.path.join(temp_dir_2, "final_sketch.png")
    final_png = os.path.join(args.output_dir, "final_sketch.png")
    
    img1 = Image.open(sketch_png1).convert("RGB")
    img2 = Image.open(sketch_png2).convert("RGB")
    arr1 = np.array(img1) / 255.0
    arr2 = np.array(img2) / 255.0
    final_arr = arr1 * arr2
    
    final_img = Image.fromarray((final_arr * 255.0).astype(np.uint8))
    final_img.save(final_png)
    
    print("\n=== RESIDUAL SKETCHING COMPLETED SUCCESSFULLY ===")
    print(f"Final Merged SVG: {final_svg}")
    print(f"Final Merged PNG: {final_png}")

if __name__ == "__main__":
    main()
