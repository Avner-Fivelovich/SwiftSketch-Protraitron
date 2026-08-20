import os
import random
from pathlib import Path
from datasets import load_dataset
from PIL import Image
import numpy as np
import io
import time

# Anchor directory to the script's location
SCRIPT_DIR = Path(__file__).parent.resolve()
download_dir = SCRIPT_DIR / "data" / "ffhq_raw_npz"
download_dir.mkdir(parents=True, exist_ok=True)

num_images = 15000

# Count how many files we already have so we don't even download them from HuggingFace!
existing_files = len(list(download_dir.glob("ffhq_batch_*.npz")))
images_to_fetch = max(0, num_images - existing_files)

print(f"Found {existing_files} existing images. Instructing Hugging Face to skip them...")
print("Waiting for Hugging Face to seek through the dataset metadata... (This usually takes 1-2 minutes of silence!)")

# Use .skip() to completely bypass the network transfer for images we already have
dataset = load_dataset("marcosv/ffhq-dataset", split="train", streaming=True).skip(existing_files).take(images_to_fetch)

def save_compressed_npz(file_path, image):
    # Resize slightly if needed to match SwiftSketch standard, though 512x512 is standard
    resized_image = image.resize((512,512))
    img_buffer = io.BytesIO()
    resized_image.save(img_buffer, format='JPEG')
    img_data = img_buffer.getvalue()
    
    # Save using np.savez_compressed
    # ControlSketch/object_sketching.py can generate mask/attn automatically if they are missing!
    np.savez_compressed(
        file_path, 
        image=img_data, 
        caption="A portrait photo of a person's face"
    )

saved_count = 0
start_time = time.time()

for idx, item in enumerate(dataset):
    if saved_count == 0:
        print("\n🚀 Dataset seek complete! Starting downloads now...")
        
    try:
        # Offset the index by the ones we skipped
        actual_idx = existing_files + idx
        output_path = download_dir / f"ffhq_batch_{actual_idx}.npz"
        
        # Skip instantly if we already downloaded this image!
        if output_path.exists():
            continue
            
        img = item['image']
        save_compressed_npz(output_path, img)
        
        saved_count += 1
        # Print the very first one, then every 10th image so we see constant progress
        if saved_count == 1 or saved_count % 10 == 0:
            elapsed = time.time() - start_time
            print(f"[{saved_count}/{images_to_fetch}] Saved .npz to: {output_path} (Elapsed: {elapsed:.1f}s)")
            
    except Exception as e:
        print(f"Failed to save image at index {idx}: {e}")

print(f"\nDone! Successfully downloaded {saved_count} FFHQ images to {download_dir}.")
