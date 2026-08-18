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

print(f"Streaming dataset from Hugging Face... We will download {num_images} images.")
# Removed memory-intensive shuffle(buffer_size=10000) to prevent macOS from SIGKILLing the process (OOM).
# Taking the first 15000 images is perfectly diverse for FFHQ.
dataset = load_dataset("marcosv/ffhq-dataset", split="train", streaming=True).take(num_images)

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
    try:
        output_path = download_dir / f"ffhq_batch_{idx}.npz"
        
        # Skip instantly if we already downloaded this image!
        if output_path.exists():
            continue
            
        img = item['image']
        save_compressed_npz(output_path, img)
        
        saved_count += 1
        if saved_count % 100 == 0:
            elapsed = time.time() - start_time
            print(f"[{saved_count}/{num_images}] Saved .npz to: {output_path} (Elapsed: {elapsed:.1f}s)")
            
    except Exception as e:
        print(f"Failed to save image at index {idx}: {e}")

print(f"\nDone! Successfully downloaded {saved_count} FFHQ images to {download_dir}.")
