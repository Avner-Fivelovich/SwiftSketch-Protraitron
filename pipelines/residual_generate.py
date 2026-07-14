import sys
import os
import cv2
import numpy as np
import torch
import mediapipe as mp
from PIL import Image
import xml.etree.ElementTree as ET

# Add SwiftSketch to path dynamically relative to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'SwiftSketch'))

# ADD THESE MISSING IMPORTS:
from SwiftSketch.utils import sketch_utils, dist_util
from SwiftSketch.generate import main as run_standard_inference
from SwiftSketch.utils.parser_util import generate_args
# Ensure mask_model is accessible here too
from SwiftSketch.generate import mask_model 

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def get_face_box(image_np):
    # 1. Initialize the FaceDetector with the .tflite model in the models/ directory
    model_path = os.path.join(ROOT_DIR, 'models', 'blaze_face_short_range.tflite')
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceDetectorOptions(base_options=base_options)
    
    with vision.FaceDetector.create_from_options(options) as detector:
        # 2. Convert BGR (OpenCV) to RGB (MediaPipe)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, 
                            data=cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
        
        # 3. Perform detection
        detection_result = detector.detect(mp_image)
        
        if not detection_result.detections:
            return None
        
        # 4. Extract bounding box (MediaPipe Tasks returns normalized coordinates)
        bbox = detection_result.detections[0].bounding_box
        h, w, _ = image_np.shape
        return (bbox.origin_x, bbox.origin_y, bbox.width, bbox.height)

def apply_mask(image_pil, bbox, landmarks):
    # 1. Get the raw mask output (which is a Torch Tensor)
    mask_tensor = sketch_utils.get_mask(image_pil, dist_util.dev(), mask_model)
    
    # 2. Safely convert to a NumPy array
    # .squeeze() removes the (1, H, W) dimension to get (H, W)
    mask_np = mask_tensor.squeeze().cpu().detach().numpy()
    
    # 3. Create a PIL image from the NumPy array
    # Map the [0, 1] float range to [0, 255] uint8 range
    mask_img = Image.fromarray((mask_np * 255).astype(np.uint8))
    
    # 4. Now perform the spatial resize using PIL
    img_np = np.array(image_pil)
    mask_img = mask_img.resize((img_np.shape[1], img_np.shape[0]), Image.Resampling.LANCZOS)
    
    # 5. Convert to binary and apply the logic
    binary_mask = (np.array(mask_img) > 128).astype(np.uint8)
    
    # ... rest of your outline logic (eroded_mask, etc) ...
    kernel = np.ones((81, 81), np.uint8)
    eroded_mask = cv2.erode(binary_mask, kernel, iterations=1)
    outline_mask = binary_mask - eroded_mask
    
    masked_img = np.full_like(img_np, 255)
    darkening_factor = 1 # 0.4  # 0.0 = pure black, 1.0 = original colour
    outline_pixels = (img_np[outline_mask == 1].astype(np.float32) * darkening_factor).clip(0, 255).astype(np.uint8)
    masked_img[outline_mask == 1] = outline_pixels
    
    return Image.fromarray(masked_img)


def boost_face_features(image_pil, bbox):
    """
    Pass 2: Refined face boost with lower CLAHE limit 
    and Gaussian smoothing to reduce 'scribble' noise.
    """
    x, y, w, h = bbox
    crop = image_pil.crop((x, y, x + w, y + h))
    crop_cv = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)
    
    # 1. Soft CLAHE (Reduced clipLimit)
    lab = cv2.cvtColor(crop_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)) # Lowered limit
    l_boosted = clahe.apply(l)
    
    # 2. Gaussian Blur to soften high-frequency edges
    l_boosted = cv2.GaussianBlur(l_boosted, (3, 3), 0)
    
    boosted_lab = cv2.merge((l_boosted, a, b))
    boosted_bgr = cv2.cvtColor(boosted_lab, cv2.COLOR_LAB2BGR)
    
    # 3. Soft circular feathering for blending
    h, w = crop_cv.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.circle(mask, (w//2, h//2), int(min(w, h)/2.2), 1.0, -1)
    mask = cv2.GaussianBlur(mask, (51, 51), 0)
    
    mask_3c = cv2.merge([mask, mask, mask])
    final_crop = (boosted_bgr * mask_3c + crop_cv * (1.0 - mask_3c)).astype(np.uint8)
    
    return Image.fromarray(cv2.cvtColor(final_crop, cv2.COLOR_BGR2RGB))
    
def get_face_landmarks_full(image_np):
    # 1. Setup FaceLandmarker options with model in the models/ directory
    model_path = os.path.join(ROOT_DIR, 'models', 'face_landmarker.task')
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1
    )
    
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        # 2. Convert BGR (OpenCV) to RGB (MediaPipe)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, 
                            data=cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
        
        # 3. Detect
        detection_result = landmarker.detect(mp_image)
        
        if not detection_result.face_landmarks:
            return None
        
        # 4. Extract coordinates from the first detected face
        h, w, _ = image_np.shape
        # detection_result.face_landmarks is a list of lists of NormalizedLandmark
        landmarks = detection_result.face_landmarks[0]
        return np.array([(int(p.x * w), int(p.y * h)) for p in landmarks])

def run_residual_sketch(args):
    img = Image.open(args.input_data).convert("RGB")
    image_np = np.array(img)
    
    # Get landmarks AND bbox
    landmarks = get_face_landmarks_full(image_np) 
    bbox = get_face_box(image_np)
    
    if not bbox or landmarks is None:
        print("No face detected! Running full-image standard inference.")
        run_standard_inference(args)
        return

    # --- PASS 1: Global Anchor (outline) ---
    masked_img = apply_mask(img, bbox, landmarks)
    masked_img.save("temp_masked.png")
    args.input_data = "temp_masked.png"
    original_output_dir = args.output_dir
    run_standard_inference(args)

    # The SVG is saved as <output_dir>/temp_masked.svg by generate.py
    pass1_svg = os.path.join(original_output_dir, "temp_masked.svg")
    
    # --- Pass 2: Combined mask — face oval (sides/chin) + silhouette (hair) ---
    FACE_OVAL_IDX = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172,  58, 132,  93, 234, 127, 162,  21,  54, 103,  67, 109
    ]

    # 1. Face oval polygon — precise boundary for sides, jaw, and chin
    oval_pts = landmarks[FACE_OVAL_IDX].astype(np.int32)
    face_poly_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
    cv2.fillPoly(face_poly_mask, [oval_pts], 255)
    # Expand 15px outward so ears/hairline edges aren't clipped
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (71, 101))
    face_poly_mask = cv2.dilate(face_poly_mask, dilation_kernel, iterations=1)

    # 2. RMBG silhouette — captures the actual hair boundary above the forehead
    mask_tensor = sketch_utils.get_mask(img, dist_util.dev(), mask_model)
    mask_np = mask_tensor.squeeze().cpu().detach().numpy()
    silhouette = (np.array(
        Image.fromarray((mask_np * 255).astype(np.uint8)).resize(
            (image_np.shape[1], image_np.shape[0]), Image.Resampling.LANCZOS
        )
    ) > 128).astype(np.uint8) * 255

    # 3. Split line: topmost y of the face oval = forehead landmark
    y_chin_limit = int(oval_pts[:, 1].max()) + 15

    # 4. Combined mask:
    #    - From top to y_chin_limit: Use silhouette (Hair/Face)
    #    - From y_chin_limit downwards: Use face_poly_mask (Base)
    upper_region = np.zeros_like(silhouette)
    upper_region[:y_chin_limit, :] = silhouette[:y_chin_limit, :]
    
    # We combine them so the silhouette takes precedence at the top, 
    # while keeping the polygon structure for the chin/neck area
    combined_mask = np.clip(
        upper_region.astype(np.int32) + face_poly_mask.astype(np.int32), 0, 255
    ).astype(np.uint8)

    # 5. Feather edges for a natural transition
    face_mask_f = cv2.GaussianBlur(combined_mask.astype(np.float32), (21, 21), 0) / 255.0

    # 6. Mean colour sampled from inside the combined mask
    inside_pixels = image_np[combined_mask > 128]
    mean_color = (inside_pixels.mean(axis=0).astype(np.uint8)
                  if len(inside_pixels) > 0
                  else np.array([200, 170, 140], dtype=np.uint8))

    # 4. CLAHE + gamma boost on the full image
    img_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    boosted_bgr = cv2.cvtColor(cv2.merge((clahe.apply(l_ch), a_ch, b_ch)), cv2.COLOR_LAB2BGR)
    gamma = 1.5
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    boosted_bgr = cv2.LUT(boosted_bgr, lut)
    full_boosted = cv2.cvtColor(boosted_bgr, cv2.COLOR_BGR2RGB)

    # 5. Blend: boosted face inside polygon, mean skin colour outside
    background = np.full_like(image_np, mean_color, dtype=np.uint8)
    alpha = face_mask_f[:, :, np.newaxis]
    final_pass_2 = (alpha * full_boosted + (1.0 - alpha) * background).astype(np.uint8)
    
    final_pass_2 = Image.fromarray(final_pass_2)
    final_pass_2.save("temp_boosted.png")
    args.input_data = "temp_boosted.png"
    run_standard_inference(args)

    # The SVG is saved as <output_dir>/temp_boosted.svg by generate.py
    pass2_svg = os.path.join(original_output_dir, "temp_boosted.svg")

    # --- Merge both passes into a single final SVG ---
    output_svg = os.path.join(original_output_dir, "final_merged.svg")
    merge_svgs(pass1_svg, pass2_svg, output_svg)
    print(f"Final merged SVG saved to: {output_svg}")

def merge_svgs(pass1_svg: str, pass2_svg: str, output_path: str) -> None:
    """
    Merge two SwiftSketch SVG outputs into a single file.

    Both passes produce SVGs with the same 224x224 viewBox, so no coordinate
    transformation is needed — we simply copy all <path> elements from both
    documents into one <svg> root.

    Pass 1 paths are written first (background/outline layer).
    Pass 2 paths are written on top (facial detail layer).
    """
    SVG_NS = "http://www.w3.org/2000/svg"
    ET.register_namespace("", SVG_NS)

    def _load_paths(svg_path: str):
        """Return the root <svg> element and a list of all <path> children."""
        if not os.path.exists(svg_path):
            print(f"Warning: SVG not found: {svg_path}")
            return None, []
        tree = ET.parse(svg_path)
        root = tree.getroot()
        # ElementTree prefixes the namespace, handle both bare and namespaced tags
        paths = root.findall(f".//{{{SVG_NS}}}path") or root.findall(".//path")
        return root, paths

    root1, paths1 = _load_paths(pass1_svg)
    root2, paths2 = _load_paths(pass2_svg)

    if root1 is None and root2 is None:
        print("Error: neither SVG file found — nothing to merge.")
        return

    # Use whichever root exists as the base; prefer pass1
    base_root = root1 if root1 is not None else root2

    # Build fresh <svg> with the same dimensions as the base
    merged_svg = ET.Element(
        f"{{{SVG_NS}}}svg",
        attrib={
            "version": base_root.get("version", "1.1"),
            "width":   base_root.get("width",  "224"),
            "height":  base_root.get("height", "224"),
        }
    )
    group = ET.SubElement(merged_svg, f"{{{SVG_NS}}}g")

    for path_el in paths1:  # Pass 1 — outline layer (bottom)
        group.append(path_el)
    for path_el in paths2:  # Pass 2 — detail layer (top)
        group.append(path_el)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    ET.indent(merged_svg, space="  ")
    tree = ET.ElementTree(merged_svg)
    ET.ElementTree(merged_svg).write(
        output_path,
        xml_declaration=True,
        encoding="unicode"
    )
    print(f"Merged {len(paths1)} pass-1 strokes + {len(paths2)} pass-2 strokes → {output_path}")

if __name__ == "__main__":
    args = generate_args()
    run_residual_sketch(args)