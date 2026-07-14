import sys
import os
import cv2
import numpy as np
import torch
import mediapipe as mp
from PIL import Image
from svgpathtools import svg2paths, wsvg

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
    # 1. Get masks
    mask_tensor = sketch_utils.get_mask(image_pil, dist_util.dev(), mask_model)
    mask_np = mask_tensor.squeeze().cpu().detach().numpy()
    img_np = np.array(image_pil)
    
    # 2. Resize and create binary_mask (Defined here!)
    mask_img = Image.fromarray((mask_np * 255).astype(np.uint8)).resize((img_np.shape[1], img_np.shape[0]), Image.Resampling.LANCZOS)
    binary_mask = (np.array(mask_img) > 128).astype(np.uint8)

    # 3. Create the 5px Outline Anchor
    kernel = np.ones((11, 11), np.uint8)
    eroded_mask = cv2.erode(binary_mask, kernel, iterations=1)
    outline_mask = binary_mask - eroded_mask

    # 4. Create "Chin Split" Mask (Below chin)
    # MediaPipe 478-point model: oval face contour is indices 10,338,297,332,284,251,389,356,
    # 454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109
    # A reliable lower-face range: landmarks y-max from the full set gives chin position
    chin_line_y = np.max(landmarks[:, 1])  # bottom-most point = chin
    split_mask = np.zeros_like(binary_mask)
    split_mask[int(chin_line_y):, :] = 1
    
    # 5. Apply to anchor: silhouette AND below chin
    final_anchor_mask = cv2.bitwise_and(outline_mask, split_mask)
    
    # 6. Fill with mean color and paint outline
    mean_color = np.array(cv2.mean(np.array(image_pil.crop(bbox)))[:3], dtype=np.uint8)
    masked_img = np.full_like(img_np, mean_color, dtype=np.uint8)
    masked_img[final_anchor_mask == 1] = img_np[final_anchor_mask == 1]
    
    return Image.fromarray(masked_img)

def boost_face_features(image_pil, bbox):
    x, y, w, h = bbox
    # Add a small padding to ensure we get the whole forehead/chin
    pad = int(h * 0.1)
    crop = image_pil.crop((x, y - pad, x + w, y + h + pad))
    crop_cv = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)
    
    # Soft CLAHE
    lab = cv2.cvtColor(crop_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_boosted = clahe.apply(l)
    
    boosted_lab = cv2.merge((l_boosted, a, b))
    boosted_bgr = cv2.cvtColor(boosted_lab, cv2.COLOR_LAB2BGR)
    
    return Image.fromarray(cv2.cvtColor(boosted_bgr, cv2.COLOR_BGR2RGB))
    
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

    # --- PASS 1: Global Anchor (5px Outline) ---
    masked_img = apply_mask(img, bbox, landmarks) # Ensure this uses the new 5px outline logic
    masked_img.save("temp_masked.png")
    args.input_data = "temp_masked.png"
    run_standard_inference(args) 
    
    # --- Pass 2: Combined mask — face oval (sides/chin) + silhouette (hair) ---
    FACE_OVAL_IDX = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172,  58, 132,  93, 234, 127, 162,  21,  54, 103,  67, 109
    ]
    oval_pts = landmarks[FACE_OVAL_IDX].astype(np.int32)
    face_poly_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
    cv2.fillPoly(face_poly_mask, [oval_pts], 255)
    # Expand 15px outward so ears/hairline edges aren't clipped
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    face_poly_mask = cv2.dilate(face_poly_mask, dilation_kernel, iterations=1)

    mask_tensor = sketch_utils.get_mask(img, dist_util.dev(), mask_model)
    mask_np = mask_tensor.squeeze().cpu().detach().numpy()
    silhouette = (np.array(
        Image.fromarray((mask_np * 255).astype(np.uint8)).resize(
            (image_np.shape[1], image_np.shape[0]), Image.Resampling.LANCZOS
        )
    ) > 128).astype(np.uint8) * 255

    y_forehead = int(oval_pts[:, 1].min())
    hair_region = np.zeros_like(silhouette)
    hair_region[:y_forehead, :] = silhouette[:y_forehead, :]
    combined_mask = np.clip(
        face_poly_mask.astype(np.int32) + hair_region.astype(np.int32), 0, 255
    ).astype(np.uint8)
    face_mask_f = cv2.GaussianBlur(combined_mask.astype(np.float32), (21, 21), 0) / 255.0

    inside_pixels = image_np[combined_mask > 128]
    mean_color = (inside_pixels.mean(axis=0).astype(np.uint8)
                  if len(inside_pixels) > 0
                  else np.array([200, 170, 140], dtype=np.uint8))

    img_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    boosted_bgr = cv2.cvtColor(cv2.merge((clahe.apply(l_ch), a_ch, b_ch)), cv2.COLOR_LAB2BGR)
    gamma = 1.5
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    boosted_bgr = cv2.LUT(boosted_bgr, lut)
    full_boosted = cv2.cvtColor(boosted_bgr, cv2.COLOR_BGR2RGB)

    background = np.full_like(image_np, mean_color, dtype=np.uint8)
    alpha = face_mask_f[:, :, np.newaxis]
    final_pass_2 = (alpha * full_boosted + (1.0 - alpha) * background).astype(np.uint8)
    
    final_pass_2 = Image.fromarray(final_pass_2)
    final_pass_2.save("temp_boosted.png")
    args.input_data = "temp_boosted.png"
    run_standard_inference(args)


def merge_svgs(anchor_svg, detail_svg, bbox, output_dir):
    x, y, w, h = bbox
    paths1, attrs1 = svg2paths(anchor_svg)
    paths2, attrs2 = svg2paths(detail_svg)
    
    # Simple translate and scale logic
    scale = w / 512.0
    # Iterate over the second set of paths and transform them
    # Note: paths2 is a list of Path objects
    # Each Path object contains a list of segments (e.g., Line, CubicBezier)
    for p in paths2:
        # A simple in-place transformation of the path's segments
        for i, segment in enumerate(p.segments):
            # Check the type of segment and apply appropriate transformation
            if isinstance(segment, Line):
                # For a Line segment: p1(x,y) -> p2(x,y)
                x1, y1 = segment.p1
                x2, y2 = segment.p2
                
                # Apply scaling and translation
                # Assuming the face crop was 512x512 originally, we need to map it back to the full image coordinates
                # However, svgpathtools deals with path data, so we just modify the coordinates
                new_x1 = x1 * scale + x
                new_y1 = y1 * scale + y
                new_x2 = x2 * scale + x
                new_y2 = y2 * scale + y
                
                # Update the segment
                p.segments[i] = Line((new_x1, new_y1), (new_x2, new_y2))
                
            elif isinstance(segment, CubicBezier):
                # For a CubicBezier segment: p1, cp1, cp2, p2
                # Each is a tuple (x, y)
                x1, y1 = segment.p1
                x2, y2 = segment.p2
                cx1, cy1 = segment.c1
                cx2, cy2 = segment.c2
                
                # Apply scaling and translation
                new_x1 = x1 * scale + x
                new_y1 = y1 * scale + y
                new_x2 = x2 * scale + x
                new_y2 = y2 * scale + y
                new_cx1 = cx1 * scale + x
                new_cy1 = cy1 * scale + y
                new_cx2 = cx2 * scale + x
                new_cy2 = cy2 * scale + y
                
                # Update the segment
                p.segments[i] = CubicBezier(
                    (new_x1, new_y1), (new_cx1, new_cy1), (new_cx2, new_cy2), (new_x2, new_y2)
                )
            
            # Note: Other segment types like QuadraticBezier, Arc, ClosePath might exist.
            # For a robust solution, you'd handle all types.
            # For this example, we focus on Line and CubicBezier as they are most common.

    # Merge the paths
    # paths1 remains as is, paths2 is now transformed
    merged_paths = paths1 + paths2
    
    # Save the merged SVG
    output_path = os.path.join(output_dir, "final_merged.svg")
    wsvg(merged_paths, filename=output_path, attributes={'width': str(w), 'height': str(h)})
    print(f"Merged SVG saved to {output_path}")

if __name__ == "__main__":
    args = generate_args()
    run_residual_sketch(args)