import os
import cv2
import numpy as np
from glob import glob
import shutil

def is_brightness_ok(img, min_brightness=90, max_brightness=210):
    """Kiểm tra độ sáng trung bình của ảnh"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    return min_brightness <= mean_brightness <= max_brightness

def preprocess_brightness(img):
    """Tăng sáng nhẹ trước khi augment"""
    return cv2.convertScaleAbs(img, alpha=1.0, beta=15)

def augment_image(img):
    augmented_images = []
    h, w = img.shape[:2]

    # --- Rotation (nhẹ) ---
    for angle in [-15, -8, 8, 15]:
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
        rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        if is_brightness_ok(rotated):
            augmented_images.append(rotated)

    # --- Flip ---
    for flip_code in [0, 1, -1]:
        flipped = cv2.flip(img, flip_code)
        if is_brightness_ok(flipped):
            augmented_images.append(flipped)

    # --- Zoom/Crop ---
    zoom_factors = [0.9, 1.1]
    for z in zoom_factors:
        if z < 1.0:
            new_size = (int(w*z), int(h*z))
            resized = cv2.resize(img, new_size, interpolation=cv2.INTER_LANCZOS4)
            pad_x = (w - new_size[0]) // 2
            pad_y = (h - new_size[1]) // 2
            zoomed = cv2.copyMakeBorder(resized, pad_y, h - new_size[1] - pad_y,
                                        pad_x, w - new_size[0] - pad_x,
                                        cv2.BORDER_REPLICATE)
        else:
            new_size = (int(w*z), int(h*z))
            resized = cv2.resize(img, new_size, interpolation=cv2.INTER_LANCZOS4)
            start_x = (new_size[0] - w) // 2
            start_y = (new_size[1] - h) // 2
            zoomed = resized[start_y:start_y + h, start_x:start_x + w]
        if is_brightness_ok(zoomed):
            augmented_images.append(zoomed)

    # --- Lighting (ưu tiên sáng) ---
    for brightness in [10, 20]:  # chỉ tăng sáng, không giảm
        lighting = cv2.convertScaleAbs(img, alpha=1.0, beta=brightness)
        if is_brightness_ok(lighting):
            augmented_images.append(lighting)
    for contrast in [1.0, 1.05]:  # giữ nguyên hoặc tăng nhẹ contrast
        lighting = cv2.convertScaleAbs(img, alpha=contrast, beta=10)
        if is_brightness_ok(lighting):
            augmented_images.append(lighting)

    # --- Filters ---
    gaussian = cv2.GaussianBlur(img, (5, 5), 0)
    if is_brightness_ok(gaussian):
        augmented_images.append(gaussian)
    median = cv2.medianBlur(img, 5)
    if is_brightness_ok(median):
        augmented_images.append(median)

    return augmented_images

def process_images(input_dir, output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    image_paths = glob(os.path.join(input_dir, '*.*'))
    
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue

        img = preprocess_brightness(img)  # tăng sáng nhẹ

        base_name = os.path.splitext(os.path.basename(img_path))[0]

        # Lưu ảnh gốc nếu đạt điều kiện sáng
        if is_brightness_ok(img):
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_orig.jpg"), img)

        # Augmentation
        augmented_imgs = augment_image(img)
        for idx, aug_img in enumerate(augmented_imgs):
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_aug{idx}.jpg"), aug_img)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Augment images for dataset creation (bright version).")
    parser.add_argument('--input_dir', type=str, required=True, help='Folder with original images')
    parser.add_argument('--output_dir', type=str, required=True, help='Folder to save augmented images')
    args = parser.parse_args()

    process_images(args.input_dir, args.output_dir)