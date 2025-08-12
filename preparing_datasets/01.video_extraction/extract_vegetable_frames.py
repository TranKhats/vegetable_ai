#!/usr/bin/env python3
"""
Script to extract frames from vegetable videos with augmentation
Structure: preparing_datasets/video_extraction/videos/{vegetable_name}/*.mov/mp4
Output: data/raw/{vegetable_name}/*.jpg

Features:
- Target size: 640x640 (YOLOv8 optimal)
- Color space conversion (RGB/HSV)
- Gaussian/Median filtering
- Augmentation: rotation, flip, zoom/crop, lighting
"""

import cv2
import numpy as np
from pathlib import Path
import time
import argparse

def convert_color_space(frame, color_space='RGB'):
    """
    Convert frame to specified color space
    
    Args:
        frame: Input frame (BGR from OpenCV)
        color_space: Target color space ('RGB', 'HSV', 'BGR')
    
    Returns:
        Converted frame
    """
    if color_space == 'RGB':
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    elif color_space == 'HSV':
        return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    else:  # BGR - default
        return frame

def apply_smoothing_filter(frame, filter_type='gaussian', kernel_size=5):
    """
    Apply smoothing filter to reduce noise
    
    Args:
        frame: Input frame
        filter_type: Type of filter ('gaussian', 'median')
        kernel_size: Size of the kernel
    
    Returns:
        Filtered frame
    """
    if filter_type == 'gaussian':
        return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
    elif filter_type == 'median':
        return cv2.medianBlur(frame, kernel_size)
    else:
        return frame

def augment_frame(frame, augmentation_types=['rotation', 'flip', 'zoom', 'lighting']):
    """
    Apply augmentation to frame
    
    Args:
        frame: Input frame
        augmentation_types: List of augmentation types
    
    Returns:
        List of (augmented_frame, suffix) tuples
    """
    augmented_frames = []
    height, width = frame.shape[:2]
    center = (width // 2, height // 2)
    
    # 5.1. ROTATION - Xoay ảnh
    if 'rotation' in augmentation_types:
        # Use smaller angles to minimize border artifacts
        rotation_angles = [-15, -8, 8, 15]
        for angle in rotation_angles:
            # Create rotation matrix around center
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Apply rotation with replicated border (extends edge pixels)
            rotated = cv2.warpAffine(frame, rotation_matrix, (width, height), 
                                   borderMode=cv2.BORDER_REPLICATE)
            
            augmented_frames.append((rotated, f"rot{angle:+d}"))
    
    # 5.2. FLIP - Lật ngang/dọc
    if 'flip' in augmentation_types:
        # Horizontal flip
        h_flipped = cv2.flip(frame, 1)
        augmented_frames.append((h_flipped, "flip_h"))
        
        # Vertical flip
        v_flipped = cv2.flip(frame, 0)
        augmented_frames.append((v_flipped, "flip_v"))
        
        # Both horizontal and vertical
        hv_flipped = cv2.flip(frame, -1)
        augmented_frames.append((hv_flipped, "flip_hv"))
    
    # 5.3. ZOOM/CROP
    if 'zoom' in augmentation_types:
        zoom_factors = [0.8, 0.9, 1.1, 1.2]
        for zoom_factor in zoom_factors:
            if zoom_factor < 1.0:
                # Zoom out - add padding with edge pixels (no reflection)
                new_size = (int(width * zoom_factor), int(height * zoom_factor))
                resized = cv2.resize(frame, new_size, interpolation=cv2.INTER_LANCZOS4)
                
                # Add padding with replicated edge pixels (no overlapping)
                pad_x = (width - new_size[0]) // 2
                pad_y = (height - new_size[1]) // 2
                
                zoomed = cv2.copyMakeBorder(resized, pad_y, height - new_size[1] - pad_y,
                                          pad_x, width - new_size[0] - pad_x,
                                          cv2.BORDER_REPLICATE)
            else:
                # Zoom in - crop center (no padding needed)
                new_size = (int(width * zoom_factor), int(height * zoom_factor))
                resized = cv2.resize(frame, new_size, interpolation=cv2.INTER_LANCZOS4)
                
                # Crop center to restore original size
                start_x = (new_size[0] - width) // 2
                start_y = (new_size[1] - height) // 2
                
                zoomed = resized[start_y:start_y + height, start_x:start_x + width]
            
            augmented_frames.append((zoomed, f"zoom{zoom_factor:.1f}"))
    
    # 5.4. LIGHTING - Thay đổi ánh sáng
    if 'lighting' in augmentation_types:
        # Brightness adjustments
        brightness_values = [-30, -15, 15, 30]
        for brightness in brightness_values:
            lit_frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=brightness)
            suffix = f"bright{brightness:+d}" if brightness > 0 else f"dark{abs(brightness)}"
            augmented_frames.append((lit_frame, suffix))
        
        # Contrast adjustments - minimal range to preserve natural vegetable colors
        contrast_factors = [0.95, 1.05]
        for contrast in contrast_factors:
            contrasted = cv2.convertScaleAbs(frame, alpha=contrast, beta=0)
            augmented_frames.append((contrasted, f"cont{contrast:.2f}"))
    
    return augmented_frames

def extract_frames_from_video(video_path, output_dir, vegetable_name, frame_interval=60, 
                            target_size=(640, 640), color_space='RGB', filter_type='gaussian', 
                            enable_augmentation=True, augmentation_types=['rotation', 'flip', 'zoom', 'lighting']):
    """
    Extract frames from a single video file
    
    Args:
        video_path: Path to video file
        output_dir: Output directory
        vegetable_name: Name of vegetable for prefix
        frame_interval: Extract every N frames
        target_size: Target size (640, 640) for YOLOv8
        color_space: Color space conversion ('RGB', 'HSV', 'BGR')
        filter_type: Smoothing filter ('gaussian', 'median')
        enable_augmentation: Enable data augmentation
        augmentation_types: Types of augmentation to apply
    
    Returns:
        Number of frames extracted
    """
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"      ❌ Cannot open: {video_path.name}")
        return 0
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"      📊 FPS: {fps:.1f}, Duration: {duration:.1f}s, Total Frames: {total_frames}")
    print(f"      🎯 Target size: {target_size[0]}x{target_size[1]} (YOLOv8 optimal)")
    print(f"      🎨 Color space: {color_space}")
    print(f"      🔧 Filter: {filter_type}")
    print(f"      📈 Augmentation: {'ENABLED' if enable_augmentation else 'DISABLED'}")
    
    frame_count = 0
    extracted_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract frame at specified intervals
        if frame_count % frame_interval == 0:
            timestamp = frame_count / fps if fps > 0 else frame_count
            
            # 1. Save RAW frame first (no processing, preserve original contrast)
            raw_filename = f"{vegetable_name}_{video_path.stem}_{extracted_count:04d}_t{timestamp:.1f}s_raw.jpg"
            raw_filepath = output_dir / raw_filename
            raw_success = cv2.imwrite(str(raw_filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
            if raw_success:
                extracted_count += 1
                print(f"      💎 Saved RAW: {raw_filename}")
            # 2. Save processed frame (no resize)
            original_filename = f"{vegetable_name}_{video_path.stem}_{extracted_count:04d}_t{timestamp:.1f}s.jpg"
            original_filepath = output_dir / original_filename
            save_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if color_space == 'RGB' else frame
            if color_space == 'HSV':
                save_frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
            success = cv2.imwrite(str(original_filepath), save_frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
            if success:
                extracted_count += 1
                print(f"      💾 Saved: {original_filename}")
            
            # 4. Apply augmentation if enabled
            if enable_augmentation:
                augmented_frames = augment_frame(frame, augmentation_types)
                
                for aug_frame, aug_suffix in augmented_frames:
                    # Convert color space for augmented frame
                    aug_final = convert_color_space(aug_frame, color_space)
                    
                    # Convert back to BGR for saving
                    aug_save_frame = cv2.cvtColor(aug_final, cv2.COLOR_RGB2BGR) if color_space == 'RGB' else aug_final
                    if color_space == 'HSV':
                        aug_save_frame = cv2.cvtColor(aug_final, cv2.COLOR_HSV2BGR)
                    
                    aug_filename = f"{vegetable_name}_{video_path.stem}_{extracted_count-1:04d}_t{timestamp:.1f}s_{aug_suffix}.jpg"
                    aug_filepath = output_dir / aug_filename
                    
                    aug_success = cv2.imwrite(str(aug_filepath), aug_save_frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                    if aug_success:
                        extracted_count += 1
                
                if len(augmented_frames) > 0:
                    print(f"      📈 Generated {len(augmented_frames)} augmented variants")
        
        frame_count += 1
    
    cap.release()
    
    elapsed_time = time.time() - start_time
    print(f"      ✅ Extracted {extracted_count} frames in {elapsed_time:.1f}s")
    
    return extracted_count

def extract_vegetable_frames(vegetable_name, frame_interval=60, target_size=(640, 640), 
                           color_space='RGB', filter_type='gaussian', enable_augmentation=True,
                           augmentation_types=['rotation', 'flip', 'zoom', 'lighting'], clean_old=True):
    """
    Extract frames from all videos of a specific vegetable
    
    Args:
        vegetable_name: Name of vegetable directory
        frame_interval: Extract every N frames
        target_size: Target size for YOLOv8 (640, 640)
        color_space: Color space conversion ('RGB', 'HSV', 'BGR')
        filter_type: Smoothing filter ('gaussian', 'median')
        enable_augmentation: Enable data augmentation
        augmentation_types: Types of augmentation to apply
        clean_old: Remove old images before extraction
    
    Returns:
        Total number of frames extracted
    """
    
    # Set up directories - updated paths after moving to preparing_datasets
    videos_dir = Path(f"videos/{vegetable_name}")
    output_dir = Path(f"../../data/raw/{vegetable_name}")
    
    if not videos_dir.exists():
        print(f"❌ Vegetable directory not found: {videos_dir}")
        return 0
    
    print(f"🥬 Processing vegetable: {vegetable_name}")
    
    # Find video files
    video_extensions = ['.mov', '.mp4', '.avi', '.mkv', '.flv', '.wmv', '.m4v']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(videos_dir.glob(f"*{ext}"))
        video_files.extend(videos_dir.glob(f"*{ext.upper()}"))
    # Remove duplicate files (same file with different case extension)
    video_files = list(set(video_files))
    
    if not video_files:
        print(f"❌ No video files found in {videos_dir}")
        return 0
    
    print(f"📁 Found {len(video_files)} video files")
    
    # Clean old images if requested
    if clean_old and output_dir.exists():
        old_images = list(output_dir.glob("*.jpg")) + list(output_dir.glob("*.jpeg")) + list(output_dir.glob("*.png"))
        if old_images:
            print(f"🗑️  Removing {len(old_images)} old images")
            for img_file in old_images:
                img_file.unlink()
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_extracted = 0
    
    # Process each video
    for i, video_file in enumerate(video_files, 1):
        print(f"🎥 Processing {i}/{len(video_files)}: {video_file.name}")
        
        extracted = extract_frames_from_video(
            video_file,
            output_dir,
            vegetable_name,
            frame_interval=frame_interval,
            target_size=target_size,
            color_space=color_space,
            filter_type=filter_type,
            enable_augmentation=enable_augmentation,
            augmentation_types=augmentation_types
        )
        
        total_extracted += extracted
    
    print(f"🎉 Completed! Total extracted: {total_extracted} images")
    print(f"📂 Output directory: {output_dir}")
    
    return total_extracted

def main():
    parser = argparse.ArgumentParser(description="Extract frames from vegetable videos with augmentation")
    parser.add_argument("--vegetable", "-v", help="Vegetable name")
    parser.add_argument("--interval", "-i", type=int, default=60, help="Frame interval (default: 60)")
    parser.add_argument("--size", "-s", type=int, default=640, help="Target image size (default: 640x640)")
    parser.add_argument("--color-space", choices=['RGB', 'HSV', 'BGR'], default='RGB', help="Color space (default: RGB)")
    parser.add_argument("--filter", choices=['gaussian', 'median'], default='gaussian', help="Smoothing filter (default: gaussian)")
    parser.add_argument("--no-augment", action="store_true", help="Disable augmentation")
    parser.add_argument("--augment-types", nargs='+', choices=['rotation', 'flip', 'zoom', 'lighting'], 
                       default=['rotation', 'flip', 'zoom', 'lighting'], help="Augmentation types")
    parser.add_argument("--keep-old", action="store_true", help="Keep old images (don't clean)")
    parser.add_argument("--list", "-l", action="store_true", help="List available vegetables")
    
    args = parser.parse_args()
    
    # List available vegetables
    if args.list:
        videos_base_dir = Path("videos")
        if videos_base_dir.exists():
            vegetable_dirs = [d.name for d in videos_base_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
            print("🥬 Available vegetables:")
            for veg in vegetable_dirs:
                print(f"   - {veg}")
        else:
            print("❌ Videos directory not found")
        return
    
    # Check if vegetable is specified
    if not args.vegetable:
        print("❌ Please specify a vegetable with --vegetable or use --list to see available options")
        return
    
    # Extract frames
    target_size = (args.size, args.size)
    enable_augmentation = not args.no_augment
    clean_old = not args.keep_old
    
    extract_vegetable_frames(
        vegetable_name=args.vegetable,
        frame_interval=args.interval,
        target_size=target_size,
        color_space=args.color_space,
        filter_type=args.filter,
        enable_augmentation=enable_augmentation,
        augmentation_types=args.augment_types,
        clean_old=clean_old
    )

if __name__ == "__main__":
    main()
