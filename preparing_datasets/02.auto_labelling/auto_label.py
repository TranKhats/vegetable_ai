#!/usr/bin/env python3
"""
Auto-label remaining images using trained YOLOv8 model
"""

from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
import os

def auto_label_remaining_images(vegetable="carrot", conf_threshold=0.5, iou_threshold=0.45):
    """
    Auto-label remaining unlabeled images using trained YOLOv8 model
    
    Args:
        vegetable: Vegetable name to process
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
    """
    
    print(f"🤖 Auto-labelling remaining {vegetable} images...")
    
    # Paths
    base_dir = Path("../../")
    images_dir = base_dir / f"data/raw/{vegetable}"
    labels_dir = base_dir / "data/yolo_labels"
    
    # Model path
    model_path = Path(f"runs/detect/{vegetable}_auto_labeller/weights/best.pt")
    
    if not model_path.exists():
        print(f"❌ Trained model not found: {model_path}")
        print("   Run train_model.py first!")
        return
    
    print(f"📦 Loading trained model: {model_path}")
    model = YOLO(str(model_path))
    
    # Create labels directory if it doesn't exist
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Find unlabeled images
    print("🔍 Scanning for unlabeled images...")
    
    all_images = list(images_dir.glob(f"{vegetable}_*.jpg"))
    labeled_images = set()
    
    for txt_file in labels_dir.glob("*.txt"):
        img_name = txt_file.stem + ".jpg"
        labeled_images.add(img_name)
    
    unlabeled_images = [img for img in all_images if img.name not in labeled_images]
    
    print(f"📊 Image statistics:")
    print(f"   📁 Total images: {len(all_images)}")
    print(f"   ✅ Already labeled: {len(labeled_images)}")
    print(f"   ❌ Unlabeled: {len(unlabeled_images)}")
    
    if not unlabeled_images:
        print("🎉 All images are already labeled!")
        return
    
    print(f"🎯 Detection parameters:")
    print(f"   🎚️  Confidence threshold: {conf_threshold}")
    print(f"   🔗 IoU threshold: {iou_threshold}")
    
    # Auto-label unlabeled images
    auto_labeled_count = 0
    no_detection_count = 0
    low_confidence_count = 0
    error_count = 0
    
    print(f"\n🚀 Starting auto-labelling {len(unlabeled_images)} images...")
    
    for i, img_path in enumerate(unlabeled_images, 1):
        try:
            # Run inference
            results = model(str(img_path), conf=conf_threshold, iou=iou_threshold, verbose=False)
            
            # Get predictions
            boxes = results[0].boxes
            
            if boxes is not None and len(boxes) > 0:
                # Create TXT file
                txt_path = labels_dir / (img_path.stem + ".txt")
                
                yolo_lines = []
                max_confidence = 0
                
                for box in boxes:
                    # Get YOLO format coordinates (normalized 0-1)
                    x_center, y_center, width, height = box.xywhn[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    max_confidence = max(max_confidence, confidence)
                    
                    # Create YOLO format line
                    yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                    yolo_lines.append(yolo_line)
                
                # Write TXT file
                with open(txt_path, 'w') as f:
                    f.write('\n'.join(yolo_lines))
                
                auto_labeled_count += 1
                
                # Check confidence level
                if max_confidence < 0.7:
                    low_confidence_count += 1
                    conf_indicator = "⚠️ "
                else:
                    conf_indicator = "✅"
                
                print(f"{conf_indicator} ({i:3d}/{len(unlabeled_images)}) {img_path.name} → {len(boxes)} objects (conf: {max_confidence:.3f})")
                
            else:
                no_detection_count += 1
                print(f"❌ ({i:3d}/{len(unlabeled_images)}) No {vegetable} detected: {img_path.name}")
        
        except Exception as e:
            error_count += 1
            print(f"💥 ({i:3d}/{len(unlabeled_images)}) Error processing {img_path.name}: {e}")
        
        # Progress indicator every 50 images
        if i % 50 == 0:
            progress = (i / len(unlabeled_images)) * 100
            print(f"📊 Progress: {progress:.1f}% ({i}/{len(unlabeled_images)})")
    
    # Final summary
    print(f"\n{'='*50}")
    print(f"🎉 AUTO-LABELLING COMPLETED!")
    print(f"{'='*50}")
    print(f"📊 Results:")
    print(f"   ✅ Successfully auto-labeled: {auto_labeled_count}")
    print(f"   ⚠️  Low confidence (< 0.7): {low_confidence_count}")
    print(f"   ❌ No detections: {no_detection_count}")
    print(f"   💥 Errors: {error_count}")
    print(f"   📁 Total processed: {len(unlabeled_images)}")
    
    total_labeled_now = len(labeled_images) + auto_labeled_count
    print(f"\n📈 Overall progress:")
    print(f"   📁 Total images: {len(all_images)}")
    print(f"   ✅ Total labeled: {total_labeled_now}")
    print(f"   📊 Completion: {(total_labeled_now / len(all_images)) * 100:.1f}%")
    
    if low_confidence_count > 0:
        print(f"\n⚠️  Recommendation:")
        print(f"   Review {low_confidence_count} low-confidence predictions manually")
        print(f"   These files might need manual correction")
    
    print(f"\n📂 Labels saved to: {labels_dir}")
    
    return {
        "auto_labeled": auto_labeled_count,
        "no_detection": no_detection_count,
        "low_confidence": low_confidence_count,
        "errors": error_count,
        "total_processed": len(unlabeled_images)
    }

def verify_auto_labels(vegetable="carrot", sample_size=5):
    """
    Verify a sample of auto-generated labels
    
    Args:
        vegetable: Vegetable name to verify
        sample_size: Number of samples to verify
    """
    
    print(f"\n🔍 Verifying auto-generated labels for {vegetable}...")
    
    base_dir = Path("../../")
    images_dir = base_dir / f"data/raw/{vegetable}"
    labels_dir = base_dir / "data/yolo_labels"
    
    # Find recently created labels (auto-generated)
    import time
    current_time = time.time()
    recent_labels = []
    
    for txt_file in labels_dir.glob("*.txt"):
        # Check if file was created recently (within last hour)
        if current_time - txt_file.stat().st_mtime < 3600:  # 1 hour
            img_file = images_dir / (txt_file.stem + ".jpg")
            if img_file.exists():
                recent_labels.append((img_file, txt_file))
    
    if not recent_labels:
        print("   No recent auto-generated labels found")
        return
    
    # Sample verification
    import random
    sample_labels = random.sample(recent_labels, min(sample_size, len(recent_labels)))
    
    print(f"   Verifying {len(sample_labels)} samples...")
    
    for img_path, txt_path in sample_labels:
        print(f"\n📄 {img_path.name}:")
        
        # Read label
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        
        print(f"   Objects detected: {len(lines)}")
        
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) == 5:
                class_id, x_center, y_center, width, height = map(float, parts)
                print(f"   Object {i+1}: class={int(class_id)}, center=({x_center:.3f}, {y_center:.3f}), size=({width:.3f}, {height:.3f})")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto-label images using trained YOLOv8 model")
    parser.add_argument("--vegetable", "-v", default="carrot",
                       help="Vegetable to auto-label (default: carrot)")
    parser.add_argument("--conf", "-c", type=float, default=0.5,
                       help="Confidence threshold (default: 0.5)")
    parser.add_argument("--iou", "-i", type=float, default=0.45,
                       help="IoU threshold for NMS (default: 0.45)")
    parser.add_argument("--verify", action="store_true",
                       help="Verify auto-generated labels")
    
    args = parser.parse_args()
    
    # Check if ultralytics is installed
    try:
        import ultralytics
    except ImportError:
        print("❌ Ultralytics not installed!")
        print("   Install with: pip install ultralytics")
        return
    
    # Auto-label images
    results = auto_label_remaining_images(
        vegetable=args.vegetable,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # Verify if requested
    if args.verify and results and results["auto_labeled"] > 0:
        verify_auto_labels(vegetable=args.vegetable)

if __name__ == "__main__":
    main()
