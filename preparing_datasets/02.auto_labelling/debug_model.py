#!/usr/bin/env python3
"""
Debug auto-labelling - test model inference capability
"""

from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np

def debug_model_inference():
    """Debug model inference step by step"""
    
    print("🔍 DEBUG: Model inference capability")
    print("="*50)
    
    # 1. Check model path
    model_path = "runs/detect/carrot_auto_labeller/weights/best.pt"
    print(f"1️⃣ Model path: {model_path}")
    print(f"   Exists: {Path(model_path).exists()}")
    
    if not Path(model_path).exists():
        print("❌ Model not found! Run training first.")
        return
    
    # 2. Load model
    try:
        model = YOLO(model_path)
        print(f"✅ Model loaded successfully")
        print(f"   Device: {model.device}")
        print(f"   Classes: {model.names}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # 3. Check input images
    base_dir = Path("../../")
    images_dir = base_dir / "data/raw/carrot"  # Note: should be "carrot" not "carot"
    
    # Try both spellings
    if not images_dir.exists():
        images_dir = base_dir / "data/raw/carot"
    
    print(f"\n2️⃣ Images directory: {images_dir}")
    print(f"   Exists: {images_dir.exists()}")
    
    if not images_dir.exists():
        print("❌ Images directory not found!")
        return
    
    # Get sample images
    sample_images = list(images_dir.glob("carrot_*.jpg"))[:5]
    print(f"   Found {len(sample_images)} sample images")
    
    if not sample_images:
        print("❌ No carrot images found!")
        return
    
    # 4. Test inference on samples
    print(f"\n3️⃣ Testing inference on sample images:")
    
    for i, img_path in enumerate(sample_images, 1):
        print(f"\n   📸 Image {i}: {img_path.name}")
        
        # Check image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"      ❌ Cannot load image")
            continue
            
        h, w, c = img.shape
        print(f"      📏 Size: {w}x{h}x{c}")
        
        # Test with different confidence thresholds
        confidence_levels = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
        
        for conf in confidence_levels:
            try:
                # Run inference
                results = model(str(img_path), conf=conf, verbose=False)
                
                if len(results) == 0:
                    print(f"      Conf {conf:0.2f}: No results object")
                    continue
                    
                boxes = results[0].boxes
                
                if boxes is not None and len(boxes) > 0:
                    print(f"      Conf {conf:0.2f}: ✅ {len(boxes)} detections")
                    
                    # Show first detection details
                    box = boxes[0]
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    coords = box.xywh[0].cpu().numpy()
                    
                    print(f"         Best: class={class_id}, conf={confidence:.4f}")
                    print(f"         Box: x={coords[0]:.1f}, y={coords[1]:.1f}, w={coords[2]:.1f}, h={coords[3]:.1f}")
                    
                    # If we found detections, break
                    if conf <= 0.1:  # Found detections with very low confidence
                        break
                        
                else:
                    print(f"      Conf {conf:0.2f}: ❌ No detections")
                    
            except Exception as e:
                print(f"      Conf {conf:0.2f}: 💥 Error: {e}")
    
    # 5. Test on training images (should work)
    print(f"\n4️⃣ Testing on training images (sanity check):")
    
    train_images_dir = Path("dataset_carrot/train/images")
    if train_images_dir.exists():
        train_images = list(train_images_dir.glob("*.jpg"))[:3]
        print(f"   Found {len(train_images)} training images")
        
        for img_path in train_images:
            print(f"\n   📸 Training image: {img_path.name}")
            
            try:
                results = model(str(img_path), conf=0.1, verbose=False)
                boxes = results[0].boxes
                
                if boxes is not None and len(boxes) > 0:
                    print(f"      ✅ {len(boxes)} detections (conf ≥ 0.1)")
                else:
                    print(f"      ❌ No detections")
                    
            except Exception as e:
                print(f"      💥 Error: {e}")
    else:
        print("   ❌ Training images not found")
    
    # 6. Model summary
    print(f"\n5️⃣ Model analysis:")
    try:
        # Validate model on validation set
        print("   Running validation...")
        val_results = model.val(data="dataset_carrot/dataset.yaml")
        
        print(f"   mAP50: {val_results.box.map50:.4f}")
        print(f"   mAP50-95: {val_results.box.map:.4f}")
        print(f"   Precision: {val_results.box.mp:.4f}")
        print(f"   Recall: {val_results.box.mr:.4f}")
        
    except Exception as e:
        print(f"   ❌ Validation failed: {e}")

def check_image_paths():
    """Check if there's a path mismatch issue"""
    
    print("\n🔍 DEBUG: Image path issues")
    print("="*30)
    
    base_dir = Path("../../")
    
    # Check both spellings
    carrot_dir = base_dir / "data/raw/carrot"
    carot_dir = base_dir / "data/raw/carot"
    
    print(f"carrot/ exists: {carrot_dir.exists()}")
    print(f"carot/ exists: {carot_dir.exists()}")
    
    if carrot_dir.exists():
        carrot_count = len(list(carrot_dir.glob("*.jpg")))
        print(f"carrot/ images: {carrot_count}")
    
    if carot_dir.exists():
        carot_count = len(list(carot_dir.glob("*.jpg")))
        print(f"carot/ images: {carot_count}")

def main():
    """Main debug function"""
    check_image_paths()
    debug_model_inference()

if __name__ == "__main__":
    main()
