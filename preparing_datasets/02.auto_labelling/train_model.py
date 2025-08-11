#!/usr/bin/env python3
"""
Train YOLOv8 model for auto-labelling remaining images
"""

from ultralytics import YOLO
import torch
from pathlib import Path
import yaml
import os

def train_auto_labelling_model(vegetable="carrot", epochs=100, imgsz=640, batch=4):
    """
    Train YOLOv8 model on manually labeled data for auto-labelling
    
    Args:
        vegetable: Vegetable name to train for
        epochs: Number of training epochs
        imgsz: Training image size
        batch: Batch size
    """
    
    print(f"🚂 Training YOLOv8 model for {vegetable} auto-labelling...")
    
    # Paths
    dataset_dir = Path(f"dataset_{vegetable}")
    yaml_path = dataset_dir / "dataset.yaml"
    
    if not yaml_path.exists():
        print(f"❌ Dataset config not found: {yaml_path}")
        print("   Run prepare_dataset.py first!")
        return None
    
    # Load dataset info
    with open(yaml_path, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    print(f"📊 Dataset info:")
    print(f"   📁 Total images: {dataset_config.get('total_images', 'Unknown')}")
    print(f"   ✅ Labeled: {dataset_config.get('labeled_images', 'Unknown')}")
    print(f"   ❌ Unlabeled: {dataset_config.get('unlabeled_images', 'Unknown')}")
    
    # Check GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Training device: {device}")
    
    if device == 'cpu':
        print("⚠️  Warning: Training on CPU will be slow. Consider using GPU.")
        batch = min(batch, 8)  # Reduce batch size for CPU
    
    # Load pretrained YOLOv8 model
    print("📦 Loading pretrained YOLOv8 model...")
    model = YOLO('yolov8l.pt')  # Large version for better accuracy

    # Create runs directory
    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)
    
    # Training parameters
    training_args = {
        'data': str(yaml_path),
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'name': f'{vegetable}_auto_labeller',
        'project': 'runs/detect',
        'patience': 20,  # Increased from 15 → 20 (more patience for convergence)
        'save': True,
        'cache': False,  # Disable caching to save memory
        'device': device,
        'workers': 4 if device == 'cuda' else 2,
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'AdamW',
        'close_mosaic': 15,  # Increased from 10 → 15 (disable mosaic later)
        
        # IMPROVED LEARNING RATE SCHEDULE
        'lr0': 0.001,  # Reduced from 0.01 → 0.001 (conservative learning rate)
        'lrf': 0.001,  # Reduced from 0.01 → 0.001 (final LR factor)
        'momentum': 0.9,  # Increased from 0.937 → 0.9 (more momentum)
        'weight_decay': 0.001,  # Increased from 0.0005 → 0.001 (more regularization)
        'warmup_epochs': 5,  # Increased from 3 → 5 (longer warmup)
        'warmup_momentum': 0.5,  # Reduced from 0.8 → 0.5 (conservative warmup)
        'warmup_bias_lr': 0.1,
        
        # IMPROVED LOSS FUNCTION TUNING for better classification
        'box': 3.0,  # Reduced from 7.5 → 3.0 (less focus on box regression)
        'cls': 1.5,  # Increased from 0.5 → 1.5 (MORE focus on classification - FIX precision!)
        'dfl': 1.0,  # Reduced from 1.5 → 1.0 (balanced distribution focal loss)
        # Removed 'pose' and 'kobj' - not needed for object detection
        'label_smoothing': 0.15,  # Increased from 0.0 → 0.15 (reduce overfitting)
        'nbs': 64,  # Nominal batch size
        
        # IMPROVED AUGMENTATION for better generalization
        'hsv_h': 0.01,  # Reduced from 0.015 → 0.01 (less color variation)
        'hsv_s': 0.5,   # Reduced from 0.7 → 0.5 (conservative saturation)
        'hsv_v': 0.3,   # Reduced from 0.4 → 0.3 (conservative brightness)
        'degrees': 5.0,  # Increased from 0.0 → 5.0 (add small rotation)
        'translate': 0.05,  # Reduced from 0.1 → 0.05 (conservative translation)
        'scale': 0.2,   # Reduced from 0.5 → 0.2 (conservative scaling)
        'shear': 2.0,   # Increased from 0.0 → 2.0 (add small shear)
        'perspective': 0.0001,  # Add tiny perspective change
        'flipud': 0.0,  # No vertical flip (carrots have orientation)
        'fliplr': 0.3,  # Reduced from 0.5 → 0.3 (less horizontal flip)
        'mosaic': 0.8,  # Reduced from 1.0 → 0.8 (less aggressive mosaic)
        'mixup': 0.1,   # Increased from 0.0 → 0.1 (add mixup for regularization)
        'copy_paste': 0.0,  # Keep disabled
    }
    
    print(f"🎯 Training parameters:")
    print(f"   📊 Epochs: {epochs}")
    print(f"   📏 Image size: {imgsz}")
    print(f"   📦 Batch size: {batch}")
    print(f"   🖥️  Device: {device}")
    
    try:
        # Train the model
        print("🚀 Starting training...")
        results = model.train(**training_args)
        
        # Get best model path
        exp_dir = Path(f"runs/detect/{vegetable}_auto_labeller")
        best_model_path = exp_dir / "weights/best.pt"
        last_model_path = exp_dir / "weights/last.pt"
        
        print(f"\n🎉 Training completed!")
        print(f"📁 Experiment directory: {exp_dir}")
        print(f"🏆 Best model: {best_model_path}")
        print(f"🔄 Last model: {last_model_path}")
        
        # Model validation
        if best_model_path.exists():
            print("\n📊 Model validation:")
            model_best = YOLO(str(best_model_path))
            val_results = model_best.val()
            
            print(f"   📈 mAP50: {val_results.box.map50:.4f}")
            print(f"   📈 mAP50-95: {val_results.box.map:.4f}")
            print(f"   📊 Precision: {val_results.box.mp:.4f}")
            print(f"   📊 Recall: {val_results.box.mr:.4f}")
        
        print(f"\n🚀 Next step:")
        print(f"   Auto-label remaining images: python auto_label.py --vegetable {vegetable}")
        
        return best_model_path, results
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None, None

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train YOLOv8 for auto-labelling")
    parser.add_argument("--vegetable", "-v", default="carrot",
                       help="Vegetable to train for (default: carrot)")
    parser.add_argument("--epochs", "-e", type=int, default=100,
                       help="Number of training epochs (default: 100)")
    parser.add_argument("--imgsz", "-s", type=int, default=640,
                       help="Training image size (default: 640)")
    parser.add_argument("--batch", "-b", type=int, default=4,
                       help="Batch size (default: 4)")
    
    args = parser.parse_args()
    
    # Check if ultralytics is installed
    try:
        import ultralytics
        print(f"📦 Ultralytics version: {ultralytics.__version__}")
    except ImportError:
        print("❌ Ultralytics not installed!")
        print("   Install with: pip install ultralytics")
        return
    
    # Train model
    model_path, results = train_auto_labelling_model(
        vegetable=args.vegetable,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch
    )
    
    if model_path:
        print(f"\n✅ Training successful! Model ready for auto-labelling.")
    else:
        print(f"\n❌ Training failed. Check the logs above.")

if __name__ == "__main__":
    main()
