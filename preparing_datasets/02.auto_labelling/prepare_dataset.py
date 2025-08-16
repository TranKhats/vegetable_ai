#!/usr/bin/env python3
"""
Prepare dataset for YOLOv8 training from manually labeled data
Structure: Use existing labeled data in yolo_labels to train model for auto-labelling
"""

import shutil
from pathlib import Path
import random
import os

def prepare_yolo_dataset(vegetable="carrot", train_ratio=0.8):
    """
    Prepare dataset structure for YOLOv8 training
    
    Args:
        vegetable: Vegetable name to process
        train_ratio: Ratio for train/val split
    """
    
    print(f"🥕 Preparing dataset for {vegetable} auto-labelling...")
    
    # Paths
    base_dir = Path("../../")
    images_dir = base_dir / f"data/raw/{vegetable}"
    labels_dir = base_dir / "data/yolo_labels"
    
    # Output structure for YOLOv8
    dataset_dir = Path(f"dataset_{vegetable}")
    train_images = dataset_dir / "train/images"
    train_labels = dataset_dir / "train/labels"
    val_images = dataset_dir / "val/images" 
    val_labels = dataset_dir / "val/labels"
    
    # Create directories
    for dir_path in [train_images, train_labels, val_images, val_labels]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Get labeled files (files with corresponding TXT)
    labeled_files = []
    unlabeled_files = []
    
    print("🔍 Scanning for labeled and unlabeled images...")
    image_files = []
    print(f"📂 Scanning directory: {images_dir}")
    image_files = images_dir.glob("*.jpg")
    # if vegetable == "all":
    #     image_files = images_dir.glob("*.jpg")
    # else:
    #     image_files = images_dir.glob(f"{vegetable}_*.jpg")
    for img_file in image_files:
        txt_name = img_file.stem + ".txt"
        txt_path = labels_dir / txt_name
        if txt_path.exists():
            labeled_files.append((img_file, txt_path))
        else:
            unlabeled_files.append(img_file)
    
    print(f"📊 Dataset statistics:")
    print(f"   ✅ Labeled images: {len(labeled_files)}")
    print(f"   ❌ Unlabeled images: {len(unlabeled_files)}")
    print(f"   📁 Total images: {len(labeled_files) + len(unlabeled_files)}")
    
    if len(labeled_files) < 10:
        print("⚠️  Warning: Very few labeled images for training!")
        return None, None, None, unlabeled_files
    
    # Split train/val
    random.seed(42)  # For reproducible splits
    random.shuffle(labeled_files)
    split_idx = int(len(labeled_files) * train_ratio)
    train_files = labeled_files[:split_idx]
    val_files = labeled_files[split_idx:]
    
    print(f"📊 Train/Val split:")
    print(f"   🚂 Train: {len(train_files)} images")
    print(f"   🔍 Val: {len(val_files)} images")
    
    # Copy train files
    print("📋 Copying training files...")
    for img_path, txt_path in train_files:
        shutil.copy(img_path, train_images / img_path.name)
        # Remap to class 0 for single-vegetable datasets
        if vegetable != "all":
            dest_txt = train_labels / txt_path.name
            with open(txt_path, 'r') as src_f, open(dest_txt, 'w') as dst_f:
                for line in src_f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        remapped = ["0", parts[1], parts[2], parts[3], parts[4]]
                        dst_f.write(" ".join(remapped) + "\n")
        else:
            shutil.copy(txt_path, train_labels / txt_path.name)
    
    # Copy val files
    print("📋 Copying validation files...")
    for img_path, txt_path in val_files:
        shutil.copy(img_path, val_images / img_path.name)
        if vegetable != "all":
            dest_txt = val_labels / txt_path.name
            with open(txt_path, 'r') as src_f, open(dest_txt, 'w') as dst_f:
                for line in src_f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        remapped = ["0", parts[1], parts[2], parts[3], parts[4]]
                        dst_f.write(" ".join(remapped) + "\n")
        else:
            shutil.copy(txt_path, val_labels / txt_path.name)

    # Remove any stale label caches to force Ultralytics to reindex
    for cache_path in [train_labels / "labels.cache", val_labels / "labels.cache"]:
        if cache_path.exists():
            try:
                cache_path.unlink()
            except Exception:
                pass
    
    # Create dataset.yaml
    if vegetable == "all":
        # Multi-class config for 'all' using classes from classes.txt if available
        classes_file = labels_dir / "classes.txt"
        if classes_file.exists():
            with open(classes_file, 'r') as cf:
                class_names = [line.strip() for line in cf if line.strip()]
        else:
            class_names = ["carrot", "potato", "orange"]

        names_lines = "\n".join([f"  {i}: {name}" for i, name in enumerate(class_names)])
        yaml_content = f"""# YOLOv8 Dataset Configuration for {vegetable} auto-labelling
path: {dataset_dir.absolute()}  # dataset root dir
train: train/images  # train images (relative to 'path')
val: val/images  # val images (relative to 'path')

# Classes
nc: {len(class_names)}  # number of classes
names:
{names_lines}

# Auto-labelling info
total_images: {len(labeled_files) + len(unlabeled_files)}
labeled_images: {len(labeled_files)}
unlabeled_images: {len(unlabeled_files)}
"""
    else:
        yaml_content = f"""# YOLOv8 Dataset Configuration for {vegetable} auto-labelling
path: {dataset_dir.absolute()}  # dataset root dir
train: train/images  # train images (relative to 'path')
val: val/images  # val images (relative to 'path')

# Classes
nc: 1  # number of classes
names:
  0: {vegetable}

# Auto-labelling info
total_images: {len(labeled_files) + len(unlabeled_files)}
labeled_images: {len(labeled_files)}
unlabeled_images: {len(unlabeled_files)}
"""
    
    yaml_path = dataset_dir / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"✅ Dataset prepared successfully!")
    print(f"   📂 Dataset directory: {dataset_dir}")
    print(f"   📄 Config file: {yaml_path}")
    print(f"   🎯 Ready for YOLOv8 training!")
    
    return dataset_dir, yaml_path, labeled_files, unlabeled_files

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare dataset for YOLOv8 auto-labelling")
    parser.add_argument("--vegetable", "-v", default="all", 
                       help="Vegetable to process (default: all)")
    parser.add_argument("--train-ratio", "-r", type=float, default=0.8,
                       help="Train/val split ratio (default: 0.8)")
    
    args = parser.parse_args()
    
    dataset_dir, yaml_path, labeled_files, unlabeled_files = prepare_yolo_dataset(
        vegetable=args.vegetable,
        train_ratio=args.train_ratio
    )
    
    if dataset_dir:
        print(f"\n🚀 Next steps:")
        print(f"   1. Train model: python train_model.py --vegetable {args.vegetable}")
        print(f"   2. Auto-label: python auto_label.py --vegetable {args.vegetable}")

if __name__ == "__main__":
    main()
