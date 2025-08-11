#!/usr/bin/env python3
"""
Complete auto-labelling workflow for vegetables using YOLOv8
Orchestrates the entire process from dataset preparation to auto-labelling
"""

import subprocess
import sys
from pathlib import Path
import argparse


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*50}")
    print(f"🚀 {description}")
    print(f"{'='*50}")
    print(f"Command: {' '.join(command)}")
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def check_requirements():
    """Check if required packages are installed"""
    print("🔍 Checking requirements...")
    
    required_packages = [
        ('ultralytics', 'ultralytics'),
        ('torch', 'torch'),
        ('cv2', 'opencv-python'),
        ('numpy', 'numpy'),
        ('yaml', 'PyYAML')
    ]
    
    missing_packages = []
    
    for package_name, pip_name in required_packages:
        try:
            __import__(package_name)
            print(f"   ✅ {package_name}")
        except ImportError:
            print(f"   ❌ {package_name} (install with: pip install {pip_name})")
            missing_packages.append(pip_name)
    
    return missing_packages

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    
    requirements_file = Path("requirements.txt")
    if requirements_file.exists():
        return run_command([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ], "Installing packages from requirements.txt")
    else:
        print("❌ requirements.txt not found!")
        return False

def complete_auto_labelling_workflow(vegetable="carrot", epochs=50, conf=0.5):
    """
    Complete workflow for auto-labelling
    
    Args:
        vegetable: Vegetable to process
        epochs: Training epochs
        conf: Confidence threshold for auto-labelling
    """
    
    print(f"\n🥕 Starting complete auto-labelling workflow for {vegetable}")
    print(f"📊 Parameters: epochs={epochs}, confidence={conf}")
    
    # Step 1: Prepare dataset
    if not run_command([
        sys.executable, "prepare_dataset.py", "--vegetable", vegetable
    ], f"Preparing dataset for {vegetable}"):
        return False
    
    # Step 2: Train model
    if not run_command([
        sys.executable, "train_model.py", 
        "--vegetable", vegetable,
        "--epochs", str(epochs)
    ], f"Training YOLOv8 model for {vegetable}"):
        return False
    
    # Step 3: Auto-label remaining images
    if not run_command([
        sys.executable, "auto_label.py",
        "--vegetable", vegetable,
        "--conf", str(conf),
        "--verify"
    ], f"Auto-labelling remaining {vegetable} images"):
        return False
    
    print(f"\n🎉 Complete auto-labelling workflow finished successfully!")
    print(f"📂 Check results in ../../data/yolo_labels/")
    
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Complete YOLOv8 auto-labelling workflow")
    
    # Workflow options
    parser.add_argument("--vegetable", "-v", default="carrot",
                       help="Vegetable to process (default: carrot)")
    parser.add_argument("--input-dir", "-i", default=None,
                       help="Custom input directory for images (e.g., 'added'). If not specified, uses 'data/raw/{vegetable}'")
    parser.add_argument("--epochs", "-e", type=int, default=50,
                       help="Training epochs (default: 50)")
    parser.add_argument("--conf", "-c", type=float, default=0.5,
                       help="Confidence threshold for auto-labelling (default: 0.5)")
    
    # Step-by-step options
    parser.add_argument("--prepare-only", action="store_true",
                       help="Only prepare dataset")
    parser.add_argument("--train-only", action="store_true",
                       help="Only train model")
    parser.add_argument("--label-only", action="store_true",
                       help="Only auto-label (requires trained model)")
    
    # Setup options
    parser.add_argument("--install-requirements", action="store_true",
                       help="Install required packages")
    parser.add_argument("--check-requirements", action="store_true",
                       help="Check if required packages are installed")
    
    args = parser.parse_args()
    
    # Handle setup commands
    if args.check_requirements:
        missing = check_requirements()
        if missing:
            print(f"\n❌ Missing packages: {', '.join(missing)}")
            print("Install with: python run_auto_labelling.py --install-requirements")
        else:
            print("\n✅ All requirements satisfied!")
        return
    
    if args.install_requirements:
        if install_requirements():
            print("\n✅ Requirements installed successfully!")
        else:
            print("\n❌ Failed to install requirements!")
        return
    
    # Check requirements before running workflow
    missing = check_requirements()
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Install with: python run_auto_labelling.py --install-requirements")
        return
    
    # Handle step-by-step commands
    if args.prepare_only:
        run_command([
            sys.executable, "prepare_dataset.py", "--vegetable", args.vegetable
        ], f"Preparing dataset for {args.vegetable}")
        return
    
    if args.train_only:
        run_command([
            sys.executable, "train_model.py",
            "--vegetable", args.vegetable,
            "--epochs", str(args.epochs)
        ], f"Training model for {args.vegetable}")
        return
    
    if args.label_only:
        # Build command with input-dir if specified
        cmd = [sys.executable, "auto_label.py",
               "--vegetable", args.vegetable,
               "--conf", str(args.conf),
               "--verify"]
        
        if args.input_dir:
            cmd.extend(["--input-dir", args.input_dir])
            
        run_command(cmd, f"Auto-labelling {args.vegetable}")
        return
    
    # Run complete workflow
    success = complete_auto_labelling_workflow(
        vegetable=args.vegetable,
        epochs=args.epochs,
        conf=args.conf
    )
    
    if success:
        print(f"\n🎉 Auto-labelling completed successfully!")
    else:
        print(f"\n❌ Auto-labelling workflow failed!")

if __name__ == "__main__":
    main()
