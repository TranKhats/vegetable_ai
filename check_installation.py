#!/usr/bin/env python3
"""
Installation check script for Vegetable AI project
"""

import sys
import subprocess
import importlib
from pathlib import Path

def check_python_version():
    """Check Python version"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Need Python 3.8+")
        return False

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name} - OK")
        return True
    except ImportError:
        print(f"❌ {package_name} - Not installed")
        return False

def install_requirements():
    """Install requirements from requirements.txt"""
    print("\n📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        return False

def check_directories():
    """Check if required directories exist"""
    print("\n📁 Checking directories...")
    
    required_dirs = [
        "data",
        "data/raw",
        "data/processed", 
        "data/labels",
        "preprocessing",
        "yolo"
    ]
    
    all_good = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path} - OK")
        else:
            print(f"❌ {dir_path} - Missing")
            all_good = False
    
    return all_good

def check_files():
    """Check if required files exist"""
    print("\n📄 Checking files...")
    
    required_files = [
        "requirements.txt",
        "main.py",
        "data/dataset.yaml",
        "preprocessing/__init__.py",
        "preprocessing/pipeline.py",
        "yolo/train.py",
        "yolo/predict.py"
    ]
    
    all_good = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path} - OK")
        else:
            print(f"❌ {file_path} - Missing")
            all_good = False
    
    return all_good

def check_gpu():
    """Check GPU availability"""
    print("\n🎮 Checking GPU...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA available - {gpu_count} GPU(s)")
            print(f"   GPU: {gpu_name}")
            return True
        else:
            print("⚠️  CUDA not available - will use CPU")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def main():
    """Main check function"""
    print("🔍 Vegetable AI Installation Check")
    print("=" * 40)
    
    checks = []
    
    # Check Python version
    checks.append(check_python_version())
    
    # Check if requirements.txt exists
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        return
    
    # Ask user if they want to install requirements
    print("\n📦 Checking Python packages...")
    
    required_packages = [
        ("torch", "torch"),
        ("torchvision", "torchvision"), 
        ("ultralytics", "ultralytics"),
        ("opencv-python", "cv2"),
        ("Pillow", "PIL"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("pandas", "pandas"),
        ("pyyaml", "yaml"),
        ("albumentations", "albumentations"),
        ("scikit-learn", "sklearn")
    ]
    
    missing_packages = []
    for package_name, import_name in required_packages:
        if not check_package(package_name, import_name):
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        install = input("\nDo you want to install missing packages? (y/n): ").lower().strip()
        if install == 'y':
            if install_requirements():
                print("\n✅ Packages installed. Re-checking...")
                # Re-check packages
                for package_name, import_name in required_packages:
                    check_package(package_name, import_name)
    
    # Check directories and files
    checks.append(check_directories())
    checks.append(check_files())
    
    # Check GPU
    check_gpu()
    
    # Final status
    print("\n" + "=" * 40)
    if all(checks):
        print("🎉 All checks passed! Your environment is ready.")
        print("\nNext steps:")
        print("1. Add your raw images to data/raw/")
        print("2. Run: python main.py")
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        print("\nCommon solutions:")
        print("- Install missing packages: pip install -r requirements.txt")
        print("- Create missing directories manually")
        print("- Check file permissions")

if __name__ == "__main__":
    main()
