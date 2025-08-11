#!/usr/bin/env python3
"""
Launch LabelImg for labeling images
"""

import subprocess
import sys
import os
from pathlib import Path

def install_labelimg():
    """Install LabelImg if not available"""
    print("📦 Installing LabelImg...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "labelImg"])
        print("✅ LabelImg installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install LabelImg")
        return False

def check_labelimg():
    """Check if LabelImg is installed"""
    try:
        import labelImg
        return True
    except ImportError:
        return False

def setup_labelimg_directories():
    """Setup directories for LabelImg"""
    base_dir = Path(".")
    
    # Create directories if they don't exist
    directories = {
        "images": base_dir / "data/raw/carrot",
        "labels": base_dir / "data/yolo_labels", 
        "classes": base_dir / "data/classes/classes.txt"
    }
    
    print("📁 Setting up LabelImg directories:")
    for name, path in directories.items():
        if name == "classes":
            if path.exists():
                print(f"   ✅ {name}: {path}")
            else:
                print(f"   ❌ {name}: {path} (missing)")
                # Create classes.txt if missing
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, 'w') as f:
                    f.write("carrot\n")
                print(f"   ✅ Created {path}")
        else:
            if path.exists() and path.is_dir():
                count = len(list(path.glob("*")))
                print(f"   ✅ {name}: {path} ({count} files)")
            else:
                print(f"   ❌ {name}: {path} (missing)")
    
    return directories

def launch_labelimg():
    """Launch LabelImg with proper configuration"""
    
    # Check if LabelImg is installed
    if not check_labelimg():
        print("❌ LabelImg not found!")
        if input("Install LabelImg? (y/n): ").lower() == 'y':
            if not install_labelimg():
                return False
        else:
            return False
    
    # Setup directories
    directories = setup_labelimg_directories()
    
    print("\n🚀 Launching LabelImg...")
    print("📋 Instructions:")
    print("   1. Set Image Dir to: data/raw/carrot")
    print("   2. Set Label Dir to: data/yolo_labels")
    print("   3. Load classes from: data/classes/classes.txt")
    print("   4. Make sure format is set to YOLO")
    print("   5. Review auto-generated labels (many may have low confidence)")
    print("\n⚠️  Note: Auto-generated labels have low confidence - review carefully!")
    
    try:
        # Try to launch LabelImg
        subprocess.run([sys.executable, "-m", "labelImg"], check=True)
    except subprocess.CalledProcessError:
        try:
            # Alternative launch method
            subprocess.run(["labelImg"], check=True)
        except subprocess.CalledProcessError:
            # Try direct Python import
            try:
                from labelImg import labelImg
                labelImg.main()
            except Exception as e:
                print(f"❌ Failed to launch LabelImg: {e}")
                print("Try installing with: pip install labelImg")
                return False
    
    return True

def main():
    """Main function"""
    print("🏷️  LabelImg Launcher for Vegetable AI")
    print("=" * 50)
    
    if launch_labelimg():
        print("✅ LabelImg launched successfully!")
    else:
        print("❌ Failed to launch LabelImg!")

if __name__ == "__main__":
    main()