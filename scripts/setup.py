"""
============================================================
 AI DeepFake Detector
 Project Setup Script
============================================================

Run:
    python -m scripts.setup
"""

import importlib
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ==========================================================
# Project Folders
# ==========================================================

FOLDERS = [
    "dataset",
    "dataset/raw",
    "dataset/processed",
    "dataset/processed/frames_real",
    "dataset/processed/frames_fake",
    "dataset/processed/faces_real",
    "dataset/processed/faces_fake",
    "trained_models",
    "reports",
    "logs",
    "temp"
]

# ==========================================================
# Required Libraries
# ==========================================================

LIBRARIES = {
    "torch": ("PyTorch", "pip install torch torchvision torchaudio"),
    "torchvision": ("TorchVision", "pip install torchvision"),
    "cv2": ("OpenCV", "pip install opencv-python"),
    "numpy": ("NumPy", "pip install numpy"),
    "pandas": ("Pandas", "pip install pandas"),
    "retinaface": ("RetinaFace", "pip install retina-face"),
    "fastapi": ("FastAPI", "pip install fastapi"),
    "tqdm": ("TQDM", "pip install tqdm"),
}

missing_packages = []

# ==========================================================
# Python Check
# ==========================================================

def check_python():

    print("\nPython Information")
    print("-" * 50)

    version = platform.python_version()

    print(f"Python Version : {version}")
    print(f"Operating System : {platform.system()} {platform.release()}")

    major, minor, *_ = map(int, version.split("."))

    if major == 3 and minor >= 13:
        print("✓ Python version is supported")
    else:
        print("⚠ Recommended Python version: 3.13 or later")


# ==========================================================
# Folder Check
# ==========================================================

def create_folders():

    print("\nProject Structure")
    print("-" * 50)

    created = 0
    existing = 0

    for folder in FOLDERS:

        path = ROOT / folder

        if path.exists():

            print(f"✓ Already Exists : {folder}")
            existing += 1

        else:

            path.mkdir(parents=True, exist_ok=True)

            print(f"➕ Created       : {folder}")
            created += 1

    return created, existing


# ==========================================================
# Library Check
# ==========================================================

def check_libraries():

    print("\nPython Libraries")
    print("-" * 50)

    installed = 0

    for module, (name, command) in LIBRARIES.items():

        try:

            lib = importlib.import_module(module)

            version = getattr(lib, "__version__", "Unknown")

            print(f"✓ {name:<15} {version}")

            installed += 1

        except ImportError:

            print(f"✖ {name:<15} NOT Installed")

            print(f"    Install: {command}")

            missing_packages.append(name)

    return installed


# ==========================================================
# Hardware Check
# ==========================================================

def check_gpu():

    print("\nHardware")
    print("-" * 50)

    try:

        import torch

        if torch.cuda.is_available():

            print("✓ GPU Available")

            print(f"GPU : {torch.cuda.get_device_name(0)}")

            print(f"CUDA: {torch.version.cuda}")

            return "GPU"

        else:

            print("✓ CPU Mode")

            return "CPU"

    except Exception:

        print("Unable to detect hardware.")

        return "Unknown"


# ==========================================================
# Summary
# ==========================================================

def summary(created, existing, installed, hardware):

    print("\n" + "=" * 60)
    print("PROJECT SUMMARY")
    print("=" * 60)

    print(f"Folders Created     : {created}")
    print(f"Folders Existing    : {existing}")
    print(f"Libraries Installed : {installed}/{len(LIBRARIES)}")
    print(f"Hardware            : {hardware}")

    if len(missing_packages) == 0:

        print("\n🎉 Environment is ready!")

    else:

        print("\n⚠ Missing Libraries:")

        for pkg in missing_packages:

            print(f"   - {pkg}")

    print("=" * 60)


# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 60)
    print("        AI DeepFake Detector - Setup")
    print("=" * 60)

    check_python()

    created, existing = create_folders()

    installed = check_libraries()

    hardware = check_gpu()

    summary(created, existing, installed, hardware)


if __name__ == "__main__":

    main()