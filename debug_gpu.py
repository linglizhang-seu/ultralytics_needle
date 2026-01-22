import sys

import torch

from ultralytics import YOLO

print(f"Python version: {sys.version}")
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
        print(f"  Memory Reserved: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")

try:
    print("\nValidating YOLO device selection...")
    # Try to load a model and move to device 1
    model = YOLO("yolo11n.pt")
    # Just check if we can pass device='1'
    print("Attempting to train for 1 epoch on device='1'...")
    model.train(data="coco8.yaml", epochs=1, imgsz=64, device="1")
    print("Train completed successfully on device='1'")
except Exception as e:
    print(f"\nERROR during YOLO execution: {e}")
