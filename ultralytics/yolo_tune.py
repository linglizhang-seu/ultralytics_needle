
import os
import pathlib
import onnx
import cv2
import albumentations as A
# import wandb
from ultralytics import YOLO
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ray import tune
# wandb.init(project="YOLO-Tuning", entity="your-entity")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.chdir(os.path.dirname(__file__))  # 切
def main():
    weight_path = (r"E:\models\lightly_train\out\my_experiment6\exported_models\exported_last.pt")
    # weight_path = r"E:\Human_Projects\yolov11\ultralytics-main\ultralytics\yolo11n-seg.pt"

    model = YOLO(r"E:/Human_Projects/yolov11/ultralytics-main/ultralytics/cfg/models/11/yolo11-seg.yaml").load(weight_path)
    search_space = {
    "lr0": (1e-5, 1e-1),
    "copy_paste": (0.0, 0.5),
    "mosaic": (0.0, 1.0),
    "warmup_epochs": (0.0, 5.0),
    "warmup_momentum": (0.0, 0.9),
}
    
    model.tune(
    data="ultralytics/datasets/yolo.yaml",
    space=search_space,
    epochs=50,
    iterations=300,
    optimizer='AdamW',
    plots=True,
    save=True,
    val=True,
    imgsz=1024,
    device='1',
    )

  
if __name__ == "__main__":
    main()
   
