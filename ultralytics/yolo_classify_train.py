import os

from ultralytics import YOLO


def main():
    # ==== 禁止联网检查 ====
    os.environ["YOLO_OFFLINE"] = "True"
    os.environ["ULTRALYTICS_HUB"] = "False"
    os.environ["ULTRALYTICS_API_KEY"] = ""

    # ==== 本地权重路径 ====
    weight_path = r"E:\Human_Projects\yolov11\ultralytics-main\ultralytics\yolo11n-cls.pt"

    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"❌ 权重文件不存在: {weight_path}")

    # ==== 载入模型 ====
    model = YOLO(weight_path)

    # ==== 开始训练 ====
    model.train(
        data="dataset",  # 你的分类数据集路径
        epochs=100,
        imgsz=224,
        device="0",
        workers=0,  # 🔴 建议在 Windows 下设为 0 以避免多进程问题
    )

    # ==== 导出ONNX ====
    path = model.export(format="onnx")
    print(f"✅ 模型已导出: {path}")


if __name__ == "__main__":
    main()
