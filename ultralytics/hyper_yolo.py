from ultralytics import YOLO


def main():
    model = YOLO("E:/Human_Projects/yolov11/ultralytics-main/ultralytics/cfg/models/11/yolo11-seg.yaml").load(
        "E:/models/lightly_train/out/my_experiment6/exported_models/exported_last.pt"
    )

    # ----------- 分割任务安全搜索空间（确保不会训练崩掉） -----------

    search_space = {
        # 训练参数
        "lr0": (1e-4, 5e-3),
        "lrf": (0.1, 1.0),
        "momentum": (0.7, 0.97),
        "weight_decay": (0.0, 0.0005),
        # warmup（seg 任务推荐小一些）
        "warmup_epochs": (0.0, 3.0),
        # 数据增强（避免 seg 崩溃）
        "degrees": (0.0, 45.0),
        "scale": (0.5, 1.2),
        "flipud": (0.0, 0.5),
        "fliplr": (0.0, 0.5),
        "mixup": (0.0, 0.15),
        "hsv_v": (0.0, 0.5),
        # mosaic — segmentation 不建议开启
        "mosaic": (0.0, 0.1),
    }

    print("🚀 开始 Tune ...")

    results = model.tune(
        data="E:/Human_Projects/yolov11/ultralytics-main/ultralytics/datasets/yolo.yaml",  # ← 必须改成你的真实路径
        epochs=50,
        iterations=40,  # 调优次数
        optimizer="AdamW",
        space=search_space,
        plots=False,  # 先关闭绘图（避免空数据报错）
        save=True,
        val=True,
        workers=0,
        project="tune_cells_seg",
        name="exp1",
    )

    print("Tune 完成:", results)


if __name__ == "__main__":
    main()
