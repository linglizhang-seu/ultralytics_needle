# import cv2
# from pathlib import Path

# root_dir = Path(r"E:\Human_Projects\yolov11\ultralytics-main\ultralytics\datasets\images")
# save_dir = Path(r"E:\Human_Projects\yolov11\ultralytics-main\ultralytics\datasets\images_rgb")
# # 数据根目录
# save_dir.mkdir(parents=True, exist_ok=True)

# # 遍历 train / val / test
# for split in ["train", "val", "test"]:
#     split_dir = root_dir / split
#     save_split_dir = save_dir / split
#     save_split_dir.mkdir(parents=True, exist_ok=True)
#     for img_path in split_dir.glob("*.*"):  # 匹配所有文件
#         # 读取图像
#         im = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)  # 保持原通道
#         if im is None:
#             print(f"无法读取 {img_path}, 跳过")
#             continue

#         # 如果是灰度图，转换为 3 通道 RGB
#         if len(im.shape) == 2 or im.shape[2] == 1:
#             im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

#         # 保存到 images_rgb 对应子目录
#         save_path = save_split_dir / img_path.name
#         cv2.imwrite(str(save_path), im)
#         print(f"已保存 {save_path}")

import torch

from ultralytics import YOLO

torch.cuda.empty_cache()
# torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()


# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.chdir(os.path.dirname(__file__))  # 切
def main():
    weight_path = r"E:\models\lightly_train\out\my_experiment_640\exported_last.pt"
    # weight_path = r"E:\Human_Projects\yolov11\ultralytics-main\ultralytics\yolo11n-seg.pt"

    model = YOLO(r"E:/Human_Projects/yolov11/ultralytics-main/ultralytics/cfg/models/11/yolo11-seg-wfu.yaml").load(
        weight_path
    )
    model.train(
        data="ultralytics/datasets/yolo.yaml",  # 你的分类数据集路径
        epochs=400,
        patience=100,
        imgsz=224,
        device="1",
        workers=0,
        # 🔴 建议在 Windows 下设为 0 以避免多进程问         # 🔴 禁用混合精度训练，防止 NaN / Inf
    )

    results = model.val(data="ultralytics/datasets/yolo.yaml")
    # from ultralytics import YOLO

    # model = YOLO(r"E:/Human_Projects/yolov11/ultralytics-main/ultralytics/runs/segment/train7_my_experiment_6/weights/best.pt")

    # results = model.train(
    #     data="ultralytics/datasets/yolo.yaml",
    #     epochs=150,
    #     freeze=3,           # 冻结前3层
    #     lr0=0.0005,         # 小学习率
    #     optimizer='AdamW',  # AdamW对微调更友好
    #     mosaic=0.0,         # 关闭强增强
    #     mixup=0.0
    # )

    print("F1 score:", results.box.f1)
    # print("F1 score curve:", results.box.f1_curve)
    # print("Overall fitness score:", results.box.fitness)
    # print("Mean average precision:", results.box.map)
    print("Mean average precision at IoU=0.50:", results.box.map50)
    print("Mean average precision at IoU=0.75:", results.box.map75)
    # print("Mean average precision for different IoU thresholds:", results.box.maps)
    # print("Mean results for different metrics:", results.box.mean_results)
    print("Mean precision:", results.box.mp)
    print("Mean recall:", results.box.mr)
    # print("Precision:", results.box.p)
    # print("Precision curve:", results.box.p_curve)
    # print("Precision values:", results.box.prec_values)
    # # print("Specific precision metrics:", results.box.px)
    # print("Recall:", results.box.r)
    # print("Recall curve:", results.box.r_curve)

    # ==== 导出ONNX ====
    model.export(format="onnx")


if __name__ == "__main__":
    main()
