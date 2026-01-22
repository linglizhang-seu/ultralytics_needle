
import onnx
import cv2
import albumentations as A
from ultralytics import YOLO
import torch
torch.cuda.empty_cache()
def main():
    # weight_path = (r"E:\models\lightly_train\out\my_experiment_512\exported_last.pt")
    # weight_path = r"E:\Human_Projects\yolov11\ultralytics-main\ultralytics\yolo11s-seg.pt" # 建议尝试 s 或 m 模型以提升精度
    weight_path = r"E:\Human_Projects\yolov11\ultralytics-main\ultralytics\yolo11s-seg.pt"

    # model = YOLO(r"E:/Human_Projects/yolov11/ultralytics-main/ultralytics/cfg/models/11/yolo11-seg-wfu.yaml").load \
    #     (weight_path)
    model = YOLO(r"E:/Human_Projects/yolov11/ultralytics-main/ultralytics/cfg/models/11/yolo11-seg.yaml").load \
        (weight_path)
    model.train(
        data="ultralytics/datasets/yolo.yaml",     # 你的分类数据集路径
        epochs=400,
        patience=100,
        imgsz=1024,
        device="1",
        workers=0,
        # ==== 基础数据增强 ====
        degrees=180.0,      # 旋转范围 (+/- 180度)
        translate=0.1,      # 平移
        scale=0.5,          # 缩放 (默认0.5)
        hsv_h=0.015,        # 色调增强 (默认 0.015)
        hsv_s=0.7,          # 饱和度增强 (默认 0.7)
        hsv_v=0.4,          # 亮度/明度增强 (默认 0.4) - 对应强度/对比度调整
        flipud=0.5,         # 上下翻转概率
        fliplr=0.5,         # 左右翻转概率
        mosaic=1.0,         # 马赛克增强 (开启，提升小目标及复杂背景鲁棒性)
        mixup=0.1,          # Mixup (减少一点，避免过度混淆)
        copy_paste=0.5,     # CopyPaste (仅分割有效) 增加复制粘贴概率，制造更多遮挡和实例
        auto_augment="randaugment", # 自动增强策略
        erasing=0.4,        # 随机擦除，模拟遮挡
        crop_fraction=1.0,  # 保持原图比例

    )

    results=model.val(data="ultralytics/datasets/yolo.yaml")

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
    print(torch.cuda.is_available())
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_name(1))
    main()

