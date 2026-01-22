import shutil
from pathlib import Path

from PIL import Image

from ultralytics import YOLO


def filter_low_confidence_images():
    # ================= 配置区域 =================
    # 1. 模型路径：请修改为您训练好的最佳权重路径，例如 'runs/segment/train20/weights/best.pt'
    # 如果还没有训练好的，先用预训练模型测试(但效果可能一般)
    model_path = r"E:\Human_Projects\yolov11\ultralytics-main\runs\segment\train13\weights\best.pt"

    # 2. 待预测的图像文件夹路径
    source_dir = r"E:\Human_Projects\yolov11\ultralytics-main\ultralytics\datasets_needle\images"

    # 3. 结果输出路径
    output_dir = r"D:\result_low_conf3"

    # 4. 筛选阈值：最大置信度低于此值的图片将被挑选出来
    confidence_threshold = 0.7
    # ===========================================

    # 如果使用的是 Windows 路径，确保路径格式正确
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    # 创建输出目录
    if output_path.exists():
        print(f"清空旧结果目录: {output_dir}")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"正在加载模型: {model_path} ...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("请检查 'model_path' 是否正确指向了 .pt 文件")
        return

    # 支持的图像扩展名
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    images = [f for f in source_path.rglob("*") if f.suffix.lower() in img_extensions]

    if not images:
        print(f"并未在 {source_dir} 找到任何图像。请检查路径。")
        return

    print(f"找到 {len(images)} 张图像，开始预测...")

    count_low_conf = 0

    # 批量预测（由于需要逐个判断逻辑，我们在循环中处理，为了速度也可以 batch=16 推理然后处理结果）
    # 这里为了代码逻辑清晰，逐张（或小批量）处理

    for i, img_file in enumerate(images):
        try:
            # 使用 PIL 读取并强制转换为 RGB，这是处理 TIF/单通道最稳健的方法
            # 避免 OpenCV 在处理特殊位深 TIF 时的各种边界情况
            img_pil = Image.open(img_file).convert("RGB")

            # 运行预测 (直接传入 PIL 对象，Ultralytics 内部会自动处理)
            # conf=0.01 是为了让模型尽量吐出结果，哪怕置信度很低
            results = model.predict(source=img_pil, conf=0.4, verbose=False, device="1")

            result = results[0]
            boxes = result.boxes

            max_conf = 0.0
            if len(boxes) > 0:
                # 获取这张图中所有检测框的最高置信度
                # sorted desc
                confs = boxes.conf.cpu().numpy()
                max_conf = confs.max()

            # 筛选逻辑：
            # 如果整张图中，最强的那个目标的置信度都不到 0.5
            # 或者根本没有检测到任何目标 (max_conf == 0.0)
            if max_conf < confidence_threshold:
                count_low_conf += 1

                # 构建输出文件名： "置信度_父文件夹_原文件名"
                # 防止不同子文件夹中有同名文件(如 1.jpg) 导致覆盖，同时保留所有结果在一个文件夹内
                # 例如: "0.35_batch1_needle_01.jpg"
                new_filename = f"{max_conf:.2f}_{img_file.parent.name}_{img_file.name}"
                dest_path = output_path / new_filename

                # 复制原始图像
                shutil.copy2(img_file, dest_path)

                # 可选：如果你想把检测结果画上去看看为什么低，可以用 result.save()
                result.save(filename=str(output_path / f"plot_{new_filename}"))

                print(f"[{i + 1}/{len(images)}] 挑选出: {img_file.name} (Max Conf: {max_conf:.2f})")

        except Exception as e:
            print(f"处理图像 {img_file.name} 时出错: {e}")

    print("-" * 50)
    print("处理完成。")
    print(f"共扫描: {len(images)} 张")
    print(f"挑选出低置信度 (<{confidence_threshold}): {count_low_conf} 张")
    print(f"结果保存在: {output_dir}")


if __name__ == "__main__":
    filter_low_confidence_images()
