import numpy as np
from shapely.geometry import Point, Polygon

from ultralytics import YOLO


def calculate_iou(box1, box2):
    """计算两个边界框的IoU."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0


def refine_needle_detection(model_path, image_path):
    # 加载模型
    model = YOLO(model_path)

    # 推理
    # conf=0.25 降低阈值以召回更多潜在目标，后续通过逻辑过滤
    results = model.predict(image_path, conf=0.15, iou=0.45)

    result = results[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()
    masks = result.masks.xy if result.masks else None

    needles = []
    tips = []

    # 1. 分类整理检测结果
    for i, class_id in enumerate(cls):
        item = {"box": boxes[i], "conf": conf[i], "mask": masks[i] if masks is not None else None, "id": i}
        if class_id == 0:  # 假设 0 是 needle (全针)
            needles.append(item)
        elif class_id == 1:  # 假设 1 是 needle_tip (针尖)
            tips.append(item)

    final_detections = []

    # 2. 逻辑关联与过滤

    # 情况A: 找到针身
    # 我们认为置信度高的针身是可靠的锚点
    valid_needles = [n for n in needles if n["conf"] > 0.4]

    for needle in valid_needles:
        # 寻找归属于该针身的针尖
        associated_tip = None
        float("inf")

        needle_poly = Polygon(needle["mask"]) if needle["mask"] is not None else None

        for tip in tips:
            # 计算距离或包含关系
            # 简单方法：检查 Tip 中心是否在 Needle 框内/掩膜内，或距离很近
            tip_center = ((tip["box"][0] + tip["box"][2]) / 2, (tip["box"][1] + tip["box"][3]) / 2)
            tip_point = Point(tip_center)

            # 如果有掩膜，使用精确几何检查
            is_contained = False
            if needle_poly:
                is_contained = needle_poly.buffer(20).contains(tip_point)  # 缓冲20像素容错
            else:
                # 简单的 Box 包含检查
                n_box = needle["box"]
                # 扩大一点针身框
                if (
                    tip_center[0] > n_box[0] - 10
                    and tip_center[0] < n_box[2] + 10
                    and tip_center[1] > n_box[1] - 10
                    and tip_center[1] < n_box[3] + 10
                ):
                    is_contained = True

            if is_contained:
                # 找到关联针尖，取置信度最高的或者距离最近的
                # 这里简单取置信度最高的
                if associated_tip is None or tip["conf"] > associated_tip["conf"]:
                    associated_tip = tip

        # 添加结果
        final_detections.append(
            {
                "type": "whole_needle",
                "box": needle["box"],
                "has_tip": associated_tip is not None,
                "tip_box": associated_tip["box"] if associated_tip else None,
            }
        )

    # 情况B: 只有针尖，没有高置信度针身 (漏检针身或针尖也是误检?)
    # 如果针尖置信度极高，可以保留，但标记为"疑似缺失针身"
    # 或者寻找低置信度的针身 (conf < 0.4) 尝试召回
    for tip in tips:
        # 检查是否已经被关联
        is_associated = False
        for fd in final_detections:
            if fd["tip_box"] is not None and np.array_equal(fd["tip_box"], tip["box"]):
                is_associated = True
                break

        if not is_associated:
            # 这是一个孤立的针尖
            # 在全集中寻找低置信度针身
            for needle in needles:  # 包括低分针身
                if needle["conf"] <= 0.4:
                    # 同样的距离/包含逻辑...
                    pass  # (此处省略重复代码，实际应用需封装)

            if tip["conf"] > 0.6:  # 对于孤立针尖，要求极高置信度才认为是真的
                final_detections.append(
                    {"type": "tip_only", "tip_box": tip["box"], "note": "High confidence tip without body"}
                )
            else:
                # 认为是漂浮物误检，丢弃
                print(f"Filtered out isolated low-conf tip: {tip['conf']}")

    return final_detections


def demo():
    # 使用示例
    # 替换为你训练好的模型路径
    # image_path = r"..."
    # res = refine_needle_detection(model_path, image_path)
    # print(res)
    pass


if __name__ == "__main__":
    demo()
