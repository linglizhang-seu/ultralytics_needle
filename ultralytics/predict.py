from unittest import result
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
import os

def fit_triangle_to_mask_cnt(mask_xy_list):
    """
    输入 mask 的轮廓点坐标 (N, 2)，返回拟合的最小外接三角形轮廓点 (3, 2) (int)
    """
    if not isinstance(mask_xy_list, np.ndarray):
        pts = np.array(mask_xy_list, dtype=np.float32)
    else:
        pts = mask_xy_list.astype(np.float32)

    # 必须 reshape 成 (-1, 1, 2) 才能给 minEnclosingTriangle 用
    pts = pts.reshape(-1, 1, 2)

    # 计算最小外接三角形
    # retval 是面积，triangle 是三角形的 3 个顶点坐标 (3, 1, 2)
    retval, triangle = cv2.minEnclosingTriangle(pts)
    
    # 转为 int 以便 cv2 绘制
    triangle_pts = np.int32(triangle)
    return triangle_pts

def run_predict():
    # 1. 模型路径：请修改为您认为效果最好的那个训练权重
    # (根据之前的文件列表，train20 似乎有 best.pt，您代码里写的是 train9，请自行确认)
    model_path = r"E:\Human_Projects\yolov11\ultralytics-main\runs\segment\train15\weights\best.pt"
    
    # 2. 测试集路径 (针尖数据集)
    source_dir = Path(r"D:\snapshot_result_images")
    
    # 3. 结果保存目录
    output_dir = Path(r"D:\snapshot_result_2")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"加载模型: {model_path}")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 扫描图像文件
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    # 检查目录是否存在
    if not source_dir.exists():
         print(f"源目录不存在: {source_dir}")
         return
         
    files = [f for f in source_dir.iterdir() if f.is_file() and f.suffix.lower() in extensions]
    
    if not files:
        print(f"在 {source_dir} 未找到图像")
        return

    print(f"开始预测 {len(files)} 张图像...")

    for i, file_path in enumerate(files):
        try:
            # ==== 关键步骤：使用 PIL 读取并转 RGB (解决 TIF 单通道报错) ====
            img = Image.open(file_path).convert('RGB')
            
            # 预测
            # 务必保证模型使用了包含 tip 和 needle 的训练权重
            results = model.predict(img, imgsz=1024, conf=0.3, verbose=False, device='1')
            for result in results:
                 # 保存原始预测结果对比
                 orig_plot = result.plot()
                 orig_save_path = output_dir / f"orig_{file_path.name}"
                 cv2.imwrite(str(orig_save_path), orig_plot)

                 # 1. 准备绘图
                 img_plot = result.orig_img.copy()
                 h, w = img_plot.shape[:2]

                 # 如果检测到了目标，只处理置信度最高的一个
                 if result.masks is not None and len(result.boxes) > 0:
                     # 找到置信度最高的索引
                     best_idx = result.boxes.conf.argmax().item()
                     
                     # 获取该索引对应的 mask
                     mask_pts = result.masks.xy[best_idx]
                     if len(mask_pts) == 0:
                         continue
                     
                     # ---- A. 形态学处理 (Morphological Operations) ----
                     # 1. 转为二值 Mask 图像
                     mask_bin = np.zeros((h, w), dtype=np.uint8)
                     pts_int = np.array(mask_pts, dtype=np.int32).reshape((-1, 1, 2))
                     cv2.fillPoly(mask_bin, [pts_int], 255)

                     # 2. 定义核 (Kernel)
                     # 根据图像分辨率调整大小，这里设为 7x7
                     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
                     
                     # 3. 开运算 (Opening): 去除毛刺/小噪点
                     mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel)
                     
                     # 4. 闭运算 (Closing): 填补内部空洞
                     mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel)

                     # 5. 重新提取轮廓
                     contours_morph, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                     
                     for cnt in contours_morph:
                         # ---- B. 拟合最小外接三角形 & 凸包 ----
                         
                         # 1. 拟合最小外接三角形
                         triangle_cnt = fit_triangle_to_mask_cnt(cnt.reshape(-1, 2))
                         
                         # [新增] 填充三角形 Mask (半透明红色) - "更改的是Mask"
                         # 绘制半透明填充
                         overlay = img_plot.copy()
                         cv2.drawContours(overlay, [triangle_cnt], -1, (0, 0, 255), -1)
                         
                         # 定义透明度
                         alpha = 0.4
                         
                         # 仅在三角形区域进行混合 (避免全图变色)
                         # 创建三角形掩码
                         mask_tri = np.zeros((h, w), dtype=np.uint8)
                         cv2.drawContours(mask_tri, [triangle_cnt], -1, 255, -1)
                         
                         # 提取区域
                         roi = img_plot[mask_tri == 255]
                         overlay_roi = overlay[mask_tri == 255]
                         
                         if roi.size > 0:
                             mixed_roi = cv2.addWeighted(roi, 1 - alpha, overlay_roi, alpha, 0)
                             img_plot[mask_tri == 255] = mixed_roi

                         # 绘制三角形边框 (红色，线宽2)
                         cv2.drawContours(img_plot, [triangle_cnt], 0, (0, 0, 255), 2)
                         
                         # 2. (可选) 绘制凸包 (蓝色，线宽1) - 用于对比
                         hull = cv2.convexHull(cnt)
                         cv2.drawContours(img_plot, [hull], 0, (255, 0, 0), 1)

                         # 3. 绘制经过形态学处理后的 Mask 轮廓 (绿色，细线)
                         cv2.drawContours(img_plot, [cnt], 0, (0, 255, 0), 1)

                     # 绘制该最佳检测的框和标签
                     box = result.boxes[best_idx]
                     x1, y1, x2, y2 = map(int, box.xyxy[0])
                     cls_id = int(box.cls)
                     conf = float(box.conf)
                     label = f"{result.names[cls_id]} {conf:.2f}"
                     
                     # 画框
                     cv2.rectangle(img_plot, (x1, y1), (x2, y2), (0, 255, 0), 1)
                     # 画标签
                     cv2.putText(img_plot, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                 # 保存结果图像
                 save_path = output_dir / file_path.name
                 print(f"保存处理后的图像: {save_path}")
                 cv2.imwrite(str(save_path), img_plot)





            # ==== 核心逻辑：高精度针尖提取 + 杂质过滤 (Advanced Filter) ====
            
            # 准备容器
            # candidate_tips = []   # 候选针尖
            # confirmed_needles = [] # 确认的全针
            
            # img_w, img_h = img.size
            
            # # 1. 第一轮：尺寸纠错与分类
            # if results[0].boxes:
            #     boxes = results[0].boxes
            #     masks = results[0].masks
                
            #     for i, box in enumerate(boxes):
            #         # 获取基础信息
            #         cls_id = int(box.cls)
            #         xywh = box.xywh[0].tolist()
            #         w, h = xywh[2], xywh[3]
            #         area_ratio = (w * h) / (img_w * img_h)
                    
            #         # 提取 Mask 多边形 (用于计算几何关系)
            #         # 注意：mask.xy 这是一个列表，如果是多段的取最长的那段
            #         poly = masks.xy[i] if masks is not None else []
            #         if len(poly) == 0: continue 

            #         from shapely.geometry import Polygon
            #         geo_poly = Polygon(poly)

            #         # --- 规则A: 尺寸修正类别 ---
            #         # 如果面积 > 3%，即便模型说是 Tip，也强制认为是 Needle
            #         if area_ratio > 0.03:
            #             confirmed_needles.append({'poly': geo_poly, 'box': box, 'mask_data': poly})
            #         else:
            #             # 只要面积小，就作为候选针尖 (即便模型说是 Needle)
            #             candidate_tips.append({'poly': geo_poly, 'box': box, 'mask_data': poly, 'conf': float(box.conf)})

            # # 2. 第二轮：包含关系验证 (排除杂质)
            # final_valid_tips = []
            
            # for tip in candidate_tips:
            #     tip_poly = tip['poly']
            #     is_valid = False
                
            #     if len(confirmed_needles) == 0:
            #         # 如果图中完全没找到全针，那么这个针尖大概率也是假的(或者全针漏检)
            #         # 保守起见，可以设一个较低的阈值保留，或者直接丢弃
            #         # 这里演示：如果没有全针作为依托，认为该针尖不可靠 (或根据您的实际情况修改)
            #         pass 
            #     else:
            #         for needle in confirmed_needles:
            #             needle_poly = needle['poly']
                        
            #             # 计算重叠
            #             # 针尖应该大部分在全针内部，或者与全针边缘紧密接触
            #             intersection = tip_poly.intersection(needle_poly).area
            #             tip_area = tip_poly.area
                        
            #             # 如果针尖有 > 50% 的面积在全针Mask内 (考虑分割边缘误差)
            #             if tip_area > 0 and (intersection / tip_area) > 0.1: # 稍微沾边就算
            #                 is_valid = True
            #                 break
                
            #     if is_valid:
            #         final_valid_tips.append(tip)
            #     else:
            #         print(f"  [过滤] 剔除一个疑似杂质的针尖 (置信度 {tip['conf']:.2f})，因为它没有连接到任何针身。")

            # # ==== 3. 结果输出 ====
            # print(f"\n图像: {file_path.name}")
            # print(f"  -> 全针(Needle)数量: {len(confirmed_needles)} (可用于定位针轴线)")
            # print(f"  -> 有效针尖(Tip)数量: {len(final_valid_tips)} (高精度，且已排除杂质)")

            # 可视化/保存逻辑...
            # 如果只想保存有效针尖，可以在这里利用 plot 也就是 final_valid_tips 里的 mask_data 绘制



                
        except Exception as e:
            print(f"处理 {file_path.name} 失败: {e}")

    print(f"预测完成！结果保存在: {output_dir}")

if __name__ == "__main__":
    run_predict()