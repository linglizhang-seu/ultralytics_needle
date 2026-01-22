
import warnings

import ultralytics
from ultralytics.models import yolo
warnings.filterwarnings('ignore')
import os
import numpy as np
from prettytable import PrettyTable
from ultralytics import YOLO
from ultralytics.utils.torch_utils import model_info
from ultralytics import settings, YOLO
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import os
import cv2

def resize_mask(mask, target_shape):
    # mask: [H, W]，target_shape: (H, W)
    return cv2.resize(mask.astype('float32'), (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST) > 0.5

# 在可视化前加上
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
def mask_iou(mask1, mask2):
    # mask1, mask2: [H, W], bool or 0/1
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return inter / (union + 1e-6)
def apply_color_mask(img, mask, color):
    # img: HWC, mask: HW, color: (B, G, R)
    if img.ndim == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    color_mask = np.zeros_like(img)
    for c in range(3):
        color_mask[..., c][mask] = color[c]
    return cv2.addWeighted(img, 1, color_mask, 0.4, 0)

def visualize_tp_fp_fn(pred_masks, gt_masks, orig_img, save_path, iou_thr=0.5):
    pred_used = np.zeros(len(pred_masks), dtype=bool)
    gt_used = np.zeros(len(gt_masks), dtype=bool)
    img_vis = orig_img.copy()

    # 1. TP: pred与gt一一匹配，IoU>阈值
    for i, p_mask in enumerate(pred_masks):
        for j, g_mask in enumerate(gt_masks):
            if not gt_used[j] and mask_iou(p_mask, g_mask) > iou_thr:
                img_vis = apply_color_mask(img_vis, p_mask > 0.5, (0, 255, 0))  # 绿色
                pred_used[i] = True
                gt_used[j] = True
                break

    # 2. FP: pred未匹配
    for i, p_mask in enumerate(pred_masks):
        if not pred_used[i]:
            img_vis = apply_color_mask(img_vis, p_mask > 0.5, (0, 0, 255))  # 红色

    # 3. FN: gt未匹配
    for j, g_mask in enumerate(gt_masks):
        if not gt_used[j]:
            img_vis = apply_color_mask(img_vis, g_mask > 0.5, (255, 0, 0))  # 蓝色

    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.imshow(img_vis[..., ::-1])
    plt.savefig(save_path)
    plt.close()
# 设置 datasets 根路径
settings.update({'datasets_dir': r"E:\Human_Projects\yolov11\ultralytics-main\ultralytics"})
def get_weight_size(path):
    stats = os.stat(path)
    return f'{stats.st_size / 1024 / 1024:.1f}'
def load_yolo_polygon_mask(label_path, img_shape):
        h, w = img_shape[:2]
        masks = []
        if not os.path.exists(label_path):
            return masks
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 7:  # 至少一个三角形
                    continue
                # 跳过class_id
                points = np.array(parts[1:], dtype=float).reshape(-1, 2)
                points[:, 0] *= w
                points[:, 1] *= h
                polygon = np.round(points).astype(np.int32)
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [polygon], 1)
                masks.append(mask.astype(bool))
        return masks
if __name__ == '__main__':
    model_path = r'E:\Human_Projects\yolov11\runs\segment\train16\weights\best.pt'
    model = YOLO(model_path) # 选择训练好的权重路径
    result = model.val(data=r"E:\Human_Projects\yolov11\ultralytics-main\ultralytics\datasets\yolo.yaml",
                        imgsz=1024,
                        split='test',
                        batch=32,
                        project='runs/val',
                        name='exp',
                            )
    # test_img_dir = r"E:\Human_Projects\yolov11\ultralytics-main\ultralytics\datasets\images\test"
    # img_list = [os.path.join(test_img_dir, f) for f in os.listdir(test_img_dir) if f.lower().endswith(('.jpg', '.png', '.tif'))]

    # for i, img_path in enumerate(img_list):


    #     pred = model(img_path)[0]  # 推理一张图片，取第一个结果
    #     img = cv2.imread(img_path)
    #     # 预测mask
    #     if hasattr(pred, "masks") and pred.masks is not None:
    #         pred_masks = pred.masks.data.cpu().numpy().astype(bool)
    #     else:
    #         pred_masks = np.zeros((0, img.shape[0], img.shape[1]), dtype=bool)
    #     img_name = os.path.splitext(os.path.basename(img_path))[0]
    #     label_path = os.path.join(
    #         r"E:\Human_Projects\yolov11\ultralytics-main\ultralytics\datasets\labels\test",
    #         img_name + ".txt"
    #     )
    #     gt_masks = load_yolo_polygon_mask(label_path, img.shape)
    #     # GT mask（假设pred有gt_masks属性，否则需自行加载）
    #     # if hasattr(pred, "gt_masks") and pred.gt_masks is not None:
    #     #     gt_masks = pred.gt_masks.data.cpu().numpy().astype(bool)
    #     # else:
    #     #     gt_masks = np.zeros((0, img.shape[0], img.shape[1]), dtype=bool)
    #     save_path = f"runs/val/exp/tpfpfn_{i}.png"
    #     h, w = img.shape[:2]
    #     pred_masks_resized = [resize_mask(m, (h, w)) for m in pred_masks]
    #     gt_masks_resized = [resize_mask(m, (h, w)) for m in gt_masks]

    #     visualize_tp_fp_fn(pred_masks_resized, gt_masks_resized, img, save_path)
        

    if model.task == 'segment': # 仅目标检测任务适用
        length = result.box.p.size
        model_names = list(result.names.values())
        preprocess_time_per_image = result.speed['preprocess']
        inference_time_per_image = result.speed['inference']
        postprocess_time_per_image = result.speed['postprocess']
        all_time_per_image = preprocess_time_per_image + inference_time_per_image + postprocess_time_per_image
        
        n_l, n_p, n_g, flops = model_info(model.model)
        
        print('-'*20 + '论文上的数据以以下结果为准' + '-'*20)
        print('-'*20 + '论文上的数据以以下结果为准' + '-'*20)
        print('-'*20 + '论文上的数据以以下结果为准' + '-'*20)
        print('-'*20 + '论文上的数据以以下结果为准' + '-'*20)
        print('-'*20 + '论文上的数据以以下结果为准' + '-'*20)

        model_info_table = PrettyTable()
        model_info_table.title = "Model Info"
        model_info_table.field_names = ["GFLOPs", "Parameters", "前处理时间/一张图", "推理时间/一张图", "后处理时间/一张图", "FPS(前处理+模型推理+后处理)", "FPS(推理)", "Model File Size"]
        model_info_table.add_row([f'{flops:.1f}', f'{n_p:,}', 
                                  f'{preprocess_time_per_image / 1000:.6f}s', f'{inference_time_per_image / 1000:.6f}s', 
                                  f'{postprocess_time_per_image / 1000:.6f}s', f'{1000 / all_time_per_image:.2f}', 
                                  f'{1000 / inference_time_per_image:.2f}', f'{get_weight_size(model_path)}MB'])
        print(model_info_table)

        model_metrice_table = PrettyTable()
        model_metrice_table.title = "Model Metrice"
        model_metrice_table.field_names = ["Class Name", "Precision", "Recall", "F1-Score", "mAP50", "mAP75", "mAP50-95"]
        for idx in range(length):
            model_metrice_table.add_row([
                                        model_names[idx], 
                                        f"{result.box.p[idx]:.4f}", 
                                        f"{result.box.r[idx]:.4f}", 
                                        f"{result.box.f1[idx]:.4f}", 
                                        f"{result.box.ap50[idx]:.4f}", 
                                        f"{result.box.all_ap[idx, 5]:.4f}", # 50 55 60 65 70 75 80 85 90 95 
                                        f"{result.box.ap[idx]:.4f}"
                                    ])
        model_metrice_table.add_row([
                                    "all(平均数据)", 
                                    f"{result.results_dict['metrics/precision(B)']:.4f}", 
                                    f"{result.results_dict['metrics/recall(B)']:.4f}", 
                                    f"{np.mean(result.box.f1[:length]):.4f}", 
                                    f"{result.results_dict['metrics/mAP50(B)']:.4f}", 
                                    f"{np.mean(result.box.all_ap[:length, 5]):.4f}", # 50 55 60 65 70 75 80 85 90 95 
                                    f"{result.results_dict['metrics/mAP50-95(B)']:.4f}"
                                ])
        print(model_metrice_table)

        with open(result.save_dir / 'paper_data.txt', 'w+') as f:
            f.write(str(model_info_table))
            f.write('\n')
            f.write(str(model_metrice_table))
        
        print('-'*20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-'*20)
        print('-'*20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-'*20)
        print('-'*20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-'*20)
        print('-'*20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-'*20)
        print('-'*20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-'*20)