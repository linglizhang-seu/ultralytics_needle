# import onnx
# from ultralytics import YOLO
# import onnxruntime as ort
# import numpy as np
# 1. 加载模型
# weight_path = r"E:\Human_Projects\yolov11\ultralytics-main\ultralytics\runs\segment\train8\weights\best.pt"
from ultralytics import YOLO

model = YOLO(r"E:\Human_Projects\yolov11\ultralytics-main\ultralytics\runs\segment\train8\weights\best.p")
model.export(format="onnx", imgsz=224, opset=13, dynamic=False, simplify=False, nms=False, device=0)

# model = YOLO(weight_path)

#     # ✅ 导出为 ONNX 格式
# onnx_path = model.export(
#     format="onnx",       # 导出格式
#     imgsz=224,           # 输入图像尺寸 (根据你训练时的尺寸)          # ONNX opset版本，可调为 11~17
#     simplify=True,       # 使用 onnx-simplifier 优化模型
#     dynamic=True,        # 支持动态batch尺寸
#     device="cpu"         # 导出设备
# )

# print(f"✅ 成功导出为: {onnx_path}")


# onnx_path = r"E:\Human_Projects\yolov11\ultralytics-main\ultralytics\runs\segment\train8\weights\best.onnx"

# onnx_model = onnx.load(onnx_path)

# for inp in onnx_model.graph.input:
#     dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
#     print(f"🧩 模型输入名: {inp.name}, 输入形状: {dims}")

# # 根据实际情况修改为 1 或 3
# x = np.random.rand(1, 1, 224, 224).astype(np.float32)

# session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
# input_name = session.get_inputs()[0].name
# outputs = session.run(None, {input_name: x})

# print("✅ ONNX 推理成功！输出 shape：", [o.shape for o in outputs])

# import onnxruntime as ort
# import numpy as np
# import cv2
# import torch

# # ============ 配置区域 ============
# onnx_path = r"E:\Human_Projects\yolov11\ultralytics-main\ultralytics\runs\segment\train8\weights\best.onnx"
# image_path = r"E:\Human_Projects\yolov11\ultralytics-main\ultralytics\cell_20240126_163217_04.tif"
# imgsz = 224
# conf_thres = 0.25
# mask_thres = 0.5
# # =================================

# # 1️⃣ 加载模型（GPU 优化）
# providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
# session = ort.InferenceSession(onnx_path, providers=providers)
# input_name = session.get_inputs()[0].name
# print(f"模型输入: {input_name}, 形状: {session.get_inputs()[0].shape}")

# # 2️⃣ 读取灰度图并预处理
# img0 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# img = cv2.resize(img0, (imgsz, imgsz))
# x = img[np.newaxis, np.newaxis, :, :].astype(np.float32) / 255.0  # [1,1,H,W]

# # 3️⃣ 推理
# outputs = session.run(None, {input_name: x})
# pred, proto = outputs

# # 4️⃣ 转为 GPU Tensor
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# pred = torch.tensor(pred, device=device)
# proto = torch.tensor(proto, device=device)

# # 转换 [1, n_ch, n_pred] -> [n_pred, n_ch]
# pred = pred.squeeze(0).permute(1, 0)
# boxes = pred[:, :4]
# scores = pred[:, 4]
# mask_coeffs = pred[:, 6:]

# # 筛选高置信度
# mask_idx = scores > conf_thres
# boxes, scores, mask_coeffs = boxes[mask_idx], scores[mask_idx], mask_coeffs[mask_idx]

# # 5️⃣ 重建掩膜（GPU）
# if boxes.shape[0] > 0:
#     proto = proto.squeeze(0)
#     masks = torch.matmul(mask_coeffs, proto[:mask_coeffs.shape[1]].view(mask_coeffs.shape[1], -1)).sigmoid()
#     masks = masks.view(-1, proto.shape[1], proto.shape[2])
#     masks = torch.nn.functional.interpolate(
#         masks.unsqueeze(1),
#         size=(imgsz, imgsz),
#         mode='bilinear',
#         align_corners=False
#     ).squeeze(1)
# else:
#     masks = torch.empty(0, imgsz, imgsz, device=device)

# # 6️⃣ 可视化
# img_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# for i, box in enumerate(boxes):
#     x1, y1, x2, y2 = box.int().cpu().numpy()
#     color = (0, 255, 0)

#     # 绘制框
#     cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
#     cv2.putText(img_vis, f"{scores[i].item():.2f}", (x1, y1 - 5),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

#     # 绘制掩膜
#     if masks.shape[0] > 0:
#         mask = (masks[i] > mask_thres).cpu().numpy().astype(np.uint8)
#         mask_color = np.zeros_like(img_vis)
#         mask_color[:, :, 1] = mask * 255
#         img_vis = cv2.addWeighted(img_vis, 1.0, mask_color, 0.5, 0)

# # 7️⃣ 显示结果
# cv2.imshow("ONNX Segmentation Result", img_vis)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


