from ultralytics import YOLO
# model = YOLO("yolo11n-seg.pt") 
model=YOLO(r'E:\Human_Projects\yolov11\ultralytics-main\ultralytics\runs\segment\train8\weights\best.pt')
source=r"E:\Human_Projects\yolov11\ultralytics-main\ultralytics\test"
results=model.predict(source, 
              save=True,
              imgsz=224,
              project='runs/segment',
              save_txt=True,
              conf=0.6,
            )
