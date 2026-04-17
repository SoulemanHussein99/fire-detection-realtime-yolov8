from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(data=r"d:\python\fire_detection\data.yaml", epochs=50, batch=2, imgsz=640)
