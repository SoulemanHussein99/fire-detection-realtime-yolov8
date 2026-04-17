# Real-Time Fire Detection System (YOLOv8 + OpenCV)

A real-time computer vision system that detects fire using a custom-trained YOLOv8 model.  
The system uses a webcam stream to perform live object detection and highlights fire instances with bounding boxes and confidence scores.

---

## Features

- Real-time fire detection using webcam
- Custom-trained YOLOv8 model
- Bounding box visualization
- Confidence score display
- Warning label when fire is detected
- Fast inference using OpenCV pipeline

---

## Project Type

- Type: Real-Time AI System
- Field: Computer Vision
- Task: Object Detection
- Deployment: Webcam (Live stream)

---

## Model Information
- Model: YOLOv8 (Ultralytics)
- Classes: Fire (1 class)
- Input: Live video frames
- Output: Bounding boxes + confidence scores

---

## Training
yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640 batch=2
