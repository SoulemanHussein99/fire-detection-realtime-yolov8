import cv2
from ultralytics import YOLO

model = YOLO(r"D:\python\projects\fire_detection\runs\detect\train\weights\best.pt")

camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = model.predict(rgb_frame, conf=0.2, verbose=False)[0]
    """"" 
    another way instead of using for loops (for box ---> cv2.putText)
    # we can use im= result.plot()
    """
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cl = int(box.cls[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 215, 0), 2)
        cv2.putText(frame, "Warning", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (125, 215, 0), 2)
    cv2.imshow("Fire Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
camera.release()
cv2.destroyAllWindows()
