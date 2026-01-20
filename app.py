from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
from ultralytics import YOLO
from collections import Counter

app = FastAPI(title="YOLO Indoor Detection API")

model = YOLO("yolov8n.pt")
class_names = model.names

@app.post("/detect")
async def detect(file: UploadFile = File(...), conf: float = 0.45):
    img_bytes = await file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(frame, conf=conf, iou=0.45, verbose=False)

    detections = []
    for box in results[0].boxes:
        cls = int(box.cls[0])
        detections.append({
            "class": class_names[cls],
            "confidence": float(box.conf[0]),
            "bbox": [int(x) for x in box.xyxy[0].tolist()]
        })

    counts = Counter(d['class'] for d in detections)
    return {"detections": detections, "counts": counts}
