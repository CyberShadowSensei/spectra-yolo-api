from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware  # <-- ADDED
import numpy as np
import cv2
from ultralytics import YOLO
from collections import Counter

app = FastAPI(title="YOLO Indoor Detection API")

# --- ADDED: This allows your website to talk to your API ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all websites to access your API
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("yolov8n.pt")
class_names = model.names

# --- ADDED: A simple check to see if the API is working ---
@app.get("/")
async def root():
    return {"status": "Spectra YOLO API is running!"}

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
            "confidence": round(float(box.conf[0]), 2), # Rounded for cleaner display
            "bbox": [int(x) for x in box.xyxy[0].tolist()]
        })

    counts = Counter(d['class'] for d in detections)
    return {"detections": detections, "counts": counts}
