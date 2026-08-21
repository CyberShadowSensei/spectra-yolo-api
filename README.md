# spectra-yolo-api

**YOLOv8 object detection FastAPI backend for the [Spectra](https://github.com/CyberShadowSensei/spectra) accessibility smart-glasses project. Deployed on Azure.**

Part of Spectra — an accessibility platform that provides real-time speech captions and vision-based hazard alerts to assist people with hearing and visual impairments.

---

## Overview

This service exposes a REST API that accepts image frames from the Spectra web client and returns a list of detected objects with bounding boxes and confidence scores. It powers the "vision alert" feature — where Spectra warns users about approaching hazards (vehicles, stairs, obstacles) identified by YOLOv8.

```
Client (browser webcam) → POST /detect → FastAPI → YOLOv8 → JSON response
```

---

## Architecture

```
spectra-yolo-api/
├── main.py          # FastAPI application + /detect endpoint
├── model.py         # YOLOv8 model loading and inference wrapper
├── requirements.txt # Python dependencies
└── Dockerfile       # Container image for Azure deployment
```

---

## API

### `POST /detect`

**Request:** `multipart/form-data` with an image file field (`file`).

**Response:**
```json
{
  "detections": [
    {
      "class": "person",
      "confidence": 0.92,
      "bbox": { "x": 120, "y": 45, "width": 80, "height": 200 }
    }
  ],
  "count": 1,
  "inference_ms": 38
}
```

### `GET /health`

Returns `{"status": "ok"}` for Azure health probes.

---

## Setup

### Local

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker

```bash
docker build -t spectra-yolo-api .
docker run -p 8000:8000 spectra-yolo-api
```

### Azure (Container Apps)

This service is deployed as an Azure Container App. The Spectra frontend
hits the Azure-hosted URL for real-time inference.

---

## Requirements

- Python ≥ 3.10
- `ultralytics` (YOLOv8)
- `fastapi` + `uvicorn`
- ONNX runtime (optional, for faster CPU inference)

See `requirements.txt` for the full dependency list.

---

## Part of Spectra

| Component | Repo | Description |
|---|---|---|
| Frontend + Azure STT | [spectra](https://github.com/CyberShadowSensei/spectra) | React app, speech captions, smart-glasses UI |
| YOLO Backend | **this repo** | Object detection API on Azure |

---

## License

MIT — see [LICENSE](./LICENSE)
