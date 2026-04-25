# 🔍 YOLO Object Detection — FastAPI Workshop

> **เว็บแอพลิเคชันสำหรับ Detect วัตถุในภาพถ่าย**  
> ใช้ FastAPI เป็น Backend + YOLO (Ultralytics) เป็น AI Model  
> Frontend ใช้ HTML/CSS/JavaScript + AJAX (Fetch API)

---

## 📌 ภาพรวมโปรเจค

โปรเจคนี้เป็นเว็บแอพลิเคชันที่:
1. ผู้ใช้อัพโหลดรูปภาพผ่านหน้าเว็บ (Browse File / Drag & Drop)
2. ระบบส่งรูปไปประมวลผลด้วย **YOLO Model** ผ่าน **FastAPI** Backend
3. YOLO ตรวจจับวัตถุในภาพ (Object Detection)
4. ส่งผลลัพธ์กลับมาแสดงที่หน้าเว็บ:
   - 🖼️ ภาพที่มี Bounding Box ล้อมรอบวัตถุ
   - 📊 สรุปจำนวนวัตถุที่พบ
   - 📋 Raw JSON Response ดิบ
   - 📑 ตารางรายละเอียดแต่ละวัตถุ

---

## ⚙️ ขั้นตอนการติดตั้ง

### 1. สร้าง Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 2. ติดตั้ง Dependencies
```bash
pip install -r requirements.txt
```

### 3. รันเซิร์ฟเวอร์
```bash
# วิธีที่ 1: รันผ่าน Python โดยตรง
python main.py

# วิธีที่ 2: รันผ่าน uvicorn (พร้อม auto-reload)
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. เปิดเบราว์เซอร์
ไปที่ **http://localhost:8000**

> 💡 **Swagger UI** (API Documentation): http://localhost:8000/docs

---

## 📁 โครงสร้างไฟล์

```
FastAPI2/
├── main.py              ← FastAPI Backend + YOLO Model
├── requirements.txt     ← Python Dependencies
├── README.md            ← เอกสารนี้
├── static/
│   ├── css/
│   │   └── style.css    ← Stylesheet (Dark Theme)
│   └── js/
│       └── app.js       ← JavaScript AJAX Logic
└── templates/
    └── index.html       ← หน้าเว็บหลัก (Jinja2 Template)
```

---

## 🔌 API Reference

### `GET /` — หน้าเว็บหลัก
แสดงหน้า HTML สำหรับอัพโหลดรูปภาพ

### `POST /api/detect` — Detect วัตถุในภาพ
**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` = ไฟล์รูปภาพ (JPG, PNG, WEBP)

**Response (JSON):**
```json
{
    "success": true,
    "annotated_image": "data:image/jpeg;base64,...",
    "detections": [
        {
            "class_id": 0,
            "label": "person",
            "confidence": 0.9234,
            "bbox": { "x1": 100, "y1": 200, "x2": 300, "y2": 500 }
        }
    ],
    "total_objects": 3,
    "summary": { "person": 2, "car": 1 },
    "processing_time": 0.245,
    "image_size": { "width": 1920, "height": 1080 },
    "model_info": { "name": "yolov8n.pt", "device": "cuda:0" }
}
```

### `GET /api/health` — Health Check
ตรวจสอบสถานะ API และ GPU/CPU

---

## 💡 แนวคิดหลัก (Key Concepts)

### 1. FastAPI
- Python web framework ที่เร็วที่สุด (async support)
- Auto-generate API documentation (Swagger UI)
- Type checking ด้วย Pydantic

### 2. YOLO (You Only Look Once)
- Deep Learning model สำหรับ Object Detection
- ใช้ Ultralytics library — `pip install ultralytics`
- รองรับทั้ง GPU (CUDA) และ CPU
- Auto-download model weights ครั้งแรกที่รัน

### 3. AJAX (Fetch API)
- ส่ง request ไปยัง API โดยไม่ต้อง reload หน้าเว็บ
- ใช้ `FormData` สำหรับ file upload
- **สำคัญ:** ห้าม set `Content-Type` header เองเมื่อใช้ FormData

### 4. Base64 Encoding
- แปลงภาพเป็น text string เพื่อส่งผ่าน JSON
- ใช้ `data:image/jpeg;base64,...` สำหรับแสดงใน `<img>` tag

---

## 🔧 การปรับแต่ง

### เปลี่ยน YOLO Model
```python
# ใน main.py — เปลี่ยน MODEL_PATH
MODEL_PATH = "yolov8s.pt"    # Small (แม่นกว่า nano)
MODEL_PATH = "yolov8m.pt"    # Medium
MODEL_PATH = "yolov8l.pt"    # Large
MODEL_PATH = "best.pt"       # Custom trained model
```

### ปรับ Confidence Threshold
```python
# ใน main.py — เพิ่ม conf parameter
results = model.predict(image, verbose=False, conf=0.5)  # ขั้นต่ำ 50%
```

### เพิ่ม CORS (Cross-Origin)
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 📝 License

สำหรับการศึกษาเท่านั้น — URU Master Program 2568
