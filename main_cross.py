"""
╔══════════════════════════════════════════════════════════════════╗
║  FastAPI + YOLO Object Detection API                           ║
║  ───────────────────────────────────────────────────────────    ║
║  Backend สำหรับรับรูปภาพ → ประมวลผลด้วย YOLO → ส่งผลลัพธ์กลับ   ║
║                                                                ║
║  Endpoints:                                                    ║
║    GET  /           → หน้าเว็บหลัก                               ║
║    POST /api/detect → รับภาพ + Detect วัตถุ                      ║
║    GET  /api/health → ตรวจสอบสถานะ API                          ║
║                                                                ║
║  URU Master Program 2568                                       ║
╚══════════════════════════════════════════════════════════════════╝
"""

import base64
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
from ultralytics import YOLO

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. สร้าง FastAPI Application
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
app = FastAPI(
    title="YOLO Cross Detection API",
    description="FastAPI + Ultralytics YOLO สำหรับ Detect กากบาทในภาพถ่าย",
    version="1.0.0",
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Setup Static Files & Templates
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Mount โฟลเดอร์ static เพื่อ serve CSS/JS
BASE_DIR = Path(__file__).resolve().parent
# app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Setup Jinja2 Templates สำหรับ render HTML
# templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. โหลด YOLO Model (ทำครั้งเดียวตอน startup)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# best_cross.pt = Model ตรวจจับกากบาท
# YOLO จะ auto-detect GPU (CUDA) ถ้ามี หรือใช้ CPU ถ้าไม่มี GPU
MODEL_PATH = "best_cross.pt"
print(f"🔄 กำลังโหลด YOLO Model: {MODEL_PATH} ...")
model = YOLO(MODEL_PATH)
print(f"✅ โหลด Model สำเร็จ! Device: {model.device}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. API Endpoints
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Endpoint: GET /
    แสดงหน้าเว็บหลักสำหรับอัพโหลดรูปภาพและดูผลลัพธ์
    ใช้ Jinja2 Templates ในการ render HTML
    """
    return FileResponse(BASE_DIR / "cross.html")


@app.post("/api/detect")
async def detect_objects(
    file: UploadFile = File(...),
    confidence: float = Form(0.25),  # ค่า Confidence Threshold จาก Frontend (default 0.25)
):
    """
    Endpoint: POST /api/detect
    ═══════════════════════════════

    รับไฟล์รูปภาพผ่าน multipart/form-data
    ประมวลผลด้วย YOLO Model
    ส่งผลลัพธ์กลับเป็น JSON ประกอบด้วย:
      - annotated_image : ภาพที่วาด Bounding Box แล้ว (Base64)
      - detections      : รายละเอียดวัตถุที่ตรวจพบ
      - total_objects    : จำนวนวัตถุทั้งหมด
      - summary          : สรุปจำนวนแต่ละประเภท
      - processing_time  : เวลาที่ใช้ประมวลผล (วินาที)
      - model_info       : ข้อมูล Model ที่ใช้

    Parameters:
      - file: ไฟล์รูปภาพ (JPG, PNG, WEBP)
      - confidence: ค่า Confidence Threshold (0.0 - 1.0, default 0.25)
    """
    # ตรวจสอบค่า confidence ให้อยู่ในช่วง 0.01 - 1.0
    confidence = max(0.01, min(1.0, confidence))
    # บันทึกเวลาเริ่มต้น
    start_time = time.time()

    try:
        # ──────────────────────────────────────────
        # Step 1: อ่านไฟล์ภาพจาก Upload
        # ──────────────────────────────────────────
        contents = await file.read()

        # ตรวจสอบว่าไฟล์ไม่ว่างเปล่า
        if len(contents) == 0:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "ไฟล์ว่างเปล่า กรุณาอัพโหลดไฟล์รูปภาพ"},
            )

        # แปลง bytes → numpy array → OpenCV image (BGR)
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # ตรวจสอบว่าภาพถูก decode สำเร็จ
        if image is None:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "ไม่สามารถอ่านไฟล์รูปภาพได้ กรุณาตรวจสอบรูปแบบไฟล์ (รองรับ JPG, PNG, WEBP)",
                },
            )

        # ──────────────────────────────────────────
        # Step 2: ประมวลผลด้วย YOLO
        # ──────────────────────────────────────────
        # model.predict() จะ return list ของ Results objects
        # verbose=False เพื่อไม่แสดง log ใน console
        # conf=confidence เพื่อกรองผลลัพธ์ตาม Confidence Threshold ที่ผู้ใช้ตั้งไว้
        results = model.predict(image, verbose=False, conf=confidence)

        # ──────────────────────────────────────────
        # Step 3: สร้างภาพที่มี Bounding Box (Annotated Image)
        # ──────────────────────────────────────────
        # results[0].plot() จะวาด bounding box, label, confidence
        # บนภาพต้นฉบับ และ return เป็น numpy array (BGR)
        annotated_frame = results[0].plot()

        # แปลง BGR numpy array → JPEG bytes → Base64 string
        # เพื่อส่งกลับผ่าน JSON (ไม่ต้อง save ไฟล์)
        _, buffer = cv2.imencode(
            ".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 90]
        )
        img_base64 = base64.b64encode(buffer).decode("utf-8")

        # ──────────────────────────────────────────
        # Step 4: ดึงข้อมูล Detection แต่ละรายการ
        # ──────────────────────────────────────────
        detections = []
        for r in results:
            for box in r.boxes:
                # xyxy = [x1, y1, x2, y2] พิกัดมุมซ้ายบน-ขวาล่าง
                b = box.xyxy[0].tolist()
                cls_id = int(box.cls[0].item())  # Class ID (ตัวเลข)
                conf = float(box.conf[0].item())  # Confidence Score (0-1)
                label = model.names[cls_id]  # Class Name (ชื่อวัตถุ)

                detections.append(
                    {
                        "class_id": cls_id,
                        "label": label,
                        "confidence": round(conf, 4),
                        "bbox": {
                            "x1": round(b[0], 2),
                            "y1": round(b[1], 2),
                            "x2": round(b[2], 2),
                            "y2": round(b[3], 2),
                        },
                    }
                )

        # ──────────────────────────────────────────
        # Step 5: สรุปผล (นับจำนวนแต่ละ class)
        # ──────────────────────────────────────────
        summary = {}
        for d in detections:
            label = d["label"]
            summary[label] = summary.get(label, 0) + 1

        # คำนวณเวลาที่ใช้
        processing_time = round(time.time() - start_time, 3)

        # ──────────────────────────────────────────
        # Step 6: ส่ง JSON Response กลับ
        # ──────────────────────────────────────────
        return JSONResponse(
            content={
                "success": True,
                "annotated_image": f"data:image/jpeg;base64,{img_base64}",
                "detections": detections,
                "total_objects": len(detections),
                "summary": summary,
                "processing_time": processing_time,
                "image_size": {
                    "width": image.shape[1],
                    "height": image.shape[0],
                },
                "model_info": {
                    "name": MODEL_PATH,
                    "device": str(model.device),
                },
                "confidence_threshold": confidence,
            }
        )

    except Exception as e:
        # จัดการ error ที่ไม่คาดคิด
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"เกิดข้อผิดพลาดภายในเซิร์ฟเวอร์: {str(e)}",
            },
        )


@app.get("/api/health")
async def health_check():
    """
    Endpoint: GET /api/health
    ตรวจสอบสถานะของ API และ Model
    แสดงข้อมูล GPU/CPU ที่ใช้งาน
    """
    import torch

    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_name": MODEL_PATH,
        "device": str(model.device),
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
        ),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Entry Point — รันเซิร์ฟเวอร์
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    import uvicorn

    # host="0.0.0.0" เพื่อให้เข้าถึงได้จากเครื่องอื่นในเครือข่าย
    # port=8000 เป็น default port ของ FastAPI
    # รันเซิร์ฟเวอร์บน Port 8001
    uvicorn.run("main_cross:app", host="0.0.0.0", port=8001, reload=True)
