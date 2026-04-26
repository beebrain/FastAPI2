"""
╔══════════════════════════════════════════════════════════════════╗
║  Answer Sheet Alignment API                                    ║
║  ───────────────────────────────────────────────────────────    ║
║  รับภาพ + 4 จุดมุมกระดาษ → Perspective Transform → Crop        ║
║                                                                ║
║  Endpoints:                                                    ║
║    GET  /           → หน้าเว็บจัดหน้ากระดาษ                      ║
║    POST /api/align  → รับภาพ + corners → ส่งภาพที่ตรงแล้วกลับ    ║
║    GET  /api/health → ตรวจสอบสถานะ API                          ║
║                                                                ║
║  URU Master Program 2568                                       ║
╚══════════════════════════════════════════════════════════════════╝
"""

import base64
import json
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="Answer Sheet Alignment API",
    description="จัดหน้ากระดาษคำตอบให้ตรงด้วย Perspective Transform",
    version="1.0.0",
)


@app.get("/", response_class=HTMLResponse)
async def home():
    return FileResponse(BASE_DIR / "align.html")


@app.post("/api/align")
async def align_sheet(
    file: UploadFile = File(...),
    corners: str = Form(...),
    output_width: int = Form(2100),
    output_height: int = Form(2970),
):
    """
    Endpoint: POST /api/align

    รับไฟล์รูปภาพ + ตำแหน่ง 4 มุมกระดาษ (pixel coords ของภาพต้นฉบับ)
    ทำ Perspective Transform เพื่อจัดหน้ากระดาษให้ตรง
    ส่งคืนภาพที่แก้ไขแล้วเป็น Base64

    Parameters:
      - file: ไฟล์รูปภาพ (JPG, PNG, WEBP)
      - corners: JSON {"tl":[x,y],"tr":[x,y],"br":[x,y],"bl":[x,y]} (pixel coords)
      - output_width: ความกว้างผลลัพธ์ (default 2100)
      - output_height: ความสูงผลลัพธ์ (default 2970)
    """
    try:
        contents = await file.read()
        if not contents:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "ไฟล์ว่างเปล่า กรุณาอัพโหลดไฟล์รูปภาพ"},
            )

        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "ไม่สามารถอ่านไฟล์ภาพได้ (รองรับ JPG, PNG, WEBP)"},
            )

        h_orig, w_orig = image.shape[:2]

        corners_data = json.loads(corners)
        tl = corners_data["tl"]
        tr = corners_data["tr"]
        br = corners_data["br"]
        bl = corners_data["bl"]

        # clamp coordinates to image bounds
        def clamp(pt):
            return [
                max(0, min(w_orig - 1, pt[0])),
                max(0, min(h_orig - 1, pt[1])),
            ]

        src = np.float32([clamp(tl), clamp(tr), clamp(br), clamp(bl)])
        dst = np.float32([
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1],
        ])

        M = cv2.getPerspectiveTransform(src, dst)
        aligned = cv2.warpPerspective(image, M, (output_width, output_height))

        _, buf = cv2.imencode(".jpg", aligned, [cv2.IMWRITE_JPEG_QUALITY, 95])
        b64 = base64.b64encode(buf).decode()

        return JSONResponse(content={
            "success": True,
            "aligned_image": f"data:image/jpeg;base64,{b64}",
            "output_size": {"width": output_width, "height": output_height},
            "original_size": {"width": w_orig, "height": h_orig},
        })

    except json.JSONDecodeError:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "รูปแบบข้อมูล corners ไม่ถูกต้อง (ต้องเป็น JSON)"},
        )
    except KeyError as e:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": f"ข้อมูล corners ไม่ครบ: {e}"},
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"เกิดข้อผิดพลาด: {str(e)}"},
        )


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "service": "Answer Sheet Alignment API"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_align:app", host="0.0.0.0", port=8002, reload=True)
