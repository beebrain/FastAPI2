# 🚀 Dual YOLO Object Detection API (FastAPI)

ระบบตรวจจับวัตถุประสิทธิภาพสูงที่ทำงานผ่าน **FastAPI** และ **YOLO v8** โดยมาพร้อมกับอินเตอร์เฟซแบบ **Single-file HTML** ที่มีความสวยงาม ทันสมัย และติดตั้งง่าย

---

## 🌟 จุดเด่นของโปรเจกต์ (Key Features)

-   **Dual Model Support**: รองรับการทำงาน 2 รูปแบบแยกพอร์ตกัน (ชุดมาตรฐาน และ ชุดตรวจจับกากบาท)
-   **Modern Glassmorphism UI**: หน้าเว็บออกแบบด้วยสไตล์กระจกใส (Glass) ทันสมัย ตอบสนองไว (Responsive)
-   **Zero Static Dependencies**: โค้ด CSS และ JS ถูกรวมอยู่ในไฟล์ HTML เดียว ไม่ต้องมีโฟลเดอร์ static
-   **Real-time Interaction**: ปรับค่า Confidence Threshold ผ่าน Slider และดูผลลัพธ์ได้ทันที
-   **GPU Acceleration**: รองรับการคำนวณผ่าน NVIDIA GPU (CUDA) พร้อมระบบเช็คสถานะอัตโนมัติ
-   **Self-contained Architecture**: โครงสร้างไฟล์สะอาด เรียบง่ายต่อการนำไปใช้งานหรือ Deploy

---

## 🏗️ โครงสร้างระบบ (Architecture)

โปรเจกต์นี้แบ่งการทำงานออกเป็น 2 เซิร์ฟเวอร์หลัก:

1.  **Standard Detection (Port 8000)**:
    -   **Model**: ยูทิลิตี้ YOLOv8 Nano (`yolov8n.pt`)
    -   **Target**: ตรวจจับวัตถุทั่วไป 80 ชนิด (คน, รถ, สัตว์, สิ่งของ)
    -   **UI File**: `standard.html`

2.  **Cross Detection (Port 8001)**:
    -   **Model**: Custom Trained Model (`best_cross.pt`)
    -   **Target**: เน้นการตรวจจับสัญลักษณ์กากบาทโดยเฉพาะ
    -   **UI File**: `cross.html`

---

## 🛠️ การติดตั้งและใช้งาน (Installation & Usage)

### 1. เตรียมสภาพแวดล้อม
```bash
# ติดตั้งไลบรารีที่จำเป็น
pip install -r requirements.txt
```

### 2. การรันเซิร์ฟเวอร์
คุณสามารถเลือกโหมดที่ต้องการรันได้:

-   **โหมดมาตรฐาน**:
    ```bash
    python main.py
    ```
    เข้าใช้งานได้ที่: [http://localhost:8000](http://localhost:8000)

-   **โหมดตรวจจับกากบาท**:
    ```bash
    python main_cross.py
    ```
    เข้าใช้งานได้ที่: [http://localhost:8001](http://localhost:8001)

---

## 📑 คู่มือการใช้งานเชิงลึก
สำหรับรายละเอียดการปรับแต่งพิกเซล, ขั้นตอนการอัพโหลด, หรือการแก้ไขปัญหาเบื้องต้น สามารถดูได้ที่ [RUN_GUIDE.md](RUN_GUIDE.md)

---

## 🤝 จัดทำโดย
**URU Master Program 2568**
Uttaradit Rajabhat University
