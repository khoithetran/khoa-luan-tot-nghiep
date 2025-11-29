import os
import uuid
import numpy as np
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path

from typing import Optional, List, Dict
import json

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image

import cv2
from collections import deque

from ultralytics import YOLO
import uvicorn
import shutil


LIVE_STREAMS: Dict[str, dict] = {}

# ================== CẤU HÌNH ĐƯỜNG DẪN & MODEL ==================

# Đường dẫn model YOLO của bạn
MODEL_PATH = r"D:\KLTN_Code\backend\yolov8s_ap.pt"

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
HISTORY_DIR = DATA_DIR / "history"
GLOBAL_DIR = HISTORY_DIR / "global"
CROPS_DIR = HISTORY_DIR / "crops"
HISTORY_JSONL = HISTORY_DIR / "history.jsonl"
VIDEOS_DIR = DATA_DIR / "videos"
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

UPDATE_POOL_DIR = DATA_DIR / "update_pool"
UPDATE_POOL_IMAGES_DIR = UPDATE_POOL_DIR / "images"
UPDATE_POOL_LABELS_DIR = UPDATE_POOL_DIR / "labels"
UPDATE_POOL_META = UPDATE_POOL_DIR / "accepted.jsonl"

UPDATE_POOL_DIR.mkdir(parents=True, exist_ok=True)
UPDATE_POOL_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
UPDATE_POOL_LABELS_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_GLOBAL_DIR = DATA_DIR / "history" / "global"
HISTORY_GLOBAL_DIR.mkdir(parents=True, exist_ok=True)

for d in [DATA_DIR, HISTORY_DIR, GLOBAL_DIR, CROPS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Load model 1 lần
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"[ERROR] Không load được model từ {MODEL_PATH}: {e}")
    model = None



# ================== FASTAPI SETUP ==================

app = FastAPI(title="PPE Safety Backend", version="0.1.0")

# Cho phép frontend Vite truy cập (localhost:5173)
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Mount static để phục vụ ảnh history
# /static/history/global/xxx.jpg, /static/history/crops/xxx.jpg
app.mount("/static", StaticFiles(directory=DATA_DIR), name="static")

# ================== Pydantic Models ==================

class DetectionBoxOut(BaseModel):
    id: str
    class_name: str
    confidence: float
    # toạ độ normalized (0–1) theo width/height của ảnh
    x: float
    y: float
    width: float
    height: float
    # giữ thêm xyxy pixel cho tiện debug nếu cần
    x1: int
    y1: int
    x2: int
    y2: int


class DetectImageResponse(BaseModel):
    boxes: List[DetectionBoxOut]
    # URL ảnh toàn cục (nếu có lưu history)
    global_image_url: Optional[str] = None
    # URL các crop người vi phạm (nếu có)
    crop_image_urls: List[str] = []
    # loại event: VI_PHAM / NGHI_NGO / NONE
    event_type: str
    # id event trong history (nếu có)
    history_event_id: Optional[str] = None

class HistoryEvent(BaseModel):
    id: str
    timestamp: str
    source: str
    type: str  # VI_PHAM / NGHI_NGO
    global_image_url: str
    crop_image_urls: List[str]
    num_violators: int

class HistoryLatestResponse(BaseModel):
    event: Optional[HistoryEvent] = None
@app.get("/api/history/latest", response_model=HistoryLatestResponse)
def get_latest_history_event(
    source: Optional[str] = Query(None, description="Lọc theo nguồn (tên video hoặc camera)"),
    types: Optional[str] = Query(None, description="Lọc theo loại, ví dụ: VI_PHAM,NGHI_NGO"),
):
    """
    Trả về history event mới nhất (ở cuối file) thỏa điều kiện:
    - Nếu 'source' được cung cấp: chỉ lấy event có source đúng chuỗi đó.
    - Nếu 'types' được cung cấp: là danh sách type phân tách bởi dấu phẩy (VD: VI_PHAM,NGHI_NGO).
    """
    if not HISTORY_JSONL.exists():
        return HistoryLatestResponse(event=None)

    type_filter: Optional[List[str]] = None
    if types:
        type_filter = [t.strip().upper() for t in types.split(",") if t.strip()]

    latest: Optional[HistoryEvent] = None

    try:
        with open(HISTORY_JSONL, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                evt = HistoryEvent(**data)
            except Exception as e:
                print(f"[WARN] Không parse được dòng history: {e}")
                continue

            if source is not None and evt.source != source:
                continue

            if type_filter is not None and evt.type not in type_filter:
                continue

            latest = evt
            break

    except Exception as e:
        print(f"[WARN] get_latest_history_event: {e}")
        return HistoryLatestResponse(event=None)

    return HistoryLatestResponse(event=latest)

class VideoDetectResponse(BaseModel):
    total_frames: int          # số frame thực sự đưa vào YOLO (đã downsample 10 FPS)
    fps_input: float           # FPS gốc của video
    fps_used: float            # 10.0 (theo yêu cầu)
    window_size: int           # 40 frame (4 giây)
    violation_events: int      # số lần kích hoạt VI_PHAM
    suspicion_events: int      # số lần kích hoạt NGHI_NGO
    events: List[HistoryEvent] # danh sách event đã lưu vào history

def normalize_class_name(name: str) -> str:
    """
    Chuẩn hóa tên lớp để so sánh:
    - lower
    - bỏ khoảng trắng
    - thay '_' bằng '-'
    """
    return name.strip().lower().replace(" ", "").replace("_", "-")


def is_head_class(name: str) -> bool:
    n = normalize_class_name(name)
    # nếu lớp 'head' có đặt đúng tên thì n == 'head'
    return n == "head"


def is_nonhelmet_class(name: str) -> bool:
    n = normalize_class_name(name)
    # hỗ trợ nhiều biến thể: non-helmet, nonhelmet, no-helmet, nohelmet
    return n in {"non-helmet", "nonhelmet", "no-helmet", "nohelmet"}


# ================== HÀM PHỤ TRỢ ==================

def generate_event_id() -> str:
    return uuid.uuid4().hex


def save_pil_image(img: Image.Image, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path), format="JPEG", quality=90)


def append_history_record(record: HistoryEvent):
    import json
    try:
        data = record.model_dump()  # Pydantic v2: trả về dict
        with open(HISTORY_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    except PermissionError as e:
        print(f"[WARN] Không ghi được history.jsonl (PermissionError): {e}")
    except Exception as e:
        print(f"[WARN] Lỗi khác khi ghi history.jsonl: {e}")

# ================== TRACKING ==================
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

def iou_xyxy(box1, box2) -> float:
    """Tính IoU giữa 2 box dạng (x1, y1, x2, y2)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


@dataclass
class Track:
    id: int
    x1: float
    y1: float
    x2: float
    y2: float
    last_frame: int
    cls_counts: Dict[str, int] = field(default_factory=dict)
    conf_sum: Dict[str, float] = field(default_factory=dict)

    def update_box(self, x1, y1, x2, y2):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

    def update_class(self, cls_name: str, conf: float):
        self.cls_counts[cls_name] = self.cls_counts.get(cls_name, 0) + 1
        self.conf_sum[cls_name] = self.conf_sum.get(cls_name, 0.0) + conf

    @property
    def main_class(self) -> str:
        """Class đã được smooth: chọn class có count lớn nhất, tie-break bằng tổng confidence."""
        if not self.cls_counts:
            return "unknown"
        # ưu tiên: count cao nhất, nếu bằng nhau -> conf_sum cao nhất
        sorted_items = sorted(
            self.cls_counts.items(),
            key=lambda kv: (kv[1], self.conf_sum.get(kv[0], 0.0)),
            reverse=True,
        )
        return sorted_items[0][0]

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return int(self.x1), int(self.y1), int(self.x2), int(self.y2)


class SimpleIOUTracker:
    """
    Tracker cực nhẹ, chỉ dùng IoU để gán ID:
    - Không có Kalman filter, không dùng appearance.
    - Đủ để giữ ID ổn định cho người đi chậm / đứng yên (công trường).
    """

    def __init__(self, iou_thresh: float = 0.3, max_age: int = 30):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.tracks: List[Track] = []
        self.next_id = 1

    def update(self, detections: List[Tuple[float, float, float, float, str, float]], frame_idx: int) -> List[Track]:
        """
        detections: list (x1, y1, x2, y2, class_name, conf)
        frame_idx: processed frame index (10 FPS)

        Trả về list track hiện tại (đã update).
        """
        # đánh dấu tất cả track là chưa matched
        unmatched_tracks = set(range(len(self.tracks)))
        matched_dets = set()

        # lưu match (track_idx -> det_idx)
        matches: List[Tuple[int, int]] = []

        # Nếu không có detection thì chỉ age track
        if detections:
            # greedy matching theo IoU (đơn giản, đủ dùng)
            for det_idx, (dx1, dy1, dx2, dy2, dcls, dconf) in enumerate(detections):
                best_iou = 0.0
                best_track_idx = -1
                for t_idx in unmatched_tracks:
                    track = self.tracks[t_idx]
                    iou_val = iou_xyxy((dx1, dy1, dx2, dy2), (track.x1, track.y1, track.x2, track.y2))
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_track_idx = t_idx

                if best_track_idx >= 0 and best_iou >= self.iou_thresh:
                    # gán detection này cho track
                    matches.append((best_track_idx, det_idx))
                    unmatched_tracks.discard(best_track_idx)
                    matched_dets.add(det_idx)

        # cập nhật track theo matches
        for t_idx, d_idx in matches:
            dx1, dy1, dx2, dy2, dcls, dconf = detections[d_idx]
            trk = self.tracks[t_idx]
            trk.update_box(dx1, dy1, dx2, dy2)
            trk.update_class(dcls, dconf)
            trk.last_frame = frame_idx

        # tạo track mới cho detection chưa match
        for det_idx, (dx1, dy1, dx2, dy2, dcls, dconf) in enumerate(detections):
            if det_idx in matched_dets:
                continue
            new_trk = Track(
                id=self.next_id,
                x1=dx1,
                y1=dy1,
                x2=dx2,
                y2=dy2,
                last_frame=frame_idx,
            )
            new_trk.update_class(dcls, dconf)
            self.tracks.append(new_trk)
            self.next_id += 1

        # xoá track quá cũ
        alive_tracks: List[Track] = []
        for trk in self.tracks:
            if frame_idx - trk.last_frame <= self.max_age:
                alive_tracks.append(trk)
        self.tracks = alive_tracks

        return self.tracks


def load_all_history() -> List[HistoryEvent]:
    if not HISTORY_JSONL.is_file():
        return []
    import json
    events: List[HistoryEvent] = []
    with open(HISTORY_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                events.append(HistoryEvent(**data))
            except Exception as e:
                print(f"[WARN] Lỗi đọc 1 dòng history: {e}")
    # sort mới nhất lên đầu
    events.sort(key=lambda e: e.timestamp, reverse=True)
    return events

tracker = SimpleIOUTracker(iou_thresh=0.4, max_age=10)

# ================== UPDATE ==================


def read_all_history() -> List[HistoryEvent]:
    events: List[HistoryEvent] = []
    if not HISTORY_JSONL.exists():
        return events
    with open(HISTORY_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                events.append(HistoryEvent(**obj))
            except Exception as e:
                print("[WARN] Lỗi parse history:", e)
    return events

# mapping class name → id theo thứ tự training
CLASS_NAME_TO_ID = {
    "helmet": 0,
    "head": 1,
    "non-helmet": 2,
}

def run_yolo_update_inference(img_path: Path):
    """
    Chạy YOLO trên ảnh global để:
      - Trả về danh sách boxes chuẩn hoá:
          class_name: 'helmet' | 'head' | 'non-helmet'
          class_id:   0 | 1 | 2
          confidence
          xc, yc, width, height (normalized, center)
          x, y, x1, y1, x2, y2 (normalized, topleft / bottomright)
      - Trả về class_counts: {'helmet': n1, 'head': n2, 'non-helmet': n3}
    """
    # TODO: đổi YOLO_MODEL thành model bạn đang dùng
    results = model(img_path)
    r = results[0]

    xywhn = r.boxes.xywhn.cpu().numpy()   # (N,4) [xc, yc, w, h]
    cls_ids = r.boxes.cls.cpu().numpy()   # (N,)
    confs   = r.boxes.conf.cpu().numpy()  # (N,)

    boxes = []
    class_counts = {
        "helmet": 0,
        "head": 0,
        "non-helmet": 0,
    }

    for i in range(len(xywhn)):
        xc, yc, w, h = map(float, xywhn[i])
        raw_cls_id = int(cls_ids[i])
        conf = float(confs[i])

        raw_name = str(r.names.get(raw_cls_id, str(raw_cls_id)))

        # Dùng bộ normalize đã có sẵn trong server
        if is_head_class(raw_name):
            name = "head"
        elif is_nonhelmet_class(raw_name):
            name = "non-helmet"
        else:
            name = "helmet"

        # Đếm thống kê
        if name in class_counts:
            class_counts[name] += 1

        # map sang id 0/1/2 cho YOLO txt
        if name == "helmet":
            cls_id = 0
        elif name == "head":
            cls_id = 1
        else:
            cls_id = 2

        # chuyển center (xc, yc, w, h) → topleft/bottomright
        x1 = xc - w / 2
        y1 = yc - h / 2
        x2 = xc + w / 2
        y2 = yc + h / 2

        # clamp cho an toàn
        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))
        x2 = max(0.0, min(1.0, x2))
        y2 = max(0.0, min(1.0, y2))

        boxes.append({
            "id": f"update_box_{i}",
            "class_name": name,
            "class_id": cls_id,
            "confidence": conf,
            # center format
            "xc": xc,
            "yc": yc,
            "width": w,
            "height": h,
            # topleft / bottomright (normalized) – giống format detect/image
            "x": x1,
            "y": y1,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        })

    return boxes, class_counts




# ================== API ENDPOINTS ==================

@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
    }


@app.post("/api/detect/image", response_model=DetectImageResponse)
async def detect_image(file: UploadFile = File(...), source: str | None = Form(None),):
    """
    Nhận 1 ảnh, chạy YOLO, trả về bounding boxes + xác định:
    - Nếu có ít nhất 1 'head'  -> VI_PHAM
    - Nếu không có 'head' nhưng có 'non-helmet' -> NGHI_NGO
    - Ngược lại -> NONE (không lưu lịch sử)

    Khi VI_PHAM hoặc NGHI_NGO:
    - Lưu ảnh toàn cục.
    - Lưu crop CHỈ cho các box 'head' và 'non-helmet' (KHÔNG crop helmet).
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model chưa được load.")

    # Chỉ cho JPG/PNG
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=415, detail="Chỉ hỗ trợ ảnh JPG/PNG.")

    # Đọc bytes
    raw_bytes = await file.read()

    # Mở bằng PIL, fallback OpenCV
    pil_img = None
    try:
        pil_img = Image.open(BytesIO(raw_bytes))
        pil_img = pil_img.convert("RGB")
    except Exception as e:
        print(f"[WARN] Pillow không đọc được ảnh: {e}")

    if pil_img is None:
        try:
            np_arr = np.frombuffer(raw_bytes, np.uint8)
            img_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img_cv is None:
                raise ValueError("cv2.imdecode trả None")
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
        except Exception as e:
            print(f"[ERROR] OpenCV cũng không đọc được ảnh: {e}")
            raise HTTPException(status_code=400, detail="Không đọc được ảnh.")

    img_np = np.array(pil_img)
    h, w = img_np.shape[:2]

    try:
        results = model.predict(img_np, imgsz=640, verbose=False)
    except Exception as e:
        print(f"[ERROR] YOLO predict lỗi: {e}")
        raise HTTPException(status_code=500, detail="Lỗi khi chạy YOLO.")

    boxes_out: List[DetectionBoxOut] = []
    has_head = False
    has_nonhelmet = False

    if len(results) > 0:
        r = results[0]
        if r.boxes is not None:
            for i, b in enumerate(r.boxes):
                cls_id = int(b.cls[0].item())
                class_name = r.names.get(cls_id, str(cls_id))
                conf = float(b.conf[0].item())

                x1, y1, x2, y2 = b.xyxy[0].tolist()
                x1_i, y1_i, x2_i, y2_i = map(int, [x1, y1, x2, y2])

                # Clamp vào ảnh
                x1_i = max(0, min(x1_i, w - 1))
                y1_i = max(0, min(y1_i, h - 1))
                x2_i = max(0, min(x2_i, w))
                y2_i = max(0, min(y2_i, h))

                bw = x2_i - x1_i
                bh = y2_i - y1_i
                if bw <= 1 or bh <= 1:
                    continue

                # normalized theo ảnh gốc
                nx = x1_i / w
                ny = y1_i / h
                nwidth = bw / w
                nheight = bh / h

                box_out = DetectionBoxOut(
                    id=f"box_{i}",
                    class_name=class_name,
                    confidence=conf,
                    x=nx,
                    y=ny,
                    width=nwidth,
                    height=nheight,
                    x1=x1_i,
                    y1=y1_i,
                    x2=x2_i,
                    y2=y2_i,
                )
                boxes_out.append(box_out)

                # đánh dấu vi phạm/nghi ngờ
                if is_head_class(class_name):
                    has_head = True
                if is_nonhelmet_class(class_name):
                    has_nonhelmet = True

    # Xác định loại sự kiện cho mode Ảnh
    event_type: str = "NONE"
    if has_head:
        event_type = "VI_PHAM"
    elif has_nonhelmet:
        event_type = "NGHI_NGO"

    # Xác định loại sự kiện cho mode Ảnh
    event_type: str = "NONE"
    if has_head:
        event_type = "VI_PHAM"
    elif has_nonhelmet:
        event_type = "NGHI_NGO"

    # Nếu đây là frame từ VIDEO (frontend gửi blob tên 'frame.jpg')thì chỉ trả kết quả, KHÔNG lưu lịch sử để tránh spam.
    is_video_frame = file.filename and file.filename.startswith("frame")
    if is_video_frame:
        return DetectImageResponse(
            event_type=event_type,
            boxes=boxes_out,
            history_event_id=None,
        )

    history_event_id = None

    # Chỉ lưu lịch sử nếu có vi phạm/nghi ngờ
    if event_type != "NONE":
        history_event_id = generate_event_id()
        timestamp = datetime.now().isoformat()

        # Lưu ảnh toàn cục
        global_filename = f"{history_event_id}.jpg"
        global_path = GLOBAL_DIR / global_filename
        save_pil_image(pil_img, global_path)
        global_url = f"/static/history/global/{global_filename}"

        # Lưu crop CHỈ cho head & non-helmet
        crop_urls: List[str] = []
        crop_idx = 0
        for b in boxes_out:
            # CHỈ CROP head & non-helmet
            if not (is_head_class(b.class_name) or is_nonhelmet_class(b.class_name)):
                continue
            if b.x2 <= b.x1 or b.y2 <= b.y1:
                continue

            try:
                crop = pil_img.crop((b.x1, b.y1, b.x2, b.y2))
            except Exception as e:
                print(f"[WARN] Lỗi crop bbox {b.id}: {e}")
                continue

            crop_filename = f"{history_event_id}_{crop_idx}.jpg"
            crop_path = CROPS_DIR / crop_filename
            save_pil_image(crop, crop_path)
            crop_urls.append(f"/static/history/crops/{crop_filename}")
            crop_idx += 1

        num_violators = len(crop_urls) if crop_urls else 1

        effective_source = source or (file.filename or "Uploaded image")

        history_record = HistoryEvent(
            id=history_event_id,
            timestamp=timestamp,
            source=file.filename or "Uploaded image",
            type=event_type,
            global_image_url=global_url,
            crop_image_urls=crop_urls,
            num_violators=num_violators,
        )
        append_history_record(history_record)

    return DetectImageResponse(
        event_type=event_type,
        boxes=boxes_out,
        history_event_id=history_event_id,
    )


@app.post("/api/detect_video", response_model=VideoDetectResponse)
async def detect_video(
    file: UploadFile = File(...),
    source: str | None = Form(None),
):
    """
    Phân tích toàn bộ video (offline) với YOLO + luật 2/3:

    - Downsample về ~10 FPS: mỗi giây xử lý ~10 frame nội dung video.
    - Cửa sổ 30 frame (≈ 3 giây).
    - Nếu trong 30 frame gần nhất:
        + Có >= 20 frame chứa 'head'  -> tạo 1 sự kiện VI_PHAM.
        + Có >= 20 frame chứa 'non-helmet' -> tạo 1 sự kiện NGHI_NGO.
    - Mỗi lần vượt ngưỡng từ dưới lên (rising edge) chỉ tạo 1 event cho chuỗi vi phạm liên tục.
    - Mỗi event được lưu vào history với:
        + Ảnh global (frame có bbox).
        + Crop từng người vi phạm / nghi ngờ.
    """

    if model is None:
        raise HTTPException(status_code=500, detail="Model chưa được load.")

    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File phải là video (mp4, avi, ...)")

    # Lưu file video tạm
    temp_dir = DATA_DIR / "temp_videos"
    temp_dir.mkdir(parents=True, exist_ok=True)

    temp_path = temp_dir / file.filename
    raw_bytes = await file.read()
    with open(temp_path, "wb") as f:
        f.write(raw_bytes)

    cap = cv2.VideoCapture(str(temp_path))
    if not cap.isOpened():
        temp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Không mở được video.")

    fps_input = cap.get(cv2.CAP_PROP_FPS)
    if not fps_input or fps_input <= 0:
        fps_input = 25.0

    target_fps = 10.0
    frame_interval = max(int(round(fps_input / target_fps)), 1)

    # Cửa sổ 30 frame (≈ 3 giây ở 10 FPS)
    window_head = deque(maxlen=30)
    window_nonhelmet = deque(maxlen=30)

    current_frame_idx = 0          # index frame gốc (30fps)
    processed_idx = 0              # index frame đã xử lý YOLO (~10fps)

    prev_head_count = 0
    prev_nonhelmet_count = 0

    violation_events = 0
    suspicion_events = 0
    events_out: list[HistoryEvent] = []

    effective_source = source or (file.filename or "Uploaded video")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_frame_idx += 1

            # downsample: chỉ xử lý 1 frame mỗi frame_interval frame gốc
            if current_frame_idx % frame_interval != 0:
                continue

            processed_idx += 1
            h, w = frame.shape[:2]

            frame_has_head = False
            frame_has_nonhelmet = False
            boxes_for_crop: list[tuple[str, int, int, int, int]] = []

            # YOLO detect
            try:
                results = model.predict(frame, imgsz=640, verbose=False)
            except Exception as e:
                print(f"[WARN] YOLO predict lỗi ở frame {current_frame_idx}: {e}")
                continue

            if len(results) > 0:
                r = results[0]
                if r.boxes is not None:
                    for b in r.boxes:
                        cls_id = int(b.cls[0].item())
                        class_name = r.names.get(cls_id, str(cls_id))
                        conf = float(b.conf[0].item())

                        x1, y1, x2, y2 = b.xyxy[0].tolist()
                        x1_i, y1_i, x2_i, y2_i = map(int, [x1, y1, x2, y2])

                        # clamp vào ảnh
                        x1_i = max(0, min(x1_i, w - 1))
                        y1_i = max(0, min(y1_i, h - 1))
                        x2_i = max(0, min(x2_i, w))
                        y2_i = max(0, min(y2_i, h))
                        bw = x2_i - x1_i
                        bh = y2_i - y1_i
                        if bw <= 1 or bh <= 1:
                            continue

                        # check class
                        if is_head_class(class_name):
                            frame_has_head = True
                        if is_nonhelmet_class(class_name):
                            frame_has_nonhelmet = True

                        # lưu box để crop nếu cần
                        if is_head_class(class_name) or is_nonhelmet_class(class_name):
                            boxes_for_crop.append((class_name, x1_i, y1_i, x2_i, y2_i))

            # cập nhật cửa sổ 30 frame
            window_head.append(1 if frame_has_head else 0)
            window_nonhelmet.append(1 if frame_has_nonhelmet else 0)

            head_count = sum(window_head)
            nonhelmet_count = sum(window_nonhelmet)

            # Frame PIL (RGB) để lưu ảnh khi cần
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # ========== NGHI_NGO: non-helmet >= 20/30, rising edge ==========
            if prev_nonhelmet_count < 20 and nonhelmet_count >= 20:
                event_id = generate_event_id()
                timestamp = datetime.now().isoformat()

                global_filename = f"{event_id}.jpg"
                global_path = GLOBAL_DIR / global_filename
                save_pil_image(pil_frame, global_path)
                global_url = f"/static/history/global/{global_filename}"

                crop_urls: list[str] = []
                crop_idx = 0
                for cls_name, x1_i, y1_i, x2_i, y2_i in boxes_for_crop:
                    if not is_nonhelmet_class(cls_name):
                        continue
                    if x2_i <= x1_i or y2_i <= y1_i:
                        continue
                    crop = pil_frame.crop((x1_i, y1_i, x2_i, y2_i))
                    crop_filename = f"{event_id}_{crop_idx}.jpg"
                    crop_path = CROPS_DIR / crop_filename
                    save_pil_image(crop, crop_path)
                    crop_urls.append(f"/static/history/crops/{crop_filename}")
                    crop_idx += 1

                num_violators = len(crop_urls) if crop_urls else 1

                history_record = HistoryEvent(
                    id=event_id,
                    timestamp=timestamp,
                    source=effective_source,
                    type="NGHI_NGO",
                    global_image_url=global_url,
                    crop_image_urls=crop_urls,
                    num_violators=num_violators,
                )
                append_history_record(history_record)
                events_out.append(history_record)
                suspicion_events += 1

            # ========== VI_PHAM: head >= 20/30, rising edge ==========
            if prev_head_count < 20 and head_count >= 20:
                event_id = generate_event_id()
                timestamp = datetime.now().isoformat()

                global_filename = f"{event_id}.jpg"
                global_path = GLOBAL_DIR / global_filename
                save_pil_image(pil_frame, global_path)
                global_url = f"/static/history/global/{global_filename}"

                crop_urls: list[str] = []
                crop_idx = 0
                for cls_name, x1_i, y1_i, x2_i, y2_i in boxes_for_crop:
                    if not is_head_class(cls_name):
                        continue
                    if x2_i <= x1_i or y2_i <= y1_i:
                        continue
                    crop = pil_frame.crop((x1_i, y1_i, x2_i, y2_i))
                    crop_filename = f"{event_id}_{crop_idx}.jpg"
                    crop_path = CROPS_DIR / crop_filename
                    save_pil_image(crop, crop_path)
                    crop_urls.append(f"/static/history/crops/{crop_filename}")
                    crop_idx += 1

                num_violators = len(crop_urls) if crop_urls else 1

                history_record = HistoryEvent(
                    id=event_id,
                    timestamp=timestamp,
                    source=effective_source,
                    type="VI_PHAM",
                    global_image_url=global_url,
                    crop_image_urls=crop_urls,
                    num_violators=num_violators,
                )
                append_history_record(history_record)
                events_out.append(history_record)
                violation_events += 1

            # cập nhật prev_* để dùng cho rising edge
            prev_head_count = head_count
            prev_nonhelmet_count = nonhelmet_count

    finally:
        cap.release()
        temp_path.unlink(missing_ok=True)

    # total_frames = số frame đã thực sự xử lý (~10 FPS)
    total_processed = processed_idx

    return VideoDetectResponse(
        total_frames=total_processed,
        fps_input=float(fps_input),
        fps_used=float(target_fps),
        window_size=30,
        violation_events=violation_events,
        suspicion_events=suspicion_events,
        events=events_out,
    )


@app.get("/api/history", response_model=List[HistoryEvent])
def get_history():
    """
    Lấy toàn bộ history (mới nhất ở đầu).
    Frontend tab Lịch sử sẽ gọi endpoint này.
    """
    return load_all_history()


@app.get("/api/history/{event_id}", response_model=HistoryEvent)
def get_history_detail(event_id: str):
    events = load_all_history()
    for e in events:
        if e.id == event_id:
            return e
    raise HTTPException(status_code=404, detail="Không tìm thấy sự kiện.")

@app.post("/api/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """
    Nhận file video từ frontend, lưu tạm và trả về video_id + file_name
    để frontend dùng khi stream.
    """
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File phải là video.")

    video_id = uuid.uuid4().hex
    safe_name = Path(file.filename).name  # tên gốc
    save_path = VIDEOS_DIR / f"{video_id}__{safe_name}"

    raw = await file.read()
    with open(save_path, "wb") as f:
        f.write(raw)

    return {"video_id": video_id, "file_name": safe_name}

@app.get("/api/stream/video")
def stream_video(
    video_id: str = Query(...),
    file_name: str = Query(...),
    source: str | None = Query(None),
):
    """
    Stream MJPEG video đã được YOLO xử lý + áp dụng luật 2/3 trong 30 frame
    THEO THỜI GIAN THẬT ~10 FPS.

    - Video đã upload trước đó vào VIDEOS_DIR qua /api/upload-video.
    - Backend:
        + Đọc video, downsample theo fps_input nếu cần.
        + Mỗi frame đã chọn:
            * Chạy YOLO.
            * Vẽ bbox trực tiếp lên frame.
            * Cập nhật cửa sổ 30 frame (≈ 3 giây thực).
            * Kiểm tra luật 2/3 (>=20/30 frame head/non-helmet) – rising edge → tạo event.
        + Dùng time.sleep để đảm bảo tốc độ gửi frame ra ~10 FPS wall-clock.
    - Frontend chỉ cần <img src="/api/stream/video?..."> để xem.
    """

    video_path = VIDEOS_DIR / f"{video_id}__{file_name}"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video không tồn tại trên server.")

    effective_source = source or file_name

    def generate():
        tracker = SimpleIOUTracker(iou_thresh=0.4, max_age=10)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            return

        fps_input = cap.get(cv2.CAP_PROP_FPS)
        if not fps_input or fps_input <= 0:
            fps_input = 25.0

        target_fps = 10.0  # 👉 MỤC TIÊU: 10 frame / 1s THỜI GIAN THẬT
        frame_interval = max(int(round(fps_input / target_fps)), 1)

        window_head = deque(maxlen=30)
        window_nonhelmet = deque(maxlen=30)

        current_frame_idx = 0      # frame gốc
        processed_idx = 0          # frame đã xử lý (khoảng 10 FPS)

        prev_head_count = 0
        prev_nonhelmet_count = 0

        # Thời gian gửi frame cuối cùng (để throttle theo thời gian thật)
        last_send_time = time.perf_counter()
        target_dt = 1.0 / target_fps  # khoảng 0.1s / frame

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                current_frame_idx += 1

                # Downsample: chỉ xử lý 1 frame mỗi frame_interval frame gốc
                if current_frame_idx % frame_interval != 0:
                    continue

                processed_idx += 1
                h, w = frame.shape[:2]

                frame_has_head = False
                frame_has_nonhelmet = False

                # 1) YOLO detect trước, gom detections
                detections_for_tracker: List[Tuple[float, float, float, float, str, float]] = []

                # ---------- YOLO detect ----------
                try:
                    results = model.predict(frame, imgsz=640, verbose=False)
                except Exception as e:
                    print(f"[WARN] YOLO predict lỗi ở frame {current_frame_idx}: {e}")
                    results = []

                if len(results) > 0:
                    r = results[0]
                    if r.boxes is not None:
                        for b in r.boxes:
                            cls_id = int(b.cls[0].item())
                            class_name = r.names.get(cls_id, str(cls_id))
                            conf = float(b.conf[0].item())

                            x1, y1, x2, y2 = b.xyxy[0].tolist()
                            x1_i, y1_i, x2_i, y2_i = map(int, [x1, y1, x2, y2])

                            # clamp
                            x1_i = max(0, min(x1_i, w - 1))
                            y1_i = max(0, min(y1_i, h - 1))
                            x2_i = max(0, min(x2_i, w))
                            y2_i = max(0, min(y2_i, h))
                            bw = x2_i - x1_i
                            bh = y2_i - y1_i
                            if bw <= 1 or bh <= 1:
                                continue

                            detections_for_tracker.append(
                                (x1_i, y1_i, x2_i, y2_i, class_name, conf)
                            )

                # 2) Cập nhật tracker bằng danh sách detection
                tracks_now = tracker.update(detections_for_tracker, processed_idx)

                # 3) Dùng track (ID + class smoothed) để:
                #    - vẽ bbox
                #    - xác định frame_has_head / frame_has_nonhelmet
                boxes_for_crop: list[tuple[str, int, int, int, int]] = []

                for trk in tracks_now:
                    # chỉ vẽ track nếu nó vừa được update ở frame processed_idx hiện tại
                    if trk.last_frame != processed_idx:
                        continue

                    tx1, ty1, tx2, ty2 = trk.bbox
                    smooth_cls = trk.main_class

                    if is_head_class(smooth_cls):
                        frame_has_head = True
                    if is_nonhelmet_class(smooth_cls):
                        frame_has_nonhelmet = True

                    if is_head_class(smooth_cls) or is_nonhelmet_class(smooth_cls):
                        boxes_for_crop.append((smooth_cls, tx1, ty1, tx2, ty2))

                    color = (0, 255, 0)
                    if is_head_class(smooth_cls):
                        color = (0, 0, 255)
                    elif is_nonhelmet_class(smooth_cls):
                        color = (0, 215, 255)

                    cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), color, 2)
                    label = f"ID{trk.id} {smooth_cls}"
                    cv2.putText(
                        frame,
                        label,
                        (tx1, max(0, ty1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                        cv2.LINE_AA,
                    )

                # ---------- Cập nhật cửa sổ 30 frame ----------
                window_head.append(1 if frame_has_head else 0)
                window_nonhelmet.append(1 if frame_has_nonhelmet else 0)

                head_count = sum(window_head)
                nonhelmet_count = sum(window_nonhelmet)

                pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # ========== NGHI_NGO: non-helmet >= 20/30, rising edge ==========
                if prev_nonhelmet_count < 20 and nonhelmet_count >= 20:
                    event_id = generate_event_id()
                    timestamp = datetime.now().isoformat()

                    global_filename = f"{event_id}.jpg"
                    global_path = GLOBAL_DIR / global_filename
                    save_pil_image(pil_frame, global_path)
                    global_url = f"/static/history/global/{global_filename}"

                    crop_urls: list[str] = []
                    crop_idx = 0
                    for cls_name, x1_i, y1_i, x2_i, y2_i in boxes_for_crop:
                        if not is_nonhelmet_class(cls_name):
                            continue
                        if x2_i <= x1_i or y2_i <= y1_i:
                            continue
                        crop = pil_frame.crop((x1_i, y1_i, x2_i, y2_i))
                        crop_filename = f"{event_id}_{crop_idx}.jpg"
                        crop_path = CROPS_DIR / crop_filename
                        save_pil_image(crop, crop_path)
                        crop_urls.append(f"/static/history/crops/{crop_filename}")
                        crop_idx += 1

                    num_violators = len(crop_urls) if crop_urls else 1

                    history_record = HistoryEvent(
                        id=event_id,
                        timestamp=timestamp,
                        source=effective_source,
                        type="NGHI_NGO",
                        global_image_url=global_url,
                        crop_image_urls=crop_urls,
                        num_violators=num_violators,
                    )
                    append_history_record(history_record)

                # ========== VI_PHAM: head >= 20/30, rising edge ==========
                if prev_head_count < 20 and head_count >= 20:
                    event_id = generate_event_id()
                    timestamp = datetime.now().isoformat()

                    global_filename = f"{event_id}.jpg"
                    global_path = GLOBAL_DIR / global_filename
                    save_pil_image(pil_frame, global_path)
                    global_url = f"/static/history/global/{global_filename}"

                    crop_urls: list[str] = []
                    crop_idx = 0
                    for cls_name, x1_i, y1_i, x2_i, y2_i in boxes_for_crop:
                        if not is_head_class(cls_name):
                            continue
                        if x2_i <= x1_i or y2_i <= y1_i:
                            continue
                        crop = pil_frame.crop((x1_i, y1_i, x2_i, y2_i))
                        crop_filename = f"{event_id}_{crop_idx}.jpg"
                        crop_path = CROPS_DIR / crop_filename
                        save_pil_image(crop, crop_path)
                        crop_urls.append(f"/static/history/crops/{crop_filename}")
                        crop_idx += 1

                    num_violators = len(crop_urls) if crop_urls else 1

                    history_record = HistoryEvent(
                        id=event_id,
                        timestamp=timestamp,
                        source=effective_source,
                        type="VI_PHAM",
                        global_image_url=global_url,
                        crop_image_urls=crop_urls,
                        num_violators=num_violators,
                    )
                    append_history_record(history_record)

                # cập nhật prev_* cho rising edge
                prev_head_count = head_count
                prev_nonhelmet_count = nonhelmet_count

                # ---------- THROTTLE: đảm bảo ~10 FPS THỜI GIAN THẬT ----------
                now = time.perf_counter()
                elapsed = now - last_send_time
                if elapsed < target_dt:
                    time.sleep(target_dt - elapsed)
                    now = time.perf_counter()
                last_send_time = now

                # encode JPEG để stream
                ret2, jpeg = cv2.imencode(".jpg", frame)
                if not ret2:
                    continue
                frame_bytes = jpeg.tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )

        finally:
            cap.release()

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )

@app.post("/api/live/start")
async def start_live_stream(
    stream_url: str = Form(...),
    source: str | None = Form(None),
):
    """
    Nhận URL stream (RTSP/RTMP/HTTP…) từ OBS và trả về live_id.
    Frontend sẽ dùng live_id để mở stream MJPEG.
    """
    live_id = uuid.uuid4().hex

    LIVE_STREAMS[live_id] = {
        "url": stream_url,
        "source": source or f"Camera {live_id[:6]}",
    }

    # (Optional) kiểm tra thử mở 1 frame cho chắc
    cap = cv2.VideoCapture(stream_url)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        LIVE_STREAMS.pop(live_id, None)
        raise HTTPException(status_code=400, detail="Không đọc được stream từ URL (OBS).")

    return {"live_id": live_id, "source": LIVE_STREAMS[live_id]["source"]}

@app.get("/api/live/stream")
def stream_live(
    live_id: str = Query(...),
):
    """
    Đọc stream từ OBS (RTSP/RTMP/HTTP) theo thời gian thực,
    YOLO + logic 2/3 (20/30 frame ≈ 3s), lưu lịch sử & stream MJPEG cho frontend.
    """

    live_cfg = LIVE_STREAMS.get(live_id)
    if not live_cfg:
        raise HTTPException(status_code=404, detail="Live ID không tồn tại.")

    stream_url = live_cfg["url"]
    effective_source = live_cfg["source"]

    def generate():
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            print(f"[ERROR] Không mở được stream: {stream_url}")
            cap.release()
            return

        # Logic 2/3 trên 30 frame (10 FPS)
        window_head = deque(maxlen=30)
        window_nonhelmet = deque(maxlen=30)

        processed_idx = 0
        prev_head_count = 0
        prev_nonhelmet_count = 0

        # throttle 10 FPS theo thời gian thật
        target_fps = 10.0
        target_dt = 1.0 / target_fps
        last_send_time = time.perf_counter()

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    # với live stream, có thể tạm sleep rồi thử lại,
                    # ở đây cho đơn giản: break nếu mất stream
                    print("[WARN] Mất frame từ stream, dừng live.")
                    break

                now = time.perf_counter()
                elapsed = now - last_send_time
                if elapsed < target_dt:
                    # chưa đủ 0.1s thì đọc tiếp frame nhưng không xử lý/gửi
                    # (hoặc time.sleep(target_dt - elapsed) nếu muốn tiết kiệm CPU)
                    continue
                last_send_time = now

                processed_idx += 1
                h, w = frame.shape[:2]

                frame_has_head = False
                frame_has_nonhelmet = False
                boxes_for_crop: list[tuple[str, int, int, int, int]] = []

                # ----- YOLO detect -----
                try:
                    results = model.predict(frame, imgsz=640, verbose=False)
                except Exception as e:
                    print(f"[WARN] YOLO predict lỗi trên live frame {processed_idx}: {e}")
                    continue

                if len(results) > 0:
                    r = results[0]
                    if r.boxes is not None:
                        for b in r.boxes:
                            cls_id = int(b.cls[0].item())
                            class_name = r.names.get(cls_id, str(cls_id))
                            conf = float(b.conf[0].item())

                            x1, y1, x2, y2 = b.xyxy[0].tolist()
                            x1_i, y1_i, x2_i, y2_i = map(int, [x1, y1, x2, y2])

                            # clamp
                            x1_i = max(0, min(x1_i, w - 1))
                            y1_i = max(0, min(y1_i, h - 1))
                            x2_i = max(0, min(x2_i, w))
                            y2_i = max(0, min(y2_i, h))
                            bw = x2_i - x1_i
                            bh = y2_i - y1_i
                            if bw <= 1 or bh <= 1:
                                continue

                            # xác định hành vi
                            if is_head_class(class_name):
                                frame_has_head = True
                            if is_nonhelmet_class(class_name):
                                frame_has_nonhelmet = True

                            # chỉ lưu những box head / non-helmet để crop nếu có event
                            if is_head_class(class_name) or is_nonhelmet_class(class_name):
                                boxes_for_crop.append((class_name, x1_i, y1_i, x2_i, y2_i))

                            # vẽ bbox
                            color = (0, 255, 0)  # helmet
                            if is_head_class(class_name):
                                color = (0, 0, 255)      # head -> đỏ
                            elif is_nonhelmet_class(class_name):
                                color = (0, 215, 255)    # non-helmet -> vàng

                            label = f"{class_name} {conf:.2f}"
                            cv2.rectangle(frame, (x1_i, y1_i), (x2_i, y2_i), color, 2)
                            cv2.putText(
                                frame,
                                label,
                                (x1_i, max(0, y1_i - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                color,
                                1,
                                cv2.LINE_AA,
                            )

                # ----- cập nhật cửa sổ 30 frame -----
                window_head.append(1 if frame_has_head else 0)
                window_nonhelmet.append(1 if frame_has_nonhelmet else 0)

                head_count = sum(window_head)
                nonhelmet_count = sum(window_nonhelmet)

                pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # ====== NGHI_NGO: non-helmet >= 20/30, rising edge ======
                if prev_nonhelmet_count < 20 and nonhelmet_count >= 20:
                    event_id = generate_event_id()
                    timestamp = datetime.now().isoformat()

                    global_filename = f"{event_id}.jpg"
                    global_path = GLOBAL_DIR / global_filename
                    save_pil_image(pil_frame, global_path)
                    global_url = f"/static/history/global/{global_filename}"

                    crop_urls: list[str] = []
                    crop_idx = 0
                    for cls_name, x1_i, y1_i, x2_i, y2_i in boxes_for_crop:
                        if not is_nonhelmet_class(cls_name):
                            continue
                        if x2_i <= x1_i or y2_i <= y1_i:
                            continue
                        crop = pil_frame.crop((x1_i, y1_i, x2_i, y2_i))
                        crop_filename = f"{event_id}_{crop_idx}.jpg"
                        crop_path = CROPS_DIR / crop_filename
                        save_pil_image(crop, crop_path)
                        crop_urls.append(f"/static/history/crops/{crop_filename}")
                        crop_idx += 1

                    num_violators = len(crop_urls) if crop_urls else 1

                    history_record = HistoryEvent(
                        id=event_id,
                        timestamp=timestamp,
                        source=effective_source,
                        type="NGHI_NGO",
                        global_image_url=global_url,
                        crop_image_urls=crop_urls,
                        num_violators=num_violators,
                    )
                    append_history_record(history_record)

                # ====== VI_PHAM: head >= 20/30, rising edge ======
                if prev_head_count < 20 and head_count >= 20:
                    event_id = generate_event_id()
                    timestamp = datetime.now().isoformat()

                    global_filename = f"{event_id}.jpg"
                    global_path = GLOBAL_DIR / global_filename
                    save_pil_image(pil_frame, global_path)
                    global_url = f"/static/history/global/{global_filename}"

                    crop_urls: list[str] = []
                    crop_idx = 0
                    for cls_name, x1_i, y1_i, x2_i, y2_i in boxes_for_crop:
                        if not is_head_class(cls_name):
                            continue
                        if x2_i <= x1_i or y2_i <= y1_i:
                            continue
                        crop = pil_frame.crop((x1_i, y1_i, x2_i, y2_i))
                        crop_filename = f"{event_id}_{crop_idx}.jpg"
                        crop_path = CROPS_DIR / crop_filename
                        save_pil_image(crop, crop_path)
                        crop_urls.append(f"/static/history/crops/{crop_filename}")
                        crop_idx += 1

                    num_violators = len(crop_urls) if crop_urls else 1

                    history_record = HistoryEvent(
                        id=event_id,
                        timestamp=timestamp,
                        source=effective_source,
                        type="VI_PHAM",
                        global_image_url=global_url,
                        crop_image_urls=crop_urls,
                        num_violators=num_violators,
                    )
                    append_history_record(history_record)

                prev_head_count = head_count
                prev_nonhelmet_count = nonhelmet_count

                # encode JPEG để stream
                ret2, jpeg = cv2.imencode(".jpg", frame)
                if not ret2:
                    continue

                frame_bytes = jpeg.tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )

        finally:
            cap.release()

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )

@app.get("/api/history/latest", response_model=HistoryLatestResponse)
def get_latest_history_event(
    source: Optional[str] = Query(None, description="Lọc theo nguồn (tên video hoặc camera)"),
    types: Optional[str] = Query(None, description="Lọc theo loại, ví dụ: VI_PHAM,NGHI_NGO"),
):
    """
    Trả về history event mới nhất (cuối file) thỏa điều kiện:
    - Nếu 'source' được cung cấp: chỉ lấy event có source đúng chuỗi đó.
    - Nếu 'types' được cung cấp: là danh sách type phân tách bởi dấu phẩy (VD: VI_PHAM,NGHI_NGO).
    """
    if not HISTORY_JSONL.exists():
        return HistoryLatestResponse(event=None)

    type_filter: Optional[List[str]] = None
    if types:
        type_filter = [t.strip().upper() for t in types.split(",") if t.strip()]

    latest: Optional[HistoryEvent] = None

    try:
        # Đọc từ cuối file lên cho nhanh
        with open(HISTORY_JSONL, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                evt = HistoryEvent(**data)
            except Exception as e:
                print(f"[WARN] Không parse được dòng history: {e}")
                continue

            if source is not None and evt.source != source:
                continue

            if type_filter is not None and evt.type not in type_filter:
                continue

            latest = evt
            break

    except Exception as e:
        print(f"[WARN] get_latest_history_event: {e}")
        return HistoryLatestResponse(event=None)

    return HistoryLatestResponse(event=latest)

def _get_event_type(e) -> str | None:
    # Ưu tiên event_type, nếu không có thì fallback sang type
    et = getattr(e, "event_type", None)
    if et is None:
        et = getattr(e, "type", None)
    return et

def _get_global_image_rel(e) -> str | None:
    giu = getattr(e, "global_image_url", None)
    if not giu:
        giu = getattr(e, "globalImageUrl", None)
    return giu

def read_marked_event_ids() -> set[str]:
    """
    Đọc file meta (UPDATE_POOL_META) để lấy danh sách event_id
    đã được người dùng đánh dấu Đúng/Sai.
    """
    ids: set[str] = set()
    if not UPDATE_POOL_META.exists():
        return ids

    with open(UPDATE_POOL_META, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                eid = obj.get("event_id")
                if eid:
                    ids.add(eid)
            except Exception:
                # bỏ qua dòng lỗi, tránh crash
                continue
    return ids

@app.get("/api/update/candidates")
def get_update_candidates(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=50),
):
    events: List[HistoryEvent] = read_all_history()

    # event nào đã xuất hiện trong update_pool (accepted.jsonl)
    # sẽ không hiển thị nữa
    marked_ids = read_marked_event_ids()

    filtered: List[HistoryEvent] = []
    for e in events:
        # bỏ qua event đã xử lý rồi (Đúng hoặc Sai)
        if e.id in marked_ids:
            continue

        et = _get_event_type(e)
        if et not in ("VI_PHAM", "NGHI_NGO"):
            continue

        giu = _get_global_image_rel(e)
        if not giu:
            continue

        filtered.append(e)

    # sắp xếp mới nhất ở trên
    filtered.sort(key=lambda e: e.timestamp, reverse=True)

    start = (page - 1) * page_size
    end = start + page_size
    page_items = filtered[start:end]

    return {
        "total": len(filtered),
        "page": page,
        "page_size": page_size,
        "items": [e.model_dump() for e in page_items],
    }


def _resolve_image_file(img_rel: str) -> Path:
    """
    Cố gắng tìm file ảnh thật từ chuỗi lưu trong global_image_url.

    Ưu tiên:
      1) Nếu img_rel là path tuyệt đối và tồn tại → dùng luôn.
      2) Map theo tên file vào thư mục D:\KLTN_Code\backend\data\history\global.
      3) Thử các biến thể tương đối: DATA_DIR / raw, DATA_DIR.parent / raw.
    """
    img_rel_norm = img_rel.replace("\\", "/").strip()

    # 1) Nếu là đường dẫn tuyệt đối
    p_abs = Path(img_rel_norm)
    if p_abs.is_absolute() and p_abs.exists():
        return p_abs

    # 2) Luôn ưu tiên tìm theo basename trong thư mục history/global
    basename = Path(img_rel_norm).name
    cand0 = HISTORY_GLOBAL_DIR / basename
    if cand0.exists():
        return cand0

    # 3) Thử các dạng tương đối
    raw = img_rel_norm.lstrip("/")

    cand1 = DATA_DIR / raw           # vd: data/history/global/xxx.jpg
    cand2 = DATA_DIR.parent / raw    # vd: backend/data/history/global/xxx.jpg

    if cand1.exists():
        return cand1
    if cand2.exists():
        return cand2

    # 4) fallback: trả về cand0 (để khi raise 404 còn in ra được)
    return cand0


@app.get("/api/update/auto-label/{event_id}")
def get_update_auto_label(event_id: str):
    events = read_all_history()
    target = None
    for e in events:
        if e.id == event_id:
            target = e
            break

    if not target:
        raise HTTPException(status_code=404, detail="Không tìm thấy event.")

    img_rel = _get_global_image_rel(target)
    if not img_rel:
        raise HTTPException(status_code=400, detail="Event không có global_image_url.")

    img_path = _resolve_image_file(img_rel)
    if not img_path.exists():
        raise HTTPException(status_code=404, detail=f"Không tìm thấy file ảnh global: {img_path}")

    boxes, class_counts = run_yolo_update_inference(img_path)

    # URL public cho frontend: nếu history đã lưu dạng /data/.... thì cứ dùng lại
    img_url = img_rel.replace("\\", "/")
    if not img_url.startswith("/"):
        # fallback: dựng URL theo kiểu /data/history/global/<basename>
        img_url = f"/data/history/global/{Path(img_url).name}"

    return {
        "event_id": target.id,
        "image_url": img_url,
        "boxes": boxes,
        "class_counts": class_counts,
    }


class UpdateMarkRequest(BaseModel):
    event_id: str
    accepted: bool

from fastapi import HTTPException

@app.post("/api/update/mark")
def post_update_mark(req: UpdateMarkRequest):
    """
    Người dùng xác nhận ĐÚNG / SAI cho một event trong lịch sử:

    - Nếu accepted = False:
        + Chỉ ghi log vào update_pool/accepted.jsonl
        + Không copy ảnh, không tạo nhãn.

    - Nếu accepted = True:
        + Tìm ảnh global tương ứng event (data/history/global)
        + Chạy lại YOLO để auto-label 3 class
        + Copy ảnh sang update_pool/images/
        + Ghi file YOLO .txt sang update_pool/labels/
        + Ghi log accepted vào update_pool/accepted.jsonl
    """
    events = read_all_history()
    target: HistoryEvent | None = None
    for e in events:
        if e.id == req.event_id:
            target = e
            break

    if not target:
        raise HTTPException(status_code=404, detail="Không tìm thấy event trong lịch sử.")

    # lấy đường dẫn ảnh global từ event (hỗ trợ cả global_image_url / globalImageUrl)
    img_rel = _get_global_image_rel(target)
    if not img_rel:
        raise HTTPException(status_code=400, detail="Event không có global_image_url.")

    # resolve sang file thật trong ổ D:\KLTN_Code\backend\data\history\global
    src_path = _resolve_image_file(img_rel)
    if not src_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Không tìm thấy file ảnh global: {src_path}"
        )

    # đảm bảo thư mục update_pool tồn tại
    UPDATE_POOL_DIR.mkdir(parents=True, exist_ok=True)
    UPDATE_POOL_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    UPDATE_POOL_LABELS_DIR.mkdir(parents=True, exist_ok=True)

    # ghi log accepted / rejected vào accepted.jsonl
    with open(UPDATE_POOL_META, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "event_id": target.id,
            "timestamp": target.timestamp,
            "source": target.source,
            "type": _get_event_type(target),
            "accepted": req.accepted,
        }, ensure_ascii=False) + "\n")

    # Nếu người dùng chọn SAI → không đưa vào update_pool
    if not req.accepted:
        return {"ok": True, "accepted": False}

    # Nếu accepted = True → chạy YOLO để tạo nhãn + copy ảnh
    try:
        boxes, class_counts = run_yolo_update_inference(src_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi YOLO khi tạo nhãn: {e}")

    # đặt tên file đích theo id event + tên gốc
    dst_stem = f"{target.id}_{src_path.stem}"
    dst_img_name = dst_stem + src_path.suffix
    dst_label_name = dst_stem + ".txt"

    dst_img_path = UPDATE_POOL_IMAGES_DIR / dst_img_name
    label_path = UPDATE_POOL_LABELS_DIR / dst_label_name

    # copy ảnh sang update_pool/images
    shutil.copy2(src_path, dst_img_path)

    # ghi file .txt theo format YOLO: class_id xc yc w h (normalized)
    with open(label_path, "w", encoding="utf-8") as f:
        for b in boxes:
            class_name = str(b.get("class_name", ""))
            cls_id = b.get("class_id")

            # fallback nếu run_yolo_update_inference không set class_id
            if cls_id is None:
                cls_id = CLASS_NAME_TO_ID.get(class_name, 0)

            xc = float(b.get("xc", 0.0))
            yc = float(b.get("yc", 0.0))
            w = float(b.get("width", 0.0))
            h = float(b.get("height", 0.0))

            # đảm bảo trong [0,1]
            xc = max(0.0, min(1.0, xc))
            yc = max(0.0, min(1.0, yc))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))

            f.write(f"{int(cls_id)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    return {
        "ok": True,
        "accepted": True,
        "image_path": str(dst_img_path),
        "label_path": str(label_path),
        "num_boxes": len(boxes),
    }



@app.get("/api/update/status")
def get_update_status():
    if not UPDATE_POOL_IMAGES_DIR.exists():
        count = 0
    else:
        count = sum(1 for _ in UPDATE_POOL_IMAGES_DIR.glob("*.*"))

    threshold = 100
    return {
        "num_images": count,
        "threshold": threshold,
        "ready": count >= threshold,
    }


# ================== MAIN (chạy trực tiếp) ==================

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
