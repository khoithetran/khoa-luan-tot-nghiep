import React, { useEffect, useState } from 'react';
import type { DetectionMode } from '../../types';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

type HistoryEvent = {
  id: string;
  timestamp: string;
  source: string;
  type: 'VI_PHAM' | 'NGHI_NGO';
  global_image_url?: string;
  globalImageUrl?: string;
  crop_image_urls?: string[];
  cropImageUrls?: string[];
  num_violators?: number;
  numViolators?: number;
};

type ApiBox = {
  id: string;
  class_name: string;
  confidence: number;

  // normalized top-left (0–1)
  x: number;
  y: number;

  // normalized size (0–1)
  width: number;
  height: number;

  // optional – cho backend cũ
  x1?: number;
  y1?: number;
  x2?: number;
  y2?: number;
  w?: number;
  h?: number;
};

export const DetectionTab: React.FC = () => {
  const [mode, setMode] = useState<DetectionMode>('image');

  // -------- IMAGE MODE --------
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState<string | null>(null);
  const [imageCropUrls, setImageCropUrls] = useState<string[]>([]);
  const [isImageDetecting, setIsImageDetecting] = useState(false);
  const [imageInfo, setImageInfo] = useState<string>('');
  const [imageBoxes, setImageBoxes] = useState<ApiBox[]>([]);

  // -------- VIDEO MODE (backend xử lý & stream) --------
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [selectedVideoName, setSelectedVideoName] = useState<string | null>(null);
  const [uploadedVideoId, setUploadedVideoId] = useState<string | null>(null);
  const [uploadedVideoName, setUploadedVideoName] = useState<string | null>(null);
  const [isUploadingVideo, setIsUploadingVideo] = useState(false);
  const [videoBackendInfo, setVideoBackendInfo] = useState<string>('');

  // -------- LIVE MODE (OBS / RTSP) --------
  const [liveUrl, setLiveUrl] = useState<string>('');
  const [liveSourceName, setLiveSourceName] = useState<string>('Camera OBS');
  const [liveId, setLiveId] = useState<string | null>(null);
  const [isConnectingLive, setIsConnectingLive] = useState(false);

  // -------- ALERTS --------
  const [videoAlert, setVideoAlert] = useState<HistoryEvent | null>(null);
  const [lastVideoAlertId, setLastVideoAlertId] = useState<string | null>(null);

  const [liveAlert, setLiveAlert] = useState<HistoryEvent | null>(null);
  const [lastLiveAlertId, setLastLiveAlertId] = useState<string | null>(null);

  // ===================== IMAGE HANDLERS =====================

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setImageFile(file);
    setImagePreviewUrl(URL.createObjectURL(file));
    setImageBoxes([]);
    setImageCropUrls([]);
    setImageInfo('');
  };

  const handleRunImageDetection = async () => {
    if (!imageFile) {
      alert('Vui lòng chọn ảnh trước.');
      return;
    }

    try {
      setIsImageDetecting(true);
      setImageCropUrls([]);
      setImageInfo('');
      setImageBoxes([]);

      const formData = new FormData();
      formData.append('file', imageFile);
      formData.append('source', imageFile.name);

      const res = await fetch(`${API_BASE}/api/detect/image`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const text = await res.text();
        console.error('Lỗi backend:', text);
        alert('Backend trả lỗi khi nhận diện ảnh.');
        return;
      }

      const data = await res.json();
      console.log('[detect/image] response:', data);

      const rawBoxes: any[] = Array.isArray(data.boxes) ? data.boxes : [];

      // ⚠️ Map chính xác theo logic cũ của video:
      // backend trả x,y,width,height là top-left + size (normalized 0–1)
      const boxes: ApiBox[] = rawBoxes.map((b, idx) => {
        // top-left
        const xTopLeft =
          typeof b.x === 'number'
            ? b.x
            : typeof b.x1 === 'number'
            ? b.x1
            : 0;

        const yTopLeft =
          typeof b.y === 'number'
            ? b.y
            : typeof b.y1 === 'number'
            ? b.y1
            : 0;

        // size
        let wNorm: number =
          typeof b.width === 'number'
            ? b.width
            : typeof b.w === 'number'
            ? b.w
            : typeof b.x2 === 'number' && typeof b.x1 === 'number'
            ? b.x2 - b.x1
            : 0;

        let hNorm: number =
          typeof b.height === 'number'
            ? b.height
            : typeof b.h === 'number'
            ? b.h
            : typeof b.y2 === 'number' && typeof b.y1 === 'number'
            ? b.y2 - b.y1
            : 0;

        // clamp lại cho an toàn
        wNorm = Math.max(0, Math.min(1, wNorm));
        hNorm = Math.max(0, Math.min(1, hNorm));

        return {
          id: b.id ?? `box_${idx}`,
          class_name: b.class_name ?? b.label ?? '',
          confidence: typeof b.confidence === 'number' ? b.confidence : b.score ?? 0,
          x: xTopLeft,
          y: yTopLeft,
          width: wNorm,
          height: hNorm,
          x1: b.x1,
          y1: b.y1,
          x2: b.x2,
          y2: b.y2,
          w: b.w,
          h: b.h,
        };
      });

      setImageBoxes(boxes);

      const infoLines: string[] = [];
      infoLines.push(`Số lượng bounding box: ${boxes.length}`);
      if (data.event_type) {
        infoLines.push(`Loại sự kiện: ${data.event_type}`);
      }
      if (data.history_event_id) {
        infoLines.push(`ID sự kiện: ${data.history_event_id}`);
      }
      infoLines.push('Nhận diện ảnh hoàn tất, bbox sẽ hiển thị ở khung bên phải.');
      setImageInfo(infoLines.join('\n'));
    } catch (err) {
      console.error(err);
      alert('Có lỗi xảy ra khi gửi ảnh lên backend.');
    } finally {
      setIsImageDetecting(false);
    }
  };

  // 🔧 Tự crop từng người vi phạm/nghi ngờ trên frontend
  useEffect(() => {
    if (!imagePreviewUrl || imageBoxes.length === 0) {
      setImageCropUrls([]);
      return;
    }

    const img = new Image();
    img.src = imagePreviewUrl;
    img.onload = () => {
      const imgW = img.naturalWidth;
      const imgH = img.naturalHeight;
      if (!imgW || !imgH) return;

      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const urls: string[] = [];

      const violatorBoxes = imageBoxes.filter((b) => {
        const cls = (b.class_name || '').toLowerCase();
        return cls.includes('head') || cls.includes('non');
      });

      for (const b of violatorBoxes) {
        const sx = Math.max(0, b.x * imgW);
        const sy = Math.max(0, b.y * imgH);
        const sw = Math.max(1, Math.min(imgW - sx, b.width * imgW));
        const sh = Math.max(1, Math.min(imgH - sy, b.height * imgH));

        canvas.width = sw;
        canvas.height = sh;
        ctx.clearRect(0, 0, sw, sh);
        ctx.drawImage(img, sx, sy, sw, sh, 0, 0, sw, sh);

        const url = canvas.toDataURL('image/jpeg', 0.9);
        urls.push(url);
      }

      setImageCropUrls(urls);
    };
  }, [imagePreviewUrl, imageBoxes]);

  // ===================== VIDEO HANDLERS =====================

  const handleVideoFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setVideoFile(file);
    setSelectedVideoName(file.name);
    setUploadedVideoId(null);
    setUploadedVideoName(null);
    setVideoBackendInfo('');
    setVideoAlert(null);
    setLastVideoAlertId(null);
  };

  const handleUploadAndProcessVideo = async () => {
    if (!videoFile) {
      alert('Vui lòng chọn file video trước.');
      return;
    }
    try {
      setIsUploadingVideo(true);
      setVideoBackendInfo('Đang upload video và khởi chạy xử lý trên backend...');
      setUploadedVideoId(null);
      setUploadedVideoName(null);
      setVideoAlert(null);
      setLastVideoAlertId(null);

      const formData = new FormData();
      formData.append('file', videoFile);

      const res = await fetch(`${API_BASE}/api/upload-video`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const text = await res.text();
        console.error('Lỗi upload-video:', text);
        alert('Backend không nhận được video.');
        return;
      }

      const data = await res.json();
      setUploadedVideoId(data.video_id);
      setUploadedVideoName(data.file_name);
      setVideoBackendInfo(
        data.message ||
          'Video đã được upload. Backend đang xử lý ~10 FPS và stream kết quả về.',
      );
    } catch (err) {
      console.error(err);
      alert('Có lỗi xảy ra khi upload video.');
    } finally {
      setIsUploadingVideo(false);
    }
  };

  // ===================== LIVE HANDLERS =====================

  const handleStartLive = async () => {
    if (!liveUrl.trim()) {
      alert('Vui lòng nhập URL stream từ OBS / DroidCam (rtsp:// hoặc http://...).');
      return;
    }

    try {
      setIsConnectingLive(true);
      setLiveId(null);
      setLiveAlert(null);
      setLastLiveAlertId(null);

      const formData = new FormData();
      formData.append('stream_url', liveUrl.trim());
      formData.append('source', liveSourceName.trim() || 'Camera OBS');

      const res = await fetch(`${API_BASE}/api/live/start`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const text = await res.text();
        console.error('Lỗi backend (start live):', text);
        alert('Backend không đọc được stream từ URL. Kiểm tra lại OBS/DroidCam & URL.');
        return;
      }

      const data = await res.json();
      setLiveId(data.live_id);
      if (data.source) {
        setLiveSourceName(data.source);
      }
    } catch (err) {
      console.error(err);
      alert('Có lỗi khi kết nối live stream.');
    } finally {
      setIsConnectingLive(false);
    }
  };

  const handleStopLive = () => {
    setLiveId(null);
    setLiveAlert(null);
    setLastLiveAlertId(null);
  };

  // ===================== POLL ALERT CHO VIDEO =====================

  useEffect(() => {
    if (mode !== 'video' || !uploadedVideoId || !uploadedVideoName) {
      setVideoAlert(null);
      return;
    }

    let isCancelled = false;
    const sourceName = uploadedVideoName;

    const poll = async () => {
      try {
        const url = `${API_BASE}/api/history/latest?source=${encodeURIComponent(
          sourceName,
        )}&types=VI_PHAM,NGHI_NGO`;
        const res = await fetch(url);
        if (!res.ok) return;
        const data = await res.json();
        if (!data.event) return;

        const evt = data.event as any as HistoryEvent;
        if (isCancelled) return;

        if (evt.id && evt.id !== lastVideoAlertId) {
          setLastVideoAlertId(evt.id);
          setVideoAlert(evt);
        }
      } catch (err) {
        console.error('Poll video history latest error:', err);
      }
    };

    poll();
    const interval = setInterval(poll, 2000);

    return () => {
      isCancelled = true;
      clearInterval(interval);
    };
  }, [mode, uploadedVideoId, uploadedVideoName, lastVideoAlertId]);

  // ===================== POLL ALERT CHO LIVE =====================

  useEffect(() => {
    if (mode !== 'live' || !liveId || !liveSourceName.trim()) {
      setLiveAlert(null);
      return;
    }

    let isCancelled = false;
    const sourceName = liveSourceName.trim();

    const poll = async () => {
      try {
        const url = `${API_BASE}/api/history/latest?source=${encodeURIComponent(
          sourceName,
        )}&types=VI_PHAM,NGHI_NGO`;
        const res = await fetch(url);
        if (!res.ok) return;
        const data = await res.json();
        if (!data.event) return;

        const evt = data.event as any as HistoryEvent;
        if (isCancelled) return;

        if (evt.id && evt.id !== lastLiveAlertId) {
          setLastLiveAlertId(evt.id);
          setLiveAlert(evt);
        }
      } catch (err) {
        console.error('Poll live history latest error:', err);
      }
    };

    poll();
    const interval = setInterval(poll, 2000);

    return () => {
      isCancelled = true;
      clearInterval(interval);
    };
  }, [mode, liveId, liveSourceName, lastLiveAlertId]);

  // ===================== RENDER LEFT PANEL =====================

  const renderLeftPanel = () => {
    if (mode === 'image') {
      // bên trái chỉ: chọn ảnh + nút + info
      return (
        <div className="space-y-4">
          <div>
            <p className="text-xs font-semibold text-slate-300 mb-2">
              Chọn ảnh để nhận diện
            </p>
            <input
              type="file"
              accept="image/*"
              onChange={handleImageChange}
              className="block w-full text-xs text-slate-200 file:mr-3 file:py-1.5 file:px-3 file:rounded-full file:border-0 file:text-xs file:font-semibold file:bg-emerald-500 file:text-white hover:file:bg-emerald-400 cursor-pointer"
            />
            {imageFile && (
              <p className="text-[11px] text-slate-400 mt-1">
                Đã chọn: <span className="font-mono">{imageFile.name}</span>
              </p>
            )}
          </div>

          <button
            onClick={handleRunImageDetection}
            disabled={!imageFile || isImageDetecting}
            className="px-4 py-1.5 bg-emerald-500 rounded-lg text-xs font-semibold text-white hover:bg-emerald-400 disabled:bg-slate-700 disabled:text-slate-400"
          >
            {isImageDetecting ? 'Đang nhận diện...' : 'Nhận diện ảnh'}
          </button>

          {imageInfo && (
            <pre className="bg-slate-950/80 border border-slate-800 rounded-xl p-2 text-[11px] text-slate-300 whitespace-pre-wrap">
              {imageInfo}
            </pre>
          )}
        </div>
      );
    }

    if (mode === 'video') {
      return (
        <div className="space-y-4">
          <div>
            <p className="text-xs font-semibold text-slate-300 mb-2">
              Chọn video để backend xử lý (~10 FPS)
            </p>
            <input
              type="file"
              accept="video/*"
              onChange={handleVideoFileChange}
              className="block w-full text-xs text-slate-200 file:mr-3 file:py-1.5 file:px-3 file:rounded-full file:border-0 file:text-xs file:font-semibold file:bg-sky-500 file:text-white hover:file:bg-sky-400 cursor-pointer"
            />
            {selectedVideoName && (
              <p className="text-[11px] text-slate-400 mt-1">
                Đã chọn: <span className="font-mono">{selectedVideoName}</span>
              </p>
            )}
          </div>

          <button
            onClick={handleUploadAndProcessVideo}
            disabled={!videoFile || isUploadingVideo}
            className="px-4 py-1.5 bg-sky-500 rounded-lg text-xs font-semibold text-white hover:bg-sky-400 disabled:bg-slate-700 disabled:text-slate-400"
          >
            {isUploadingVideo ? 'Đang upload & khởi chạy...' : 'Upload & xử lý bằng backend'}
          </button>

          {videoBackendInfo && (
            <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-2 text-[11px] text-slate-300">
              {videoBackendInfo}
            </div>
          )}

          <div className="text-[11px] text-slate-500 border-t border-slate-800 pt-3 space-y-1">
            <p>
              • Tốc độ xử lý mục tiêu: <span className="font-mono">~10 FPS</span> theo
              thời gian thực.
            </p>
            <p>
              • Cửa sổ đánh giá: <span className="font-mono">30 frame</span> (≈3s). Nếu
              ≥20 frame chứa <code>head</code> hoặc <code>non-helmet</code> sẽ ghi 1 sự
              kiện VI PHẠM / NGHI NGỜ.
            </p>
          </div>
        </div>
      );
    }

    // mode === 'live'
    return (
      <div className="space-y-4">
        <div>
          <p className="text-xs font-semibold text-slate-300 mb-1">
            Camera trực tiếp (OBS / DroidCam → Backend)
          </p>
          <p className="text-[11px] text-slate-500 mb-3">
            Cấu hình OBS/DroidCam phát RTSP/HTTP, sau đó dán URL vào đây. Backend sẽ đọc
            stream, chạy YOLO (~10 FPS), áp dụng luật 2/3 trong 30 frame (≈3s) và lưu
            lịch sử.
          </p>

          <label className="block text-xs mb-1">Tên nguồn (camera)</label>
          <input
            type="text"
            value={liveSourceName}
            onChange={(e) => setLiveSourceName(e.target.value)}
            className="w-full rounded-lg bg-slate-900 border border-slate-700 px-3 py-1.5 text-xs text-slate-100 focus:outline-none focus:ring-1 focus:ring-blue-500"
            placeholder="Ví dụ: Cổng chính công trường"
          />

          <label className="block text-xs mt-3 mb-1">
            URL stream từ OBS / DroidCam (RTSP / HTTP)
          </label>
          <input
            type="text"
            value={liveUrl}
            onChange={(e) => setLiveUrl(e.target.value)}
            className="w-full rounded-lg bg-slate-900 border border-slate-700 px-3 py-1.5 text-xs text-slate-100 focus:outline-none focus:ring-1 focus:ring-blue-500"
            placeholder="vd: rtsp://127.0.0.1:8554/live hoặc http://IP_DROIDCAM:4747/video"
          />
        </div>

        <div className="flex flex-wrap gap-2">
          <button
            className="px-3 py-1.5 bg-emerald-500 rounded-lg text-xs font-medium hover:bg-emerald-400 disabled:bg-slate-700 disabled:text-slate-400"
            disabled={isConnectingLive}
            onClick={handleStartLive}
          >
            {isConnectingLive ? 'Đang kết nối...' : 'Kết nối'}
          </button>
          <button
            className="px-3 py-1.5 bg-slate-800 rounded-lg text-xs hover:bg-slate-700 disabled:bg-slate-700 disabled:text-slate-400"
            disabled={!liveId}
            onClick={handleStopLive}
          >
            Ngắt kết nối
          </button>
        </div>

        <div className="text-[11px] text-slate-500 border-t border-slate-800 pt-3 space-y-1">
          <p>
            • Khi phát hiện hành vi vi phạm/nghi ngờ theo luật 2/3 trong 3s, cảnh báo sẽ
            bật ở khung bên phải và sự kiện được lưu vào tab Lịch sử.
          </p>
        </div>
      </div>
    );
  };

  // ===================== RENDER RIGHT PANEL =====================

  const renderRightPanel = () => {
    if (mode === 'image') {
      // dùng style giống video: border mỏng, màu RGBA
      const colorMap: Record<string, string> = {
        helmet: 'rgba(34,197,94,0.9)', // xanh
        head: 'rgba(248,113,113,0.95)', // đỏ
        'non-helmet': 'rgba(251,191,36,0.95)', // vàng
      };

      const getColor = (cls: string) => {
        const k = (cls || '').toLowerCase();
        if (k.includes('head')) return colorMap.head;
        if (k.includes('non')) return colorMap['non-helmet'];
        return colorMap.helmet;
      };

      const getBoxStyle = (b: ApiBox) => {
        if (
          typeof b.x === 'number' &&
          typeof b.y === 'number' &&
          typeof b.width === 'number' &&
          typeof b.height === 'number' &&
          b.width > 0 &&
          b.height > 0
        ) {
          return {
            left: b.x * 100,
            top: b.y * 100,
            width: b.width * 100,
            height: b.height * 100,
          };
        }
        return null;
      };

      return (
        <div className="space-y-4">
          <div className="bg-slate-950/80 rounded-2xl border border-slate-800 h-80 md:h-[360px] flex items-center justify-center text-sm text-slate-500 shadow-inner shadow-black/60 overflow-hidden">
            {imagePreviewUrl && imageBoxes.length > 0 ? (
              <div className="relative w-full h-full flex items-center justify-center">
                <div className="relative max-h-72 md:max-h-[340px] w-auto">
                  <img
                    src={imagePreviewUrl}
                    alt="Kết quả nhận diện"
                    className="max-h-72 md:max-h-[340px] w-auto block"
                  />
                  <div className="absolute inset-0 pointer-events-none">
                    {imageBoxes.map((b) => {
                      const style = getBoxStyle(b);
                      if (!style) return null;
                      const color = getColor(b.class_name || '');

                      return (
                        <div
                          key={b.id}
                          style={{
                            position: 'absolute',
                            left: `${style.left}%`,
                            top: `${style.top}%`,
                            width: `${style.width}%`,
                            height: `${style.height}%`,
                            border: `1.5px solid ${color}`,
                            boxShadow: '0 0 0 1px rgba(0,0,0,0.35)',
                          }}
                        >
                          <div
                            style={{
                              position: 'absolute',
                              left: 0,
                              top: 0,
                              transform: 'translateY(-100%)',
                              backgroundColor: 'rgba(15,23,42,0.9)',
                              color: 'white',
                              padding: '1px 4px',
                              fontSize: '10px',
                              borderRadius: '4px',
                              display: 'inline-flex',
                              alignItems: 'center',
                              gap: 4,
                            }}
                          >
                            <span>{b.class_name}</span>
                            {typeof b.confidence === 'number' && (
                              <span style={{ opacity: 0.7 }}>
                                {b.confidence.toFixed(2)}
                              </span>
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            ) : (
              <span>
                Ảnh kết quả (có bounding box) sẽ hiển thị tại đây sau khi bạn bấm{' '}
                <span className="font-semibold">“Nhận diện ảnh”</span>.
              </span>
            )}
          </div>

          {/* Ảnh crop từng người vi phạm / nghi ngờ */}
          <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-3 text-xs text-slate-300">
            <p className="font-semibold mb-2">Ảnh crop từng người vi phạm/nghi ngờ</p>
            {imageBoxes.length === 0 ? (
              <p className="text-slate-500 text-[11px]">
                Chưa có kết quả. Bấm “Nhận diện ảnh” để xem các đối tượng vi phạm/nghi
                ngờ.
              </p>
            ) : imageCropUrls.length === 0 ? (
              <p className="text-slate-500 text-[11px]">
                Đã nhận diện nhưng chưa tạo được crop (có thể không có đối tượng head /
                non-helmet hoặc ảnh đang load). Kiểm tra lại một ảnh khác có vi phạm để
                quan sát.
              </p>
            ) : (
              <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                {imageCropUrls.map((url, idx) => (
                  <div
                    key={idx}
                    className="bg-slate-900/80 border border-slate-800 rounded-lg p-1 flex items-center justify-center"
                  >
                    <img
                      src={url}
                      alt={`Crop ${idx + 1}`}
                      className="w-full h-24 object-contain rounded"
                    />
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      );
    }

    if (mode === 'video') {
      return (
        <div className="space-y-4">
          <div className="bg-slate-950/80 rounded-2xl border border-slate-800 h-80 md:h-[360px] flex items-center justify-center text-sm text-slate-500 shadow-inner shadow-black/60 overflow-hidden">
            {uploadedVideoId && uploadedVideoName ? (
              <img
                src={`${API_BASE}/api/stream/video?video_id=${uploadedVideoId}&file_name=${encodeURIComponent(
                  uploadedVideoName,
                )}`}
                alt="Video stream từ backend"
                className="w-full h-full object-contain"
              />
            ) : (
              <span>
                Chưa có video đang xử lý. Vui lòng chọn video và bấm{' '}
                <span className="font-semibold">“Upload & xử lý bằng backend”</span>.
              </span>
            )}
          </div>

          <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-3 text-xs">
            {videoAlert ? (
              <div className="flex items-start gap-2">
                <div
                  className={
                    'w-2 h-2 mt-1 rounded-full ' +
                    (videoAlert.type === 'VI_PHAM' ? 'bg-red-500' : 'bg-amber-400')
                  }
                />
                <div className="space-y-1">
                  <p className="font-semibold text-slate-100">
                    {videoAlert.type === 'VI_PHAM'
                      ? '⚠️ Phát hiện VI PHẠM trên video'
                      : '⚠️ Phát hiện NGHI NGỜ VI PHẠM trên video'}
                  </p>
                  <p className="text-slate-300">
                    Nguồn: <span className="font-mono">{videoAlert.source}</span>
                  </p>
                  <p className="text-slate-400">
                    Thời gian:{' '}
                    <span className="font-mono">{videoAlert.timestamp}</span>
                  </p>
                  <p className="text-slate-400">
                    Số đối tượng liên quan:{' '}
                    <span className="font-mono">
                      {videoAlert.numViolators ?? videoAlert.num_violators ?? '?'}
                    </span>
                  </p>
                  <p className="text-[11px] text-slate-500">
                    Chi tiết và ảnh global/crop xem trong tab{' '}
                    <span className="font-semibold">Lịch sử</span>.
                  </p>
                </div>
              </div>
            ) : (
              <p className="text-slate-400">
                Video hiển thị ở đây đã được backend gắn sẵn bounding box theo kết quả
                nhận diện. Trong quá trình xử lý, backend đồng thời áp dụng luật{' '}
                <span className="font-mono">2/3</span> trong{' '}
                <span className="font-mono">30 frame</span> (≈3s) để phát hiện VI PHẠM /
                NGHI NGỜ và tự động ghi vào Lịch sử. Khi có sự kiện mới, cảnh báo sẽ
                hiển thị trực tiếp tại đây.
              </p>
            )}
          </div>
        </div>
      );
    }

    // mode === 'live'
    return (
      <div className="space-y-4">
        <div className="bg-slate-950/80 rounded-2xl border border-slate-800 h-80 md:h-[360px] flex items-center justify-center text-sm text-slate-500 shadow-inner shadow-black/60 overflow-hidden">
          {liveId ? (
            <img
              src={`${API_BASE}/api/live/stream?live_id=${liveId}`}
              alt="Live stream từ OBS/DroidCam"
              className="w-full h-full object-contain"
            />
          ) : (
            <span>
              Chưa có live stream. Nhập URL từ OBS/DroidCam và bấm{' '}
              <span className="font-semibold">Kết nối</span>.
            </span>
          )}
        </div>

        <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-3 text-xs">
          {liveAlert ? (
            <div className="flex items-start gap-2">
              <div
                className={
                  'w-2 h-2 mt-1 rounded-full ' +
                  (liveAlert.type === 'VI_PHAM' ? 'bg-red-500' : 'bg-amber-400')
                }
              />
              <div className="space-y-1">
                <p className="font-semibold text-slate-100">
                  {liveAlert.type === 'VI_PHAM'
                    ? '⚠️ Live: Phát hiện VI PHẠM'
                    : '⚠️ Live: Phát hiện NGHI NGỜ VI PHẠM'}
                </p>
                <p className="text-slate-300">
                  Nguồn: <span className="font-mono">{liveAlert.source}</span>
                </p>
                <p className="text-slate-400">
                  Thời gian:{' '}
                  <span className="font-mono">{liveAlert.timestamp}</span>
                </p>
                <p className="text-slate-400">
                  Số đối tượng liên quan:{' '}
                  <span className="font-mono">
                    {liveAlert.numViolators ?? liveAlert.num_violators ?? '?'}
                  </span>
                </p>
                <p className="text-[11px] text-slate-500">
                  Bạn có thể mở tab <span className="font-semibold">Lịch sử</span> để xem
                  lại ảnh global/crop chi tiết.
                </p>
              </div>
            </div>
          ) : (
            <p className="text-slate-400">
              Backend đang đọc stream trực tiếp từ OBS/DroidCam, gắn bounding box theo
              YOLO. Nếu hành vi vi phạm/nghi ngờ xuất hiện ≥2/3 thời gian trong 3s, hệ
              thống sẽ tự động phát sinh sự kiện, lưu vào Lịch sử và bật cảnh báo tại
              đây.
            </p>
          )}
        </div>
      </div>
    );
  };

  // ===================== MAIN RETURN =====================

  return (
    <div className="w-full h-full flex flex-col gap-4">
      {/* Header: legend class */}
      <div className="flex flex-wrap items-center justify-between gap-3 text-xs">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1">
            <span className="inline-block w-3 h-3 rounded-full bg-green-400 border border-green-300" />
            <span className="text-[11px] text-slate-200 font-medium">helmet</span>
          </div>
          <div className="flex items-center gap-1">
            <span className="inline-block w-3 h-3 rounded-full bg-red-500 border border-red-400" />
            <span className="text-[11px] text-slate-200 font-medium">head</span>
          </div>
          <div className="flex items-center gap-1">
            <span className="inline-block w-3 h-3 rounded-full bg-amber-400 border border-amber-300" />
            <span className="text-[11px] text-slate-200 font-medium">non-helmet</span>
          </div>
        </div>
      </div>

      {/* Thanh chọn mode */}
      <div className="flex items-center gap-2 text-xs">
        <button
          onClick={() => setMode('image')}
          className={
            'px-3 py-1.5 rounded-full border text-xs font-medium ' +
            (mode === 'image'
              ? 'bg-emerald-500 border-emerald-400 text-white'
              : 'bg-slate-900 border-slate-700 text-slate-300 hover:bg-slate-800')
          }
        >
          Ảnh
        </button>
        <button
          onClick={() => setMode('video')}
          className={
            'px-3 py-1.5 rounded-full border text-xs font-medium ' +
            (mode === 'video'
              ? 'bg-sky-500 border-sky-400 text-white'
              : 'bg-slate-900 border-slate-700 text-slate-300 hover:bg-slate-800')
          }
        >
          Video
        </button>
        <button
          onClick={() => setMode('live')}
          className={
            'px-3 py-1.5 rounded-full border text-xs font-medium ' +
            (mode === 'live'
              ? 'bg-purple-500 border-purple-400 text-white'
              : 'bg-slate-900 border-slate-700 text-slate-300 hover:bg-slate-800')
          }
        >
          Camera trực tiếp
        </button>
      </div>

      {/* 2 cột chính */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 items-start">
        <div>{renderLeftPanel()}</div>
        <div>{renderRightPanel()}</div>
      </div>
    </div>
  );
};
