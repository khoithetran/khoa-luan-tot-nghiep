// src/components/tabs/UpdateTab.tsx
import React, { useEffect, useState } from 'react';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

type HistoryEvent = {
  id: string;
  timestamp: string;
  source: string;
  event_type: 'VI_PHAM' | 'NGHI_NGO';
  global_image_url?: string;
  globalImageUrl?: string;
  num_violators?: number;
  numViolators?: number;
};

type CandidatesResponse = {
  total: number;
  page: number;
  page_size: number;
  items: HistoryEvent[];
};

type ApiBox = {
  id: string;
  class_name: string;
  class_id?: number;
  confidence: number;
  x: number;       // normalized top-left (0–1)
  y: number;
  width: number;   // normalized size (0–1)
  height: number;
  xc?: number;     // normalized center
  yc?: number;
};

export const UpdateTab: React.FC = () => {
  const [candidates, setCandidates] = useState<HistoryEvent[]>([]);
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);
  const [pageSize] = useState(1); // xem từng event một

  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState<{ num_images: number; threshold: number; ready: boolean } | null>(null);

  // auto-label cho event hiện tại
  const [autoImageUrl, setAutoImageUrl] = useState<string | null>(null);
  const [autoBoxes, setAutoBoxes] = useState<ApiBox[]>([]);
  const [classCounts, setClassCounts] = useState<Record<string, number>>({});
  const [autoLoading, setAutoLoading] = useState(false);

  const toAbs = (p?: string) => {
    if (!p) return '';
    if (p.startsWith('http://') || p.startsWith('https://')) return p;
    return `${API_BASE}${p}`;
  };

  const fetchStatus = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/update/status`);
      if (!res.ok) return;
      const data = await res.json();
      setStatus(data);
    } catch (e) {
      console.error(e);
    }
  };

  const fetchCandidates = async (pageNum: number) => {
    try {
      setLoading(true);
      const res = await fetch(
        `${API_BASE}/api/update/candidates?page=${pageNum}&page_size=${pageSize}`,
      );
      if (!res.ok) {
        console.error('Lỗi load candidates:', await res.text());
        return;
      }
      const data: CandidatesResponse = await res.json();
      setCandidates(data.items);
      setTotal(data.total);
      setPage(data.page);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const fetchAutoLabel = async (eventId: string) => {
    try {
      setAutoLoading(true);
      setAutoBoxes([]);
      setClassCounts({});
      setAutoImageUrl(null);

      const res = await fetch(`${API_BASE}/api/update/auto-label/${eventId}`);
      if (!res.ok) {
        console.error('Lỗi auto-label:', await res.text());
        return;
      }
      const data = await res.json();
      setAutoBoxes(data.boxes || []);
      setClassCounts(data.class_counts || {});
      setAutoImageUrl(toAbs(data.image_url));
    } catch (err) {
      console.error(err);
    } finally {
      setAutoLoading(false);
    }
  };

  useEffect(() => {
    fetchCandidates(1);
    fetchStatus();
  }, []);

  // mỗi khi current event thay đổi → chạy auto-label
  const current = candidates[0];

  useEffect(() => {
    if (current?.id) {
      fetchAutoLabel(current.id);
    } else {
      setAutoBoxes([]);
      setClassCounts({});
      setAutoImageUrl(null);
    }
  }, [current?.id]);

  const handleMark = async (accepted: boolean) => {
    if (!current) return;
    try {
      const res = await fetch(`${API_BASE}/api/update/mark`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ event_id: current.id, accepted }),
      });
      if (!res.ok) {
        console.error('Lỗi mark update:', await res.text());
        alert('Backend báo lỗi khi ghi kết quả ĐÚNG/SAI.');
        return;
      }
      // load event tiếp theo + cập nhật status
      await fetchCandidates(page + 1);
      await fetchStatus();
    } catch (err) {
      console.error(err);
      alert('Có lỗi khi gửi ĐÚNG/SAI.');
    }
  };

  // ====== RENDER BBOX OVERLAY ======
  const renderBoxesOverlay = (boxes: ApiBox[]) => {
    const colorMap: Record<string, string> = {
      helmet: 'rgba(34,197,94,0.9)',
      head: 'rgba(248,113,113,0.95)',
      'non-helmet': 'rgba(251,191,36,0.95)',
    };

    const getColor = (cls: string) => {
      const k = (cls || '').toLowerCase();
      if (k.includes('head')) return colorMap.head;
      if (k.includes('non')) return colorMap['non-helmet'];
      return colorMap.helmet;
    };

    return (
      <div className="absolute inset-0 pointer-events-none">
        {boxes.map((b) => {
          if (
            typeof b.x !== 'number' ||
            typeof b.y !== 'number' ||
            typeof b.width !== 'number' ||
            typeof b.height !== 'number' ||
            b.width <= 0 ||
            b.height <= 0
          ) {
            return null;
          }

          const color = getColor(b.class_name || '');
          const left = b.x * 100;
          const top = b.y * 100;
          const width = b.width * 100;
          const height = b.height * 100;

          return (
            <div
              key={b.id}
              style={{
                position: 'absolute',
                left: `${left}%`,
                top: `${top}%`,
                width: `${width}%`,
                height: `${height}%`,
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
                  <span style={{ opacity: 0.7 }}>{b.confidence.toFixed(2)}</span>
                )}
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  const helmetCount = classCounts['helmet'] ?? 0;
  const headCount = classCounts['head'] ?? 0;
  const nonHelmetCount = classCounts['non-helmet'] ?? classCounts['non_helmet'] ?? 0;

  return (
    <div className="w-full h-full flex flex-col gap-4">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 text-xs">
        <div>
          <p className="font-semibold text-slate-100">
            Cập nhật mô hình từ dữ liệu thực tế (Update Data Pool)
          </p>
          <p className="text-[11px] text-slate-400">
            Hệ thống tự dùng mô hình YOLOv8s hiện tại để gán nhãn 3 lớp (helmet / head /
            non-helmet) trên các khung hình VI_PHẠM / NGHI_NGỜ. Người dùng chỉ cần kiểm
            tra lại và xác nhận ĐÚNG/S
            AI, các ảnh + nhãn sẽ được lưu vào <code>update_pool/</code> để fine-tune
            sau này.
          </p>
        </div>
        {status && (
          <div className="text-[11px] text-slate-300 bg-slate-900/80 border border-slate-700 rounded-lg px-3 py-1.5">
            Đã chấp nhận:{' '}
            <span className="font-mono font-semibold">
              {status.num_images} / {status.threshold}
            </span>{' '}
            ảnh
            {status.ready && (
              <span className="ml-1 text-emerald-400 font-semibold">
                (ĐÃ ĐỦ để fine-tune)
              </span>
            )}
          </div>
        )}
      </div>

      {/* Nội dung chính */}
      <div className="grid grid-cols-1 md:grid-cols-[minmax(0,1.1fr)_minmax(0,0.9fr)] gap-4 items-start">
        {/* Ảnh + bbox */}
        <div className="bg-slate-950/80 border border-slate-800 rounded-2xl p-4 md:p-5 shadow-xl shadow-black/40">
          {loading && (
            <p className="text-xs text-slate-400 mb-2">Đang tải danh sách sự kiện...</p>
          )}

          {!current ? (
            <p className="text-xs text-slate-400">
              Hiện chưa có sự kiện VI_PHẠM / NGHI_NGỜ nào hoặc bạn đã duyệt hết. Hãy để
              hệ thống chạy nhận diện thêm để thu thập dữ liệu mới.
            </p>
          ) : (
            <>
              <div className="flex items-center justify-between mb-3 text-[11px] text-slate-400">
                <div>
                  <p>
                    Nguồn:{' '}
                    <span className="font-mono text-slate-200">
                      {current.source}
                    </span>
                  </p>
                  <p>
                    Thời gian:{' '}
                    <span className="font-mono">{current.timestamp}</span>
                  </p>
                  <p>
                    Loại sự kiện:{' '}
                    <span
                      className={
                        current.event_type === 'VI_PHAM'
                          ? 'text-red-400 font-semibold'
                          : 'text-amber-300 font-semibold'
                      }
                    >
                      {current.event_type === 'VI_PHAM'
                        ? 'VI PHẠM'
                        : 'NGHI NGỜ VI PHẠM'}
                    </span>
                  </p>
                  <p>
                    Số đối tượng liên quan (history):{' '}
                    <span className="font-mono">
                      {current.numViolators ?? current.num_violators ?? '?'}
                    </span>
                  </p>
                </div>
                <div className="text-right">
                  <p>
                    Tổng candidate:{' '}
                    <span className="font-mono">{total}</span>
                  </p>
                  <p>
                    Đang xem:{' '}
                    <span className="font-mono">{page}</span>
                  </p>
                </div>
              </div>

              <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-2 flex items-center justify-center min-h-[220px] relative overflow-hidden">
                {autoLoading && (
                  <div className="absolute inset-0 flex items-center justify-center text-[11px] text-slate-400 bg-black/30">
                    Đang auto-label bằng YOLO...
                  </div>
                )}
                {autoImageUrl ? (
                  <div className="relative max-h-80 w-auto">
                    <img
                      src={autoImageUrl}
                      alt="Ảnh vi phạm"
                      className="max-h-80 w-auto object-contain rounded-lg"
                    />
                    {renderBoxesOverlay(autoBoxes)}
                  </div>
                ) : (
                  <span className="text-xs text-slate-500">
                    Không tải được ảnh global cho sự kiện này.
                  </span>
                )}
              </div>
            </>
          )}
        </div>

        {/* Câu hỏi ĐÚNG/SAI + đếm class */}
        <div className="bg-slate-950/80 border border-slate-800 rounded-2xl p-4 md:p-5 shadow-xl shadow-black/40 space-y-4 text-xs text-slate-300">
          <p className="font-semibold">
            Hành vi/nhãn được mô hình gán cho khung hình này có{' '}
            <span className="underline">ĐÚNG</span> không?
          </p>

          <div className="text-[11px] text-slate-400 space-y-1">
            <p>
              • Bảng dưới đây cho biết mô hình hiện đang gán:{' '}
              <span className="font-mono">helmet / head / non-helmet</span> bao nhiêu
              box.
            </p>
            <p>
              • Nếu thấy hợp lý (heatmap, số lượng class cân đối với mắt thường), bạn
              có thể chọn{' '}
              <span className="font-semibold text-emerald-400">
                ĐÚNG – thêm vào Update Pool
              </span>
              .
            </p>
            <p>
              • Nếu mô hình gán nhãn sai quá nhiều (ví dụ helmet bị gán thành head), hãy
              chọn{' '}
              <span className="font-semibold text-red-400">
                SAI – bỏ qua event này
              </span>
              .
            </p>
          </div>

          {/* Đếm box từng class */}
          <div className="bg-slate-900/80 border border-slate-800 rounded-xl px-3 py-2 text-[11px]">
            <p className="font-semibold text-slate-100 mb-1">
              Thống kê box theo class (auto-label)
            </p>
            <div className="flex flex-wrap gap-3">
              <div className="flex items-center gap-1">
                <span className="inline-block w-2.5 h-2.5 rounded-full bg-green-400 border border-green-300" />
                <span>helmet:</span>
                <span className="font-mono font-semibold text-slate-100">
                  {helmetCount}
                </span>
              </div>
              <div className="flex items-center gap-1">
                <span className="inline-block w-2.5 h-2.5 rounded-full bg-red-500 border border-red-400" />
                <span>head:</span>
                <span className="font-mono font-semibold text-slate-100">
                  {headCount}
                </span>
              </div>
              <div className="flex items-center gap-1">
                <span className="inline-block w-2.5 h-2.5 rounded-full bg-amber-400 border border-amber-300" />
                <span>non-helmet:</span>
                <span className="font-mono font-semibold text-slate-100">
                  {nonHelmetCount}
                </span>
              </div>
            </div>
          </div>

          {/* Nút ĐÚNG / SAI */}
          <div className="flex gap-2">
            <button
              onClick={() => handleMark(true)}
              disabled={!current}
              className="px-4 py-1.5 rounded-lg bg-emerald-500 hover:bg-emerald-400 text-xs font-semibold disabled:bg-slate-700 disabled:text-slate-500"
            >
              ✅ ĐÚNG – Lưu ảnh + nhãn vào Update Pool
            </button>
            <button
              onClick={() => handleMark(false)}
              disabled={!current}
              className="px-4 py-1.5 rounded-lg bg-red-500 hover:bg-red-400 text-xs font-semibold disabled:bg-slate-700 disabled:text-slate-500"
            >
              ❌ SAI – bỏ qua event này
            </button>
          </div>

          {status && status.ready ? (
            <div className="mt-3 text-[11px] text-emerald-300 bg-emerald-500/10 border border-emerald-500/40 rounded-lg px-3 py-2">
              ✅ Đã đủ{' '}
              <span className="font-mono">{status.num_images}</span> ảnh trong Update
              Data Pool. Bạn có thể:
              <br />
              1) Đem thư mục <code>update_pool/images, update_pool/labels</code> lên
              Kaggle/Colab để fine-tune YOLOv8s_ap.pt.
              <br />
              2) Mô tả quy trình này trong khóa luận như một vòng lặp cải thiện mô hình
              dựa trên dữ liệu thực tế.
            </div>
          ) : (
            <p className="text-[11px] text-slate-500">
              Hệ thống sẽ tiếp tục thu thập sự kiện mới từ tab{' '}
              <span className="font-semibold">Nhận diện</span>. Bạn có thể quay lại tab
              này bất kỳ lúc nào để duyệt thêm dữ liệu và xây dựng tập Update Pool lớn
              hơn.
            </p>
          )}
        </div>
      </div>
    </div>
  );
};
