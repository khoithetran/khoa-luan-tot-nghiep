// src/components/tabs/HistoryTab.tsx
import React, { useEffect, useState } from 'react';
import { useAppStore } from '../../store/appStore';
import type { HistoryEvent, ViolationType } from '../../types';

const API_BASE = 'http://localhost:8000';

export const HistoryTab: React.FC = () => {
  const { historyEvents, setHistoryEvents } = useAppStore();
  const [selected, setSelected] = useState<HistoryEvent | null>(null);
  const [filterType, setFilterType] = useState<'ALL' | ViolationType>('ALL');
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);

  const fetchHistory = async () => {
    try {
      setLoading(true);
      setLoadError(null);

      const res = await fetch(`${API_BASE}/api/history`);
      if (!res.ok) {
        const text = await res.text();
        console.error('Lỗi load history backend:', text);
        setLoadError('Không load được lịch sử từ backend.');
        return;
      }

      const data = await res.json();
      console.log('History từ backend:', data);

      // Map snake_case (backend) -> camelCase (frontend)
      const mapped: HistoryEvent[] = (data || []).map((e: any) => ({
        id: e.id,
        timestamp: e.timestamp,
        source: e.source,
        type: e.type,
        globalImageUrl: e.global_image_url,
        cropImageUrls: e.crop_image_urls ?? [],
        numViolators:
          e.num_violators ??
          (Array.isArray(e.crop_image_urls) ? e.crop_image_urls.length : 0),
      }));

      setHistoryEvents(mapped);
      // Nếu đang chọn event mà không còn thì bỏ chọn
      if (selected && !mapped.find((ev) => ev.id === selected.id)) {
        setSelected(null);
      }
    } catch (err) {
      console.error(err);
      setLoadError('Có lỗi khi gọi API lịch sử.');
    } finally {
      setLoading(false);
    }
  };

  // Gọi lần đầu khi mở tab
  useEffect(() => {
    fetchHistory();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const filtered = historyEvents.filter((e) =>
    filterType === 'ALL' ? true : e.type === filterType
  );

  const handleSelect = (event: HistoryEvent) => {
    setSelected(event);
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)] gap-6">
      {/* Bảng lịch sử */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-4">
        <div className="flex items-center justify-between mb-3 gap-2">
          <div>
            <h2 className="text-sm font-semibold">Lịch sử vi phạm</h2>
            <p className="text-[11px] text-slate-500">
              Dữ liệu lấy từ backend FastAPI (/api/history).
            </p>
          </div>
          <div className="flex items-center gap-2">
            <select
              value={filterType}
              onChange={(e) => setFilterType(e.target.value as 'ALL' | ViolationType)}
              className="bg-slate-950 border border-slate-700 rounded-lg px-2 py-1 text-xs"
            >
              <option value="ALL">Tất cả</option>
              <option value="VI_PHAM">VI PHẠM</option>
              <option value="NGHI_NGO">NGHI NGỜ</option>
            </select>
            <button
              onClick={fetchHistory}
              className="px-3 py-1 text-[11px] rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200"
              disabled={loading}
            >
              {loading ? 'Đang tải...' : 'Làm mới'}
            </button>
          </div>
        </div>

        {loadError && (
          <p className="text-[11px] text-red-400 mb-2">{loadError}</p>
        )}

        <div className="max-h-[420px] overflow-auto text-xs">
          <table className="w-full border-collapse">
            <thead className="sticky top-0 bg-slate-950">
              <tr className="text-slate-400 border-b border-slate-800">
                <th className="text-left py-2 pr-2">Thời gian</th>
                <th className="text-left py-2 pr-2">Nguồn</th>
                <th className="text-left py-2 pr-2">Loại</th>
                <th className="text-right py-2">Số người</th>
              </tr>
            </thead>
            <tbody>
              {filtered.length === 0 && !loading && (
                <tr>
                  <td
                    colSpan={4}
                    className="text-center py-4 text-slate-500 text-xs"
                  >
                    Chưa có sự kiện nào (hoặc chưa có ảnh vi phạm).
                  </td>
                </tr>
              )}

              {filtered.map((e) => (
                <tr
                  key={e.id}
                  className={`border-b border-slate-800 cursor-pointer hover:bg-slate-900/80 ${
                    selected?.id === e.id ? 'bg-slate-900' : ''
                  }`}
                  onClick={() => handleSelect(e)}
                >
                  <td className="py-2 pr-2 align-top">
                    {new Date(e.timestamp).toLocaleString()}
                  </td>
                  <td className="py-2 pr-2 align-top">{e.source}</td>
                  <td className="py-2 pr-2 align-top">
                    <span
                      className={`px-2 py-0.5 rounded-full text-[10px] font-semibold
                        ${
                          e.type === 'VI_PHAM'
                            ? 'bg-red-500/20 text-red-400'
                            : 'bg-amber-400/20 text-amber-400'
                        }`}
                    >
                      {e.type === 'VI_PHAM' ? 'VI PHẠM' : 'NGHI NGỜ'}
                    </span>
                  </td>
                  <td className="py-2 text-right align-top">{e.numViolators}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Panel chi tiết */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-4">
        <h2 className="text-sm font-semibold mb-3">Chi tiết sự kiện</h2>

        {!selected ? (
          <p className="text-xs text-slate-500">
            Chọn 1 dòng ở bảng bên trái để xem chi tiết.
          </p>
        ) : (
          <div className="space-y-3 text-xs">
            <div>
              <p className="text-slate-400">Thời gian</p>
              <p className="font-medium">
                {new Date(selected.timestamp).toLocaleString()}
              </p>
            </div>

            <div>
              <p className="text-slate-400">Nguồn</p>
              <p className="font-medium">{selected.source}</p>
            </div>

            <div>
              <p className="text-slate-400">Loại sự kiện</p>
              <p className="font-medium">
                {selected.type === 'VI_PHAM' ? 'VI PHẠM' : 'NGHI NGỜ VI PHẠM'}
              </p>
            </div>

            <div>
              <p className="text-slate-400 mb-1">Ảnh toàn cục</p>
              <div className="bg-slate-950 border border-slate-800 rounded-xl overflow-hidden">
                {selected.globalImageUrl ? (
                  <img
                    src={`${API_BASE}${selected.globalImageUrl}`}
                    alt="global"
                    className="w-full h-40 object-contain bg-black"
                  />
                ) : (
                  <div className="h-40 flex items-center justify-center text-slate-500 text-[11px]">
                    Không có ảnh toàn cục.
                  </div>
                )}
              </div>
            </div>

            <div>
              <p className="text-slate-400 mb-1">
                Ảnh crop từng người vi phạm ({selected.cropImageUrls.length})
              </p>
              <div className="grid grid-cols-3 gap-2">
                {selected.cropImageUrls.map((url, idx) => (
                  <div
                    key={idx}
                    className="bg-slate-950 border border-slate-800 rounded-lg overflow-hidden h-20"
                  >
                    <img
                      src={`${API_BASE}${url}`}
                      alt={`crop ${idx + 1}`}
                      className="w-full h-full object-cover"
                    />
                  </div>
                ))}
                {selected.cropImageUrls.length === 0 && (
                  <p className="text-[11px] text-slate-500">
                    Chưa lưu crop cho sự kiện này.
                  </p>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
