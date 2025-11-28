// src/components/tabs/UpdateTab.tsx
import React, { useMemo, useState } from 'react';
import { useAppStore } from '../../store/appStore';
import type { HistoryEvent } from '../../types';

const UPDATE_THRESHOLD = 100;

export const UpdateTab: React.FC = () => {
  const { historyEvents, updatePool, addToUpdatePool, clearUpdatePool } =
    useAppStore();

  const [currentIndex, setCurrentIndex] = useState(0);
  const [isTraining, setIsTraining] = useState(false);

  const candidates: HistoryEvent[] = useMemo(
    () => historyEvents,
    [historyEvents]
  );

  const current = candidates[currentIndex] ?? null;
  const progress = Math.min(updatePool.length, UPDATE_THRESHOLD);

  const handleMarkCorrect = () => {
    if (!current) return;
    addToUpdatePool(current.globalImageUrl);
    if (currentIndex < candidates.length - 1) {
      setCurrentIndex((i) => i + 1);
    }
  };

  const handleSkip = () => {
    if (currentIndex < candidates.length - 1) {
      setCurrentIndex((i) => i + 1);
    }
  };

  const handleFineTune = async () => {
    if (updatePool.length < UPDATE_THRESHOLD) return;
    // TODO: gọi API backend để:
    // 1) Export dataset từ Update Pool
    // 2) Chạy script fine-tune YOLOv8s với 100 ảnh này
    setIsTraining(true);
    try {
      // await api.startFineTune(updatePool);
      await new Promise((res) => setTimeout(res, 2000)); // fake
      alert(
        'Fine-tune hoàn tất (demo). Thực tế bạn sẽ load model mới yolov8s_ap_ft.pt từ server.'
      );
      clearUpdatePool();
    } catch (err) {
      console.error(err);
      alert('Có lỗi khi fine-tune.');
    } finally {
      setIsTraining(false);
    }
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-[minmax(0,1.3fr)_minmax(0,0.7fr)] gap-6">
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-4">
        <h2 className="text-sm font-semibold mb-3">
          Duyệt lại ảnh vi phạm / nghi ngờ
        </h2>

        {!current ? (
          <p className="text-xs text-slate-500">
            Không còn ảnh trong lịch sử để duyệt.
          </p>
        ) : (
          <div className="space-y-4 text-xs">
            <div className="flex items-center justify-between">
              <p className="text-slate-400">
                Ảnh {currentIndex + 1} / {candidates.length}
              </p>
              <p>
                Nguồn:{' '}
                <span className="font-medium text-slate-100">
                  {current.source}
                </span>
              </p>
            </div>

            <div className="bg-slate-950 border border-slate-800 rounded-xl overflow-hidden">
              {/* TODO: <img src={current.globalImageUrl} ... /> */}
              <div className="h-56 flex items-center justify-center text-slate-500 text-[11px]">
                Preview ảnh toàn cục (global frame từ lịch sử)
              </div>
            </div>

            <p className="font-medium text-sm">
              Hành vi vi phạm/nghi ngờ vi phạm được phát hiện có ĐÚNG không?
            </p>
            <div className="flex gap-3">
              <button
                className="px-4 py-2 rounded-lg text-sm font-medium bg-green-600 hover:bg-green-500"
                onClick={handleMarkCorrect}
              >
                ĐÚNG – thêm vào Update Pool
              </button>
              <button
                className="px-4 py-2 rounded-lg text-sm font-medium bg-red-600/80 hover:bg-red-500/80"
                onClick={handleSkip}
              >
                SAI / bỏ qua
              </button>
            </div>
          </div>
        )}
      </div>

      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-4 flex flex-col gap-4">
        <div>
          <h2 className="text-sm font-semibold mb-2">Update Data Pool</h2>
          <p className="text-xs text-slate-400 mb-2">
            Số ảnh đã xác nhận ĐÚNG: {updatePool.length}
          </p>
          <div className="h-2 bg-slate-800 rounded-full overflow-hidden mb-2">
            <div
              className="h-full bg-blue-500 transition-all"
              style={{ width: `${(progress / UPDATE_THRESHOLD) * 100}%` }}
            />
          </div>
          <p className="text-[11px] text-slate-500">
            Khi đạt đủ {UPDATE_THRESHOLD} ảnh, bạn có thể dùng để fine-tune mô hình
            hiện tại (pseudo-labeling).
          </p>
        </div>

        <div className="mt-auto">
          <button
            className={`w-full px-4 py-2 rounded-lg text-sm font-medium transition
              ${
                updatePool.length >= UPDATE_THRESHOLD && !isTraining
                  ? 'bg-blue-600 hover:bg-blue-500'
                  : 'bg-slate-700 cursor-not-allowed text-slate-300'
              }`}
            disabled={updatePool.length < UPDATE_THRESHOLD || isTraining}
            onClick={handleFineTune}
          >
            {isTraining
              ? 'Đang fine-tune mô hình...'
              : `Fine-tune mô hình với ${UPDATE_THRESHOLD} ảnh`}
          </button>
          <p className="text-[11px] text-slate-500 mt-2">
            Lưu ý: Quá trình fine-tune thực tế sẽ chạy ở backend (Python + YOLOv8),
            cần GPU để nhanh. Trong Khóa luận bạn có thể lựa chọn:
            tự động gọi training từ web, hoặc chỉ export dataset và train bằng
            notebook/server riêng.
          </p>
        </div>
      </div>
    </div>
  );
};
