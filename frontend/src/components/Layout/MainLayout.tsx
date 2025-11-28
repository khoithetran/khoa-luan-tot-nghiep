import React from 'react';
import { useAppStore } from '../../store/appStore';
import type { AppTab } from '../../types';

const tabLabels: Record<AppTab, string> = {
  detection: 'Nhận diện',
  history: 'Lịch sử',
  update: 'Cập nhật mô hình',
};

export const MainLayout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { currentTab, setCurrentTab } = useAppStore();

  const renderTabButton = (tab: AppTab) => (
    <button
      key={tab}
      onClick={() => setCurrentTab(tab)}
      className={`px-4 py-2 rounded-full text-sm font-medium transition
        ${
          currentTab === tab
            ? 'bg-blue-500 text-white shadow-lg shadow-blue-500/30'
            : 'bg-slate-800/70 text-slate-200 hover:bg-slate-700/90'
        }`}
    >
      {tabLabels[tab]}
    </button>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-slate-100">
      <div className="max-w-6xl mx-auto px-4 md:px-6 lg:px-8 py-6 md:py-8">
        {/* Header */}
        <header className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-6 md:mb-8">
          <div>
            <div className="inline-flex items-center gap-2 px-2 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/40 mb-2">
              <span className="inline-block h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
              <span className="text-[11px] uppercase tracking-wide text-emerald-300">
                PPE Safety – Graduation Project
              </span>
            </div>
            <h1 className="text-2xl md:text-3xl font-semibold tracking-tight">
              Ứng dụng Nhận diện Vi phạm An toàn Lao động
            </h1>
            <p className="text-xs md:text-sm text-slate-400 mt-1">
              Model: <span className="font-mono text-slate-200">yolov8s_ap.pt</span> ·
              lớp: <span className="font-mono">helmet / head / non-helmet</span>
            </p>
          </div>
          <nav className="flex items-center gap-2 bg-slate-900/60 border border-slate-800 rounded-full px-2 py-1 shadow-lg shadow-black/30">
            {(['detection', 'history', 'update'] as AppTab[]).map(renderTabButton)}
          </nav>
        </header>

        {/* Nội dung chính */}
        <main className="rounded-3xl border border-slate-800/80 bg-slate-950/60 shadow-2xl shadow-black/40 p-4 md:p-6">
          {children}
        </main>
      </div>
    </div>
  );
};
