// src/store/appStore.ts
import { create } from 'zustand';
import type { AppTab, HistoryEvent } from '../types';

interface AppState {
  currentTab: AppTab;
  setCurrentTab: (tab: AppTab) => void;

  historyEvents: HistoryEvent[];
  addHistoryEvent: (event: HistoryEvent) => void;
  setHistoryEvents: (events: HistoryEvent[]) => void;  // 👈 thêm dòng này

  updatePool: string[];
  addToUpdatePool: (imageUrl: string) => void;
  removeFromUpdatePool: (imageUrl: string) => void;
  clearUpdatePool: () => void;
}

export const useAppStore = create<AppState>((set) => ({
  currentTab: 'detection',
  setCurrentTab: (tab) => set({ currentTab: tab }),

  historyEvents: [],
  addHistoryEvent: (event) =>
    set((state) => ({
      historyEvents: [event, ...state.historyEvents],
    })),
  setHistoryEvents: (events) => set({ historyEvents: events }),   // 👈 thêm

  updatePool: [],
  addToUpdatePool: (imageUrl) =>
    set((state) => ({
      updatePool: state.updatePool.includes(imageUrl)
        ? state.updatePool
        : [...state.updatePool, imageUrl],
    })),
  removeFromUpdatePool: (imageUrl) =>
    set((state) => ({
      updatePool: state.updatePool.filter((url) => url !== imageUrl),
    })),
  clearUpdatePool: () => set({ updatePool: [] }),
}));

