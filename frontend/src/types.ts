export type AppTab = 'detection' | 'history' | 'update';
export type DetectionMode = 'image' | 'video' | 'live';
export type DetectionClass = 'helmet' | 'head' | 'non-helmet';
export type ViolationType = 'VI_PHAM' | 'NGHI_NGO';

export interface DetectionBox {
  id: string;
  className: DetectionClass;
  confidence: number;
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface HistoryEvent {
  id: string;
  timestamp: string;
  source: string;
  type: ViolationType;
  globalImageUrl: string;
  cropImageUrls: string[];
  numViolators: number;
}
