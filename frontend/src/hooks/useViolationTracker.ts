// src/hooks/useViolationTracker.ts
import { useState, useCallback, useRef } from 'react';
import type { ViolationType } from '../types';

interface FrameSummary {
  hasHead: boolean;
  hasNonHelmet: boolean;
}

interface ViolationState {
  status: 'NORMAL' | 'VI_PHAM' | 'NGHI_NGO';
  headCountInWindow: number;
  windowSize: number;
  lastTriggeredAt?: string;
}

const FPS = 10;
const WINDOW_SEC = 4;
const WINDOW_SIZE = FPS * WINDOW_SEC; // 40
const MIN_HEAD_FRAMES = 30;

export function useViolationTracker(onViolation?: (type: ViolationType) => void) {
  // lưu 40 frame gần nhất trong ref (không cần trigger re-render khi thay đổi)
  const headFlagsRef = useRef<number[]>([]);

  const [state, setState] = useState<ViolationState>({
    status: 'NORMAL',
    headCountInWindow: 0,
    windowSize: WINDOW_SIZE,
    lastTriggeredAt: undefined,
  });

  const pushFrame = useCallback(
    (summary: FrameSummary) => {
      const { hasHead, hasNonHelmet } = summary;

      // cập nhật mảng headFlags trong ref
      const prevFlags = headFlagsRef.current;
      const updated = [...prevFlags, hasHead ? 1 : 0];
      if (updated.length > WINDOW_SIZE) updated.shift();
      headFlagsRef.current = updated;

      const headCount = updated.reduce((a, b) => a + b, 0);
      const now = new Date().toISOString();

      setState((old) => {
        let status = old.status;
        let lastTriggeredAt = old.lastTriggeredAt;

        // NGHI_NGO (non-helmet) -> kích hoạt ngay
        if (hasNonHelmet) {
          status = 'NGHI_NGO';
          lastTriggeredAt = now;
          onViolation && onViolation('NGHI_NGO');
        }

        // VI_PHAM (head >= 30 frame trong 4s)
        if (updated.length === WINDOW_SIZE && headCount >= MIN_HEAD_FRAMES) {
          status = 'VI_PHAM';
          lastTriggeredAt = now;
          onViolation && onViolation('VI_PHAM');
        }

        return {
          ...old,
          status,
          headCountInWindow: headCount,
          lastTriggeredAt,
        };
      });
    },
    [onViolation]
  );

  const reset = useCallback(() => {
    headFlagsRef.current = [];
    setState({
      status: 'NORMAL',
      headCountInWindow: 0,
      windowSize: WINDOW_SIZE,
      lastTriggeredAt: undefined,
    });
  }, []);

  return { state, pushFrame, reset };
}
