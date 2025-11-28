// src/App.tsx
import React from 'react';
import { MainLayout } from './components/Layout/MainLayout';
import { DetectionTab } from './components/tabs/DetectionTab';
import { HistoryTab } from './components/tabs/HistoryTab';
import { UpdateTab } from './components/tabs/UpdateTab';
import { useAppStore } from './store/appStore';

const App: React.FC = () => {
  const { currentTab } = useAppStore();

  const renderTab = () => {
    switch (currentTab) {
      case 'detection':
        return <DetectionTab />;
      case 'history':
        return <HistoryTab />;
      case 'update':
        return <UpdateTab />;
      default:
        return null;
    }
  };

  return <MainLayout>{renderTab()}</MainLayout>;
};

export default App;
