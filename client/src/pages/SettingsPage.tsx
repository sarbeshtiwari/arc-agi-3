import React, { useState, useEffect } from 'react';
import { useTheme } from '../hooks/useTheme';
import { Video, Sun, Moon, Settings, Shield, Loader2 } from 'lucide-react';

function Toggle({ enabled, onToggle, disabled = false, color = 'blue' }) {
  const bgColor = {
    blue: enabled ? 'bg-blue-600' : 'bg-gray-300 dark:bg-gray-700',
    red: enabled ? 'bg-red-600' : 'bg-gray-300 dark:bg-gray-700',
    amber: enabled ? 'bg-amber-500' : 'bg-gray-300 dark:bg-gray-700',
  }[color] || (enabled ? 'bg-blue-600' : 'bg-gray-300 dark:bg-gray-700');

  return (
    <button
      onClick={onToggle}
      disabled={disabled}
      className={`relative w-11 h-6 rounded-full transition-colors ${bgColor} ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
    >
      <span className={`absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white shadow transition-transform ${enabled ? 'translate-x-5' : 'translate-x-0'}`} />
    </button>
  );
}

function SettingRow({ icon: Icon, iconColor, title, description, children }) {
  return (
    <div className="flex items-center justify-between py-4">
      <div className="flex items-center gap-4">
        <div className={`w-10 h-10 rounded-lg flex items-center justify-center bg-gray-100 dark:bg-gray-800 ${iconColor}`}>
          <Icon size={18} />
        </div>
        <div>
          <p className="text-sm font-medium text-gray-900 dark:text-white">{title}</p>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">{description}</p>
        </div>
      </div>
      <div className="flex items-center gap-3">
        {children}
      </div>
    </div>
  );
}

export default function SettingsPage() {
  const { theme, toggleTheme } = useTheme();
  const [recordingEnabled, setRecordingEnabled] = useState(true);
  const [loadingRecording, setLoadingRecording] = useState(true);
  const [togglingRecording, setTogglingRecording] = useState(false);

  useEffect(() => {
    fetch('/api/settings/recording_enabled')
      .then(r => r.json())
      .then(data => setRecordingEnabled(data.value === 'true'))
      .catch(() => {})
      .finally(() => setLoadingRecording(false));
  }, []);

  const toggleRecording = async () => {
    const newVal = !recordingEnabled;
    setRecordingEnabled(newVal);
    setTogglingRecording(true);
    try {
      const res = await fetch('/api/settings/recording_enabled', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${localStorage.getItem('arc_token')}` },
        body: JSON.stringify({ value: String(newVal) }),
      });
      const data = await res.json();
      setRecordingEnabled(data.value === 'true');
    } catch {
      setRecordingEnabled(!newVal);
    } finally {
      setTogglingRecording(false);
    }
  };

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Settings</h1>
        <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">Platform-wide configuration</p>
      </div>

      <div className="max-w-2xl">
        <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800">
          <div className="px-5 py-3 border-b border-gray-200 dark:border-gray-800">
            <h2 className="text-sm font-semibold text-gray-900 dark:text-white">General</h2>
          </div>
          <div className="px-5 divide-y divide-gray-100 dark:divide-gray-800/50">
            <SettingRow
              icon={Video}
              iconColor={recordingEnabled ? 'text-red-400' : 'text-gray-400'}
              title="Video Recording"
              description="Allow players to record gameplay sessions from the play page"
            >
              {loadingRecording ? (
                <Loader2 size={16} className="animate-spin text-gray-400" />
              ) : (
                <>
                  <span className={`text-[11px] font-mono ${recordingEnabled ? 'text-red-400' : 'text-gray-500'}`}>
                    {recordingEnabled ? 'ON' : 'OFF'}
                  </span>
                  <Toggle enabled={recordingEnabled} onToggle={toggleRecording} disabled={togglingRecording} color="red" />
                </>
              )}
            </SettingRow>

            <SettingRow
              icon={theme === 'dark' ? Moon : Sun}
              iconColor={theme === 'dark' ? 'text-blue-400' : 'text-amber-500'}
              title="Theme"
              description="Switch between light and dark mode for the admin panel"
            >
              <span className={`text-[11px] font-mono ${theme === 'dark' ? 'text-blue-400' : 'text-amber-500'}`}>
                {theme === 'dark' ? 'DARK' : 'LIGHT'}
              </span>
              <Toggle enabled={theme === 'dark'} onToggle={toggleTheme} color={theme === 'dark' ? 'blue' : 'amber'} />
            </SettingRow>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 mt-6">
          <div className="px-5 py-3 border-b border-gray-200 dark:border-gray-800">
            <h2 className="text-sm font-semibold text-gray-900 dark:text-white">System</h2>
          </div>
          <div className="px-5 divide-y divide-gray-100 dark:divide-gray-800/50">
            <SettingRow
              icon={Shield}
              iconColor="text-gray-400"
              title="Admin Access"
              description="Platform version and system information"
            >
              <span className="text-xs text-gray-500 font-mono">v1.0.0</span>
            </SettingRow>
          </div>
        </div>
      </div>
    </div>
  );
}
