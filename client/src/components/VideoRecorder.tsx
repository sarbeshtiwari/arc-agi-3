import React, { useRef, useState, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { useVideoRecorder } from '../hooks/useVideoRecorder';
import { gamesAPI } from '../api/client';
import {
  Video, Square, Download, Trash2, X, Circle, CheckCircle, VideoOff
} from 'lucide-react';

interface VideoRecorderProps {
  gameId: string;
  isEphemeral?: boolean;
  playerName?: string;
  isGameOver?: boolean;
  isPlaying?: boolean;
  onPromptDismissed?: () => void;
}

function fmt(s: number): string {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, '0')}`;
}

function fmtSize(bytes: number): string {
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(0) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

export default function VideoRecorder({
  gameId, isEphemeral = false, playerName, isGameOver = false, isPlaying = false,
  onPromptDismissed,
}: VideoRecorderProps) {
  const {
    isRecording, recordingTime, hasRecording, serverFilename, fileExt,
    startRecording, stopRecording, clearRecording,
  } = useVideoRecorder();

  const [showPrompt, setShowPrompt] = useState(false);
  const [promptDismissed, setPromptDismissed] = useState(false);
  const [recordingEnabled, setRecordingEnabled] = useState<boolean | null>(null);
  const autoStoppedRef = useRef(false);

  // Check if recording is enabled globally
  useEffect(() => {
    fetch('/api/settings/recording_enabled')
      .then(r => r.json())
      .then(data => setRecordingEnabled(data.value === 'true'))
      .catch(() => setRecordingEnabled(true));
  }, []);

  // Show record prompt when game starts AND setting is loaded
  // If recording disabled, skip the prompt and start the timer immediately
  useEffect(() => {
    if (!isPlaying || promptDismissed) return;
    if (recordingEnabled === null) return; // still loading setting

    if (recordingEnabled && !isEphemeral) {
      // Recording is ON — show the prompt
      setShowPrompt(true);
    } else {
      // Recording is OFF or ephemeral — skip prompt, start timer
      setPromptDismissed(true);
      onPromptDismissed?.();
    }
  }, [isPlaying, recordingEnabled]);

  // Auto-stop on game end
  useEffect(() => {
    if (isGameOver && isRecording && !autoStoppedRef.current) {
      autoStoppedRef.current = true;
      setTimeout(() => stopRecording(), 1500);
    }
    if (!isGameOver) autoStoppedRef.current = false;
  }, [isGameOver, isRecording, stopRecording]);

  const handlePromptRecord = async () => {
    setShowPrompt(false);
    setPromptDismissed(true);
    onPromptDismissed?.();
    autoStoppedRef.current = false;
    if (!isEphemeral) {
      await startRecording(gameId, playerName);
    }
  };

  const handlePromptSkip = () => {
    setShowPrompt(false);
    setPromptDismissed(true);
    onPromptDismissed?.();
  };

  const handleManualStart = async () => {
    autoStoppedRef.current = false;
    if (!isEphemeral) {
      await startRecording(gameId, playerName);
    }
  };

  const handleStop = () => stopRecording();

  const handleDownload = () => {
    if (serverFilename) {
      const url = gamesAPI.getVideoUrl(gameId, serverFilename);
      const a = document.createElement('a');
      a.href = url;
      a.download = serverFilename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
  };

  const handleDelete = async () => {
    if (serverFilename) {
      try { await gamesAPI.deleteVideo(gameId, serverFilename); } catch {}
    }
    clearRecording();
  };

  return (
    <>
      {/* ── Record Prompt (on game start) ── */}
      {showPrompt && createPortal(
        <div style={{
          position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
          zIndex: 9998, display: 'flex', alignItems: 'flex-start', justifyContent: 'center',
          paddingTop: '80px', pointerEvents: 'none',
        }}>
          <div style={{ pointerEvents: 'auto' }}
            className="bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-2xl shadow-2xl p-6 max-w-sm w-full mx-4 animate-slide-down"
          >
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-xl bg-red-500/15 border border-red-500/30 flex items-center justify-center">
                <Video size={20} className="text-red-400" />
              </div>
              <div>
                <h3 className="text-gray-900 dark:text-white font-semibold text-sm">Record this session?</h3>
                <p className="text-gray-500 dark:text-gray-400 text-xs mt-0.5">
                  720p video streamed to server. No RAM/storage used on your device.
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={handlePromptRecord}
                disabled={isEphemeral}
                className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 bg-red-600 hover:bg-red-700 disabled:bg-gray-200 dark:disabled:bg-gray-700 disabled:text-gray-400 dark:disabled:text-gray-500 text-white rounded-xl text-sm font-medium transition-colors"
              >
                <Video size={16} /> {isEphemeral ? 'Not available' : 'Record'}
              </button>
              <button
                onClick={handlePromptSkip}
                className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-xl text-sm transition-colors"
              >
                <VideoOff size={16} /> Skip
              </button>
            </div>
            {isEphemeral && (
              <p className="text-[10px] text-gray-400 dark:text-gray-500 mt-2 text-center">Recording is only available for games on the platform.</p>
            )}
          </div>
        </div>,
        document.body
      )}

      {/* ── Inline header buttons (only when recording is enabled) ── */}

      {/* Manual record (after skipping prompt) */}
      {recordingEnabled && !isRecording && !hasRecording && promptDismissed && !isEphemeral && (
        <button
          onClick={handleManualStart}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 border border-gray-300 dark:border-gray-700 text-gray-700 dark:text-gray-300 rounded-lg text-xs transition-colors"
        >
          <Video size={14} /> Record
        </button>
      )}

      {/* Recording indicator */}
      {isRecording && (
        <button
          onClick={handleStop}
          className="flex items-center gap-2 px-3 py-1.5 bg-red-600/20 hover:bg-red-600/30 border border-red-500/40 text-red-400 rounded-lg text-xs transition-colors"
        >
          <Circle size={8} className="fill-red-500 text-red-500 animate-pulse" />
          <span className="font-mono tabular-nums">{fmt(recordingTime)}</span>
          <Square size={10} />
        </button>
      )}

      {/* Recording saved on server */}
      {!isRecording && hasRecording && serverFilename && (
        <div className="flex items-center gap-1">
          <span className="flex items-center gap-1 px-2 py-1 text-[10px] text-green-400 bg-green-500/10 border border-green-500/20 rounded-md">
            <CheckCircle size={10} /> Saved
          </span>
          <button
            onClick={handleDownload}
            className="flex items-center gap-1 px-2 py-1 text-[10px] text-blue-400 hover:bg-blue-500/10 border border-blue-500/20 rounded-md transition-colors"
            title="Download recording"
          >
            <Download size={10} /> .{fileExt}
          </button>
          <button
            onClick={handleDelete}
            className="flex items-center gap-1 px-2 py-1 text-[10px] text-gray-500 dark:text-gray-400 hover:text-red-400 hover:bg-red-500/10 border border-gray-300 dark:border-gray-700 rounded-md transition-colors"
            title="Delete recording"
          >
            <Trash2 size={10} />
          </button>
        </div>
      )}

      {/* Prompt animation */}
      <style>{`
        @keyframes slideDown {
          from { opacity: 0; transform: translateY(-20px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-slide-down { animation: slideDown 0.3s ease-out; }
      `}</style>
    </>
  );
}
