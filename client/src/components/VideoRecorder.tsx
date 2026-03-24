import React, { useRef, useState, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { useVideoRecorder } from '../hooks/useVideoRecorder';
import { gamesAPI } from '../api/client';
import {
  Video, Square, Download, Upload, Trash2, X, Circle, Play,
  CheckCircle, VideoOff
} from 'lucide-react';

interface VideoRecorderProps {
  gameId: string;
  isEphemeral?: boolean;
  playerName?: string;
  isGameOver?: boolean;
  isPlaying?: boolean;
  onPromptDismissed?: () => void;  // called when user clicks Record or Skip
}

function fmt(s: number): string {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, '0')}`;
}

function fmtSize(bytes: number): string {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

export default function VideoRecorder({
  gameId, isEphemeral = false, playerName, isGameOver = false, isPlaying = false,
  onPromptDismissed,
}: VideoRecorderProps) {
  const {
    isRecording, recordingTime, videoBlob, videoUrl, fileExt,
    startRecording, stopRecording, downloadVideo, clearRecording,
  } = useVideoRecorder();

  const [showPreview, setShowPreview] = useState(false);
  const [showPrompt, setShowPrompt] = useState(false);
  const [promptDismissed, setPromptDismissed] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const autoStoppedRef = useRef(false);

  // Show record prompt when game starts playing
  useEffect(() => {
    if (isPlaying && !promptDismissed && !isRecording && !videoBlob) {
      setShowPrompt(true);
    }
  }, [isPlaying]);

  // Auto-stop recording when game ends
  useEffect(() => {
    if (isGameOver && isRecording && !autoStoppedRef.current) {
      autoStoppedRef.current = true;
      setTimeout(() => {
        stopRecording();
        setShowPreview(true);
        setUploadResult(null);
      }, 1500);
    }
    if (!isGameOver) {
      autoStoppedRef.current = false;
    }
  }, [isGameOver, isRecording, stopRecording]);

  const handlePromptRecord = async () => {
    setShowPrompt(false);
    setPromptDismissed(true);
    onPromptDismissed?.();
    autoStoppedRef.current = false;
    await startRecording();
  };

  const handlePromptSkip = () => {
    setShowPrompt(false);
    setPromptDismissed(true);
    onPromptDismissed?.();
  };

  const handleManualStart = async () => {
    autoStoppedRef.current = false;
    await startRecording();
  };

  const handleStop = () => {
    stopRecording();
    setShowPreview(true);
    setUploadResult(null);
  };

  const handleDownload = () => {
    downloadVideo(`${gameId}_${playerName || 'gameplay'}`);
  };

  const handleUpload = async () => {
    if (!videoBlob || isEphemeral) return;
    setUploading(true);
    setUploadResult(null);
    try {
      const ext = fileExt || 'mp4';
      const formData = new FormData();
      formData.append('video', videoBlob, `${gameId}_recording.${ext}`);
      formData.append('player_name', playerName || 'anonymous');
      await gamesAPI.uploadVideo(gameId, formData);
      setUploadResult('success');
    } catch (err: any) {
      setUploadResult(err.response?.data?.detail || 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  const handleClose = () => setShowPreview(false);
  const handleClear = () => {
    clearRecording();
    setShowPreview(false);
    setUploadResult(null);
    setPromptDismissed(false); // allow prompt again on next game
  };

  return (
    <>
      {/* ── Record Prompt Popup (shown on game start) ── */}
      {showPrompt && createPortal(
        <div style={{
          position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
          zIndex: 9998, display: 'flex', alignItems: 'flex-start', justifyContent: 'center',
          paddingTop: '80px', pointerEvents: 'none',
        }}>
          <div style={{ pointerEvents: 'auto' }}
            className="bg-gray-900 border border-gray-700 rounded-2xl shadow-2xl p-6 max-w-sm w-full mx-4 animate-slide-down"
          >
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-xl bg-red-500/15 border border-red-500/30 flex items-center justify-center">
                <Video size={20} className="text-red-400" />
              </div>
              <div>
                <h3 className="text-white font-semibold text-sm">Record this session?</h3>
                <p className="text-gray-400 text-xs mt-0.5">
                  Capture your gameplay as {fileExt === 'mp4' ? 'MP4' : 'video'}. Auto-stops on completion.
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={handlePromptRecord}
                className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 bg-red-600 hover:bg-red-700 text-white rounded-xl text-sm font-medium transition-colors"
              >
                <Video size={16} /> Record
              </button>
              <button
                onClick={handlePromptSkip}
                className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 bg-gray-800 hover:bg-gray-700 text-gray-300 rounded-xl text-sm transition-colors"
              >
                <VideoOff size={16} /> Skip
              </button>
            </div>
          </div>
        </div>,
        document.body
      )}

      {/* ── Inline header buttons ── */}

      {/* Manual record button (only if prompt was skipped and no recording exists) */}
      {!isRecording && !videoBlob && promptDismissed && (
        <button
          onClick={handleManualStart}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-gray-800 hover:bg-gray-700 border border-gray-700 text-gray-300 rounded-lg text-xs transition-colors"
          title="Record gameplay"
        >
          <Video size={14} /> Record
        </button>
      )}

      {/* Recording indicator + stop */}
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

      {/* Preview trigger */}
      {!isRecording && videoBlob && !showPreview && (
        <button
          onClick={() => setShowPreview(true)}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-600/20 hover:bg-blue-600/30 border border-blue-500/40 text-blue-400 rounded-lg text-xs transition-colors"
        >
          <Play size={14} /> Preview ({fmtSize(videoBlob.size)})
        </button>
      )}

      {/* ── Fullscreen Preview Modal ── */}
      {showPreview && videoUrl && createPortal(
        <div style={{
          position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
          zIndex: 9999, display: 'flex', flexDirection: 'column',
          background: '#000', height: '100vh', width: '100vw',
        }}>
          {/* Top bar */}
          <div style={{
            flexShrink: 0, display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            padding: '12px 24px', background: '#030712', borderBottom: '1px solid #1f2937',
          }}>
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2 px-3 py-1 bg-blue-600/20 border border-blue-500/30 rounded-lg">
                <Video size={14} className="text-blue-400" />
                <span className="text-sm font-medium text-blue-400">Recording</span>
              </div>
              <span className="text-white font-mono font-bold text-sm">{gameId}</span>
              {playerName && <span className="text-gray-400 text-sm">{playerName}</span>}
            </div>
            <div className="flex items-center gap-4">
              <div className="hidden sm:flex items-center gap-3 text-xs text-gray-400">
                <span className="px-2 py-1 bg-gray-800 rounded font-mono">{fmt(recordingTime)}</span>
                <span className="px-2 py-1 bg-gray-800 rounded font-mono">{fmtSize(videoBlob?.size || 0)}</span>
                <span className="px-2 py-1 bg-gray-800 rounded uppercase">{fileExt}</span>
              </div>
              <button onClick={handleClose} className="p-2 rounded-lg hover:bg-gray-800 text-gray-400 hover:text-white transition-colors">
                <X size={18} />
              </button>
            </div>
          </div>

          {/* Video */}
          <div style={{
            flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
            overflow: 'hidden', padding: '16px', minHeight: 0, background: '#000',
          }}>
            <video
              ref={videoRef}
              src={videoUrl}
              controls
              autoPlay
              style={{ maxWidth: '100%', maxHeight: '100%', borderRadius: '8px', objectFit: 'contain' }}
            />
          </div>

          {/* Bottom bar */}
          <div style={{
            flexShrink: 0, padding: '16px 24px', background: '#030712', borderTop: '1px solid #1f2937',
          }}>
            {uploadResult === 'success' && (
              <div className="mb-3 flex items-center gap-2 px-4 py-2.5 bg-green-500/10 border border-green-500/30 rounded-lg">
                <CheckCircle size={16} className="text-green-400" />
                <span className="text-sm text-green-400">Video uploaded to server successfully.</span>
              </div>
            )}
            {uploadResult && uploadResult !== 'success' && (
              <div className="mb-3 px-4 py-2.5 bg-red-500/10 border border-red-500/30 rounded-lg text-sm text-red-400">
                {uploadResult}
              </div>
            )}
            <div className="flex items-center gap-3">
              <button onClick={handleDownload} className="flex items-center gap-2 px-5 py-2.5 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium transition-colors">
                <Download size={16} /> Download .{fileExt}
              </button>
              {!isEphemeral && (
                <button onClick={handleUpload} disabled={uploading || uploadResult === 'success'} className="flex items-center gap-2 px-5 py-2.5 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-800 disabled:text-gray-600 disabled:cursor-not-allowed text-white rounded-lg text-sm font-medium transition-colors">
                  <Upload size={16} /> {uploading ? 'Uploading...' : uploadResult === 'success' ? 'Uploaded' : 'Push to Server'}
                </button>
              )}
              <div className="flex-1" />
              <button onClick={handleClear} className="flex items-center gap-2 px-4 py-2.5 bg-gray-800 hover:bg-gray-700 text-gray-400 hover:text-red-400 rounded-lg text-sm transition-colors">
                <Trash2 size={14} /> Discard
              </button>
              <button onClick={handleClose} className="flex items-center gap-2 px-4 py-2.5 bg-gray-800 hover:bg-gray-700 text-gray-300 rounded-lg text-sm transition-colors">
                Close
              </button>
            </div>
          </div>
        </div>,
        document.body
      )}

      {/* Prompt slide-down animation */}
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
