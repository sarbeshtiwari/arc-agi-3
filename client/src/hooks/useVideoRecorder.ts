import { useState, useRef, useCallback, useEffect } from 'react';
import { gamesAPI } from '../api/client';

interface RecorderState {
  isRecording: boolean;
  recordingTime: number;
  hasRecording: boolean;
  serverFilename: string | null;
  fileExt: string;
}

// ── Quality: 720p / 30fps / 4Mbps — good quality, low RAM ──
const BITRATE = 4_000_000;
const MAX_DURATION_SEC = 300; // 5 min auto-stop

function pickCodec(): { mimeType: string; ext: string } {
  const mp4 = ['video/mp4;codecs=avc1.42E01E', 'video/mp4;codecs=avc1', 'video/mp4'];
  for (const c of mp4) {
    if (MediaRecorder.isTypeSupported(c)) return { mimeType: c, ext: 'mp4' };
  }
  const webm = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'];
  for (const c of webm) {
    if (MediaRecorder.isTypeSupported(c)) return { mimeType: c, ext: 'webm' };
  }
  return { mimeType: 'video/webm', ext: 'webm' };
}

export function useVideoRecorder() {
  const [state, setState] = useState<RecorderState>({
    isRecording: false,
    recordingTime: 0,
    hasRecording: false,
    serverFilename: null,
    fileExt: 'mp4',
  });

  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const startTimeRef = useRef<number>(0);
  const gameIdRef = useRef<string>('');
  const recordingIdRef = useRef<string>('');
  const chunkQueueRef = useRef<Promise<any>>(Promise.resolve());

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      if (recorderRef.current && recorderRef.current.state !== 'inactive') recorderRef.current.stop();
      if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop());
    };
  }, []);

  const startRecording = useCallback(async (gameId: string, playerName?: string): Promise<boolean> => {
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getDisplayMedia({
        video: { displaySurface: 'browser', width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30, max: 30 } } as any,
        audio: false,
        preferCurrentTab: true,
      } as any);
    } catch {
      return false;
    }

    streamRef.current = stream;
    gameIdRef.current = gameId;
    const codec = pickCodec();

    // Start server-side recording session
    try {
      const res = await gamesAPI.videoStart(gameId, playerName || '', codec.ext);
      recordingIdRef.current = res.data.recording_id;
    } catch {
      recordingIdRef.current = '';
      stream.getTracks().forEach(t => t.stop());
      streamRef.current = null;
      return false;
    }

    const recorder = new MediaRecorder(stream, { mimeType: codec.mimeType, videoBitsPerSecond: BITRATE });

    // Stream each chunk to server immediately — zero RAM accumulation
    recorder.ondataavailable = (e) => {
      if (e.data.size > 0 && recordingIdRef.current) {
        const blob = e.data;
        chunkQueueRef.current = chunkQueueRef.current.then(async () => {
          try {
            await gamesAPI.videoChunk(gameIdRef.current, recordingIdRef.current, blob);
          } catch (err) {
            console.error('[Recorder] Chunk upload failed:', err);
          }
        });
      }
    };

    recorder.onstop = async () => {
      if (timerRef.current) clearInterval(timerRef.current);
      if (streamRef.current) { streamRef.current.getTracks().forEach(t => t.stop()); streamRef.current = null; }

      // Wait for all pending chunks to upload
      await chunkQueueRef.current;

      // Finalize on server
      if (recordingIdRef.current) {
        try {
          const res = await gamesAPI.videoEnd(gameIdRef.current, recordingIdRef.current);
          setState(prev => ({
            ...prev, isRecording: false,
            hasRecording: !res.data.deleted,
            serverFilename: res.data.deleted ? null : res.data.filename,
          }));
        } catch {
          setState(prev => ({ ...prev, isRecording: false, hasRecording: false }));
        }
      } else {
        setState(prev => ({ ...prev, isRecording: false }));
      }
    };

    stream.getVideoTracks()[0].addEventListener('ended', () => {
      if (recorderRef.current && recorderRef.current.state !== 'inactive') recorderRef.current.stop();
    });

    recorder.start(2000); // 2-second chunks
    recorderRef.current = recorder;

    startTimeRef.current = Date.now();
    timerRef.current = setInterval(() => {
      const elapsed = (Date.now() - startTimeRef.current) / 1000;
      setState(prev => ({ ...prev, recordingTime: elapsed }));
      if (elapsed >= MAX_DURATION_SEC) {
        if (recorderRef.current && recorderRef.current.state !== 'inactive') recorderRef.current.stop();
      }
    }, 200);

    setState({ isRecording: true, recordingTime: 0, hasRecording: false, serverFilename: null, fileExt: codec.ext });
    return true;
  }, []);

  const stopRecording = useCallback(() => {
    if (recorderRef.current && recorderRef.current.state !== 'inactive') recorderRef.current.stop();
  }, []);

  const clearRecording = useCallback(() => {
    recordingIdRef.current = '';
    setState({ isRecording: false, recordingTime: 0, hasRecording: false, serverFilename: null, fileExt: 'mp4' });
  }, []);

  return {
    isRecording: state.isRecording,
    recordingTime: state.recordingTime,
    hasRecording: state.hasRecording,
    serverFilename: state.serverFilename,
    fileExt: state.fileExt,
    startRecording,
    stopRecording,
    clearRecording,
  };
}
