import { useState, useRef, useCallback, useEffect } from 'react';

interface RecorderState {
  isRecording: boolean;
  recordingTime: number;
  videoBlob: Blob | null;
  videoUrl: string | null;
  fileExt: string;
}

const BITRATE = 8_000_000;

function pickCodec(): { mimeType: string; ext: string } {
  // Prefer MP4 (H.264) - supported in Chrome 120+, Edge, Safari
  const mp4 = [
    'video/mp4;codecs=avc1.42E01E',    // H.264 Baseline
    'video/mp4;codecs=avc1',
    'video/mp4',
  ];
  for (const c of mp4) {
    if (MediaRecorder.isTypeSupported(c)) return { mimeType: c, ext: 'mp4' };
  }
  // Fallback to WebM
  const webm = [
    'video/webm;codecs=vp9',
    'video/webm;codecs=vp8',
    'video/webm',
  ];
  for (const c of webm) {
    if (MediaRecorder.isTypeSupported(c)) return { mimeType: c, ext: 'webm' };
  }
  return { mimeType: 'video/webm', ext: 'webm' };
}

export function useVideoRecorder() {
  const [state, setState] = useState<RecorderState>({
    isRecording: false,
    recordingTime: 0,
    videoBlob: null,
    videoUrl: null,
    fileExt: 'mp4',
  });

  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const startTimeRef = useRef<number>(0);
  const codecRef = useRef<{ mimeType: string; ext: string }>({ mimeType: '', ext: 'mp4' });

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      if (recorderRef.current && recorderRef.current.state !== 'inactive') {
        recorderRef.current.stop();
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(t => t.stop());
      }
      if (state.videoUrl) URL.revokeObjectURL(state.videoUrl);
    };
  }, []);

  const startRecording = useCallback(async (): Promise<boolean> => {
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getDisplayMedia({
        video: {
          displaySurface: 'browser',
          width: { ideal: 1920 },
          height: { ideal: 1080 },
          frameRate: { ideal: 60, max: 60 },
        } as any,
        audio: false,
        preferCurrentTab: true,
      } as any);
    } catch {
      return false; // User cancelled
    }

    streamRef.current = stream;
    chunksRef.current = [];
    if (state.videoUrl) URL.revokeObjectURL(state.videoUrl);

    const codec = pickCodec();
    codecRef.current = codec;

    const recorder = new MediaRecorder(stream, {
      mimeType: codec.mimeType,
      videoBitsPerSecond: BITRATE,
    });

    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data);
    };

    recorder.onstop = () => {
      if (timerRef.current) clearInterval(timerRef.current);
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(t => t.stop());
        streamRef.current = null;
      }
      const blob = new Blob(chunksRef.current, { type: codec.mimeType });
      const url = URL.createObjectURL(blob);
      setState(prev => ({
        ...prev, isRecording: false, videoBlob: blob, videoUrl: url, fileExt: codec.ext,
      }));
    };

    stream.getVideoTracks()[0].addEventListener('ended', () => {
      if (recorderRef.current && recorderRef.current.state !== 'inactive') {
        recorderRef.current.stop();
      }
    });

    recorder.start(1000);
    recorderRef.current = recorder;

    startTimeRef.current = Date.now();
    timerRef.current = setInterval(() => {
      setState(prev => ({ ...prev, recordingTime: (Date.now() - startTimeRef.current) / 1000 }));
    }, 200);

    setState({ isRecording: true, recordingTime: 0, videoBlob: null, videoUrl: null, fileExt: codec.ext });
    return true;
  }, [state.videoUrl]);

  const stopRecording = useCallback(() => {
    if (recorderRef.current && recorderRef.current.state !== 'inactive') {
      recorderRef.current.stop();
    }
  }, []);

  const downloadVideo = useCallback((filename: string = 'gameplay') => {
    if (!state.videoBlob) return;
    const ext = state.fileExt;
    const a = document.createElement('a');
    a.href = URL.createObjectURL(state.videoBlob);
    a.download = `${filename}_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.${ext}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(a.href);
  }, [state.videoBlob, state.fileExt]);

  const clearRecording = useCallback(() => {
    if (state.videoUrl) URL.revokeObjectURL(state.videoUrl);
    chunksRef.current = [];
    setState({ isRecording: false, recordingTime: 0, videoBlob: null, videoUrl: null, fileExt: 'mp4' });
  }, [state.videoUrl]);

  return {
    isRecording: state.isRecording,
    recordingTime: state.recordingTime,
    videoBlob: state.videoBlob,
    videoUrl: state.videoUrl,
    fileExt: state.fileExt,
    startRecording,
    stopRecording,
    downloadVideo,
    clearRecording,
  };
}
