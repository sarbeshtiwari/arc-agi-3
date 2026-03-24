import React, { useRef, useEffect, useState } from 'react';
import { gamesAPI } from '../api/client';

const COLORS = [
  '#FFFFFF', '#CCCCCC', '#999999', '#666666', '#333333', '#000000',
  '#E53AA3', '#FF7BCC', '#F93C31', '#1E93FF', '#88D8F1', '#FFDC00',
  '#FF851B', '#921231', '#4FCC30', '#A356D6',
];

export default function GamePreviewCanvas({ gameId, size = 140 }) {
  const canvasRef = useRef(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    gamesAPI.getPublicPreview(gameId)
      .then((res) => {
        if (!cancelled) setPreview(res.data);
      })
      .catch(() => {})
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [gameId]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !preview?.grid) return;

    const ctx = canvas.getContext('2d');
    const { grid, width, height } = preview;
    const cs = Math.max(1, Math.floor(size / Math.max(width, height)));
    const cw = width * cs;
    const ch = height * cs;

    canvas.width = cw;
    canvas.height = ch;

    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, cw, ch);

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const value = grid[y]?.[x] ?? 0;
        ctx.fillStyle = COLORS[value] || COLORS[0];
        ctx.fillRect(x * cs, y * cs, cs, cs);
      }
    }
  }, [preview, size]);

  if (loading) {
    return (
      <div
        className="animate-pulse bg-gray-800 rounded-lg"
        style={{ width: size, height: size }}
      />
    );
  }

  if (!preview?.grid) {
    return (
      <div
        className="bg-gray-800 rounded-lg flex items-center justify-center"
        style={{ width: size, height: size }}
      >
        <span className="text-gray-600 text-xs">No preview</span>
      </div>
    );
  }

  return (
    <canvas
      ref={canvasRef}
      className="rounded-lg"
      style={{
        width: size,
        height: size,
        imageRendering: 'pixelated',
      }}
    />
  );
}
