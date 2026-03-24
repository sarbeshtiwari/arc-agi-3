import React, { useRef, useEffect, useCallback, forwardRef, useImperativeHandle } from 'react';

// ARC-AGI-3 color palette (0-15)
// Source: app.py -> _PALETTE_HEX
const COLORS = [
  '#FFFFFF', // 0  - White
  '#CCCCCC', // 1  - Off-white
  '#999999', // 2  - Light Grey
  '#666666', // 3  - Grey
  '#333333', // 4  - Dark Grey
  '#000000', // 5  - Black
  '#E53AA3', // 6  - Magenta
  '#FF7BCC', // 7  - Pink
  '#F93C31', // 8  - Red
  '#1E93FF', // 9  - Blue
  '#88D8F1', // 10 - Light Blue
  '#FFDC00', // 11 - Yellow
  '#FF851B', // 12 - Orange
  '#921231', // 13 - Maroon
  '#4FCC30', // 14 - Green
  '#A356D6', // 15 - Purple
];

const GameCanvas = forwardRef(function GameCanvas({ 
  grid, 
  width, 
  height, 
  onCellClick, 
  cellSize = 0,
  showGrid = false,
  maxCanvasWidth = 640,
  maxCanvasHeight = 640,
  interactive = true,
}: any, ref: any) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Expose the canvas element to parent via ref
  useImperativeHandle(ref, () => canvasRef.current);

  const computedCellSize = cellSize > 0
    ? cellSize
    : Math.max(4, Math.min(
        Math.floor(maxCanvasWidth / (width || 8)),
        Math.floor(maxCanvasHeight / (height || 8))
      ));

  const canvasWidth = (width || 8) * computedCellSize;
  const canvasHeight = (height || 8) * computedCellSize;

  const drawGrid = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !grid) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const cs = computedCellSize;

    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);

    for (let y = 0; y < (height || 0); y++) {
      for (let x = 0; x < (width || 0); x++) {
        const value = grid[y]?.[x] ?? 0;
        const color = COLORS[value] || COLORS[0];
        ctx.fillStyle = color;
        ctx.fillRect(x * cs, y * cs, cs, cs);
      }
    }

    if (showGrid && cs > 8) {
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
      ctx.lineWidth = 1;
      for (let x = 0; x <= (width || 0); x++) {
        ctx.beginPath();
        ctx.moveTo(x * cs + 0.5, 0);
        ctx.lineTo(x * cs + 0.5, canvasHeight);
        ctx.stroke();
      }
      for (let y = 0; y <= (height || 0); y++) {
        ctx.beginPath();
        ctx.moveTo(0, y * cs + 0.5);
        ctx.lineTo(canvasWidth, y * cs + 0.5);
        ctx.stroke();
      }
    }
  }, [grid, width, height, computedCellSize, canvasWidth, canvasHeight, showGrid]);

  useEffect(() => {
    drawGrid();
  }, [drawGrid]);

  const handleClick = useCallback((e: any) => {
    if (!interactive || !onCellClick) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const pixelX = (e.clientX - rect.left) * scaleX;
    const pixelY = (e.clientY - rect.top) * scaleY;
    const cellX = Math.floor(pixelX / computedCellSize);
    const cellY = Math.floor(pixelY / computedCellSize);
    if (cellX >= 0 && cellX < (width || 0) && cellY >= 0 && cellY < (height || 0)) {
      onCellClick(cellX, cellY);
    }
  }, [interactive, onCellClick, computedCellSize, width, height]);

  return (
    <div ref={containerRef} className="inline-block">
      <canvas
        ref={canvasRef}
        width={canvasWidth}
        height={canvasHeight}
        onClick={handleClick}
        className={`rounded-lg border border-gray-700 ${interactive ? 'cursor-crosshair' : ''}`}
        style={{
          maxWidth: `${maxCanvasWidth}px`,
          maxHeight: `${maxCanvasHeight}px`,
          width: `${Math.min(canvasWidth, maxCanvasWidth)}px`,
          height: `${Math.min(canvasHeight, maxCanvasHeight)}px`,
          imageRendering: 'pixelated',
        }}
      />
    </div>
  );
});

export default GameCanvas;
