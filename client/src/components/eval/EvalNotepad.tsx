import React, { useState, useEffect, useRef, useCallback } from 'react';
import { StickyNote, Copy, Check } from 'lucide-react';

interface EvalNotepadProps {
  content: string;
  gameId: string | null;
  modelId: string | null;
  runIndex: number;
}

export function EvalNotepad({ content, gameId, modelId, runIndex }: EvalNotepadProps) {
  const [flash, setFlash] = useState(false);
  const [copied, setCopied] = useState(false);
  const prevContentRef = useRef(content);

  // Flash animation on content update
  useEffect(() => {
    if (content !== prevContentRef.current && content) {
      setFlash(true);
      const timer = setTimeout(() => setFlash(false), 600);
      prevContentRef.current = content;
      return () => clearTimeout(timer);
    }
    prevContentRef.current = content;
  }, [content]);

  const handleCopy = useCallback(() => {
    if (!content) return;
    navigator.clipboard.writeText(content).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }, [content]);

  return (
    <div
      className={`border bg-gray-900 rounded-lg overflow-hidden transition-all duration-300 ${
        flash
          ? 'border-amber-500/60 shadow-[0_0_12px_rgba(245,158,11,0.15)]'
          : 'border-gray-800'
      }`}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-gray-800">
        <div className="flex items-center gap-2">
          <StickyNote className="h-3.5 w-3.5 text-amber-400" />
          <span className="text-[11px] font-mono font-semibold uppercase tracking-widest text-gray-300">
            Notepad
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[9px] font-mono text-gray-600">
            {gameId || '—'} | {modelId || '—'} | Run {runIndex + 1}
          </span>
          <button
            onClick={handleCopy}
            disabled={!content}
            className="flex items-center gap-1 text-[9px] font-mono text-gray-500 hover:text-gray-300 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
            title="Copy notepad"
          >
            {copied ? (
              <Check className="h-3 w-3 text-emerald-400" />
            ) : (
              <Copy className="h-3 w-3" />
            )}
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="bg-gray-950 p-3 min-h-[120px] max-h-[300px] overflow-y-auto">
        {content ? (
          <pre className="text-[11px] font-mono text-gray-300 leading-relaxed whitespace-pre-wrap break-words">
            {content}
          </pre>
        ) : (
          <div className="flex items-center justify-center h-20">
            <span className="text-[10px] font-mono text-gray-700">
              Notepad empty
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
