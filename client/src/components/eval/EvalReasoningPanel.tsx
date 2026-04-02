import React, { useRef, useEffect, useCallback } from 'react';
import { Brain, Zap, AlertTriangle, Copy, Check } from 'lucide-react';
import type { EvalTurnEntry } from '../../hooks/useEvalStream';

interface EvalReasoningPanelProps {
  timeline: EvalTurnEntry[];
}

export function EvalReasoningPanel({ timeline }: EvalReasoningPanelProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [copied, setCopied] = React.useState(false);

  // Auto-scroll to bottom on new entries
  useEffect(() => {
    const el = scrollRef.current;
    if (el) {
      el.scrollTop = el.scrollHeight;
    }
  }, [timeline.length]);

  const handleCopy = useCallback(() => {
    const text = timeline
      .map((entry) => {
        const prefix = `[Turn ${entry.turn}]`;
        if (entry.type === 'thinking') return `${prefix} [THINKING]`;
        if (entry.type === 'action') return `${prefix} ACTION: ${entry.action}\n  Reasoning: ${entry.reasoning || '(none)'}`;
        if (entry.type === 'result') return `${prefix} RESULT: score=${entry.score ?? '?'}`;
        if (entry.type === 'message') return `${prefix} MESSAGE: ${entry.reasoning || ''}`;
        return `${prefix} ${entry.type}`;
      })
      .join('\n');
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }, [timeline]);

  const typeStyles: Record<string, { icon: typeof Brain; color: string; bg: string; border: string }> = {
    thinking: {
      icon: Brain,
      color: 'text-blue-400',
      bg: 'bg-blue-400/5',
      border: 'border-blue-500/20',
    },
    action: {
      icon: Zap,
      color: 'text-emerald-400',
      bg: 'bg-emerald-400/5',
      border: 'border-emerald-500/20',
    },
    result: {
      icon: Zap,
      color: 'text-gray-500 dark:text-gray-400',
      bg: 'bg-gray-100/50 dark:bg-gray-800/50',
      border: 'border-gray-300/50 dark:border-gray-700/50',
    },
    message: {
      icon: AlertTriangle,
      color: 'text-amber-400',
      bg: 'bg-amber-400/5',
      border: 'border-amber-500/20',
    },
  };

  return (
    <div className="border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 rounded-lg overflow-hidden flex flex-col h-[600px]">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2.5 border-b border-gray-200 dark:border-gray-800 shrink-0">
        <div className="flex items-center gap-2">
          <Brain className="h-3.5 w-3.5 text-blue-400" />
          <span className="text-[11px] font-mono font-semibold uppercase tracking-widest text-gray-700 dark:text-gray-300">
            Reasoning
          </span>
          {timeline.length > 0 && (
            <span className="text-[9px] font-mono text-gray-400 dark:text-gray-600">
              {timeline.length} entries
            </span>
          )}
        </div>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1 text-[9px] font-mono text-gray-400 dark:text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 transition-colors"
          title="Copy all reasoning"
        >
          {copied ? (
            <Check className="h-3 w-3 text-emerald-400" />
          ) : (
            <Copy className="h-3 w-3" />
          )}
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>

      {/* Scrollable content */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto p-2 space-y-1.5"
      >
        {timeline.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <span className="text-[11px] font-mono text-gray-400 dark:text-gray-600">
              No reasoning yet...
            </span>
          </div>
        ) : (
          timeline.map((entry, i) => {
            const style = typeStyles[entry.type] || typeStyles.message;
            const Icon = style.icon;

            return (
              <div
                key={i}
                className={`border rounded px-2.5 py-2 ${style.bg} ${style.border}`}
              >
                <div className="flex items-center gap-2 mb-1">
                  <Icon className={`h-3 w-3 shrink-0 ${style.color}`} />
                  <span className={`text-[10px] font-mono font-semibold uppercase ${style.color}`}>
                    {entry.type}
                  </span>
                  <span className="text-[9px] font-mono text-gray-400 dark:text-gray-600">
                    Turn {entry.turn}
                  </span>
                  {entry.action && (
                    <>
                      <span className="text-gray-300 dark:text-gray-700">|</span>
                      <span className="text-[10px] font-mono text-emerald-600 dark:text-emerald-300 font-semibold">
                        {entry.action}
                      </span>
                    </>
                  )}
                  {entry.score !== undefined && (
                    <>
                      <span className="text-gray-300 dark:text-gray-700">|</span>
                      <span className="text-[10px] font-mono text-amber-600 dark:text-amber-300">
                        Score: {entry.score}
                      </span>
                    </>
                  )}
                </div>
                {entry.reasoning && (
                  <p className="text-[11px] font-mono text-gray-700 dark:text-gray-300 leading-relaxed whitespace-pre-wrap pl-5">
                    {entry.reasoning}
                  </p>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
