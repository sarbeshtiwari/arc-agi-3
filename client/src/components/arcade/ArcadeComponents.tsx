import React from 'react';

export function StatPill({ icon: Icon, value, color = 'cyan' }: any) {
  const colors: Record<string, string> = {
    cyan: 'neon-text-cyan',
    green: 'neon-text-green',
    magenta: 'neon-text-magenta',
    yellow: 'neon-text-yellow',
    orange: 'neon-text-orange',
    red: 'text-red-400',
    white: 'text-gray-200',
    blue: 'neon-text-blue',
  };
  return (
    <div className="flex items-center gap-1.5 px-2.5 py-1 bg-cabinet-panel/80 rounded border border-cabinet-trim/60">
      <Icon size={12} className={colors[color] || colors.cyan} />
      <span className={`font-pixel text-[9px] tabular-nums ${colors[color] || colors.cyan}`}>{value}</span>
    </div>
  );
}

export function OverlayStat({ label, value }: any) {
  return (
    <div className="text-center">
      <p className="text-xl font-bold font-pixel neon-text-cyan">{value}</p>
      <p className="text-[8px] font-pixel text-neon-cyan/50 uppercase tracking-wider mt-1">{label}</p>
    </div>
  );
}

export function DPadButton({ icon: Icon, action, active, available, disabled, onClick, kbd }: any) {
  const enabled = available.includes(action) && !disabled;
  return (
    <button
      onClick={onClick}
      disabled={!enabled}
      title={kbd}
      className={`w-12 h-12 rounded-full flex items-center justify-center transition-all duration-100 ${
        !enabled
          ? 'bg-cabinet-panel/40 text-gray-600 cursor-not-allowed'
          : active
            ? 'arcade-btn arcade-btn-blue scale-90 shadow-neon-cyan text-white'
            : 'arcade-btn text-gray-300 hover:text-white hover:shadow-neon-cyan/50 active:scale-90'
      }`}
    >
      <Icon size={18} />
    </button>
  );
}

export function ActionButton({ icon: Icon, label, action, active, available, disabled, onClick, kbd, spinning }: any) {
  const enabled = available.includes(action) && !disabled;
  return (
    <button
      onClick={onClick}
      disabled={!enabled}
      title={`${label} (${kbd})`}
      className={`h-11 px-3.5 rounded-full flex items-center gap-1.5 transition-all duration-100 font-pixel text-[8px] ${
        !enabled
          ? 'bg-cabinet-panel/40 text-gray-600 cursor-not-allowed'
          : active
            ? 'arcade-btn arcade-btn-red text-white scale-95 shadow-neon-magenta'
            : 'arcade-btn text-gray-300 hover:text-white active:scale-95'
      }`}
    >
      <Icon size={14} className={spinning ? 'animate-spin' : ''} />
      <span className="hidden sm:inline">{label}</span>
    </button>
  );
}

export function KbdRow({ keys, desc }: any) {
  return (
    <div className="flex items-center justify-between">
      <div className="flex gap-1">
        {keys.split(' ').map((k: string) => (
          <kbd key={k} className="px-1.5 py-0.5 bg-cabinet-bezel border border-cabinet-trim rounded text-[9px] font-pixel text-neon-cyan/70">{k}</kbd>
        ))}
      </div>
      <span className="text-[8px] font-pixel text-gray-500">{desc}</span>
    </div>
  );
}

export function ArcadeCabinet({ title, subtitle, children, controls, sideContent }: {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
  controls?: React.ReactNode;
  sideContent?: React.ReactNode;
}) {
  return (
    <div className="flex items-start justify-center gap-4 w-full">
      <div className="hidden lg:block w-3 self-stretch side-art rounded-l-lg mt-10 mb-16" />

      <div className="flex flex-col items-center max-w-[640px] w-full">
        <div className="arcade-marquee w-full px-6 py-3 flex items-center justify-center">
          <div className="text-center">
            <h1 className="font-pixel text-sm neon-text-cyan animate-marquee-glow truncate max-w-[500px]">
              {title}
            </h1>
            {subtitle && (
              <p className="font-pixel text-[8px] text-neon-magenta/70 mt-1">{subtitle}</p>
            )}
          </div>
        </div>

        <div className="screen-bezel w-full">
          <div className="crt-screen scanline-sweep rounded-lg overflow-hidden bg-cabinet-screen">
            {children}
          </div>
        </div>

        {controls && (
          <div className="control-panel w-full px-4 py-3">
            {controls}
          </div>
        )}

        <div className="flex items-center gap-3 mt-3 mb-1">
          <div className="coin-slot" />
          <span className="font-pixel text-[7px] text-neon-yellow/60 animate-coin-blink">
            INSERT COIN
          </span>
          <div className="coin-slot" />
        </div>
      </div>

      {sideContent ? (
        <div className="hidden xl:block w-52 self-stretch mt-10">
          {sideContent}
        </div>
      ) : (
        <div className="hidden lg:block w-3 self-stretch side-art rounded-r-lg mt-10 mb-16" />
      )}
    </div>
  );
}
