import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { playerAPI } from '../api/client';
import GameCanvas from '../components/GameCanvas';
import VideoRecorder from '../components/VideoRecorder';
import {
  ArrowUp, ArrowDown, ArrowLeft, ArrowRight,
  RotateCcw, Undo, Zap, Trophy, Square, X as XIcon,
  ChevronLeft, Skull, Clock, Layers, Home, Target, AlertTriangle,
} from 'lucide-react';
import { useTheme } from '../hooks/useTheme';
import { StatPill, DPadButton, ActionButton, KbdRow, ArcadeCabinet } from '../components/arcade/ArcadeComponents';

const ACTION_KEYS = {
  w: 'ACTION1', arrowup: 'ACTION1',
  s: 'ACTION2', arrowdown: 'ACTION2',
  a: 'ACTION3', arrowleft: 'ACTION3',
  d: 'ACTION4', arrowright: 'ACTION4',
  ' ': 'ACTION5', f: 'ACTION5',
};

function formatTime(seconds) {
  if (!seconds || seconds < 0) return '0:00';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

export default function DirectPlayPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const { theme, toggleTheme } = useTheme();
  const { gameFile, metadataFile, playerName } = location.state || {};

  const [sessionGuid, setSessionGuid] = useState(null);
  const [frame, setFrame] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [gameId, setGameId] = useState('custom');
  const [totalLevels, setTotalLevels] = useState(0);

  const [elapsed, setElapsed] = useState(0);
  const timerRef = useRef(null);
  const startTimeRef = useRef(null);
  const lastActionTime = useRef(0);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const ACTION_THROTTLE_MS = 50;

  const [resetAnim, setResetAnim] = useState(false);
  const [gameOverAnim, setGameOverAnim] = useState(false);
  const [canvasShake, setCanvasShake] = useState(false);
  const [lastAction, setLastAction] = useState(null);
  const prevStateRef = useRef(null);
  const errorTimerRef = useRef(null);

  // Level transition
  const prevLevelRef = useRef(null);
  const [levelClearAnim, setLevelClearAnim] = useState(null);

  const isPlaying = !!sessionGuid;
  const isGameOver = frame?.state === 'WIN' || frame?.state === 'GAME_OVER';
  const isWin = frame?.state === 'WIN';
  const serverTotalTime = frame?.metadata?.total_time || 0;
  const completedLevels = frame?.metadata?.completed_levels || [];
  const currentLevelStats = frame?.metadata?.current_level_stats;

  const setErrorWithDismiss = useCallback((msg: string | null, timeout = 6000) => {
    if (errorTimerRef.current) clearTimeout(errorTimerRef.current);
    setError(msg);
    if (msg && timeout > 0) {
      errorTimerRef.current = setTimeout(() => setError(null), timeout);
    }
  }, []);

  useEffect(() => () => { if (errorTimerRef.current) clearTimeout(errorTimerRef.current); }, []);

  // Timer
  const startTimer = useCallback(() => {
    startTimeRef.current = Date.now();
    setElapsed(0);
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = setInterval(() => {
      if (startTimeRef.current) setElapsed((Date.now() - startTimeRef.current) / 1000);
    }, 100);
  }, []);

  const stopTimer = useCallback(() => {
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
  }, []);

  useEffect(() => () => { if (timerRef.current) clearInterval(timerRef.current); }, []);

  useEffect(() => {
    const prev = prevStateRef.current;
    const next = frame?.state;
    if (prev && prev !== next) {
      if (next === 'WIN' || next === 'GAME_OVER') {
        stopTimer();
        setGameOverAnim(true);
        if (next === 'GAME_OVER') { setCanvasShake(true); setTimeout(() => setCanvasShake(false), 500); }
      }
    }
    prevStateRef.current = next;
  }, [frame?.state, stopTimer]);

  // Level transition detection
  useEffect(() => {
    const currentLevel = frame?.level;
    if (currentLevel == null) return;
    const prevLevel = prevLevelRef.current;
    prevLevelRef.current = currentLevel;
    if (prevLevel != null && currentLevel > prevLevel && frame?.state !== 'WIN') {
      const completed = frame?.metadata?.completed_levels || [];
      const clearedStats = completed.find(ls => ls.level === prevLevel) || { level: prevLevel, actions: 0, time: 0 };
      setLevelClearAnim({
        level: prevLevel + 1,
        nextLevel: currentLevel + 1,
        actions: clearedStats.actions,
        time: clearedStats.time,
      });
      setTimeout(() => setLevelClearAnim(null), 2000);
    }
  }, [frame?.level, frame?.state]);

  // Redirect if no files
  useEffect(() => {
    if (!gameFile || !metadataFile) { navigate('/', { replace: true }); return; }
    // Parse metadata to get total levels
    metadataFile.text().then((metaText) => {
      try {
        const meta = JSON.parse(metaText);
        if (meta.baseline_actions?.length) setTotalLevels(meta.baseline_actions.length);
      } catch (e) {}
    });
    doStart();
  }, []);

  async function doStart() {
    setLoading(true); setError(null); setGameOverAnim(false);
    try {
      const res = await playerAPI.ephemeralStart(gameFile, metadataFile, playerName);
      setSessionGuid(res.data.metadata?.session_guid);
      setFrame(res.data);
      setGameId(res.data.metadata?.game_id || 'custom');
      prevStateRef.current = res.data.state;
      // Timer starts when record prompt is dismissed
    } catch (err) {
      setErrorWithDismiss(err.response?.data?.detail || 'Failed to start game', 0);
    } finally { setLoading(false); }
  }

  const sendAction = useCallback(async (action, x = null, y = null) => {
    if (!sessionGuid) return;
    const now = Date.now();
    if (now - lastActionTime.current < ACTION_THROTTLE_MS) return;
    lastActionTime.current = now;
    setLastAction(action);
    setTimeout(() => setLastAction(null), 150);
    setError(null);
    try {
      const res = await playerAPI.ephemeralAction(sessionGuid, action, x, y);
      setFrame(res.data);
    } catch (err) {
      const detail = err.response?.data?.detail || '';
      if (err.response?.status === 404 || detail.includes('No game instance')) {
        setSessionGuid(null);
        setFrame(null);
        prevStateRef.current = null;
        setTimeout(() => doStart(), 200);
        return;
      }
      setErrorWithDismiss(detail || `Action '${action}' failed`);
    }
  }, [sessionGuid]);

  const endGame = useCallback(async () => {
    if (!sessionGuid) return;
    stopTimer();
    try { await playerAPI.ephemeralEnd(sessionGuid); } catch (e) {}
    setSessionGuid(null); setFrame(null); setGameOverAnim(false);
    prevStateRef.current = null; setElapsed(0);
  }, [sessionGuid, stopTimer]);

  const resetGame = useCallback(async () => {
    if (!sessionGuid) return;
    setResetAnim(true);
    setTimeout(() => setResetAnim(false), 400);
    try {
      const res = await playerAPI.ephemeralAction(sessionGuid, 'RESET');
      setFrame(res.data); setGameOverAnim(false);
      prevStateRef.current = res.data.state;
      startTimer();
    } catch (err) { setErrorWithDismiss(err.response?.data?.detail || 'Reset failed'); }
  }, [sessionGuid, startTimer]);


  // Keyboard
  useEffect(() => {
    if (!isPlaying) return;
    const handle = (e) => {
      const key = e.key.toLowerCase();
      if (key === 'r' && !e.ctrlKey && !e.metaKey) { e.preventDefault(); resetGame(); return; }
      if (isGameOver) return;
      if (key === 'z' && !e.ctrlKey && !e.metaKey) { e.preventDefault(); if (frame?.available_actions?.includes('ACTION7')) sendAction('ACTION7'); return; }
      const action = ACTION_KEYS[key];
      if (action) { e.preventDefault(); if (frame?.available_actions?.includes(action)) sendAction(action); }
    };
    window.addEventListener('keydown', handle);
    return () => window.removeEventListener('keydown', handle);
  }, [isPlaying, isGameOver, frame, sendAction, resetGame]);

  const handleCellClick = useCallback((x, y) => {
    if (!isPlaying || isGameOver) return;
    if (frame?.available_actions?.includes('ACTION6')) sendAction('ACTION6', x, y);
  }, [isPlaying, isGameOver, frame, sendAction]);

  const available = frame?.available_actions || [];

  if (loading && !isPlaying) {
    return (
      <div className="min-h-screen bg-cabinet-body starfield flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-neon-cyan mx-auto" />
          <p className="font-pixel text-[9px] neon-text-cyan mt-4 animate-neon-pulse">LOADING...</p>
        </div>
      </div>
    );
  }

  if (error && !isPlaying) {
    return (
      <div className="min-h-screen bg-cabinet-body starfield flex items-center justify-center">
        <div className="text-center">
          <XIcon size={48} className="mx-auto text-neon-magenta mb-4" />
          <p className="font-pixel text-[10px] text-neon-magenta mb-2">FAILED TO LOAD GAME</p>
          <p className="font-pixel text-[7px] text-gray-500 mb-6">{error}</p>
          <Link to="/" className="font-pixel text-[9px] neon-text-cyan hover:text-neon-cyan">← BACK TO GAMES</Link>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-cabinet-body starfield flex flex-col">
      {/* ── Top Bar ── */}
      <header className="bg-cabinet-panel/90 backdrop-blur-sm border-b border-cabinet-trim/60 sticky top-0 z-30">
        <div className="max-w-6xl mx-auto px-4 h-12 flex items-center gap-3">
          <Link to="/" onClick={() => { if (sessionGuid) endGame(); }} className="p-1 rounded-md hover:bg-cabinet-bezel text-neon-cyan/60 hover:text-neon-cyan transition-colors">
            <ChevronLeft size={18} />
          </Link>
          <span className="neon-text-cyan font-pixel text-[9px]">{gameId}</span>
          <span className="px-2 py-0.5 bg-neon-magenta/20 text-neon-magenta font-pixel text-[7px] rounded-full">DIRECT</span>
          {playerName && <span className="font-pixel text-[7px] text-neon-magenta/70">({playerName})</span>}

          <div className="flex-1" />

          {isPlaying && (
            <>
              <StatPill icon={Layers} value={`Lv ${isWin && totalLevels ? totalLevels : (frame?.level ?? 0) + 1}${totalLevels ? ' / ' + totalLevels : ''}`} color="blue" />
              <StatPill icon={Target} value={frame?.total_actions || 0} color="green" />
              <StatPill
                icon={Clock}
                value={formatTime(isGameOver ? serverTotalTime : elapsed)}
                color={isWin ? 'green' : isGameOver ? 'red' : 'cyan'}
              />
               <VideoRecorder
                gameId={gameId}
                isPlaying={isPlaying}
                isEphemeral={true}
                playerName={playerName || undefined}
                isGameOver={isGameOver}
                onPromptDismissed={startTimer}
              />
              <button
                onClick={async () => { await endGame(); navigate('/'); }}
                className="ml-2 flex items-center gap-1 px-2.5 py-1 font-pixel text-[8px] text-red-400 hover:bg-red-500/10 border border-red-500/30 rounded transition-colors"
              >
                <Square size={10} /> END
              </button>
            </>
          )}
        </div>
      </header>

      {/* ── Main Area ── */}
      <main className="flex-1 flex items-start justify-center px-4 py-6">
        <div className="w-full max-w-5xl">
          {!isPlaying ? (
            <div className="flex items-center justify-center py-20">
              <div className="text-center">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-neon-cyan mx-auto" />
                <p className="font-pixel text-[8px] neon-text-cyan mt-3 animate-neon-pulse">STARTING GAME...</p>
              </div>
            </div>
          ) : (
            <ArcadeCabinet
              title={gameId || 'ARCADE'}
              subtitle={playerName || undefined}
              controls={
                <div className="flex items-center justify-center gap-2">
                  <DPadButton icon={ArrowLeft} action="ACTION3" active={lastAction === 'ACTION3'} available={available} disabled={isGameOver} onClick={() => sendAction('ACTION3')} kbd="A" />
                  <div className="flex flex-col gap-1.5">
                    <DPadButton icon={ArrowUp} action="ACTION1" active={lastAction === 'ACTION1'} available={available} disabled={isGameOver} onClick={() => sendAction('ACTION1')} kbd="W" />
                    <DPadButton icon={ArrowDown} action="ACTION2" active={lastAction === 'ACTION2'} available={available} disabled={isGameOver} onClick={() => sendAction('ACTION2')} kbd="S" />
                  </div>
                  <DPadButton icon={ArrowRight} action="ACTION4" active={lastAction === 'ACTION4'} available={available} disabled={isGameOver} onClick={() => sendAction('ACTION4')} kbd="D" />

                  <div className="w-px h-10 bg-cabinet-trim/40 mx-2" />

                  <ActionButton icon={Zap} label="Action" action="ACTION5" active={lastAction === 'ACTION5'} available={available} disabled={isGameOver} onClick={() => sendAction('ACTION5')} kbd="Space" />
                  <ActionButton icon={Undo} label="Undo" action="ACTION7" active={lastAction === 'ACTION7'} available={available} disabled={isGameOver} onClick={() => sendAction('ACTION7')} kbd="Z" />
                  <ActionButton icon={RotateCcw} label="Reset" action="_reset" available={['_reset']} disabled={false} onClick={resetGame} kbd="R" spinning={resetAnim} />

                  <div className="w-px h-10 bg-cabinet-trim/40 mx-2" />

                  {/* Level indicator */}
                  <div className="flex items-center gap-1.5 px-3 h-11 bg-cabinet-panel/80 rounded-full border border-cabinet-trim/60">
                    <Layers size={14} className="neon-text-blue" />
                    <span className="font-pixel text-[8px] neon-text-cyan">
                      {isWin && totalLevels ? totalLevels : (frame?.level ?? 0) + 1}{totalLevels ? ` / ${totalLevels}` : ''}
                    </span>
                  </div>
                </div>
              }
              sideContent={
                <div className="space-y-3">
                  {/* Level Progress */}
                  {(completedLevels.length > 0 || currentLevelStats) && (
                    <div className="bg-cabinet-panel/80 rounded-lg border border-cabinet-trim/40 p-3.5">
                      <h3 className="font-pixel text-[7px] neon-text-yellow uppercase tracking-wider mb-2.5">LEVELS</h3>
                      <div className="space-y-1.5">
                        {completedLevels.map((ls, i) => (
                          <div key={i} className="bg-cabinet-bezel/50 rounded-lg p-2">
                            <div className="flex items-center gap-2 text-[11px]">
                              <span className="w-4 h-4 rounded-full bg-neon-green/20 border border-neon-green/40 flex items-center justify-center text-neon-green font-pixel text-[6px] shrink-0">{ls.level + 1}</span>
                              <span className="font-pixel text-[7px] text-gray-400 flex-1">{ls.actions}m</span>
                              <span className="font-pixel text-[7px] neon-text-cyan tabular-nums">{formatTime(ls.time)}</span>
                            </div>
                            <div className="flex items-center gap-2 mt-1 ml-6 text-[10px] text-gray-500">
                              {ls.game_overs > 0 && <span className="text-red-400 font-pixel text-[6px]">{ls.game_overs} death{ls.game_overs > 1 ? 's' : ''}</span>}
                              {ls.lives_used > 0 && <span className="font-pixel text-[6px]">{ls.lives_used} lives</span>}
                              {ls.resets > 0 && <span className="font-pixel text-[6px]">{ls.resets} reset{ls.resets > 1 ? 's' : ''}</span>}
                            </div>
                          </div>
                        ))}
                        {currentLevelStats && !isGameOver && (
                          <div className="bg-neon-cyan/5 border border-neon-cyan/20 rounded-lg p-2">
                            <div className="flex items-center gap-2 text-[11px]">
                              <span className="w-4 h-4 rounded-full bg-neon-cyan/20 border border-neon-cyan/40 flex items-center justify-center neon-text-cyan font-pixel text-[6px] shrink-0 animate-pulse">{currentLevelStats.level + 1}</span>
                              <span className="font-pixel text-[7px] text-gray-400 flex-1">{currentLevelStats.actions}m</span>
                              <span className="font-pixel text-[7px] neon-text-cyan tabular-nums">{formatTime(currentLevelStats.time)}</span>
                            </div>
                            <div className="flex items-center gap-2 mt-1 ml-6 text-[10px] text-gray-500">
                              {currentLevelStats.game_overs > 0 && <span className="text-red-400 font-pixel text-[6px]">{currentLevelStats.game_overs} death{currentLevelStats.game_overs > 1 ? 's' : ''}</span>}
                              {currentLevelStats.lives_used > 0 && <span className="font-pixel text-[6px]">{currentLevelStats.lives_used} lives</span>}
                              {currentLevelStats.resets > 0 && <span className="font-pixel text-[6px]">{currentLevelStats.resets} reset{currentLevelStats.resets > 1 ? 's' : ''}</span>}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Keyboard shortcuts */}
                  <div className="bg-cabinet-panel/80 rounded-lg border border-cabinet-trim/40 p-3.5">
                    <h3 className="font-pixel text-[7px] neon-text-yellow uppercase tracking-wider mb-2.5">SHORTCUTS</h3>
                    <div className="space-y-1.5">
                      <KbdRow keys="W A S D" desc="Move" />
                      <KbdRow keys="Space" desc="Action" />
                      <KbdRow keys="Click" desc="Coordinate" />
                      <KbdRow keys="Z" desc="Undo" />
                      <KbdRow keys="R" desc="Reset level" />
                    </div>
                  </div>
                </div>
              }
            >
              <div className="relative">
                <div className={`p-2 transition-all duration-300 ${
                  resetAnim ? 'opacity-0 scale-95' : 'opacity-100 scale-100'
                } ${canvasShake ? 'animate-shake' : ''}`}>
                  <GameCanvas
                    ref={canvasRef}
                    grid={frame?.grid || []}
                    width={frame?.width || 8}
                    height={frame?.height || 8}
                    onCellClick={handleCellClick}
                    interactive={!isGameOver}
                    maxCanvasWidth={560}
                    maxCanvasHeight={560}
                  />
                </div>

                {/* ── Level Clear: Glitch/Digital transition ── */}
                {levelClearAnim && !isGameOver && (
                  <div className="absolute inset-0 z-15 pointer-events-none overflow-hidden">
                    <div className="absolute inset-0 glitch-scanlines" />
                    <div className="absolute inset-0 glitch-rgb-shift" />
                    <div className="absolute inset-0">
                      {[...Array(6)].map((_, i) => (
                        <div key={i} className="glitch-bar absolute w-full bg-neon-cyan/10"
                          style={{ height: `${2 + Math.random() * 8}px`, top: `${10 + Math.random() * 80}%`, animationDelay: `${Math.random() * 0.3}s`, animationDuration: `${0.1 + Math.random() * 0.15}s` }} />
                      ))}
                    </div>
                    <div className="absolute inset-0 glitch-noise" />
                    <div className="absolute inset-0 glitch-blackout" />
                    <div className="absolute inset-0 flex items-center justify-center glitch-text-container">
                      <div className="text-center">
                        <div className="glitch-level-text font-pixel text-xl neon-text-cyan relative">
                          <span className="glitch-text-main">LEVEL {levelClearAnim.level}</span>
                          <span className="glitch-text-r absolute inset-0 text-red-500/70" aria-hidden>LEVEL {levelClearAnim.level}</span>
                          <span className="glitch-text-b absolute inset-0 text-neon-blue/70" aria-hidden>LEVEL {levelClearAnim.level}</span>
                        </div>
                        <div className="glitch-stats flex items-center justify-center gap-3 mt-2">
                          <span className="font-pixel text-[8px] neon-text-green">{levelClearAnim.actions} MOVES</span>
                          <span className="text-cabinet-trim">|</span>
                          <span className="font-pixel text-[8px] neon-text-green">{formatTime(levelClearAnim.time)}</span>
                        </div>
                        <div className="glitch-next font-pixel text-[7px] text-gray-500 mt-3 tracking-widest uppercase">
                          &gt; Loading level {levelClearAnim.nextLevel}_
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Game Over / Win overlay */}
                {isGameOver && gameOverAnim && (
                  <div className="absolute inset-0 z-20 overflow-hidden">
                    <div className="absolute inset-0" style={{
                      background: isWin
                        ? 'radial-gradient(ellipse at center, rgba(0,255,255,0.1) 0%, rgba(0,0,0,0.95) 70%)'
                        : 'radial-gradient(ellipse at center, rgba(255,0,100,0.1) 0%, rgba(0,0,0,0.95) 70%)',
                    }} />

                    {isWin && (
                      <div className="absolute inset-0 pointer-events-none">
                        {[...Array(20)].map((_, i) => (
                          <div key={i} className="sparkle absolute rounded-full" style={{
                            width: `${3 + Math.random() * 6}px`, height: `${3 + Math.random() * 6}px`,
                            left: `${10 + Math.random() * 80}%`, top: `${10 + Math.random() * 80}%`,
                            background: ['#00ffff', '#ff00ff', '#00ff66', '#ffff00', '#ff6600'][i % 5],
                            animationDelay: `${Math.random() * 2}s`, animationDuration: `${1.5 + Math.random() * 2}s`,
                          }} />
                        ))}
                      </div>
                    )}

                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className="text-center relative z-10">
                        {isWin ? (
                          <>
                            <div className="mx-auto mb-5">
                              <div className="w-24 h-24 rounded-full flex items-center justify-center mx-auto relative">
                                <div className="absolute inset-0 rounded-full bg-neon-cyan/10 border-2 border-neon-cyan/50 win-ring-pulse" />
                                <div className="absolute inset-1 rounded-full bg-neon-cyan/5" />
                                <Trophy size={36} className="neon-text-cyan win-trophy relative z-10" />
                              </div>
                            </div>
                            <h2 className="font-pixel text-lg neon-text-cyan mb-1 win-title">VICTORY!</h2>
                            <p className="font-pixel text-[7px] text-neon-cyan/50 mb-6 win-subtitle">ALL LEVELS COMPLETE</p>
                            <div className="flex items-center justify-center gap-6 mb-6">
                              <div className="win-stat">
                                <div className="bg-cabinet-panel/90 border border-neon-cyan/20 rounded-lg px-5 py-3 text-center">
                                  <p className="font-pixel text-sm neon-text-cyan">{formatTime(serverTotalTime)}</p>
                                  <p className="font-pixel text-[6px] text-neon-cyan/50 uppercase tracking-widest mt-1">TIME</p>
                                </div>
                              </div>
                              <div className="win-stat" style={{ animationDelay: '0.1s' }}>
                                <div className="bg-cabinet-panel/90 border border-neon-cyan/20 rounded-lg px-5 py-3 text-center">
                                  <p className="font-pixel text-sm neon-text-cyan">{frame?.total_actions}</p>
                                  <p className="font-pixel text-[6px] text-neon-cyan/50 uppercase tracking-widest mt-1">ACTIONS</p>
                                </div>
                              </div>
                            </div>
                            <Link to="/" className="inline-flex items-center gap-2 px-6 py-2.5 arcade-btn arcade-btn-green font-pixel text-[8px] text-white rounded-lg shadow-neon-green win-btn">
                              <Home size={15} /> BACK TO GAMES
                            </Link>
                          </>
                        ) : (
                          <>
                            <div className="gameover-icon mx-auto mb-4">
                              <div className="w-20 h-20 rounded-full flex items-center justify-center mx-auto relative">
                                <div className="absolute inset-0 rounded-full bg-neon-magenta/10 border-2 border-neon-magenta/30" />
                                <Skull size={32} className="text-neon-magenta relative z-10" />
                              </div>
                            </div>
                            <h2 className="font-pixel text-base text-neon-magenta mb-1 gameover-title">GAME OVER</h2>
                            <p className="font-pixel text-[7px] text-neon-magenta/40 mb-5">BETTER LUCK NEXT TIME</p>
                            <div className="flex items-center justify-center gap-5 mb-5">
                              <div className="bg-cabinet-panel/90 border border-neon-magenta/20 rounded-lg px-5 py-3 text-center">
                                <p className="font-pixel text-sm text-neon-magenta">{formatTime(serverTotalTime)}</p>
                                <p className="font-pixel text-[6px] text-neon-magenta/50 uppercase tracking-widest mt-1">TIME</p>
                              </div>
                              <div className="bg-cabinet-panel/90 border border-neon-magenta/20 rounded-lg px-5 py-3 text-center">
                                <p className="font-pixel text-sm text-neon-magenta">{frame?.total_actions}</p>
                                <p className="font-pixel text-[6px] text-neon-magenta/50 uppercase tracking-widest mt-1">ACTIONS</p>
                              </div>
                            </div>
                            <Link to="/" className="inline-flex items-center gap-1.5 px-5 py-2 arcade-btn font-pixel text-[8px] text-gray-300 rounded-lg transition-colors">
                              <Home size={14} /> BACK TO GAMES
                            </Link>
                          </>
                        )}
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {error && (
                <div className="mt-3 mx-2 bg-neon-magenta/10 border border-neon-magenta/40 rounded-lg px-3 py-2.5 flex items-start gap-2">
                  <XIcon size={14} className="text-neon-magenta shrink-0 mt-0.5" />
                  <div className="flex-1 min-w-0">
                    <p className="font-pixel text-[8px] text-neon-magenta">ERROR</p>
                    <p className="font-pixel text-[7px] text-neon-magenta/70 mt-0.5 break-words">{error}</p>
                  </div>
                  <button onClick={() => setError(null)} className="text-neon-magenta/60 hover:text-neon-magenta shrink-0">
                    <XIcon size={12} />
                  </button>
                </div>
              )}

              {frame?.action_map_inferred && (
                <div className="mt-3 mx-2 bg-neon-yellow/10 border border-neon-yellow/40 rounded-lg px-3 py-2.5 flex items-start gap-2">
                  <AlertTriangle size={14} className="neon-text-yellow shrink-0 mt-0.5" />
                  <div className="flex-1 min-w-0">
                    <p className="font-pixel text-[8px] neon-text-yellow">ACTION MAP INFERRED</p>
                    <p className="font-pixel text-[7px] text-neon-yellow/70 mt-0.5">Controls were auto-mapped and may not match the intended layout.</p>
                  </div>
                </div>
              )}
            </ArcadeCabinet>
          )}
        </div>
      </main>

      <style>{`
        @keyframes shake { 0%,100%{transform:translateX(0)} 10%,30%,50%,70%,90%{transform:translateX(-4px)} 20%,40%,60%,80%{transform:translateX(4px)} }
        .animate-shake { animation: shake 0.5s ease-in-out; }
        @keyframes sparkle { 0%{opacity:0;transform:scale(0) translateY(0)} 30%{opacity:1;transform:scale(1) translateY(-10px)} 100%{opacity:0;transform:scale(0.5) translateY(-40px)} }
        .sparkle { animation: sparkle 2.5s ease-out infinite; }
        @keyframes winTrophy { 0%{opacity:0;transform:scale(0) rotate(-30deg)} 50%{transform:scale(1.2) rotate(5deg)} 100%{opacity:1;transform:scale(1) rotate(0)} }
        .win-trophy { animation: winTrophy 0.6s cubic-bezier(0.34,1.56,0.64,1) forwards; }
        @keyframes ringPulse { 0%,100%{transform:scale(1);opacity:1} 50%{transform:scale(1.08);opacity:0.8} }
        .win-ring-pulse { animation: ringPulse 2s ease-in-out infinite; }
        @keyframes winTitle { 0%{opacity:0;transform:translateY(15px)} 100%{opacity:1;transform:translateY(0)} }
        .win-title { animation: winTitle 0.5s ease-out 0.3s both; }
        .win-subtitle { animation: winTitle 0.5s ease-out 0.4s both; }
        @keyframes winStat { 0%{opacity:0;transform:translateY(20px) scale(0.9)} 100%{opacity:1;transform:translateY(0) scale(1)} }
        .win-stat { animation: winStat 0.5s ease-out 0.5s both; }
        .win-btn { animation: winTitle 0.4s ease-out 0.7s both; }
        @keyframes gameoverIcon { 0%{opacity:0;transform:scale(1.5)} 100%{opacity:1;transform:scale(1)} }
        .gameover-icon { animation: gameoverIcon 0.4s ease-out; }
        .gameover-title { animation: winTitle 0.4s ease-out 0.2s both; }

        .glitch-scanlines {
          background: repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,255,255,0.03) 2px,rgba(0,255,255,0.03) 4px);
          animation: glitchScanlines 2s steps(1) forwards;
        }
        @keyframes glitchScanlines { 0%{opacity:1} 70%{opacity:1} 100%{opacity:0} }

        .glitch-rgb-shift { mix-blend-mode:screen; animation: rgbShift 0.4s steps(2) 3; }
        @keyframes rgbShift {
          0%{box-shadow:-3px 0 0 rgba(255,0,0,0.15) inset,3px 0 0 rgba(0,255,255,0.15) inset}
          25%{box-shadow:2px 0 0 rgba(255,0,0,0.2) inset,-2px 0 0 rgba(0,255,255,0.2) inset}
          50%{box-shadow:-4px 0 0 rgba(255,0,0,0.15) inset,4px 0 0 rgba(0,0,255,0.15) inset}
          75%{box-shadow:1px 0 0 rgba(0,255,0,0.2) inset,-3px 0 0 rgba(255,0,255,0.15) inset}
          100%{box-shadow:none}
        }
        .glitch-bar { animation: glitchBar 0.15s steps(1) 4; }
        @keyframes glitchBar { 0%{transform:translateX(0);opacity:0} 20%{transform:translateX(-20px);opacity:1} 40%{transform:translateX(15px);opacity:0.7} 60%{transform:translateX(-8px);opacity:1} 80%{transform:translateX(5px);opacity:0.5} 100%{transform:translateX(0);opacity:0} }
        .glitch-noise { animation: glitchNoise 0.3s steps(3) 2; background-size:100px 100px; }
        @keyframes glitchNoise { 0%{opacity:0;background:url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.4'/%3E%3C/svg%3E")} 33%{opacity:0.4} 66%{opacity:0.2} 100%{opacity:0} }
        .glitch-blackout { animation: glitchBlackout 2s steps(1) forwards; background:#000; }
        @keyframes glitchBlackout { 0%{opacity:0} 15%{opacity:0} 16%{opacity:1} 18%{opacity:0.6} 20%{opacity:1} 22%{opacity:0.8} 25%{opacity:1} 75%{opacity:1} 80%{opacity:0.6} 85%{opacity:0.3} 90%{opacity:0} 100%{opacity:0} }
        .glitch-text-container { animation: glitchTextContainer 2s steps(1) forwards; }
        @keyframes glitchTextContainer { 0%{opacity:0} 25%{opacity:0} 26%{opacity:1} 75%{opacity:1} 85%{opacity:0} 100%{opacity:0} }
        .glitch-text-main { position:relative; z-index:2; }
        .glitch-text-r { animation: glitchTextR 0.3s steps(2) infinite; }
        .glitch-text-b { animation: glitchTextB 0.3s steps(2) 0.1s infinite; }
        @keyframes glitchTextR { 0%{transform:translate(0);clip-path:inset(20% 0 40% 0)} 25%{transform:translate(-3px,1px);clip-path:inset(50% 0 10% 0)} 50%{transform:translate(2px,-1px);clip-path:inset(10% 0 60% 0)} 75%{transform:translate(-1px,2px);clip-path:inset(40% 0 20% 0)} 100%{transform:translate(0);clip-path:inset(30% 0 30% 0)} }
        @keyframes glitchTextB { 0%{transform:translate(0);clip-path:inset(60% 0 5% 0)} 25%{transform:translate(3px,-1px);clip-path:inset(10% 0 50% 0)} 50%{transform:translate(-2px,1px);clip-path:inset(30% 0 30% 0)} 75%{transform:translate(1px,-2px);clip-path:inset(5% 0 60% 0)} 100%{transform:translate(0);clip-path:inset(40% 0 20% 0)} }
        .glitch-stats { animation: glitchFadeIn 0.3s ease-out 0.6s both; }
        .glitch-next { animation: glitchBlink 0.5s steps(1) 0.8s infinite; }
        @keyframes glitchFadeIn { 0%{opacity:0;transform:translateY(5px)} 100%{opacity:1;transform:translateY(0)} }
        @keyframes glitchBlink { 0%,49%{opacity:1} 50%,100%{opacity:0.3} }
      `}</style>
    </div>
  );
}
