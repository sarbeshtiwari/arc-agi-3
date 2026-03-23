import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { playerAPI } from '../api/client';
import GameCanvas from '../components/GameCanvas';
import {
  ArrowUp, ArrowDown, ArrowLeft, ArrowRight,
  RotateCcw, Undo, Zap, Trophy, Square, X as XIcon,
  ChevronLeft, Skull, Clock, Layers, Home, Target
} from 'lucide-react';

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
  const ACTION_THROTTLE_MS = 50;

  const [resetAnim, setResetAnim] = useState(false);
  const [gameOverAnim, setGameOverAnim] = useState(false);
  const [canvasShake, setCanvasShake] = useState(false);
  const [lastAction, setLastAction] = useState(null);
  const prevStateRef = useRef(null);

  // Level transition
  const prevLevelRef = useRef(null);
  const [levelClearAnim, setLevelClearAnim] = useState(null);

  const isPlaying = !!sessionGuid;
  const isGameOver = frame?.state === 'WIN' || frame?.state === 'GAME_OVER';
  const isWin = frame?.state === 'WIN';
  const serverTotalTime = frame?.metadata?.total_time || 0;
  const completedLevels = frame?.metadata?.completed_levels || [];
  const currentLevelStats = frame?.metadata?.current_level_stats;

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
      startTimer();
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to start game');
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
        // Server lost the session -- auto-recover
        setSessionGuid(null);
        setFrame(null);
        prevStateRef.current = null;
        setTimeout(() => doStart(), 200);
        return;
      }
      setError(detail || 'Action failed');
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
    } catch (err) { setError(err.response?.data?.detail || 'Reset failed'); }
  }, [sessionGuid, startTimer]);


  // Keyboard
  useEffect(() => {
    if (!isPlaying) return;
    const handle = (e) => {
      const key = e.key.toLowerCase();
      if (key === 'r' && !e.ctrlKey && !e.metaKey) { e.preventDefault(); resetGame(); return; }
      if (isGameOver) return;
      if ((e.ctrlKey || e.metaKey) && key === 'z') { e.preventDefault(); if (frame?.available_actions?.includes('ACTION7')) sendAction('ACTION7'); return; }
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
      <div className="min-h-screen bg-gray-950 flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-500" />
      </div>
    );
  }

  if (error && !isPlaying) {
    return (
      <div className="min-h-screen bg-gray-950 flex items-center justify-center">
        <div className="text-center">
          <XIcon size={48} className="mx-auto text-red-400 mb-4" />
          <p className="text-red-400 font-medium mb-2">Failed to load game</p>
          <p className="text-gray-500 text-sm mb-6">{error}</p>
          <Link to="/" className="text-blue-400 hover:text-blue-300 text-sm">Back to home</Link>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-950 flex flex-col">
      {/* ── Top Bar ── */}
      <header className="bg-gray-900/80 backdrop-blur-sm border-b border-gray-800 sticky top-0 z-30">
        <div className="max-w-6xl mx-auto px-4 h-12 flex items-center gap-3">
          <Link to="/" onClick={() => { if (sessionGuid) endGame(); }} className="p-1 rounded-md hover:bg-gray-800 text-gray-500 hover:text-white transition-colors">
            <ChevronLeft size={18} />
          </Link>
          <span className="text-white font-mono font-bold text-sm">{gameId}</span>
          <span className="px-2 py-0.5 bg-purple-500/20 text-purple-400 text-[10px] font-medium rounded-full">Direct Play</span>
          {playerName && <span className="text-blue-400 text-xs">({playerName})</span>}

          <div className="flex-1" />

          {isPlaying && (
            <>
              <StatPill icon={Layers} value={`Lv ${isWin && totalLevels ? totalLevels : (frame?.level ?? 0) + 1}${totalLevels ? ' / ' + totalLevels : ''}`} color="blue" />
              <StatPill icon={Target} value={frame?.total_actions || 0} color="green" />
              <StatPill
                icon={Clock}
                value={formatTime(isGameOver ? serverTotalTime : elapsed)}
                color={isWin ? 'green' : isGameOver ? 'red' : 'white'}
              />
              <button
                onClick={async () => { await endGame(); navigate('/'); }}
                className="ml-2 flex items-center gap-1 px-2.5 py-1 text-[11px] text-red-400 hover:bg-red-500/10 border border-red-500/20 rounded-md transition-colors"
              >
                <Square size={10} /> End
              </button>
            </>
          )}
        </div>
      </header>

      {/* ── Main Area ── */}
      <main className="flex-1 flex items-start justify-center px-4 py-6">
        <div className="w-full max-w-4xl">
          {!isPlaying ? (
            <div className="flex items-center justify-center py-20">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-500" />
            </div>
          ) : (
            <div className="flex gap-5 items-start">
              {/* ── Canvas ── */}
              <div className="flex-1 min-w-0">
                <div className={`relative rounded-xl overflow-hidden border-2 transition-all duration-300 ${
                  isWin ? 'border-green-500/50 shadow-[0_0_30px_rgba(74,222,128,0.15)]' :
                  isGameOver ? 'border-red-500/50 shadow-[0_0_30px_rgba(248,113,113,0.15)]' :
                  'border-gray-800'
                }`}>
                  <div className={`bg-gray-900 p-3 transition-all duration-300 ${
                    resetAnim ? 'opacity-0 scale-95' : 'opacity-100 scale-100'
                  } ${canvasShake ? 'animate-shake' : ''}`}>
                    <GameCanvas
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
                          <div key={i} className="glitch-bar absolute w-full bg-white/10"
                            style={{ height: `${2 + Math.random() * 8}px`, top: `${10 + Math.random() * 80}%`, animationDelay: `${Math.random() * 0.3}s`, animationDuration: `${0.1 + Math.random() * 0.15}s` }} />
                        ))}
                      </div>
                      <div className="absolute inset-0 glitch-noise" />
                      <div className="absolute inset-0 glitch-blackout" />
                      <div className="absolute inset-0 flex items-center justify-center glitch-text-container">
                        <div className="text-center">
                          <div className="glitch-level-text font-mono text-3xl font-black text-cyan-400 relative">
                            <span className="glitch-text-main">LEVEL {levelClearAnim.level}</span>
                            <span className="glitch-text-r absolute inset-0 text-red-500/70" aria-hidden>LEVEL {levelClearAnim.level}</span>
                            <span className="glitch-text-b absolute inset-0 text-blue-500/70" aria-hidden>LEVEL {levelClearAnim.level}</span>
                          </div>
                          <div className="glitch-stats flex items-center justify-center gap-3 mt-2">
                            <span className="font-mono text-xs text-green-400/80">{levelClearAnim.actions} MOVES</span>
                            <span className="text-gray-600">|</span>
                            <span className="font-mono text-xs text-green-400/80">{formatTime(levelClearAnim.time)}</span>
                          </div>
                          <div className="glitch-next font-mono text-[10px] text-gray-500 mt-3 tracking-widest uppercase">
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
                          ? 'radial-gradient(ellipse at center, rgba(16,185,129,0.15) 0%, rgba(0,0,0,0.95) 70%)'
                          : 'radial-gradient(ellipse at center, rgba(239,68,68,0.15) 0%, rgba(0,0,0,0.95) 70%)',
                      }} />

                      {isWin && (
                        <div className="absolute inset-0 pointer-events-none">
                          {[...Array(20)].map((_, i) => (
                            <div key={i} className="sparkle absolute rounded-full" style={{
                              width: `${3 + Math.random() * 6}px`, height: `${3 + Math.random() * 6}px`,
                              left: `${10 + Math.random() * 80}%`, top: `${10 + Math.random() * 80}%`,
                              background: ['#10b981', '#fbbf24', '#60a5fa', '#f472b6', '#a78bfa'][i % 5],
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
                                  <div className="absolute inset-0 rounded-full bg-gradient-to-br from-emerald-500/20 to-teal-500/20 border-2 border-emerald-400/50 win-ring-pulse" />
                                  <div className="absolute inset-1 rounded-full bg-gradient-to-br from-emerald-500/10 to-transparent" />
                                  <Trophy size={36} className="text-emerald-400 win-trophy relative z-10" />
                                </div>
                              </div>
                              <h2 className="text-3xl font-black text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 via-teal-300 to-cyan-400 mb-1 win-title">Victory!</h2>
                              <p className="text-emerald-400/60 text-sm mb-6 win-subtitle">You completed all levels</p>
                              <div className="flex items-center justify-center gap-6 mb-6">
                                <div className="win-stat">
                                  <div className="bg-gray-900/80 backdrop-blur-sm border border-emerald-500/20 rounded-xl px-5 py-3 text-center">
                                    <p className="text-2xl font-bold text-white font-mono">{formatTime(serverTotalTime)}</p>
                                    <p className="text-[10px] text-emerald-400/70 uppercase tracking-widest mt-0.5">Total Time</p>
                                  </div>
                                </div>
                                <div className="win-stat" style={{ animationDelay: '0.1s' }}>
                                  <div className="bg-gray-900/80 backdrop-blur-sm border border-emerald-500/20 rounded-xl px-5 py-3 text-center">
                                    <p className="text-2xl font-bold text-white font-mono">{frame?.total_actions}</p>
                                    <p className="text-[10px] text-emerald-400/70 uppercase tracking-widest mt-0.5">Actions</p>
                                  </div>
                                </div>
                              </div>
                              <Link to="/" className="inline-flex items-center gap-2 px-6 py-2.5 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white text-sm font-medium rounded-xl transition-all shadow-lg shadow-emerald-500/20 win-btn">
                                <Home size={15} /> Back to Games
                              </Link>
                            </>
                          ) : (
                            <>
                              <div className="gameover-icon mx-auto mb-4">
                                <div className="w-20 h-20 rounded-full flex items-center justify-center mx-auto relative">
                                  <div className="absolute inset-0 rounded-full bg-red-500/10 border-2 border-red-500/30" />
                                  <Skull size={32} className="text-red-400 relative z-10" />
                                </div>
                              </div>
                              <h2 className="text-2xl font-bold text-red-400 mb-1 gameover-title">Game Over</h2>
                              <p className="text-red-400/50 text-sm mb-5">Better luck next time</p>
                              <div className="flex items-center justify-center gap-5 mb-5">
                                <div className="bg-gray-900/80 backdrop-blur-sm border border-red-500/15 rounded-xl px-5 py-3 text-center">
                                  <p className="text-xl font-bold text-white font-mono">{formatTime(serverTotalTime)}</p>
                                  <p className="text-[10px] text-red-400/60 uppercase tracking-widest mt-0.5">Time</p>
                                </div>
                                <div className="bg-gray-900/80 backdrop-blur-sm border border-red-500/15 rounded-xl px-5 py-3 text-center">
                                  <p className="text-xl font-bold text-white font-mono">{frame?.total_actions}</p>
                                  <p className="text-[10px] text-red-400/60 uppercase tracking-widest mt-0.5">Actions</p>
                                </div>
                              </div>
                              <Link to="/" className="inline-flex items-center gap-1.5 px-5 py-2 bg-gray-800 hover:bg-gray-700 text-gray-300 text-sm rounded-lg transition-colors">
                                <Home size={14} /> Back to Games
                              </Link>
                            </>
                          )}
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                {error && (
                  <div className="mt-3 bg-red-500/10 border border-red-500/30 rounded-lg p-2.5">
                    <p className="text-xs text-red-400">{error}</p>
                  </div>
                )}

                {/* ── Inline Controls ── */}
                <div className="mt-4 flex items-center justify-center gap-2">
                  <DPadButton icon={ArrowLeft} action="ACTION3" active={lastAction === 'ACTION3'} available={available} disabled={isGameOver} onClick={() => sendAction('ACTION3')} kbd="A" />
                  <div className="flex flex-col gap-1.5">
                    <DPadButton icon={ArrowUp} action="ACTION1" active={lastAction === 'ACTION1'} available={available} disabled={isGameOver} onClick={() => sendAction('ACTION1')} kbd="W" />
                    <DPadButton icon={ArrowDown} action="ACTION2" active={lastAction === 'ACTION2'} available={available} disabled={isGameOver} onClick={() => sendAction('ACTION2')} kbd="S" />
                  </div>
                  <DPadButton icon={ArrowRight} action="ACTION4" active={lastAction === 'ACTION4'} available={available} disabled={isGameOver} onClick={() => sendAction('ACTION4')} kbd="D" />

                  <div className="w-px h-10 bg-gray-800 mx-2" />

                  <ActionButton icon={Zap} label="Action" action="ACTION5" active={lastAction === 'ACTION5'} available={available} disabled={isGameOver} onClick={() => sendAction('ACTION5')} kbd="Space" />
                  <ActionButton icon={Undo} label="Undo" action="ACTION7" active={lastAction === 'ACTION7'} available={available} disabled={isGameOver} onClick={() => sendAction('ACTION7')} kbd="Ctrl+Z" />
                  <ActionButton icon={RotateCcw} label="Reset" action="_reset" available={['_reset']} disabled={false} onClick={resetGame} kbd="R" spinning={resetAnim} />

                  <div className="w-px h-10 bg-gray-800 mx-2" />

                  {/* Level indicator */}
                  <div className="flex items-center gap-1.5 px-3 h-11 bg-gray-800/60 rounded-xl border border-gray-700/50">
                    <Layers size={14} className="text-blue-400" />
                    <span className="text-xs font-mono text-white font-medium">
                      {isWin && totalLevels ? totalLevels : (frame?.level ?? 0) + 1}{totalLevels ? ` / ${totalLevels}` : ''}
                    </span>
                  </div>
                </div>
              </div>

              {/* ── Right Panel ── */}
              <div className="w-52 shrink-0 space-y-3 hidden md:block">
                {(completedLevels.length > 0 || currentLevelStats) && (
                  <div className="bg-gray-900 rounded-xl border border-gray-800 p-3.5">
                    <h3 className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider mb-2.5">Levels</h3>
                    <div className="space-y-1.5">
                      {completedLevels.map((ls, i) => (
                        <div key={i} className="bg-gray-800/50 rounded-lg p-2">
                          <div className="flex items-center gap-2 text-[11px]">
                            <span className="w-4 h-4 rounded-full bg-green-500/20 border border-green-500/40 flex items-center justify-center text-green-400 text-[8px] font-bold shrink-0">{ls.level + 1}</span>
                            <span className="text-gray-400 flex-1">{ls.actions}m</span>
                            <span className="text-gray-500 font-mono tabular-nums">{formatTime(ls.time)}</span>
                          </div>
                          <div className="flex items-center gap-2 mt-1 ml-6 text-[10px] text-gray-500">
                            {ls.game_overs > 0 && <span className="text-red-400">{ls.game_overs} death{ls.game_overs > 1 ? 's' : ''}</span>}
                            {ls.lives_used > 0 && <span>{ls.lives_used} lives</span>}
                            {ls.resets > 0 && <span>{ls.resets} reset{ls.resets > 1 ? 's' : ''}</span>}
                          </div>
                        </div>
                      ))}
                      {currentLevelStats && !isGameOver && (
                        <div className="bg-blue-500/5 border border-blue-500/20 rounded-lg p-2">
                          <div className="flex items-center gap-2 text-[11px]">
                            <span className="w-4 h-4 rounded-full bg-blue-500/20 border border-blue-500/40 flex items-center justify-center text-blue-400 text-[8px] font-bold shrink-0 animate-pulse">{currentLevelStats.level + 1}</span>
                            <span className="text-gray-400 flex-1">{currentLevelStats.actions}m</span>
                            <span className="text-blue-400 font-mono tabular-nums">{formatTime(currentLevelStats.time)}</span>
                          </div>
                          <div className="flex items-center gap-2 mt-1 ml-6 text-[10px] text-gray-500">
                            {currentLevelStats.game_overs > 0 && <span className="text-red-400">{currentLevelStats.game_overs} death{currentLevelStats.game_overs > 1 ? 's' : ''}</span>}
                            {currentLevelStats.lives_used > 0 && <span>{currentLevelStats.lives_used} lives</span>}
                            {currentLevelStats.resets > 0 && <span>{currentLevelStats.resets} reset{currentLevelStats.resets > 1 ? 's' : ''}</span>}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                <div className="bg-gray-900 rounded-xl border border-gray-800 p-3.5">
                  <h3 className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider mb-2.5">Shortcuts</h3>
                  <div className="space-y-1.5">
                    <KbdRow keys="W A S D" desc="Move" />
                    <KbdRow keys="Space" desc="Action" />
                    <KbdRow keys="Click" desc="Coordinate" />
                    <KbdRow keys="Ctrl+Z" desc="Undo" />
                    <KbdRow keys="R" desc="Reset level" />
                  </div>
                </div>
              </div>
            </div>
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

// ── Sub-components (same as PublicPlayPage) ──

function StatPill({ icon: Icon, value, color = 'white' }: any) {
  const colors = { blue: 'text-blue-400', green: 'text-green-400', red: 'text-red-400', white: 'text-white' };
  return (
    <div className="flex items-center gap-1.5 px-2.5 py-1 bg-gray-800/60 rounded-md border border-gray-700/50">
      <Icon size={12} className={colors[color]} />
      <span className={`font-mono text-xs font-medium tabular-nums ${colors[color]}`}>{value}</span>
    </div>
  );
}

function OverlayStat({ label, value }: any) {
  return (
    <div className="text-center">
      <p className="text-xl font-bold text-white font-mono">{value}</p>
      <p className="text-[10px] text-gray-500 uppercase tracking-wider">{label}</p>
    </div>
  );
}

function DPadButton({ icon: Icon, action, active, available, disabled, onClick, kbd }: any) {
  const enabled = available.includes(action) && !disabled;
  return (
    <button onClick={onClick} disabled={!enabled} title={kbd}
      className={`w-11 h-11 rounded-xl flex items-center justify-center transition-all duration-100 ${
        !enabled ? 'bg-gray-800/30 text-gray-700 cursor-not-allowed' :
        active ? 'bg-blue-600 text-white scale-90 shadow-lg shadow-blue-500/20' :
        'bg-gray-800 hover:bg-gray-700 text-gray-300 border border-gray-700 active:scale-90'
      }`}>
      <Icon size={18} />
    </button>
  );
}

function ActionButton({ icon: Icon, label, action, active, available, disabled, onClick, kbd, spinning }: any) {
  const enabled = available.includes(action) && !disabled;
  return (
    <button onClick={onClick} disabled={!enabled} title={`${label} (${kbd})`}
      className={`h-11 px-3 rounded-xl flex items-center gap-1.5 transition-all duration-100 text-xs font-medium ${
        !enabled ? 'bg-gray-800/30 text-gray-700 cursor-not-allowed' :
        active ? 'bg-blue-600 text-white scale-95' :
        'bg-gray-800 hover:bg-gray-700 text-gray-300 border border-gray-700 active:scale-95'
      }`}>
      <Icon size={14} className={spinning ? 'animate-spin' : ''} />
      <span className="hidden sm:inline">{label}</span>
    </button>
  );
}

function KbdRow({ keys, desc }: any) {
  return (
    <div className="flex items-center justify-between">
      <div className="flex gap-1">
        {keys.split(' ').map((k) => (
          <kbd key={k} className="px-1.5 py-0.5 bg-gray-800 border border-gray-700 rounded text-[10px] font-mono text-gray-400">{k}</kbd>
        ))}
      </div>
      <span className="text-[10px] text-gray-500">{desc}</span>
    </div>
  );
}
