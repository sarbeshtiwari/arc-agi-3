import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { gamesAPI, playerAPI } from '../api/client';
import GameCanvas from '../components/GameCanvas';
import VideoRecorder from '../components/VideoRecorder';
import {
  ArrowUp, ArrowDown, ArrowLeft, ArrowRight,
  RotateCcw, Undo, Zap, Trophy, Square, Play,
  ChevronLeft, Skull, Clock, Layers, Home, Target, X as XIcon, AlertTriangle
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

export default function GamePlayPage() {
  const { gameId } = useParams();
  const navigate = useNavigate();
  const [game, setGame] = useState(null);
  const [gameLoading, setGameLoading] = useState(true);

  const [sessionGuid, setSessionGuid] = useState(null);
  const [frame, setFrame] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

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

  const isPlaying = !!sessionGuid;
  const isGameOver = frame?.state === 'WIN' || frame?.state === 'GAME_OVER';
  const isWin = frame?.state === 'WIN';
  const serverTotalTime = frame?.metadata?.total_time || 0;
  const completedLevels = frame?.metadata?.completed_levels || [];
  const currentLevelStats = frame?.metadata?.current_level_stats;
  const totalLevels = game?.baseline_actions?.length || 0;

  const setErrorWithDismiss = useCallback((msg: string | null, timeout = 6000) => {
    if (errorTimerRef.current) clearTimeout(errorTimerRef.current);
    setError(msg);
    if (msg && timeout > 0) {
      errorTimerRef.current = setTimeout(() => setError(null), timeout);
    }
  }, []);

  useEffect(() => () => { if (errorTimerRef.current) clearTimeout(errorTimerRef.current); }, []);

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

  useEffect(() => {
    gamesAPI.get(gameId)
      .then((res) => setGame(res.data))
      .catch(() => navigate('/admin/games'))
      .finally(() => setGameLoading(false));
  }, [gameId, navigate]);

  async function doStartGame(startLevel = 0) {
    setLoading(true); setError(null); setGameOverAnim(false);
    try {
      const seed = Math.floor(Math.random() * 1000000);
      const res = await playerAPI.start(gameId, seed, startLevel);
      setSessionGuid(res.data.metadata?.session_guid);
      setFrame(res.data);
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
      const res = await playerAPI.action(sessionGuid, action, x, y);
      setFrame(res.data);
    } catch (err) {
      const detail = err.response?.data?.detail || '';
      if (err.response?.status === 404 || detail.includes('No game instance')) {
        setErrorWithDismiss('Session lost (server restarted). Click Start Game to begin a new session.', 0);
        setSessionGuid(null); setFrame(null); prevStateRef.current = null;
        return;
      }
      setErrorWithDismiss(detail || `Action '${action}' failed`);
    }
  }, [sessionGuid]);

  const endGame = useCallback(async () => {
    if (!sessionGuid) return;
    stopTimer();
    try { await playerAPI.end(sessionGuid); } catch (e) {}
    setSessionGuid(null); setFrame(null); setGameOverAnim(false);
    prevStateRef.current = null; setElapsed(0);
  }, [sessionGuid, stopTimer]);

  const resetGame = useCallback(async () => {
    if (!sessionGuid) return;
    setResetAnim(true);
    setTimeout(() => setResetAnim(false), 400);
    try {
      const res = await playerAPI.action(sessionGuid, 'RESET');
      setFrame(res.data); setGameOverAnim(false);
      prevStateRef.current = res.data.state;
      startTimer();
    } catch (err) { setErrorWithDismiss(err.response?.data?.detail || 'Reset failed'); }
  }, [sessionGuid, startTimer]);

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

  if (gameLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500" />
      </div>
    );
  }

  return (
    <div className="min-h-[calc(100vh-48px)]">
      {/* Header */}
      <div className="flex items-center gap-3 mb-5">
        <Link to={`/admin/games/${gameId}`} className="p-1.5 rounded-lg hover:bg-gray-800 text-gray-500 hover:text-white transition-colors">
          <ChevronLeft size={18} />
        </Link>
        <div className="flex-1 min-w-0">
          <h1 className="text-lg font-bold text-white truncate">{game?.name || gameId}</h1>
          <span className="text-xs text-gray-500 font-mono">{gameId}</span>
        </div>

        {isPlaying && (
          <div className="flex items-center gap-2">
            <StatPill icon={Layers} value={`Lv ${isWin && totalLevels ? totalLevels : (frame?.level ?? 0) + 1}${totalLevels ? ' / ' + totalLevels : ''}`} color="blue" />
            <StatPill icon={Target} value={frame?.total_actions || 0} color="green" />
            <StatPill icon={Clock} value={formatTime(isGameOver ? serverTotalTime : elapsed)} color={isWin ? 'green' : isGameOver ? 'red' : 'white'} />
             <VideoRecorder
              gameId={gameId}
              isPlaying={isPlaying}
              isEphemeral={false}
              isGameOver={isGameOver}
              onPromptDismissed={startTimer}
            />
            <button
              onClick={async () => { await endGame(); navigate(`/admin/games/${gameId}`); }}
              className="flex items-center gap-1 px-2.5 py-1 text-[11px] text-red-400 hover:bg-red-500/10 border border-red-500/20 rounded-md transition-colors"
            >
              <Square size={10} /> End
            </button>
          </div>
        )}
      </div>

      {!isPlaying ? (
        <div className="flex items-center justify-center py-20">
          <div className="text-center">
            <div className="w-20 h-20 rounded-2xl bg-blue-600/20 border border-blue-500/30 flex items-center justify-center mx-auto mb-5">
              <Play size={36} className="text-blue-400 ml-1" />
            </div>
            <h2 className="text-xl font-bold text-white mb-2">Ready to Play</h2>
            <p className="text-gray-500 text-sm mb-6">{game?.name || gameId}</p>
            <button
              onClick={() => doStartGame()}
              disabled={loading}
              className="px-8 py-3 bg-blue-600 hover:bg-blue-500 disabled:bg-blue-600/50 text-white font-medium rounded-xl transition-colors"
            >
              {loading ? 'Starting...' : 'Start Game'}
            </button>
          </div>
        </div>
      ) : (
        <div className="flex gap-5 items-start">
          {/* Canvas */}
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
                              <Trophy size={36} className="text-emerald-400 win-trophy relative z-10" />
                            </div>
                          </div>
                          <h2 className="text-3xl font-black text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 via-teal-300 to-cyan-400 mb-1 win-title">Victory!</h2>
                          <p className="text-emerald-400/60 text-sm mb-6 win-subtitle">All levels completed</p>
                          <div className="flex items-center justify-center gap-6 mb-6">
                            <div className="win-stat"><div className="bg-gray-900/80 backdrop-blur-sm border border-emerald-500/20 rounded-xl px-5 py-3 text-center"><p className="text-2xl font-bold text-white font-mono">{formatTime(serverTotalTime)}</p><p className="text-[10px] text-emerald-400/70 uppercase tracking-widest mt-0.5">Time</p></div></div>
                            <div className="win-stat" style={{animationDelay:'0.1s'}}><div className="bg-gray-900/80 backdrop-blur-sm border border-emerald-500/20 rounded-xl px-5 py-3 text-center"><p className="text-2xl font-bold text-white font-mono">{frame?.total_actions}</p><p className="text-[10px] text-emerald-400/70 uppercase tracking-widest mt-0.5">Actions</p></div></div>
                          </div>
                          <Link to={`/admin/games/${gameId}`} className="inline-flex items-center gap-2 px-6 py-2.5 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white text-sm font-medium rounded-xl transition-all shadow-lg shadow-emerald-500/20 win-btn">
                            <ChevronLeft size={15} /> Back to Game
                          </Link>
                        </>
                      ) : (
                        <>
                          <div className="gameover-icon mx-auto mb-4"><div className="w-20 h-20 rounded-full flex items-center justify-center mx-auto relative"><div className="absolute inset-0 rounded-full bg-red-500/10 border-2 border-red-500/30" /><Skull size={32} className="text-red-400 relative z-10" /></div></div>
                          <h2 className="text-2xl font-bold text-red-400 mb-1 gameover-title">Game Over</h2>
                          <p className="text-red-400/50 text-sm mb-5">Better luck next time</p>
                          <div className="flex items-center justify-center gap-5 mb-5">
                            <div className="bg-gray-900/80 backdrop-blur-sm border border-red-500/15 rounded-xl px-5 py-3 text-center"><p className="text-xl font-bold text-white font-mono">{formatTime(serverTotalTime)}</p><p className="text-[10px] text-red-400/60 uppercase tracking-widest mt-0.5">Time</p></div>
                            <div className="bg-gray-900/80 backdrop-blur-sm border border-red-500/15 rounded-xl px-5 py-3 text-center"><p className="text-xl font-bold text-white font-mono">{frame?.total_actions}</p><p className="text-[10px] text-red-400/60 uppercase tracking-widest mt-0.5">Actions</p></div>
                          </div>
                          <Link to={`/admin/games/${gameId}`} className="inline-flex items-center gap-1.5 px-5 py-2 bg-gray-800 hover:bg-gray-700 text-gray-300 text-sm rounded-lg transition-colors"><ChevronLeft size={14} /> Back to Game</Link>
                        </>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Controls bar */}
            <div className="mt-4 flex items-center justify-center gap-2">
              <DPadButton icon={ArrowLeft} action="ACTION3" active={lastAction === 'ACTION3'} available={available} disabled={isGameOver} onClick={() => sendAction('ACTION3')} kbd="A" />
              <div className="flex flex-col gap-1.5">
                <DPadButton icon={ArrowUp} action="ACTION1" active={lastAction === 'ACTION1'} available={available} disabled={isGameOver} onClick={() => sendAction('ACTION1')} kbd="W" />
                <DPadButton icon={ArrowDown} action="ACTION2" active={lastAction === 'ACTION2'} available={available} disabled={isGameOver} onClick={() => sendAction('ACTION2')} kbd="S" />
              </div>
              <DPadButton icon={ArrowRight} action="ACTION4" active={lastAction === 'ACTION4'} available={available} disabled={isGameOver} onClick={() => sendAction('ACTION4')} kbd="D" />
              <div className="w-px h-10 bg-gray-800 mx-2" />
              <ActionBtn icon={Zap} label="Action" action="ACTION5" active={lastAction === 'ACTION5'} available={available} disabled={isGameOver} onClick={() => sendAction('ACTION5')} kbd="Space" />
              <ActionBtn icon={Undo} label="Undo" action="ACTION7" active={lastAction === 'ACTION7'} available={available} disabled={isGameOver} onClick={() => sendAction('ACTION7')} kbd="Z" />
              <ActionBtn icon={RotateCcw} label="Reset" action="_reset" available={['_reset']} disabled={false} onClick={resetGame} kbd="R" spinning={resetAnim} />
              <div className="w-px h-10 bg-gray-800 mx-2" />
              <div className="flex items-center gap-1.5 px-3 h-11 bg-gray-800/60 rounded-xl border border-gray-700/50">
                <Layers size={14} className="text-blue-400" />
                <span className="text-xs font-mono text-white font-medium">
                  {isWin && totalLevels ? totalLevels : (frame?.level ?? 0) + 1}{totalLevels ? ` / ${totalLevels}` : ''}
                </span>
              </div>
            </div>

            {error && (
              <div className="mt-3 bg-red-950/60 border border-red-500/40 rounded-lg px-3 py-2.5 flex items-start gap-2">
                <XIcon size={14} className="text-red-400 shrink-0 mt-0.5" />
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-medium text-red-400">Error</p>
                  <p className="text-[11px] text-red-300/80 mt-0.5 break-words font-mono">{error}</p>
                </div>
                <button onClick={() => setError(null)} className="text-red-400/60 hover:text-red-400 shrink-0">
                  <XIcon size={12} />
                </button>
              </div>
            )}

            {frame?.action_map_inferred && (
              <div className="mt-3 bg-amber-950/60 border border-amber-500/40 rounded-lg px-3 py-2.5 flex items-start gap-2">
                <AlertTriangle size={14} className="text-amber-400 shrink-0 mt-0.5" />
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-medium text-amber-400">Action Map Inferred</p>
                  <p className="text-[11px] text-amber-300/80 mt-0.5">This game does not define an ACTION_MAP. Controls were auto-mapped and may not match the intended layout.</p>
                </div>
              </div>
            )}
          </div>

          {/* Right Panel */}
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
      `}</style>
    </div>
  );
}

function StatPill({ icon: Icon, value, color = 'white' }: any) {
  const colors = { blue: 'text-blue-400', green: 'text-green-400', red: 'text-red-400', white: 'text-white' };
  return (
    <div className="flex items-center gap-1.5 px-2.5 py-1 bg-gray-800/60 rounded-md border border-gray-700/50">
      <Icon size={12} className={colors[color]} />
      <span className={`font-mono text-xs font-medium tabular-nums ${colors[color]}`}>{value}</span>
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

function ActionBtn({ icon: Icon, label, action, active, available, disabled, onClick, kbd, spinning }: any) {
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
