import { useState, useCallback, useRef } from 'react';
import { playerAPI } from '../api/client';

// ARC-AGI-3 color palette as hex
// Source: app.py -> _PALETTE_HEX
export const ARC_COLORS = {
  0: '#FFFFFF',   // White
  1: '#CCCCCC',   // Off-white
  2: '#999999',   // Light Grey
  3: '#666666',   // Grey
  4: '#333333',   // Dark Grey
  5: '#000000',   // Black
  6: '#E53AA3',   // Magenta
  7: '#FF7BCC',   // Pink
  8: '#F93C31',   // Red
  9: '#1E93FF',   // Blue
  10: '#88D8F1',  // Light Blue
  11: '#FFDC00',  // Yellow
  12: '#FF851B',  // Orange
  13: '#921231',  // Maroon
  14: '#4FCC30',  // Green
  15: '#A356D6',  // Purple
};

export function useGameEngine() {
  const [sessionGuid, setSessionGuid] = useState(null);
  const [frame, setFrame] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const sessionRef = useRef(null);

  const startGame = useCallback(async (gameId, seed = 0, startLevel = 0) => {
    setLoading(true);
    setError(null);
    try {
      const res = await playerAPI.start(gameId, seed, startLevel);
      const data = res.data;
      const guid = data.metadata?.session_guid;
      setSessionGuid(guid);
      sessionRef.current = guid;
      setFrame(data);
      return data;
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to start game');
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const sendAction = useCallback(async (action, x = null, y = null) => {
    if (!sessionRef.current) {
      setError('No active session');
      return null;
    }
    setError(null);
    try {
      const res = await playerAPI.action(sessionRef.current, action, x, y);
      setFrame(res.data);
      return res.data;
    } catch (err) {
      const detail = err.response?.data?.detail || '';
      if (err.response?.status === 404 || detail.includes('No game instance')) {
        setError('Session lost (server restarted). Please start a new game.');
        sessionRef.current = null;
        setFrame(null);
        return null;
      }
      setError(detail || 'Action failed');
      return null;
    }
  }, []);

  const endGame = useCallback(async () => {
    if (!sessionRef.current) return;
    try {
      await playerAPI.end(sessionRef.current);
    } catch (err) {
      // ignore
    }
    setSessionGuid(null);
    sessionRef.current = null;
    setFrame(null);
  }, []);

  const resetGame = useCallback(async () => {
    if (!sessionRef.current) return null;
    try {
      const res = await playerAPI.action(sessionRef.current, 'RESET');
      setFrame(res.data);
      return res.data;
    } catch (err) {
      setError(err.response?.data?.detail || 'Reset failed');
      return null;
    }
  }, []);

  return {
    sessionGuid,
    frame,
    loading,
    error,
    startGame,
    sendAction,
    endGame,
    resetGame,
    isPlaying: !!sessionRef.current,
    isGameOver: frame?.state === 'WIN' || frame?.state === 'GAME_OVER',
  };
}
