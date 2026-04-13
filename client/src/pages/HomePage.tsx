import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { gamesAPI } from '../api/client';
import GamePreviewCanvas from '../components/GamePreviewCanvas';

import {
  Search, Gamepad2, BarChart3, Clock, X, User, ArrowRight,
  Upload, FileCode, FileJson, Zap, ChevronRight, ChevronDown, Layers,
  Play, Trophy, Flame, ArrowUpDown, Grid3X3, List, Tag
} from 'lucide-react';
import { useTheme } from '../hooks/useTheme';

export default function HomePage() {
  const [games, setGames] = useState([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [selectedGame, setSelectedGame] = useState(null);
  const [playerName, setPlayerName] = useState('');
  const [activeTab, setActiveTab] = useState('games');
  const [publicStats, setPublicStats] = useState(null);
  const [sortBy, setSortBy] = useState('name');
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [activeCategory, setActiveCategory] = useState<string | null>(null);
  const [sortOpen, setSortOpen] = useState(false);
  const sortRef = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();
  const nameInputRef = useRef(null);
  const { theme, toggleTheme } = useTheme();

  // Direct play state
  const [gameFile, setGameFile] = useState(null);
  const [metadataFile, setMetadataFile] = useState(null);
  const gameFileRef = useRef(null);
  const metadataFileRef = useRef(null);

  useEffect(() => {
    const saved = localStorage.getItem('arc_player_name');
    if (saved) setPlayerName(saved);
  }, []);

  useEffect(() => {
    gamesAPI.listPublic()
      .then((res) => setGames(res.data))
      .catch(() => {})
      .finally(() => setLoading(false));
    gamesAPI.getPublicStats()
      .then((res) => setPublicStats(res.data))
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (selectedGame && nameInputRef.current) {
      nameInputRef.current.focus();
    }
  }, [selectedGame]);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (sortRef.current && !sortRef.current.contains(e.target as Node)) setSortOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const sortOptions = [
    { value: 'name', label: 'Name A-Z' },
    { value: 'popular', label: 'Most Played' },
    { value: 'winrate', label: 'Win Rate' },
    { value: 'newest', label: 'Newest' },
  ] as const;

  const filtered = games.filter((g) =>
    g.game_id.toLowerCase().includes(search.toLowerCase()) ||
    g.name.toLowerCase().includes(search.toLowerCase()) ||
    (g.description || '').toLowerCase().includes(search.toLowerCase()) ||
    (g.tags || []).some((t) => t.toLowerCase().includes(search.toLowerCase()))
  ).filter((g) =>
    !activeCategory || (g.tags?.length > 0 ? g.tags[0] : 'Uncategorized') === activeCategory
  ).sort((a, b) => {
    switch (sortBy) {
      case 'name': return (a.name || a.game_id).localeCompare(b.name || b.game_id);
      case 'popular': return (b.total_plays || 0) - (a.total_plays || 0);
      case 'winrate': {
        const wrA = a.total_plays > 0 ? a.total_wins / a.total_plays : 0;
        const wrB = b.total_plays > 0 ? b.total_wins / b.total_plays : 0;
        return wrB - wrA;
      }
      case 'newest': return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
      default: return 0;
    }
  });

  const allCategories = [...new Set(
    games.map((g) => g.tags?.length > 0 ? g.tags[0] : 'Uncategorized')
  )].sort((a, b) => {
    if (a === 'Uncategorized') return 1;
    if (b === 'Uncategorized') return -1;
    return a.localeCompare(b);
  });

  const groupedByCategory = filtered.reduce<Record<string, typeof filtered>>((acc, game) => {
    const category = game.tags?.length > 0 ? game.tags[0] : 'Uncategorized';
    if (!acc[category]) acc[category] = [];
    acc[category].push(game);
    return acc;
  }, {});

  const sortedCategories = Object.keys(groupedByCategory).sort((a, b) => {
    if (a === 'Uncategorized') return 1;
    if (b === 'Uncategorized') return -1;
    return a.localeCompare(b);
  });

  const handleGameClick = (game) => {
    setSelectedGame(game);
  };

  const handlePlay = () => {
    if (playerName.trim()) {
      localStorage.setItem('arc_player_name', playerName.trim());
    }
    navigate(`/play/${selectedGame.game_id}${playerName.trim() ? `?name=${encodeURIComponent(playerName.trim())}` : ''}`);
  };

  const handleSkip = () => {
    navigate(`/play/${selectedGame.game_id}`);
  };

  const handleDirectPlay = () => {
    if (!gameFile || !metadataFile) return;
    navigate('/play-direct', { state: { gameFile, metadataFile, playerName: playerName.trim() || null } });
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') handlePlay();
    if (e.key === 'Escape') setSelectedGame(null);
  };

  return (
    <div className="min-h-screen bg-gray-950">
      {/* Header */}
      <header className="border-b border-gray-800/80 bg-gray-950/90 backdrop-blur-md sticky top-0 z-30">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center">
              <Gamepad2 size={20} className="text-cyan-400" />
            </div>
            <div>
              <h1 className="font-pixel text-sm text-cyan-400 tracking-tight">ARC-AGI ARCADE</h1>
              <p className="text-[11px] text-gray-500 mt-0.5">Interactive Reasoning Games</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <a href="/login" className="flex items-center gap-1.5 px-3 py-1.5 bg-cyan-500/15 text-cyan-400 hover:bg-cyan-500/25 rounded-lg text-xs font-medium transition-colors">Login</a>
          </div>
        </div>
      </header>

      {/* Tab switcher + search */}
      <div className="max-w-6xl mx-auto px-6 pt-6">
        <div className="flex flex-wrap items-center gap-3 mb-5">
          {/* Tabs */}
          <div className="flex bg-gray-900 border border-gray-800 rounded-lg overflow-hidden">
            <button
              onClick={() => setActiveTab('games')}
              className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium transition-colors ${
                activeTab === 'games'
                  ? 'bg-cyan-500/15 text-cyan-400 border-b-2 border-cyan-400'
                  : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800/60'
              }`}
            >
              <Gamepad2 size={16} /> Games
            </button>
            <button
              onClick={() => setActiveTab('direct')}
              className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium transition-colors ${
                activeTab === 'direct'
                  ? 'bg-fuchsia-500/15 text-fuchsia-400 border-b-2 border-fuchsia-400'
                  : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800/60'
              }`}
            >
              <Upload size={16} /> Play Your Own
            </button>
          </div>

          {activeTab === 'games' && (
            <div className="relative flex-1 max-w-xs">
              <Search size={16} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-gray-500" />
              <input
                type="text"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search games..."
                className="w-full pl-10 pr-4 py-2.5 bg-gray-900 border border-gray-800 rounded-lg text-sm text-gray-200 placeholder-gray-600 focus:outline-none focus:border-cyan-500/50 focus:ring-1 focus:ring-cyan-500/20 transition-all"
              />
              {search && (
                <button onClick={() => setSearch('')} className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-300">
                  <X size={14} />
                </button>
              )}
            </div>
          )}

          {activeTab === 'games' && (
            <div className="flex items-center gap-2 ml-auto">
              <div className="relative" ref={sortRef}>
                <button
                  onClick={() => setSortOpen(!sortOpen)}
                  className="flex items-center gap-2 pl-3 pr-3 py-2 bg-gray-900 border border-gray-800 rounded-lg text-sm text-gray-300 cursor-pointer hover:border-gray-700 focus:outline-none focus:border-cyan-500/50 transition-colors"
                >
                  <ArrowUpDown size={14} className="text-gray-500" />
                  {sortOptions.find(o => o.value === sortBy)?.label}
                  <ChevronDown size={14} className={`text-gray-500 transition-transform ${sortOpen ? 'rotate-180' : ''}`} />
                </button>
                {sortOpen && (
                  <div className="absolute top-full right-0 mt-1 w-40 bg-gray-900 border border-gray-700 rounded-lg shadow-xl shadow-black/40 py-1 z-40">
                    {sortOptions.map((opt) => (
                      <button
                        key={opt.value}
                        onClick={() => { setSortBy(opt.value); setSortOpen(false); }}
                        className={`w-full text-left px-3 py-2 text-sm transition-colors ${
                          sortBy === opt.value
                            ? 'text-cyan-400 bg-cyan-500/10'
                            : 'text-gray-300 hover:bg-gray-800'
                        }`}
                      >
                        {opt.label}
                      </button>
                    ))}
                  </div>
                )}
              </div>

              <div className="flex bg-gray-900 border border-gray-800 rounded-lg overflow-hidden">
                <button
                  onClick={() => setViewMode('grid')}
                  className={`p-2 transition-colors ${viewMode === 'grid' ? 'bg-cyan-500/10 text-cyan-400' : 'text-gray-500 hover:text-gray-300'}`}
                  title="Grid view"
                >
                  <Grid3X3 size={16} />
                </button>
                <button
                  onClick={() => setViewMode('list')}
                  className={`p-2 transition-colors ${viewMode === 'list' ? 'bg-cyan-500/10 text-cyan-400' : 'text-gray-500 hover:text-gray-300'}`}
                  title="List view"
                >
                  <List size={16} />
                </button>
              </div>

              <span className="hidden md:block text-xs text-gray-500 tabular-nums">{filtered.length} games</span>
            </div>
          )}
        </div>
      </div>

      {/* Content */}
      <main className="max-w-6xl mx-auto px-6 pb-16">

        {/* ── Games Tab ── */}
        {activeTab === 'games' && (
          <>
            {/* Category filter pills */}
            {allCategories.length > 1 && (
              <div className="flex items-center gap-2 mb-5 overflow-x-auto pb-1 scrollbar-none">
                <Tag size={14} className="text-gray-500 shrink-0" />
                <button
                  onClick={() => setActiveCategory(null)}
                  className={`px-3 py-1.5 rounded-full text-xs font-medium whitespace-nowrap transition-all ${
                    !activeCategory
                      ? 'bg-cyan-500/15 text-cyan-400 border border-cyan-500/30'
                      : 'bg-gray-900 text-gray-400 hover:bg-gray-800 border border-gray-800'
                  }`}
                >
                  All
                </button>
                {allCategories.map((cat) => (
                  <button
                    key={cat}
                    onClick={() => setActiveCategory(activeCategory === cat ? null : cat)}
                    className={`px-3 py-1.5 rounded-full text-xs font-medium whitespace-nowrap transition-all ${
                      activeCategory === cat
                        ? 'bg-cyan-500/15 text-cyan-400 border border-cyan-500/30'
                        : 'bg-gray-900 text-gray-400 hover:bg-gray-800 border border-gray-800'
                    }`}
                  >
                    {cat}
                  </button>
                ))}
              </div>
            )}

            {/* Featured Game + Platform Stats */}
            {publicStats && publicStats.total_plays > 0 && !search && (
              <div className="mb-8 grid grid-cols-1 lg:grid-cols-3 gap-4">
                {publicStats.featured_game && (
                  <div
                    className="lg:col-span-2 relative bg-gray-900/80 rounded-2xl border border-gray-800 p-6 overflow-hidden cursor-pointer group hover:border-cyan-500/30 transition-all duration-300"
                    onClick={() => {
                      const g = games.find(x => x.game_id === publicStats.featured_game.game_id);
                      if (g) handleGameClick(g);
                    }}
                  >
                    <div className="absolute top-0 right-0 w-64 h-64 bg-cyan-500/5 rounded-full blur-[100px]" />
                    <div className="relative z-10">
                      <span className="inline-flex items-center gap-1.5 px-2.5 py-1 bg-amber-500/10 text-amber-400 rounded-full text-xs font-medium mb-3">
                        <Trophy size={12} /> Most Played
                      </span>
                      <h3 className="font-pixel text-lg text-cyan-400 mb-2 group-hover:text-cyan-300 transition-colors">
                        {publicStats.featured_game.name}
                      </h3>
                      <div className="flex items-center gap-4 text-sm text-gray-400 mb-4">
                        <span>{publicStats.featured_game.total_plays} plays</span>
                        <span>{publicStats.featured_game.total_wins} wins</span>
                        <span>{publicStats.featured_game.levels} levels</span>
                        <span className="text-emerald-400 font-medium">{publicStats.featured_game.win_rate}% win rate</span>
                      </div>
                      {publicStats.featured_game.top_performer && (
                        <div className="flex items-center gap-2 text-xs text-gray-500">
                          <span>Top performer:</span>
                          <span className="text-amber-400 font-medium">{publicStats.featured_game.top_performer.player}</span>
                          <span>({publicStats.featured_game.top_performer.time}s, {publicStats.featured_game.top_performer.actions} moves)</span>
                        </div>
                      )}
                      <button className="mt-5 inline-flex items-center gap-2 px-5 py-2.5 bg-cyan-500 hover:bg-cyan-400 text-gray-950 font-semibold text-sm rounded-lg transition-colors shadow-lg shadow-cyan-500/20">
                        <Play size={16} /> Play Now <ArrowRight size={14} className="group-hover:translate-x-0.5 transition-transform" />
                      </button>
                    </div>
                  </div>
                )}

                <div className="space-y-3">
                  <div className="bg-gray-900/80 rounded-2xl border border-gray-800 p-5">
                    <h4 className="font-pixel text-[10px] text-amber-400 uppercase tracking-widest mb-4">Platform Stats</h4>
                    <div className="space-y-3.5">
                      {[
                        { label: 'Total Games', value: publicStats.total_games, icon: Gamepad2, color: 'text-cyan-400' },
                        { label: 'Total Plays', value: publicStats.total_plays, icon: BarChart3, color: 'text-fuchsia-400' },
                        { label: 'Win Rate', value: `${publicStats.win_rate}%`, icon: Trophy, color: 'text-emerald-400' },
                      ].map(({ label, value, icon: Icon, color }) => (
                        <div key={label} className="flex items-center gap-3">
                          <div className={`w-9 h-9 rounded-lg bg-gray-800/80 flex items-center justify-center ${color}`}>
                            <Icon size={16} />
                          </div>
                          <div className="flex-1">
                            <p className={`text-lg font-bold ${color}`}>{value}</p>
                            <p className="text-xs text-gray-500">{label}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {publicStats.top_games?.length > 1 && (
                     <div className="bg-gray-900/80 rounded-2xl border border-gray-800 p-5">
                      <h4 className="font-pixel text-[10px] text-amber-400 uppercase tracking-widest mb-4">Top Games</h4>
                      <div className="space-y-2.5">
                        {publicStats.top_games.slice(0, 3).map((g, i) => (
                          <div key={g.game_id} className="flex items-center gap-3 text-sm">
                            <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold shrink-0 ${
                              i === 0 ? 'bg-amber-500/15 text-amber-400' :
                              i === 1 ? 'bg-gray-700/50 text-gray-300' :
                              'bg-orange-500/15 text-orange-400'
                            }`}>{i + 1}</span>
                            <span className="text-gray-300 truncate flex-1">{g.name}</span>
                            <span className="text-gray-500 text-xs tabular-nums">{g.total_plays} plays</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {loading ? (
              <div className="flex justify-center py-20">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-cyan-400"></div>
              </div>
            ) : filtered.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-24 px-4">
                {search ? (
                  <div className="text-center">
                    <div className="w-16 h-16 rounded-2xl bg-gray-900 border border-gray-800 flex items-center justify-center mx-auto mb-5">
                      <Search size={28} className="text-gray-600" />
                    </div>
                    <p className="text-base text-gray-300 font-medium">No results for &ldquo;{search}&rdquo;</p>
                    <p className="text-sm text-gray-500 mt-1.5">Try a different search term or browse all games</p>
                    <button onClick={() => setSearch('')} className="mt-4 px-5 py-2 bg-gray-800 hover:bg-gray-700 text-gray-300 text-sm rounded-lg transition-colors">
                      Clear Search
                    </button>
                  </div>
                ) : (
                  <div className="text-center max-w-md">
                    <div className="relative w-28 h-28 mx-auto mb-6">
                      <div className="absolute inset-0 rounded-2xl bg-cyan-500/5 border border-cyan-500/10 rotate-6" />
                      <div className="absolute inset-0 rounded-2xl bg-fuchsia-500/5 border border-fuchsia-500/10 -rotate-3" />
                      <div className="absolute inset-0 rounded-2xl bg-gray-900 border border-gray-800 flex items-center justify-center">
                        <Gamepad2 size={40} className="text-gray-600" />
                      </div>
                    </div>

                    <h3 className="font-pixel text-base text-cyan-400 mb-3">INSERT COIN</h3>
                    <p className="text-sm text-gray-400 leading-relaxed mb-6">
                      Games will appear here once they're uploaded and activated by an admin. You can also test your own game right now.
                    </p>

                    <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
                      <button
                        onClick={() => setActiveTab('direct')}
                        className="flex items-center gap-2 px-5 py-2.5 bg-cyan-500 hover:bg-cyan-400 text-gray-950 font-semibold text-sm rounded-lg transition-colors shadow-lg shadow-cyan-500/20"
                      >
                        <Upload size={16} /> Play Your Own
                      </button>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="space-y-8">
                {sortedCategories.map((category) => (
                  <div key={category}>
                    {!activeCategory && (
                      <div className="flex items-center gap-3 mb-4">
                        <h2 className="font-pixel text-xs text-cyan-400">{category}</h2>
                        <span className="text-xs text-gray-500 bg-gray-900 px-2 py-0.5 rounded-full border border-gray-800">
                          {groupedByCategory[category].length}
                        </span>
                        <div className="flex-1 h-px bg-gray-800/60" />
                      </div>
                    )}

                    {viewMode === 'grid' ? (
                      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                        {groupedByCategory[category].map((game, index) => {
                          const numLevels = game.baseline_actions?.length || 0;
                          const winRate = game.total_plays > 0 ? Math.round((game.total_wins / game.total_plays) * 100) : null;
                          const categoryTag = game.tags?.length > 0 ? game.tags[0] : null;
                          return (
                            <motion.div
                              key={game.game_id}
                              initial={{ opacity: 0, y: 16 }}
                              animate={{ opacity: 1, y: 0 }}
                              transition={{ delay: Math.min(index * 0.04, 0.4) }}
                            >
                            <button
                              onClick={() => handleGameClick(game)}
                              className="w-full group relative bg-gray-900/80 rounded-xl border border-gray-800 hover:border-cyan-500/30 transition-all duration-300 overflow-hidden text-left hover:shadow-lg hover:shadow-cyan-500/5"
                            >
                              <div className="relative aspect-[4/3] bg-gray-950 overflow-hidden">
                                <div className="absolute inset-0 flex items-center justify-center">
                                  <GamePreviewCanvas gameId={game.game_id} size={320} />
                                </div>

                                <div className="absolute top-2 left-2 right-2 flex items-start justify-between z-10">
                                  {numLevels > 0 && (
                                    <span className="flex items-center gap-1 px-2 py-0.5 bg-black/70 backdrop-blur-sm rounded-md text-[11px] text-cyan-400 font-medium">
                                      <Layers size={10} /> {numLevels}
                                    </span>
                                  )}
                                  {game.total_plays > 10 && (
                                    <span className="flex items-center gap-1 px-2 py-0.5 bg-black/70 backdrop-blur-sm rounded-md text-[11px] text-orange-400">
                                      <Flame size={10} />
                                    </span>
                                  )}
                                </div>

                                <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/30 to-transparent opacity-0 group-hover:opacity-100 transition-all duration-300 flex items-center justify-center">
                                  <div className="flex flex-col items-center gap-1.5 translate-y-3 group-hover:translate-y-0 transition-transform duration-300">
                                    <div className="w-11 h-11 rounded-full bg-cyan-500/20 border border-cyan-500/40 flex items-center justify-center shadow-lg shadow-cyan-500/20">
                                      <Play size={20} className="text-cyan-400 ml-0.5" />
                                    </div>
                                    <span className="text-xs text-cyan-400 font-semibold">PLAY</span>
                                  </div>
                                </div>

                                <div className="absolute bottom-0 left-0 right-0 h-12 bg-gradient-to-t from-gray-900 to-transparent" />
                              </div>

                              <div className="px-3.5 py-3 relative">
                                <div className="flex items-start justify-between gap-2 mb-1.5">
                                  <h3 className="text-gray-200 text-sm font-medium truncate group-hover:text-cyan-400 transition-colors">
                                    {game.name || game.game_id}
                                  </h3>
                                  {winRate !== null && winRate > 0 && (
                                    <span className={`shrink-0 px-1.5 py-0.5 rounded text-xs font-bold ${
                                      winRate >= 50 ? 'bg-emerald-500/15 text-emerald-400' :
                                      winRate >= 20 ? 'bg-amber-500/15 text-amber-400' :
                                      'bg-fuchsia-500/15 text-fuchsia-400'
                                    }`}>
                                      {winRate}%
                                    </span>
                                  )}
                                </div>

                                {game.description && (
                                  <p className="text-xs text-gray-500 line-clamp-1 mb-2">{game.description}</p>
                                )}

                                <div className="flex items-center gap-2 text-[11px]">
                                  {categoryTag && (
                                    <span className="px-1.5 py-0.5 rounded bg-cyan-500/10 text-cyan-400">{categoryTag}</span>
                                  )}
                                  <span className="text-gray-500">{game.total_plays} plays</span>
                                  {game.total_wins > 0 && (
                                    <span className="text-gray-500">{game.total_wins} wins</span>
                                  )}
                                </div>
                              </div>
                            </button>
                            </motion.div>
                          );
                        })}
                      </div>
                    ) : (
                      <div className="space-y-2">
                        {groupedByCategory[category].map((game, index) => {
                          const numLevels = game.baseline_actions?.length || 0;
                          const winRate = game.total_plays > 0 ? Math.round((game.total_wins / game.total_plays) * 100) : null;
                          const categoryTag = game.tags?.length > 0 ? game.tags[0] : null;
                          return (
                            <motion.div
                              key={game.game_id}
                              initial={{ opacity: 0, x: -8 }}
                              animate={{ opacity: 1, x: 0 }}
                              transition={{ delay: Math.min(index * 0.03, 0.3) }}
                            >
                            <button
                              onClick={() => handleGameClick(game)}
                              className="w-full group flex items-center gap-4 p-3 bg-gray-900/80 rounded-xl border border-gray-800 hover:border-cyan-500/30 transition-all text-left hover:shadow-lg hover:shadow-cyan-500/5"
                            >
                              <div className="w-16 h-16 rounded-lg bg-gray-950 overflow-hidden shrink-0 border border-gray-800">
                                <GamePreviewCanvas gameId={game.game_id} size={64} />
                              </div>

                              <div className="flex-1 min-w-0">
                                <div className="flex items-center gap-2 mb-0.5">
                                  <h3 className="text-gray-200 text-sm font-medium truncate group-hover:text-cyan-400 transition-colors">
                                    {game.name || game.game_id}
                                  </h3>
                                  {categoryTag && (
                                    <span className="px-1.5 py-0.5 rounded bg-cyan-500/10 text-cyan-400 text-[11px] shrink-0">{categoryTag}</span>
                                  )}
                                </div>
                                {game.description && (
                                  <p className="text-xs text-gray-500 line-clamp-1">{game.description}</p>
                                )}
                              </div>

                              <div className="flex items-center gap-4 shrink-0 text-sm text-gray-400">
                                {numLevels > 0 && (
                                  <span className="flex items-center gap-1 text-cyan-400">
                                    <Layers size={14} /> {numLevels}
                                  </span>
                                )}
                                <span className="tabular-nums text-xs">{game.total_plays} plays</span>
                                {winRate !== null && winRate > 0 && (
                                  <span className={`px-1.5 py-0.5 rounded text-xs font-bold ${
                                    winRate >= 50 ? 'bg-emerald-500/15 text-emerald-400' :
                                    winRate >= 20 ? 'bg-amber-500/15 text-amber-400' :
                                    'bg-fuchsia-500/15 text-fuchsia-400'
                                  }`}>
                                    {winRate}%
                                  </span>
                                )}
                                <ChevronRight size={16} className="text-gray-600 group-hover:text-cyan-400 transition-colors" />
                              </div>
                            </button>
                            </motion.div>
                          );
                        })}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </>
        )}

        {/* ── Direct Play Tab ── */}
        {activeTab === 'direct' && (
          <div className="max-w-xl mx-auto">
            <div className="bg-gray-900/80 rounded-2xl border border-gray-800 p-8">
              <div className="text-center mb-8">
                <div className="w-14 h-14 rounded-xl bg-fuchsia-500/10 border border-fuchsia-500/20 flex items-center justify-center mx-auto mb-4">
                  <Zap size={24} className="text-fuchsia-400" />
                </div>
                <h2 className="font-pixel text-base text-fuchsia-400 mb-2">PLAY YOUR OWN</h2>
                <p className="text-sm text-gray-400">Upload a game.py and metadata.json to play instantly. Nothing is saved.</p>
              </div>

              {/* File uploads */}
              <div className="grid grid-cols-2 gap-3 mb-6">
                <DropZone
                  file={gameFile}
                  setFile={setGameFile}
                  fileRef={gameFileRef}
                  accept=".py"
                  icon={FileCode}
                  label="game.py"
                  hint="Game logic file"
                />
                <DropZone
                  file={metadataFile}
                  setFile={setMetadataFile}
                  fileRef={metadataFileRef}
                  accept=".json"
                  icon={FileJson}
                  label="metadata.json"
                  hint='Must have "game_id"'
                />
              </div>

              {/* Player name */}
              <div className="mb-6">
                <label className="flex items-center gap-1.5 text-xs text-gray-400 mb-2">
                  <User size={14} /> Your name <span className="text-gray-600">(optional)</span>
                </label>
                <input
                  type="text"
                  value={playerName}
                  onChange={(e) => setPlayerName(e.target.value)}
                  placeholder="For your own reference"
                  maxLength={50}
                  className="w-full px-4 py-2.5 bg-gray-950 border border-gray-800 rounded-lg text-sm text-gray-200 placeholder-gray-600 focus:outline-none focus:border-cyan-500/50 focus:ring-1 focus:ring-cyan-500/20"
                />
              </div>

              {/* Play button */}
              <button
                onClick={handleDirectPlay}
                disabled={!gameFile || !metadataFile}
                className="w-full px-4 py-3 bg-cyan-500 hover:bg-cyan-400 disabled:opacity-30 disabled:cursor-not-allowed text-gray-950 font-semibold text-sm rounded-lg transition-colors flex items-center justify-center gap-2 shadow-lg shadow-cyan-500/20"
              >
                <Zap size={16} /> Play Now
              </button>

              {/* Format hint */}
              <div className="mt-6 p-4 bg-gray-950/60 border border-gray-800 rounded-lg">
                <p className="text-xs text-gray-400 font-medium mb-1.5">Expected format</p>
                <div className="text-xs text-gray-500 space-y-1">
                  <p><strong className="text-cyan-400">game.py</strong> — Class extending ARCBaseGame with step() method</p>
                  <p><strong className="text-cyan-400">metadata.json</strong> — {`{"game_id": "xx00"}`} e.g. "cw45", "ls20"</p>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Player Name Modal (for registered games) */}
      <AnimatePresence>
      {selectedGame && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4" onClick={() => setSelectedGame(null)}>
          <div className="absolute inset-0 bg-black/80 backdrop-blur-sm" />
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ type: 'spring', stiffness: 300, damping: 25 }}
            className="relative bg-gray-900 rounded-2xl border border-gray-700 shadow-2xl w-full max-w-sm overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            <button onClick={() => setSelectedGame(null)} className="absolute top-3 right-3 p-1 rounded-lg hover:bg-gray-800 text-gray-500">
              <X size={16} />
            </button>
            <div className="p-6">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-12 h-12 rounded-xl bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center shrink-0">
                  <Gamepad2 size={22} className="text-cyan-400" />
                </div>
                <div className="min-w-0">
                  <p className="font-pixel text-sm text-cyan-400">{selectedGame.game_id}</p>
                  <div className="flex items-center gap-3 text-xs text-gray-500 mt-1">
                    {selectedGame.baseline_actions?.length > 0 && (
                      <span className="flex items-center gap-1">
                        <Layers size={12} className="text-cyan-400" />
                        {selectedGame.baseline_actions.length} levels
                      </span>
                    )}
                    {selectedGame.tags && selectedGame.tags.length > 0 && (
                      <span className="truncate">{selectedGame.tags.slice(0, 3).join(', ')}</span>
                    )}
                  </div>
                </div>
              </div>
              <div className="mb-5">
                <label className="flex items-center gap-1.5 text-xs text-gray-400 mb-2">
                  <User size={14} /> Your name <span className="text-gray-600">(optional)</span>
                </label>
                <input
                  ref={nameInputRef}
                  type="text"
                  value={playerName}
                  onChange={(e) => setPlayerName(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Enter your name for the leaderboard"
                  maxLength={50}
                  className="w-full px-4 py-3 bg-gray-950 border border-gray-800 rounded-lg text-sm text-gray-200 placeholder-gray-600 focus:outline-none focus:border-cyan-500/50 focus:ring-1 focus:ring-cyan-500/20"
                />
              </div>
              <div className="flex gap-2">
                <button
                  onClick={handlePlay}
                  className="flex-1 px-4 py-3 bg-emerald-500 hover:bg-emerald-400 text-gray-950 font-semibold text-sm rounded-lg transition-colors flex items-center justify-center gap-2 shadow-lg shadow-emerald-500/20"
                >
                  Play <ArrowRight size={14} />
                </button>
                <button
                  onClick={handleSkip}
                  className="px-4 py-3 bg-gray-800 hover:bg-gray-700 text-gray-400 text-sm font-medium rounded-lg transition-colors"
                >
                  Skip
                </button>
              </div>
            </div>
          </motion.div>
        </div>
      )}
      </AnimatePresence>
    </div>
  );
}

function DropZone({ file, setFile, fileRef, accept, icon: Icon, label, hint }) {
  return (
    <div
      onClick={() => fileRef.current?.click()}
      className={`border-2 border-dashed rounded-xl p-5 text-center cursor-pointer transition-all ${
        file
          ? 'border-emerald-500/40 bg-emerald-500/5'
          : 'border-gray-700 hover:border-cyan-500/30 hover:bg-gray-800/30'
      }`}
    >
      <input
        ref={fileRef}
        type="file"
        accept={accept}
        onChange={(e) => setFile(e.target.files[0] || null)}
        className="hidden"
      />
      {file ? (
        <div>
          <p className="text-sm text-emerald-400 font-medium truncate">{file.name}</p>
          <p className="text-xs text-gray-500 mt-1">{(file.size / 1024).toFixed(1)} KB</p>
          <button
            type="button"
            onClick={(e) => { e.stopPropagation(); setFile(null); }}
            className="text-xs text-gray-500 hover:text-fuchsia-400 mt-1"
          >
            Remove
          </button>
        </div>
      ) : (
        <>
          <Icon size={28} className="mx-auto text-gray-600 mb-2" />
          <p className="text-sm text-gray-400">{label}</p>
          <p className="text-xs text-gray-600 mt-1">{hint}</p>
        </>
      )}
    </div>
  );
}
