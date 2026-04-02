import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { gamesAPI, requestsAPI } from '../api/client';
import GamePreviewCanvas from '../components/GamePreviewCanvas';
import GameUploadForm from '../components/GameUploadForm';
import {
  Search, Gamepad2, BarChart3, Clock, X, User, ArrowRight,
  Upload, FileCode, FileJson, Zap, ChevronRight, Send, Layers,
  Play, Trophy, Flame, Sun, Moon, ArrowUpDown, Grid3X3, List, Tag
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
  const navigate = useNavigate();
  const nameInputRef = useRef(null);
  const { theme, toggleTheme } = useTheme();

  // Direct play state
  const [gameFile, setGameFile] = useState(null);
  const [metadataFile, setMetadataFile] = useState(null);
  const gameFileRef = useRef(null);
  const metadataFileRef = useRef(null);

  // Request upload state
  const [requestSubmitting, setRequestSubmitting] = useState(false);
  const [requestSuccess, setRequestSuccess] = useState('');
  const [requestError, setRequestError] = useState('');

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
    // Store files in sessionStorage as a signal, navigate to ephemeral play page
    // We'll pass files via state
    navigate('/play-direct', { state: { gameFile, metadataFile, playerName: playerName.trim() || null } });
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') handlePlay();
    if (e.key === 'Escape') setSelectedGame(null);
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-950">
      {/* Header */}
      <header className="border-b border-gray-200 dark:border-gray-800 bg-gray-50/80 dark:bg-gray-950/80 backdrop-blur-sm sticky top-0 z-30">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-blue-600/20 border border-blue-500/30 flex items-center justify-center">
              <Gamepad2 size={18} className="text-blue-400" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-gray-900 dark:text-white tracking-tight">ARC-AGI-3</h1>
              <p className="text-[10px] text-gray-400 dark:text-gray-500 -mt-0.5">Interactive Reasoning Games</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={toggleTheme}
              className="w-8 h-8 rounded-lg border border-gray-200 dark:border-gray-700 flex items-center justify-center text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
              title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
            >
              {theme === 'dark' ? <Sun size={14} /> : <Moon size={14} />}
            </button>
            <a href="/admin" className="text-xs text-gray-400 dark:text-gray-600 hover:text-gray-600 dark:hover:text-gray-400 transition-colors">Admin</a>
          </div>
        </div>
      </header>

      {/* Tab switcher + search */}
      <div className="max-w-6xl mx-auto px-6 pt-6">
        <div className="flex items-center gap-4 mb-5">
          {/* Tabs */}
          <div className="flex bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-xl overflow-hidden">
            <button
              onClick={() => setActiveTab('games')}
              className={`flex items-center gap-2 px-5 py-2.5 text-sm font-medium transition-colors ${
                activeTab === 'games'
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-500 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-800'
              }`}
            >
              <Gamepad2 size={16} /> Games
            </button>
            <button
              onClick={() => setActiveTab('direct')}
              className={`flex items-center gap-2 px-5 py-2.5 text-sm font-medium transition-colors ${
                activeTab === 'direct'
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-500 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-800'
              }`}
            >
              <Upload size={16} /> Play Your Own
            </button>
            <button
              onClick={() => setActiveTab('request')}
              className={`flex items-center gap-2 px-5 py-2.5 text-sm font-medium transition-colors ${
                activeTab === 'request'
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-500 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-800'
              }`}
            >
              <Send size={16} /> Request Upload
            </button>
          </div>

          {/* Search (only for games tab) */}
          {activeTab === 'games' && (
            <div className="relative flex-1 max-w-xs">
              <Search size={16} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-gray-400 dark:text-gray-500" />
              <input
                type="text"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search games..."
                className="w-full pl-10 pr-4 py-2.5 bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-xl text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-gray-600 focus:outline-none focus:border-blue-500/50 text-sm transition-all"
              />
              {search && (
                <button onClick={() => setSearch('')} className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 dark:text-gray-500 hover:text-gray-600 dark:hover:text-gray-300">
                  <X size={14} />
                </button>
              )}
            </div>
          )}

          {activeTab === 'games' && (
            <div className="flex items-center gap-2 ml-auto">
              <div className="relative">
                <select
                  value={sortBy}
                  onChange={(e) => setSortBy(e.target.value)}
                  className="appearance-none pl-8 pr-8 py-2 bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-lg text-gray-700 dark:text-gray-300 text-xs font-medium cursor-pointer hover:border-gray-300 dark:hover:border-gray-700 focus:outline-none focus:border-blue-500/50 transition-colors"
                >
                  <option value="name">Name A–Z</option>
                  <option value="popular">Most Played</option>
                  <option value="winrate">Win Rate</option>
                  <option value="newest">Newest</option>
                </select>
                <ArrowUpDown size={13} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-gray-400 dark:text-gray-500 pointer-events-none" />
              </div>

              <div className="flex bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-lg overflow-hidden">
                <button
                  onClick={() => setViewMode('grid')}
                  className={`p-2 transition-colors ${viewMode === 'grid' ? 'bg-blue-600/10 text-blue-500' : 'text-gray-400 dark:text-gray-500 hover:text-gray-600 dark:hover:text-gray-300'}`}
                  title="Grid view"
                >
                  <Grid3X3 size={14} />
                </button>
                <button
                  onClick={() => setViewMode('list')}
                  className={`p-2 transition-colors ${viewMode === 'list' ? 'bg-blue-600/10 text-blue-500' : 'text-gray-400 dark:text-gray-500 hover:text-gray-600 dark:hover:text-gray-300'}`}
                  title="List view"
                >
                  <List size={14} />
                </button>
              </div>

              <span className="hidden md:block text-[11px] text-gray-400 dark:text-gray-600 tabular-nums">{filtered.length}</span>
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
                <Tag size={13} className="text-gray-400 dark:text-gray-500 shrink-0" />
                <button
                  onClick={() => setActiveCategory(null)}
                  className={`px-3 py-1.5 rounded-full text-xs font-medium whitespace-nowrap transition-all ${
                    !activeCategory
                      ? 'bg-blue-600 text-white shadow-sm shadow-blue-600/20'
                      : 'bg-gray-100 dark:bg-gray-800/60 text-gray-500 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-800 border border-gray-200/50 dark:border-gray-700/50'
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
                        ? 'bg-blue-600 text-white shadow-sm shadow-blue-600/20'
                        : 'bg-gray-100 dark:bg-gray-800/60 text-gray-500 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-800 border border-gray-200/50 dark:border-gray-700/50'
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
                {/* Featured game hero */}
                {publicStats.featured_game && (
                  <div
                    className="lg:col-span-2 relative bg-gradient-to-br from-blue-600/10 via-white dark:via-gray-900 to-purple-600/5 rounded-2xl border border-blue-500/20 p-6 overflow-hidden cursor-pointer group"
                    onClick={() => {
                      const g = games.find(x => x.game_id === publicStats.featured_game.game_id);
                      if (g) handleGameClick(g);
                    }}
                  >
                    <div className="absolute top-0 right-0 w-64 h-64 bg-blue-500/5 rounded-full blur-[80px]" />
                    <div className="relative z-10">
                      <span className="inline-flex items-center gap-1.5 px-2.5 py-1 bg-yellow-500/15 text-yellow-400 rounded-full text-[10px] font-semibold mb-3">
                        <Trophy size={10} /> Most Played
                      </span>
                      <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-1 group-hover:text-blue-300 transition-colors">
                        {publicStats.featured_game.name}
                      </h3>
                      <div className="flex items-center gap-4 text-sm text-gray-500 dark:text-gray-400 mb-4">
                        <span>{publicStats.featured_game.total_plays} plays</span>
                        <span>{publicStats.featured_game.total_wins} wins</span>
                        <span>{publicStats.featured_game.levels} levels</span>
                        <span className="text-emerald-400">{publicStats.featured_game.win_rate}% win rate</span>
                      </div>
                      {publicStats.featured_game.top_performer && (
                        <div className="flex items-center gap-2 text-xs">
                          <span className="text-gray-400 dark:text-gray-500">Top performer:</span>
                          <span className="text-yellow-400 font-medium">{publicStats.featured_game.top_performer.player}</span>
                          <span className="text-gray-400 dark:text-gray-600">({publicStats.featured_game.top_performer.time}s, {publicStats.featured_game.top_performer.actions} moves)</span>
                        </div>
                      )}
                      <div className="mt-4 inline-flex items-center gap-2 px-4 py-2 bg-blue-600/20 group-hover:bg-blue-600/30 border border-blue-500/30 text-blue-400 rounded-xl text-sm font-medium transition-colors">
                        <Play size={14} /> Play Now <ArrowRight size={14} className="group-hover:translate-x-0.5 transition-transform" />
                      </div>
                    </div>
                  </div>
                )}

                {/* Platform stats sidebar */}
                <div className="space-y-3">
                  <div className="bg-white dark:bg-gray-900 rounded-2xl border border-gray-200 dark:border-gray-800 p-5">
                    <h4 className="text-[10px] font-semibold text-gray-400 dark:text-gray-500 uppercase tracking-widest mb-3">Platform Stats</h4>
                    <div className="space-y-3">
                      {[
                        { label: 'Total Games', value: publicStats.total_games, icon: Gamepad2, color: 'text-blue-400' },
                        { label: 'Total Plays', value: publicStats.total_plays, icon: BarChart3, color: 'text-purple-400' },
                        { label: 'Win Rate', value: `${publicStats.win_rate}%`, icon: Trophy, color: 'text-emerald-400' },
                      ].map(({ label, value, icon: Icon, color }) => (
                        <div key={label} className="flex items-center gap-3">
                          <div className={`w-8 h-8 rounded-lg bg-gray-100 dark:bg-gray-800 flex items-center justify-center ${color}`}>
                            <Icon size={15} />
                          </div>
                          <div className="flex-1">
                            <p className="text-gray-900 dark:text-white font-bold text-sm">{value}</p>
                            <p className="text-[10px] text-gray-400 dark:text-gray-500">{label}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Leaderboard snippet */}
                  {publicStats.top_games?.length > 1 && (
                     <div className="bg-white dark:bg-gray-900 rounded-2xl border border-gray-200 dark:border-gray-800 p-5">
                      <h4 className="text-[10px] font-semibold text-gray-400 dark:text-gray-500 uppercase tracking-widest mb-3">Top Games</h4>
                      <div className="space-y-2">
                        {publicStats.top_games.slice(0, 3).map((g, i) => (
                          <div key={g.game_id} className="flex items-center gap-2.5 text-xs">
                            <span className={`w-5 h-5 rounded-full flex items-center justify-center text-[9px] font-bold shrink-0 ${
                              i === 0 ? 'bg-yellow-500/20 text-yellow-400' :
                              i === 1 ? 'bg-gray-500/20 text-gray-300' :
                              'bg-orange-500/20 text-orange-400'
                            }`}>{i + 1}</span>
                            <span className="text-gray-700 dark:text-gray-300 truncate flex-1">{g.name}</span>
                            <span className="text-gray-400 dark:text-gray-500 font-mono">{g.total_plays}</span>
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
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
              </div>
            ) : filtered.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-24 px-4">
                {search ? (
                  <div className="text-center">
                    <div className="w-20 h-20 rounded-2xl bg-gray-100/50 dark:bg-gray-800/50 border border-gray-200/50 dark:border-gray-700/50 flex items-center justify-center mx-auto mb-5">
                      <Search size={32} className="text-gray-400 dark:text-gray-600" />
                    </div>
                    <p className="text-lg font-medium text-gray-500 dark:text-gray-400">No results for "{search}"</p>
                    <p className="text-sm text-gray-400 dark:text-gray-600 mt-1">Try a different search term or browse all games</p>
                    <button onClick={() => setSearch('')} className="mt-4 px-5 py-2 bg-blue-600/20 hover:bg-blue-600/30 text-blue-400 border border-blue-500/30 rounded-lg text-sm transition-colors">
                      Clear Search
                    </button>
                  </div>
                ) : (
                  <div className="text-center max-w-md">
                    {/* Animated empty state illustration */}
                    <div className="relative w-32 h-32 mx-auto mb-8">
                      <div className="absolute inset-0 rounded-3xl bg-gradient-to-br from-blue-500/10 to-purple-500/10 border border-blue-500/20 rotate-6 animate-tilt" />
                      <div className="absolute inset-0 rounded-3xl bg-gradient-to-br from-purple-500/10 to-pink-500/10 border border-purple-500/20 -rotate-3 animate-tilt-reverse" />
                       <div className="absolute inset-0 rounded-3xl bg-gray-50 dark:bg-gray-900 border border-gray-200/50 dark:border-gray-700/50 flex items-center justify-center">
                        <Gamepad2 size={48} className="text-gray-400 dark:text-gray-600" />
                      </div>
                    </div>

                    <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2">No games yet</h3>
                    <p className="text-gray-400 dark:text-gray-500 text-sm leading-relaxed mb-8">
                      Games will appear here once they're uploaded and activated by an admin. You can also test your own game right now.
                    </p>

                    <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
                      <button
                        onClick={() => setActiveTab('direct')}
                        className="flex items-center gap-2 px-5 py-2.5 bg-blue-600 hover:bg-blue-500 text-white rounded-xl text-sm font-medium transition-colors shadow-lg shadow-blue-600/20"
                      >
                        <Upload size={16} /> Play Your Own Game
                      </button>
                      <button
                        onClick={() => setActiveTab('request')}
                        className="flex items-center gap-2 px-5 py-2.5 bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-700 rounded-xl text-sm font-medium transition-colors"
                      >
                        <Send size={16} /> Submit a Game
                      </button>
                    </div>

                    <style>{`
                      @keyframes tilt { 0%,100%{transform:rotate(6deg)} 50%{transform:rotate(8deg)} }
                      @keyframes tiltReverse { 0%,100%{transform:rotate(-3deg)} 50%{transform:rotate(-5deg)} }
                      .animate-tilt { animation: tilt 4s ease-in-out infinite; }
                      .animate-tilt-reverse { animation: tiltReverse 4s ease-in-out infinite; }
                    `}</style>
                  </div>
                )}
              </div>
            ) : (
              <div className="space-y-8">
                {sortedCategories.map((category) => (
                  <div key={category}>
                    {!activeCategory && (
                      <div className="flex items-center gap-3 mb-4">
                        <h2 className="text-sm font-semibold text-gray-900 dark:text-white">{category}</h2>
                        <span className="text-[10px] text-gray-400 dark:text-gray-500 bg-gray-100 dark:bg-gray-800 px-2 py-0.5 rounded-full">
                          {groupedByCategory[category].length}
                        </span>
                        <div className="flex-1 h-px bg-gray-200 dark:bg-gray-800" />
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
                              className="w-full group relative bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 hover:border-blue-500/30 transition-all duration-300 overflow-hidden text-left hover:shadow-lg hover:shadow-blue-500/5 hover:-translate-y-0.5"
                            >
                              <div className="relative aspect-[4/3] bg-gray-50 dark:bg-gray-950 overflow-hidden">
                                <div className="absolute inset-0 flex items-center justify-center">
                                  <GamePreviewCanvas gameId={game.game_id} size={320} />
                                </div>

                                <div className="absolute top-2 left-2 right-2 flex items-start justify-between z-10">
                                  {numLevels > 0 && (
                                    <span className="flex items-center gap-1 px-2 py-0.5 bg-black/60 backdrop-blur-sm rounded-md text-[10px] text-blue-300 font-medium">
                                      <Layers size={9} /> {numLevels}
                                    </span>
                                  )}
                                  {game.total_plays > 10 && (
                                    <span className="flex items-center gap-1 px-2 py-0.5 bg-black/60 backdrop-blur-sm rounded-md text-[10px] text-orange-300 font-medium">
                                      <Flame size={9} />
                                    </span>
                                  )}
                                </div>

                                <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/30 to-transparent opacity-0 group-hover:opacity-100 transition-all duration-300 flex items-center justify-center">
                                  <div className="flex flex-col items-center gap-1.5 translate-y-3 group-hover:translate-y-0 transition-transform duration-300">
                                    <div className="w-10 h-10 rounded-full bg-blue-600 flex items-center justify-center shadow-lg shadow-blue-500/30">
                                      <Play size={18} className="text-white ml-0.5" />
                                    </div>
                                    <span className="text-white text-[11px] font-semibold">Play</span>
                                  </div>
                                </div>

                                <div className="absolute bottom-0 left-0 right-0 h-12 bg-gradient-to-t from-white dark:from-gray-900 to-transparent" />
                              </div>

                              <div className="px-3.5 py-3 relative">
                                <div className="flex items-start justify-between gap-2 mb-1.5">
                                  <h3 className="text-gray-900 dark:text-white font-semibold text-[13px] truncate group-hover:text-blue-500 dark:group-hover:text-blue-400 transition-colors">
                                    {game.name || game.game_id}
                                  </h3>
                                  {winRate !== null && winRate > 0 && (
                                    <span className={`shrink-0 px-1.5 py-0.5 rounded text-[10px] font-semibold ${
                                      winRate >= 50 ? 'bg-emerald-500/15 text-emerald-500 dark:text-emerald-400' :
                                      winRate >= 20 ? 'bg-yellow-500/15 text-yellow-600 dark:text-yellow-400' :
                                      'bg-red-500/15 text-red-500 dark:text-red-400'
                                    }`}>
                                      {winRate}%
                                    </span>
                                  )}
                                </div>

                                {game.description && (
                                  <p className="text-[11px] text-gray-400 dark:text-gray-500 line-clamp-1 mb-2">{game.description}</p>
                                )}

                                <div className="flex items-center gap-2 text-[10px]">
                                  {categoryTag && (
                                    <span className="px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-600 dark:text-blue-400 font-medium">{categoryTag}</span>
                                  )}
                                  <span className="text-gray-400 dark:text-gray-500">{game.total_plays} plays</span>
                                  {game.total_wins > 0 && (
                                    <span className="text-gray-400 dark:text-gray-500">{game.total_wins} wins</span>
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
                              className="w-full group flex items-center gap-4 p-3 bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 hover:border-blue-500/30 transition-all text-left hover:shadow-md hover:shadow-blue-500/5"
                            >
                              <div className="w-16 h-16 rounded-lg bg-gray-50 dark:bg-gray-950 overflow-hidden shrink-0 border border-gray-100 dark:border-gray-800">
                                <GamePreviewCanvas gameId={game.game_id} size={64} />
                              </div>

                              <div className="flex-1 min-w-0">
                                <div className="flex items-center gap-2 mb-0.5">
                                  <h3 className="text-gray-900 dark:text-white font-semibold text-sm truncate group-hover:text-blue-500 dark:group-hover:text-blue-400 transition-colors">
                                    {game.name || game.game_id}
                                  </h3>
                                  {categoryTag && (
                                    <span className="px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-600 dark:text-blue-400 text-[10px] font-medium shrink-0">{categoryTag}</span>
                                  )}
                                </div>
                                {game.description && (
                                  <p className="text-[11px] text-gray-400 dark:text-gray-500 line-clamp-1">{game.description}</p>
                                )}
                              </div>

                              <div className="flex items-center gap-4 shrink-0 text-xs text-gray-400 dark:text-gray-500">
                                {numLevels > 0 && (
                                  <span className="flex items-center gap-1 text-blue-400">
                                    <Layers size={12} /> {numLevels}
                                  </span>
                                )}
                                <span className="tabular-nums">{game.total_plays} plays</span>
                                {winRate !== null && winRate > 0 && (
                                  <span className={`px-1.5 py-0.5 rounded text-[10px] font-semibold ${
                                    winRate >= 50 ? 'bg-emerald-500/15 text-emerald-500 dark:text-emerald-400' :
                                    winRate >= 20 ? 'bg-yellow-500/15 text-yellow-600 dark:text-yellow-400' :
                                    'bg-red-500/15 text-red-500 dark:text-red-400'
                                  }`}>
                                    {winRate}%
                                  </span>
                                )}
                                <ChevronRight size={14} className="text-gray-300 dark:text-gray-600 group-hover:text-blue-400 transition-colors" />
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
            <div className="bg-white dark:bg-gray-900 rounded-2xl border border-gray-200 dark:border-gray-800 p-8">
              <div className="text-center mb-8">
                <div className="w-14 h-14 rounded-2xl bg-purple-600/20 border border-purple-500/30 flex items-center justify-center mx-auto mb-4">
                  <Zap size={24} className="text-purple-400" />
                </div>
                <h2 className="text-xl font-bold text-gray-900 dark:text-white">Play Your Own Game</h2>
                <p className="text-gray-500 dark:text-gray-400 text-sm mt-1">Upload a game.py and metadata.json to play instantly. Nothing is saved.</p>
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
                <label className="flex items-center gap-1.5 text-sm text-gray-500 dark:text-gray-400 mb-2">
                  <User size={14} /> Your name <span className="text-gray-400 dark:text-gray-600">(optional)</span>
                </label>
                <input
                  type="text"
                  value={playerName}
                  onChange={(e) => setPlayerName(e.target.value)}
                  placeholder="For your own reference"
                  maxLength={50}
                  className="w-full px-4 py-2.5 bg-gray-100 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-xl text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-gray-600 focus:outline-none focus:border-blue-500/50 text-sm"
                />
              </div>

              {/* Play button */}
              <button
                onClick={handleDirectPlay}
                disabled={!gameFile || !metadataFile}
                className="w-full px-4 py-3 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-100 dark:disabled:bg-gray-800 disabled:text-gray-400 dark:disabled:text-gray-600 disabled:cursor-not-allowed text-white font-medium rounded-xl text-sm transition-colors flex items-center justify-center gap-2"
              >
                <Zap size={16} /> Play Now
              </button>

              {/* Format hint */}
              <div className="mt-6 p-4 bg-gray-100/50 dark:bg-gray-800/50 rounded-xl">
                <p className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1.5">Expected format</p>
                <div className="text-[11px] text-gray-400 dark:text-gray-500 space-y-0.5">
                  <p><strong className="text-gray-500 dark:text-gray-400">game.py</strong> — Class extending ARCBaseGame with step() method</p>
                  <p><strong className="text-gray-500 dark:text-gray-400">metadata.json</strong> — {`{"game_id": "xx00"}`} e.g. "cw45", "ls20"</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ── Request Upload Tab ── */}
        {activeTab === 'request' && (
          <GameUploadForm
            mode="request"
            onSubmit={async (formData) => {
              setRequestSubmitting(true);
              setRequestSuccess('');
              setRequestError('');
              try {
                await requestsAPI.submit(formData);
                setRequestSuccess('Game submitted for review! You will be notified once approved.');
                return true;
              } catch (err) {
                setRequestError(err.response?.data?.detail || 'Submission failed');
                return false;
              } finally {
                setRequestSubmitting(false);
              }
            }}
            submitting={requestSubmitting}
            successMessage={requestSuccess}
            errorMessage={requestError}
          />
        )}
      </main>

      {/* Player Name Modal (for registered games) */}
      <AnimatePresence>
      {selectedGame && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4" onClick={() => setSelectedGame(null)}>
          <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" />
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ type: 'spring', stiffness: 300, damping: 25 }}
            className="relative bg-white dark:bg-gray-900 rounded-2xl border border-gray-200 dark:border-gray-800 shadow-2xl w-full max-w-sm overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            <button onClick={() => setSelectedGame(null)} className="absolute top-3 right-3 p-1 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-400 dark:text-gray-500">
              <X size={16} />
            </button>
            <div className="p-6">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-12 h-12 rounded-xl bg-blue-600/20 border border-blue-500/30 flex items-center justify-center shrink-0">
                  <Gamepad2 size={22} className="text-blue-400" />
                </div>
                <div className="min-w-0">
                  <p className="text-gray-900 dark:text-white font-mono font-bold text-lg">{selectedGame.game_id}</p>
                  <div className="flex items-center gap-3 text-xs text-gray-400 dark:text-gray-500 mt-0.5">
                    {selectedGame.baseline_actions?.length > 0 && (
                      <span className="flex items-center gap-1">
                        <Layers size={11} className="text-blue-400" />
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
                <label className="flex items-center gap-1.5 text-sm text-gray-500 dark:text-gray-400 mb-2">
                  <User size={14} /> Your name <span className="text-gray-400 dark:text-gray-600">(optional)</span>
                </label>
                <input
                  ref={nameInputRef}
                  type="text"
                  value={playerName}
                  onChange={(e) => setPlayerName(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Enter your name for the leaderboard"
                  maxLength={50}
                  className="w-full px-4 py-3 bg-gray-100 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-xl text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-gray-600 focus:outline-none focus:border-blue-500/50 text-sm"
                />
              </div>
              <div className="flex gap-2">
                <button
                  onClick={handlePlay}
                  className="flex-1 px-4 py-3 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-xl text-sm transition-colors flex items-center justify-center gap-2"
                >
                  Play <ArrowRight size={16} />
                </button>
                <button
                  onClick={handleSkip}
                  className="px-4 py-3 bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-500 dark:text-gray-400 rounded-xl text-sm transition-colors"
                >
                  Skip
                </button>
              </div>
            </div>
          </motion.div>
        </div>
      )}
      </AnimatePresence>

      <style>{`
        @keyframes modal-in {
          from { opacity: 0; transform: scale(0.95) translateY(8px); }
          to { opacity: 1; transform: scale(1) translateY(0); }
        }
        .animate-modal-in { animation: modal-in 0.2s ease-out; }
      `}</style>
    </div>
  );
}

function DropZone({ file, setFile, fileRef, accept, icon: Icon, label, hint }) {
  return (
    <div
      onClick={() => fileRef.current?.click()}
      className={`border-2 border-dashed rounded-xl p-5 text-center cursor-pointer transition-colors ${
        file
          ? 'border-green-500/30 bg-green-500/5'
          : 'border-gray-300 dark:border-gray-700 hover:border-purple-500/40 hover:bg-gray-100/50 dark:hover:bg-gray-800/50'
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
          <p className="text-sm font-medium text-green-400 truncate">{file.name}</p>
          <p className="text-[10px] text-gray-400 dark:text-gray-500 mt-0.5">{(file.size / 1024).toFixed(1)} KB</p>
          <button
            type="button"
            onClick={(e) => { e.stopPropagation(); setFile(null); }}
            className="text-[10px] text-gray-400 dark:text-gray-500 hover:text-red-400 mt-1"
          >
            Remove
          </button>
        </div>
      ) : (
        <>
          <Icon size={28} className="mx-auto text-gray-400 dark:text-gray-600 mb-2" />
          <p className="text-xs font-medium text-gray-500 dark:text-gray-400">{label}</p>
          <p className="text-[10px] text-gray-400 dark:text-gray-600 mt-0.5">{hint}</p>
        </>
      )}
    </div>
  );
}

