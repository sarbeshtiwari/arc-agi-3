import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { gamesAPI, analyticsAPI } from '../api/client';
import { 
  Search, Power, PowerOff, Trash2, 
  Play, Eye, X, Gamepad2, Download
} from 'lucide-react';

export default function GamesPage() {
  const [games, setGames] = useState([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [filter, setFilter] = useState('all');
  const [error, setError] = useState('');

  // Export state
  const [exportFilter, setExportFilter] = useState('all');
  const [exportDate, setExportDate] = useState('');
  const [exportDateFrom, setExportDateFrom] = useState('');
  const [exportDateTo, setExportDateTo] = useState('');
  const [exporting, setExporting] = useState(false);
  const [exportingGames, setExportingGames] = useState(false);

  const handleExportAll = async () => {
    setExporting(true);
    try {
      const params = { filter: exportFilter };
      if (exportFilter === 'date') params.date = exportDate;
      if (exportFilter === 'range') { params.date_from = exportDateFrom; params.date_to = exportDateTo; }
      const res = await analyticsAPI.exportAllExcel(params);
      const url = window.URL.createObjectURL(new Blob([res.data]));
      const a = document.createElement('a');
      a.href = url;
      a.download = `all_games_sessions_${exportFilter}.xlsx`;
      a.click();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to export');
      setTimeout(() => setError(''), 5000);
    }
    setExporting(false);
  };

  const handleExportGamesList = async () => {
    setExportingGames(true);
    try {
      const res = await analyticsAPI.exportGamesList();
      const url = window.URL.createObjectURL(new Blob([res.data]));
      const a = document.createElement('a');
      a.href = url;
      a.download = 'games_list.xlsx';
      a.click();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to export');
      setTimeout(() => setError(''), 5000);
    }
    setExportingGames(false);
  };

  const fetchGames = () => {
    setLoading(true);
    gamesAPI.list()
      .then((res) => setGames(res.data))
      .catch(() => {})
      .finally(() => setLoading(false));
  };

  useEffect(() => { fetchGames(); }, []);

  const handleToggle = async (gameId, currentActive) => {
    try {
      await gamesAPI.toggle(gameId, !currentActive);
      setGames((prev) =>
        prev.map((g) =>
          g.game_id === gameId ? { ...g, is_active: !currentActive } : g
        )
      );
    } catch (err) {
      setError(err.response?.data?.detail || 'Action failed');
      setTimeout(() => setError(''), 5000);
    }
  };

  const handleDelete = async (gameId) => {
    if (!confirm(`Delete game "${gameId}"? This cannot be undone.`)) return;
    try {
      await gamesAPI.delete(gameId);
      setGames((prev) => prev.filter((g) => g.game_id !== gameId));
    } catch (err) {
      setError(err.response?.data?.detail || 'Action failed');
      setTimeout(() => setError(''), 5000);
    }
  };

  const filtered = games.filter((g) => {
    const term = search.toLowerCase();
    const matchSearch =
      g.name?.toLowerCase().includes(term) ||
      g.game_id.toLowerCase().includes(term);
    const matchFilter =
      filter === 'all' ||
      (filter === 'active' && g.is_active) ||
      (filter === 'inactive' && !g.is_active);
    return matchSearch && matchFilter;
  });

  const formatDate = (dateStr) => {
    if (!dateStr) return '—';
    const d = new Date(dateStr);
    return d.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    });
  };

  const filterTabs = [
    { key: 'all', label: 'All' },
    { key: 'active', label: 'Active' },
    { key: 'inactive', label: 'Inactive' },
  ];

  return (
    <div className="min-h-screen bg-gray-950 p-6">
      {error && <div className="mb-4 px-4 py-2 bg-red-500/10 border border-red-500/30 rounded-lg text-sm text-red-400">{error}</div>}
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-2">
            <Gamepad2 size={24} className="text-blue-400" />
            Games
          </h1>
          <p className="text-gray-400 text-sm mt-1">
            Manage your ARC-AGI-3 game environments ({games.length} total)
          </p>
        </div>
      </div>

      {/* Export bar */}
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }} className="flex flex-wrap items-center gap-2 mb-6 p-3 bg-gray-900 rounded-xl border border-gray-800">
        <Download size={14} className="text-gray-400" />
        <span className="text-xs text-gray-400 font-medium">Export All Sessions:</span>

        <select
          value={exportFilter}
          onChange={(e) => setExportFilter(e.target.value)}
          className="px-2 py-1.5 bg-gray-800 border border-gray-700 rounded-lg text-xs text-white focus:outline-none focus:border-blue-500/50"
        >
          <option value="all">All Time</option>
          <option value="today">Today</option>
          <option value="date">Specific Date</option>
          <option value="range">Date Range</option>
        </select>

        {exportFilter === 'date' && (
          <input
            type="date"
            value={exportDate}
            onChange={(e) => setExportDate(e.target.value)}
            className="px-2 py-1.5 bg-gray-800 border border-gray-700 rounded-lg text-xs text-white focus:outline-none focus:border-blue-500/50"
          />
        )}

        {exportFilter === 'range' && (
          <>
            <input
              type="date"
              value={exportDateFrom}
              onChange={(e) => setExportDateFrom(e.target.value)}
              className="px-2 py-1.5 bg-gray-800 border border-gray-700 rounded-lg text-xs text-white focus:outline-none focus:border-blue-500/50"
            />
            <span className="text-xs text-gray-500">to</span>
            <input
              type="date"
              value={exportDateTo}
              onChange={(e) => setExportDateTo(e.target.value)}
              className="px-2 py-1.5 bg-gray-800 border border-gray-700 rounded-lg text-xs text-white focus:outline-none focus:border-blue-500/50"
            />
          </>
        )}

        <button
          onClick={handleExportAll}
          disabled={exporting || (exportFilter === 'date' && !exportDate) || (exportFilter === 'range' && (!exportDateFrom || !exportDateTo))}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded-lg text-xs font-medium transition-colors"
        >
          {exporting ? (
            <><div className="animate-spin rounded-full h-3 w-3 border-b-2 border-white" /> Exporting...</>
          ) : (
            <><Download size={12} /> Sessions .xlsx</>
          )}
        </button>

        <div className="w-px h-6 bg-gray-700 mx-1" />

        <button
          onClick={handleExportGamesList}
          disabled={exportingGames}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded-lg text-xs font-medium transition-colors"
        >
          {exportingGames ? (
            <><div className="animate-spin rounded-full h-3 w-3 border-b-2 border-white" /> Exporting...</>
          ) : (
            <><Download size={12} /> Games List .xlsx</>
          )}
        </button>
      </motion.div>

      {/* Search & Filter Bar */}
      <div className="flex items-center gap-4 mb-6">
        <div className="relative flex-1 max-w-sm">
          <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search by name or ID..."
            className="w-full pl-10 pr-8 py-2 bg-gray-900 border border-gray-800 rounded-lg text-white text-sm placeholder-gray-500 focus:outline-none focus:border-blue-500 transition-colors"
          />
          {search && (
            <button
              onClick={() => setSearch('')}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-300"
            >
              <X size={14} />
            </button>
          )}
        </div>
        <div className="flex bg-gray-900 border border-gray-800 rounded-lg overflow-hidden">
          {filterTabs.map((tab) => (
            <button
              key={tab.key}
              onClick={() => setFilter(tab.key)}
              className={`px-4 py-2 text-sm font-medium transition-colors ${
                filter === tab.key
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-400 hover:text-white hover:bg-gray-800'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
        <span className="text-sm text-gray-500 ml-auto">
          {filtered.length} result{filtered.length !== 1 ? 's' : ''}
        </span>
      </div>

      {/* Table */}
      {loading ? (
        <div className="flex justify-center py-20">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500" />
        </div>
      ) : filtered.length === 0 ? (
        <div className="text-center py-20 bg-gray-900 rounded-xl border border-gray-800">
          <Gamepad2 size={48} className="mx-auto text-gray-700 mb-4" />
          <p className="text-gray-400 mb-2">No games found</p>
          {search ? (
            <button
              onClick={() => setSearch('')}
              className="text-blue-400 hover:text-blue-300 text-sm"
            >
              Clear search
            </button>
          ) : (
            <Link
              to="/admin/games/upload"
              className="text-blue-400 hover:text-blue-300 text-sm"
            >
              Upload your first game
            </Link>
          )}
        </div>
      ) : (
        <div className="bg-gray-900 rounded-xl border border-gray-800 overflow-hidden">
          <table className="w-full text-sm text-left">
            <thead>
              <tr className="border-b border-gray-800 text-gray-400 text-xs uppercase tracking-wider">
                <th className="px-4 py-3 font-medium">Game ID</th>
                <th className="px-4 py-3 font-medium">Status</th>
                <th className="px-4 py-3 font-medium">Tags</th>
                <th className="px-4 py-3 font-medium text-right">Plays</th>
                <th className="px-4 py-3 font-medium text-right">Wins</th>
                <th className="px-4 py-3 font-medium text-right">FPS</th>
                <th className="px-4 py-3 font-medium">Created</th>
                <th className="px-4 py-3 font-medium text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {filtered.map((game) => (
                <tr
                  key={game.game_id}
                  className="hover:bg-gray-800/50 transition-colors"
                >
                  {/* Game ID */}
                  <td className="px-4 py-3">
                    <Link
                      to={`/admin/games/${game.game_id}`}
                      className="text-blue-400 hover:text-blue-300 font-mono text-xs transition-colors"
                      title={game.name || game.game_id}
                    >
                      {game.game_id}
                    </Link>
                    {game.name && game.name !== game.game_id && (
                      <p className="text-gray-500 text-xs mt-0.5 truncate max-w-[200px]">
                        {game.name}
                      </p>
                    )}
                    {game.description && (
                      <p className="text-gray-600 text-xs mt-0.5 truncate max-w-[250px]">
                        {game.description}
                      </p>
                    )}
                  </td>

                  {/* Status */}
                  <td className="px-4 py-3">
                    <span
                      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium ${
                        game.is_active
                          ? 'bg-green-500/15 text-green-400'
                          : 'bg-red-500/15 text-red-400'
                      }`}
                    >
                      <span
                        className={`w-1.5 h-1.5 rounded-full ${
                          game.is_active ? 'bg-green-400' : 'bg-red-400'
                        }`}
                      />
                      {game.is_active ? 'Active' : 'Inactive'}
                    </span>
                  </td>

                  {/* Tags */}
                  <td className="px-4 py-3">
                    <div className="flex flex-wrap gap-1 max-w-[200px]">
                      {game.tags && game.tags.length > 0 ? (
                        <>
                          {game.tags.slice(0, 2).map((tag) => (
                            <span
                              key={tag}
                              className="px-1.5 py-0.5 bg-gray-800 border border-gray-700 rounded text-[11px] text-gray-400"
                            >
                              {tag}
                            </span>
                          ))}
                          {game.tags.length > 2 && (
                            <span className="text-[11px] text-gray-600">+{game.tags.length - 2}</span>
                          )}
                        </>
                      ) : (
                        <span className="text-gray-600 text-xs">—</span>
                      )}
                    </div>
                  </td>

                  {/* Plays */}
                  <td className="px-4 py-3 text-right text-gray-300 tabular-nums">
                    {game.total_plays ?? 0}
                  </td>

                  {/* Wins */}
                  <td className="px-4 py-3 text-right text-gray-300 tabular-nums">
                    {game.total_wins ?? 0}
                  </td>

                  {/* FPS */}
                  <td className="px-4 py-3 text-right text-gray-300 tabular-nums">
                    {game.default_fps ?? '—'}
                  </td>

                  {/* Created */}
                  <td className="px-4 py-3 text-gray-400 text-xs whitespace-nowrap">
                    {formatDate(game.created_at)}
                  </td>

                  {/* Actions */}
                  <td className="px-4 py-3">
                    <div className="flex items-center justify-end gap-1">
                      <Link
                        to={`/admin/games/${game.game_id}/play`}
                        className="p-1.5 rounded-md text-green-400 hover:bg-green-500/10 transition-colors"
                        title="Play"
                      >
                        <Play size={15} />
                      </Link>
                      <Link
                        to={`/admin/games/${game.game_id}`}
                        className="p-1.5 rounded-md text-blue-400 hover:bg-blue-500/10 transition-colors"
                        title="Details"
                      >
                        <Eye size={15} />
                      </Link>
                      <button
                        onClick={() => handleToggle(game.game_id, game.is_active)}
                        className={`p-1.5 rounded-md transition-colors ${
                          game.is_active
                            ? 'text-orange-400 hover:bg-orange-500/10'
                            : 'text-green-400 hover:bg-green-500/10'
                        }`}
                        title={game.is_active ? 'Deactivate' : 'Activate'}
                      >
                        {game.is_active ? <PowerOff size={15} /> : <Power size={15} />}
                      </button>
                      <button
                        onClick={() => handleDelete(game.game_id)}
                        className="p-1.5 rounded-md text-red-400 hover:bg-red-500/10 transition-colors"
                        title="Delete"
                      >
                        <Trash2 size={15} />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
