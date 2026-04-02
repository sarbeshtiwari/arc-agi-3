import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { gamesAPI, analyticsAPI } from '../api/client';
import ConfirmModal from '../components/ConfirmModal';
import { 
  Search, Power, PowerOff, Trash2, 
  Play, Eye, X, Gamepad2, Download,
  ChevronLeft, ChevronRight, CheckSquare, Square, MinusSquare
} from 'lucide-react';

export default function GamesPage() {
  const [games, setGames] = useState([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [filter, setFilter] = useState('all');
  const [error, setError] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const PAGE_SIZE = 15;

  const [exportFilter, setExportFilter] = useState('all');
  const [exportDate, setExportDate] = useState('');
  const [exportDateFrom, setExportDateFrom] = useState('');
  const [exportDateTo, setExportDateTo] = useState('');
  const [exporting, setExporting] = useState(false);
  const [exportingGames, setExportingGames] = useState(false);

  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [bulkAction, setBulkAction] = useState<'activate' | 'deactivate' | 'delete' | null>(null);
  const [bulkLoading, setBulkLoading] = useState(false);

  const handleExportAll = async () => {
    setExporting(true);
    try {
      const params: any = { filter: exportFilter };
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
      setSelectedIds((prev) => { const next = new Set(prev); next.delete(gameId); return next; });
    } catch (err) {
      setError(err.response?.data?.detail || 'Action failed');
      setTimeout(() => setError(''), 5000);
    }
  };

  const toggleSelect = (gameId: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(gameId)) next.delete(gameId);
      else next.add(gameId);
      return next;
    });
  };

  const toggleSelectAll = (pageGames: any[]) => {
    const pageIds = pageGames.map((g) => g.game_id);
    const allSelected = pageIds.every((id) => selectedIds.has(id));
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (allSelected) {
        pageIds.forEach((id) => next.delete(id));
      } else {
        pageIds.forEach((id) => next.add(id));
      }
      return next;
    });
  };

  const handleBulkAction = async () => {
    if (!bulkAction || selectedIds.size === 0) return;
    setBulkLoading(true);
    const ids = Array.from(selectedIds);
    let failCount = 0;

    try {
      if (bulkAction === 'activate') {
        await Promise.allSettled(ids.map((id) => gamesAPI.toggle(id, true))).then((results) => {
          results.forEach((r) => { if (r.status === 'rejected') failCount++; });
        });
        setGames((prev) => prev.map((g) => selectedIds.has(g.game_id) ? { ...g, is_active: true } : g));
      } else if (bulkAction === 'deactivate') {
        await Promise.allSettled(ids.map((id) => gamesAPI.toggle(id, false))).then((results) => {
          results.forEach((r) => { if (r.status === 'rejected') failCount++; });
        });
        setGames((prev) => prev.map((g) => selectedIds.has(g.game_id) ? { ...g, is_active: false } : g));
      } else if (bulkAction === 'delete') {
        await Promise.allSettled(ids.map((id) => gamesAPI.delete(id))).then((results) => {
          results.forEach((r) => { if (r.status === 'rejected') failCount++; });
        });
        setGames((prev) => prev.filter((g) => !selectedIds.has(g.game_id)));
      }

      if (failCount > 0) {
        setError(`${failCount} of ${ids.length} actions failed`);
        setTimeout(() => setError(''), 5000);
      }
    } catch (err) {
      setError('Bulk action failed');
      setTimeout(() => setError(''), 5000);
    } finally {
      setBulkLoading(false);
      setBulkAction(null);
      setSelectedIds(new Set());
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

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const safePage = Math.min(currentPage, totalPages);
  const paginatedGames = filtered.slice((safePage - 1) * PAGE_SIZE, safePage * PAGE_SIZE);

  const pageIds = paginatedGames.map((g) => g.game_id);
  const allPageSelected = pageIds.length > 0 && pageIds.every((id) => selectedIds.has(id));
  const somePageSelected = pageIds.some((id) => selectedIds.has(id));

  useEffect(() => { setCurrentPage(1); }, [search, filter]);
  useEffect(() => { setSelectedIds(new Set()); }, [search, filter]);

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
    <div className="min-h-screen bg-gray-50 dark:bg-gray-950 p-6">
      {error && <div className="mb-4 px-4 py-2 bg-red-500/10 border border-red-500/30 rounded-lg text-sm text-red-400">{error}</div>}

      <AnimatePresence>
        {selectedIds.size > 0 && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.2 }}
            className="mb-4 flex items-center gap-3 px-4 py-2.5 bg-blue-600/10 border border-blue-500/30 rounded-xl"
          >
            <span className="text-sm font-medium text-blue-400">
              {selectedIds.size} game{selectedIds.size !== 1 ? 's' : ''} selected
            </span>
            <div className="w-px h-5 bg-blue-500/30" />
            <button
              onClick={() => setBulkAction('activate')}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg text-green-400 hover:bg-green-500/10 transition-colors"
            >
              <Power size={14} /> Activate
            </button>
            <button
              onClick={() => setBulkAction('deactivate')}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg text-orange-400 hover:bg-orange-500/10 transition-colors"
            >
              <PowerOff size={14} /> Deactivate
            </button>
            <button
              onClick={() => setBulkAction('delete')}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg text-red-400 hover:bg-red-500/10 transition-colors"
            >
              <Trash2 size={14} /> Delete
            </button>
            <button
              onClick={() => setSelectedIds(new Set())}
              className="ml-auto flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg text-gray-400 dark:text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
            >
              <X size={14} /> Clear
            </button>
          </motion.div>
        )}
      </AnimatePresence>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
            <Gamepad2 size={24} className="text-blue-400" />
            Games
          </h1>
          <p className="text-gray-500 dark:text-gray-400 text-sm mt-1">
            Manage your ARC-AGI-3 game environments ({games.length} total)
          </p>
        </div>
      </div>

      {/* Export bar */}
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }} className="flex flex-wrap items-center gap-2 mb-6 p-3 bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800">
        <Download size={14} className="text-gray-500 dark:text-gray-400" />
        <span className="text-xs text-gray-500 dark:text-gray-400 font-medium">Export All Sessions:</span>

        <select
          value={exportFilter}
          onChange={(e) => setExportFilter(e.target.value)}
          className="px-2 py-1.5 bg-gray-100 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg text-xs text-gray-900 dark:text-white focus:outline-none focus:border-blue-500/50"
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
            className="px-2 py-1.5 bg-gray-100 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg text-xs text-gray-900 dark:text-white focus:outline-none focus:border-blue-500/50"
          />
        )}

        {exportFilter === 'range' && (
          <>
            <input
              type="date"
              value={exportDateFrom}
              onChange={(e) => setExportDateFrom(e.target.value)}
              className="px-2 py-1.5 bg-gray-100 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg text-xs text-gray-900 dark:text-white focus:outline-none focus:border-blue-500/50"
            />
            <span className="text-xs text-gray-400 dark:text-gray-500">to</span>
            <input
              type="date"
              value={exportDateTo}
              onChange={(e) => setExportDateTo(e.target.value)}
              className="px-2 py-1.5 bg-gray-100 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg text-xs text-gray-900 dark:text-white focus:outline-none focus:border-blue-500/50"
            />
          </>
        )}

        <button
          onClick={handleExportAll}
          disabled={exporting || (exportFilter === 'date' && !exportDate) || (exportFilter === 'range' && (!exportDateFrom || !exportDateTo))}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-200 dark:disabled:bg-gray-700 disabled:text-gray-400 dark:disabled:text-gray-500 text-white rounded-lg text-xs font-medium transition-colors"
        >
          {exporting ? (
            <><div className="animate-spin rounded-full h-3 w-3 border-b-2 border-white" /> Exporting...</>
          ) : (
            <><Download size={12} /> Sessions .xlsx</>
          )}
        </button>

        <div className="w-px h-6 bg-gray-300 dark:bg-gray-700 mx-1" />

        <button
          onClick={handleExportGamesList}
          disabled={exportingGames}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-200 dark:disabled:bg-gray-700 disabled:text-gray-400 dark:disabled:text-gray-500 text-white rounded-lg text-xs font-medium transition-colors"
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
          <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400 dark:text-gray-500" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search by name or ID..."
            className="w-full pl-10 pr-8 py-2 bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-lg text-gray-900 dark:text-white text-sm placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none focus:border-blue-500 transition-colors"
          />
          {search && (
            <button
              onClick={() => setSearch('')}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 dark:text-gray-500 hover:text-gray-600 dark:hover:text-gray-300"
            >
              <X size={14} />
            </button>
          )}
        </div>
        <div className="flex bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-lg overflow-hidden">
          {filterTabs.map((tab) => (
            <button
              key={tab.key}
              onClick={() => setFilter(tab.key)}
              className={`px-4 py-2 text-sm font-medium transition-colors ${
                filter === tab.key
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-500 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-800'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
        <span className="text-sm text-gray-400 dark:text-gray-500 ml-auto">
          {filtered.length} result{filtered.length !== 1 ? 's' : ''}
        </span>
      </div>

      {/* Table */}
      {loading ? (
        <div className="flex justify-center py-20">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500" />
        </div>
      ) : filtered.length === 0 ? (
        <div className="text-center py-20 bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800">
          <Gamepad2 size={48} className="mx-auto text-gray-300 dark:text-gray-700 mb-4" />
          <p className="text-gray-500 dark:text-gray-400 mb-2">No games found</p>
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
        <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 overflow-hidden">
          <table className="w-full text-sm text-left">
            <thead>
              <tr className="border-b border-gray-200 dark:border-gray-800 text-gray-500 dark:text-gray-400 text-xs uppercase tracking-wider">
                <th className="px-4 py-3 w-10">
                  <button onClick={() => toggleSelectAll(paginatedGames)} className="p-0.5 rounded hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors">
                    {allPageSelected ? (
                      <CheckSquare size={16} className="text-blue-500" />
                    ) : somePageSelected ? (
                      <MinusSquare size={16} className="text-blue-500" />
                    ) : (
                      <Square size={16} />
                    )}
                  </button>
                </th>
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
            <tbody className="divide-y divide-gray-200 dark:divide-gray-800">
              {paginatedGames.map((game) => (
                <tr
                  key={game.game_id}
                  className="hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors"
                >
                  <td className="px-4 py-3 w-10">
                    <button onClick={() => toggleSelect(game.game_id)} className="p-0.5 rounded hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors">
                      {selectedIds.has(game.game_id) ? (
                        <CheckSquare size={16} className="text-blue-500" />
                      ) : (
                        <Square size={16} className="text-gray-400 dark:text-gray-600" />
                      )}
                    </button>
                  </td>
                  <td className="px-4 py-3">
                    <Link
                      to={`/admin/games/${game.game_id}`}
                      className="text-blue-400 hover:text-blue-300 font-mono text-xs transition-colors"
                      title={game.name || game.game_id}
                    >
                      {game.game_id}
                    </Link>
                    {game.name && game.name !== game.game_id && (
                      <p className="text-gray-400 dark:text-gray-500 text-xs mt-0.5 truncate max-w-[200px]">
                        {game.name}
                      </p>
                    )}
                    {game.description && (
                      <p className="text-gray-400 dark:text-gray-600 text-xs mt-0.5 truncate max-w-[250px]">
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
                              className="px-1.5 py-0.5 bg-gray-100 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded text-[11px] text-gray-500 dark:text-gray-400"
                            >
                              {tag}
                            </span>
                          ))}
                          {game.tags.length > 2 && (
                            <span className="text-[11px] text-gray-400 dark:text-gray-600">+{game.tags.length - 2}</span>
                          )}
                        </>
                      ) : (
                        <span className="text-gray-400 dark:text-gray-600 text-xs">—</span>
                      )}
                    </div>
                  </td>

                  {/* Plays */}
                  <td className="px-4 py-3 text-right text-gray-700 dark:text-gray-300 tabular-nums">
                    {game.total_plays ?? 0}
                  </td>

                  {/* Wins */}
                  <td className="px-4 py-3 text-right text-gray-700 dark:text-gray-300 tabular-nums">
                    {game.total_wins ?? 0}
                  </td>

                  {/* FPS */}
                  <td className="px-4 py-3 text-right text-gray-700 dark:text-gray-300 tabular-nums">
                    {game.default_fps ?? '—'}
                  </td>

                  {/* Created */}
                  <td className="px-4 py-3 text-gray-500 dark:text-gray-400 text-xs whitespace-nowrap">
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

          {totalPages > 1 && (
            <div className="flex items-center justify-between px-4 py-3 border-t border-gray-200 dark:border-gray-800">
              <span className="text-xs text-gray-400 dark:text-gray-500">
                Showing {(safePage - 1) * PAGE_SIZE + 1}–{Math.min(safePage * PAGE_SIZE, filtered.length)} of {filtered.length}
              </span>
              <div className="flex items-center gap-1">
                <button
                  onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                  disabled={safePage <= 1}
                  className="p-1.5 rounded-md text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                >
                  <ChevronLeft size={16} />
                </button>
                {Array.from({ length: totalPages }, (_, i) => i + 1).map((page) => (
                  <button
                    key={page}
                    onClick={() => setCurrentPage(page)}
                    className={`min-w-[32px] px-2 py-1 rounded-md text-xs font-medium transition-colors ${
                      page === safePage
                        ? 'bg-blue-600 text-white'
                        : 'text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 hover:text-gray-900 dark:hover:text-white'
                    }`}
                  >
                    {page}
                  </button>
                ))}
                <button
                  onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                  disabled={safePage >= totalPages}
                  className="p-1.5 rounded-md text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                >
                  <ChevronRight size={16} />
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      <ConfirmModal
        open={bulkAction !== null}
        onClose={() => setBulkAction(null)}
        onConfirm={handleBulkAction}
        loading={bulkLoading}
        variant={bulkAction === 'delete' ? 'danger' : 'warning'}
        title={
          bulkAction === 'activate' ? `Activate ${selectedIds.size} games?` :
          bulkAction === 'deactivate' ? `Deactivate ${selectedIds.size} games?` :
          `Delete ${selectedIds.size} games?`
        }
        message={
          bulkAction === 'delete'
            ? 'This will permanently delete the selected games and their files. This cannot be undone.'
            : `This will ${bulkAction} all selected games.`
        }
        confirmText={
          bulkAction === 'activate' ? 'Activate All' :
          bulkAction === 'deactivate' ? 'Deactivate All' :
          'Delete All'
        }
      />
    </div>
  );
}
