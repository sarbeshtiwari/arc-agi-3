import React, { useState, useEffect } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { gamesAPI, analyticsAPI } from '../api/client';
import ConfirmModal from '../components/ConfirmModal';
import {
  ChevronLeft, ChevronDown, ChevronRight, Play, Power, PowerOff, Code, BarChart3, 
  Clock, Tag, Trash2, Edit, Settings, Eye, Activity, Layers, Heart, User, Skull,
  Download, Calendar, Video
} from 'lucide-react';

function formatTime(seconds: any) {
  if (!seconds || seconds < 0) return '0:00';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

export default function GameDetailPage() {
  const { gameId } = useParams();
  const navigate = useNavigate();
  const [game, setGame] = useState(null);
  const [analytics, setAnalytics] = useState(null);
  const [sessions, setSessions] = useState([]);
  const [source, setSource] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('overview');
  const [editing, setEditing] = useState(false);
  const [editForm, setEditForm] = useState<any>({ name: '', description: '', tags: '' });
  const [newGameFile, setNewGameFile] = useState(null);
  const [newMetadataFile, setNewMetadataFile] = useState(null);
  const [fileUploading, setFileUploading] = useState(false);
  const [videos, setVideos] = useState<any[]>([]);
  const [videosLoading, setVideosLoading] = useState(false);

  useEffect(() => {
    Promise.all([
      gamesAPI.get(gameId).then(r => r.data),
      analyticsAPI.game(gameId).then(r => r.data).catch(() => null),
      analyticsAPI.sessions(gameId, 20).then(r => r.data).catch(() => []),
    ]).then(([gameData, analyticsData, sessionsData]) => {
      setGame(gameData);
      setAnalytics(analyticsData);
      setSessions(sessionsData || []);
      setEditForm({
        name: gameData.name,
        description: gameData.description || '',
        game_rules: gameData.game_rules || '',
        game_owner_name: gameData.game_owner_name || '',
        game_drive_link: gameData.game_drive_link || '',
        game_video_link: gameData.game_video_link || '',
        default_fps: gameData.default_fps || 5,
        tags: (gameData.tags || []).join(', '),
      });
    }).catch(() => navigate('/admin/games'))
    .finally(() => setLoading(false));
  }, [gameId, navigate]);

  const [toggleError, setToggleError] = useState('');

  const handleToggle = async () => {
    if (!game) return;
    setToggleError('');
    try {
      const res = await gamesAPI.toggle(gameId, !game.is_active);
      setGame(res.data);
    } catch (err) {
      const msg = err.response?.data?.detail || 'Failed to toggle game status';
      setToggleError(msg);
      setTimeout(() => setToggleError(''), 5000);
    }
  };

  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);

  const handleDelete = async () => {
    try {
      await gamesAPI.delete(gameId);
      navigate('/admin/games');
    } catch (err) {
      setToggleError(err.response?.data?.detail || 'Failed to delete game');
      setTimeout(() => setToggleError(''), 5000);
    }
    setShowDeleteConfirm(false);
  };

  const handleSave = async () => {
    try {
      // Save metadata fields
      const res = await gamesAPI.update(gameId, {
        name: editForm.name,
        description: editForm.description,
        game_rules: editForm.game_rules,
        game_owner_name: editForm.game_owner_name,
        game_drive_link: editForm.game_drive_link,
        game_video_link: editForm.game_video_link,
        default_fps: parseInt(editForm.default_fps) || 5,
        tags: editForm.tags.split(',').map(t => t.trim()).filter(Boolean),
      });
      let updatedGame = res.data;

      // Upload files if any were selected
      if (newGameFile || newMetadataFile) {
        setFileUploading(true);
        const formData = new FormData();
        if (newGameFile) formData.append('game_file', newGameFile);
        if (newMetadataFile) formData.append('metadata_file', newMetadataFile);
        try {
          const fileRes = await gamesAPI.updateFiles(gameId, formData);
          updatedGame = fileRes.data;
        } catch (fileErr) {
          setToggleError(fileErr.response?.data?.detail || 'Failed to update game files');
          setTimeout(() => setToggleError(''), 5000);
          setFileUploading(false);
          // Still save the metadata changes
          setGame(updatedGame);
          return;
        }
        setFileUploading(false);
      }

      setGame(updatedGame);
      setEditing(false);
      setNewGameFile(null);
      setNewMetadataFile(null);
      setSaveSuccess(true);
      setTimeout(() => setSaveSuccess(false), 3000);
    } catch (err) {
      setToggleError(err.response?.data?.detail || 'Failed to save changes');
      setTimeout(() => setToggleError(''), 5000);
    }
  };

  const loadSource = async () => {
    if (source) return;
    try {
      const res = await gamesAPI.getSource(gameId);
      setSource(res.data);
    } catch {
      // error loading source silently ignored
    }
  };

  const loadVideos = async () => {
    if (videos.length > 0) return; // already loaded
    setVideosLoading(true);
    try {
      const res = await gamesAPI.listVideos(gameId);
      setVideos(res.data);
    } catch (err) {
      console.error(err);
    } finally {
      setVideosLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  if (!game) return null;

  const tabs = [
    { id: 'overview', label: 'Overview', icon: Eye },
    { id: 'analytics', label: 'Analytics', icon: BarChart3 },
    { id: 'source', label: 'Source Code', icon: Code },
    { id: 'sessions', label: 'Sessions', icon: Activity },
    { id: 'videos', label: 'Videos', icon: Video },
  ];

  return (
    <div>
      {/* Header */}
      <div className="flex items-center gap-4 mb-6">
        <Link to="/admin/games" className="p-2 rounded-lg hover:bg-gray-800 text-gray-400">
          <ChevronLeft size={20} />
        </Link>
        <div className="flex-1">
          <div className="flex items-center gap-3">
            <h1 className="text-xl font-bold text-white">{game.name}</h1>
            <span className={`px-2 py-1 rounded text-xs font-medium ${
              game.is_active ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
            }`}>
              {game.is_active ? 'Active' : 'Inactive'}
            </span>
          </div>
          <p className="text-sm text-gray-400 font-mono">{game.game_id}</p>
        </div>
        <div className="flex gap-2">
          <Link
            to={`/admin/games/${gameId}/play`}
            className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg text-sm transition-colors"
          >
            <Play size={14} /> Play
          </Link>
          <button
            onClick={handleToggle}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm transition-colors border ${
              game.is_active
                ? 'bg-orange-600/20 border-orange-500/30 text-orange-400 hover:bg-orange-600/30'
                : 'bg-green-600/20 border-green-500/30 text-green-400 hover:bg-green-600/30'
            }`}
          >
            {game.is_active ? <PowerOff size={14} /> : <Power size={14} />}
            {game.is_active ? 'Deactivate' : 'Activate'}
          </button>
          <button
            onClick={() => setShowDeleteConfirm(true)}
            className="flex items-center gap-2 px-4 py-2 bg-red-600/20 border border-red-500/30 text-red-400 hover:bg-red-600/30 rounded-lg text-sm transition-colors"
          >
            <Trash2 size={14} /> Delete
          </button>
        </div>

      <ConfirmModal
        open={showDeleteConfirm}
        onClose={() => setShowDeleteConfirm(false)}
        onConfirm={handleDelete}
        title={`Delete "${gameId}"?`}
        message="This will permanently delete the game, all play sessions, and analytics data. This action cannot be undone."
        confirmText="Delete Game"
        variant="danger"
      />
      </div>

      {/* Status messages */}
      {toggleError && (
        <div className="mb-4 px-4 py-2 bg-red-500/10 border border-red-500/30 rounded-lg text-sm text-red-400">
          {toggleError}
        </div>
      )}
      {saveSuccess && (
        <div className="mb-4 px-4 py-2 bg-green-500/10 border border-green-500/30 rounded-lg text-sm text-green-400">
          Game details updated successfully
        </div>
      )}

      {/* Tabs */}
      <div className="flex border-b border-gray-800 mb-6">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => {
              setActiveTab(tab.id);
              if (tab.id === 'source') loadSource();
              if (tab.id === 'videos') loadVideos();
            }}
            className={`flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
              activeTab === tab.id
                ? 'border-blue-500 text-blue-400'
                : 'border-transparent text-gray-400 hover:text-white'
            }`}
          >
            <tab.icon size={16} /> {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {activeTab === 'overview' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Info */}
          <div className="lg:col-span-2 bg-gray-900 rounded-xl border border-gray-800 p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-white">Game Information</h2>
              <button
                onClick={() => setEditing(!editing)}
                className="flex items-center gap-1 text-sm text-blue-400 hover:text-blue-300"
              >
                <Edit size={14} /> {editing ? 'Cancel' : 'Edit'}
              </button>
            </div>

            {editing ? (
              <div className="space-y-4">
                <div>
                  <label className="block text-sm text-gray-400 mb-1">Name</label>
                  <input
                    value={editForm.name}
                    onChange={(e) => setEditForm(f => ({ ...f, name: e.target.value }))}
                    className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm"
                  />
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-1">Description</label>
                  <textarea
                    value={editForm.description}
                    onChange={(e) => setEditForm(f => ({ ...f, description: e.target.value }))}
                    rows={3}
                    className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm resize-none"
                  />
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-1">Game Rules</label>
                  <textarea
                    value={editForm.game_rules}
                    onChange={(e) => setEditForm(f => ({ ...f, game_rules: e.target.value }))}
                    rows={4}
                    className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm resize-none"
                    placeholder="How to play, controls, objectives..."
                  />
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-1">Game Owner Name</label>
                  <input
                    value={editForm.game_owner_name}
                    onChange={(e) => setEditForm(f => ({ ...f, game_owner_name: e.target.value }))}
                    className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm"
                    placeholder="Who created this game?"
                  />
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-sm text-gray-400 mb-1">Drive Link</label>
                    <input
                      type="url"
                      value={editForm.game_drive_link}
                      onChange={(e) => setEditForm(f => ({ ...f, game_drive_link: e.target.value }))}
                      className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm"
                      placeholder="https://drive.google.com/..."
                    />
                  </div>
                  <div>
                    <label className="block text-sm text-gray-400 mb-1">Video Link</label>
                    <input
                      type="url"
                      value={editForm.game_video_link}
                      onChange={(e) => setEditForm(f => ({ ...f, game_video_link: e.target.value }))}
                      className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm"
                      placeholder="https://youtube.com/..."
                    />
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-sm text-gray-400 mb-1">Default FPS</label>
                    <input
                      type="number"
                      value={editForm.default_fps}
                      onChange={(e) => setEditForm(f => ({ ...f, default_fps: e.target.value }))}
                      className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm"
                      min={1} max={60}
                    />
                  </div>
                  <div>
                    <label className="block text-sm text-gray-400 mb-1">Tags (comma-separated)</label>
                    <input
                      value={editForm.tags}
                      onChange={(e) => setEditForm(f => ({ ...f, tags: e.target.value }))}
                      className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm"
                    />
                  </div>
                </div>

                {/* File Updates */}
                <div className="border-t border-gray-700/50 pt-4 mt-4">
                  <p className="text-sm text-gray-400 mb-3 font-medium">Update Game Files <span className="text-gray-600 font-normal">(optional)</span></p>
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="block text-sm text-gray-500 mb-1">game.py</label>
                      <label className={`flex items-center gap-2 px-3 py-2.5 rounded-lg border cursor-pointer transition-colors text-sm ${
                        newGameFile
                          ? 'bg-blue-500/10 border-blue-500/30 text-blue-400'
                          : 'bg-gray-800 border-gray-700 text-gray-400 hover:border-gray-600'
                      }`}>
                        <Code size={14} />
                        <span className="truncate">{newGameFile ? newGameFile.name : 'Choose file...'}</span>
                        <input
                          type="file"
                          accept=".py"
                          className="hidden"
                          onChange={(e) => setNewGameFile(e.target.files[0] || null)}
                        />
                      </label>
                    </div>
                    <div>
                      <label className="block text-sm text-gray-500 mb-1">metadata.json</label>
                      <label className={`flex items-center gap-2 px-3 py-2.5 rounded-lg border cursor-pointer transition-colors text-sm ${
                        newMetadataFile
                          ? 'bg-blue-500/10 border-blue-500/30 text-blue-400'
                          : 'bg-gray-800 border-gray-700 text-gray-400 hover:border-gray-600'
                      }`}>
                        <Settings size={14} />
                        <span className="truncate">{newMetadataFile ? newMetadataFile.name : 'Choose file...'}</span>
                        <input
                          type="file"
                          accept=".json"
                          className="hidden"
                          onChange={(e) => setNewMetadataFile(e.target.files[0] || null)}
                        />
                      </label>
                    </div>
                  </div>
                  {(newGameFile || newMetadataFile) && (
                    <p className="text-[11px] text-yellow-400/70 mt-2">
                      Files will be replaced on save. Make sure the new files are valid.
                    </p>
                  )}
                </div>

                <div className="flex gap-3 pt-2">
                  <button
                    onClick={handleSave}
                    disabled={fileUploading}
                    className="px-5 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-600/50 text-white rounded-lg text-sm font-medium transition-colors flex items-center gap-2"
                  >
                    {fileUploading ? (
                      <>
                        <div className="animate-spin rounded-full h-3.5 w-3.5 border-b-2 border-white" />
                        Uploading...
                      </>
                    ) : (
                      'Save Changes'
                    )}
                  </button>
                  <button
                    onClick={() => { setEditing(false); setNewGameFile(null); setNewMetadataFile(null); }}
                    className="px-5 py-2 bg-gray-800 hover:bg-gray-700 text-gray-300 rounded-lg text-sm transition-colors"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-gray-400 mb-1">Name</p>
                    <p className="text-white text-sm font-medium">{game.name || game.game_id}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-400 mb-1">Game Owner</p>
                    <p className="text-white text-sm">{game.game_owner_name || '--'}</p>
                  </div>
                </div>
                <div>
                  <p className="text-sm text-gray-400 mb-1">Description</p>
                  <p className="text-white text-sm">{game.description || 'No description'}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-400 mb-1">Game Rules</p>
                  <p className="text-white text-sm whitespace-pre-wrap">{game.game_rules || 'No rules specified'}</p>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-gray-400 mb-1">Drive Link</p>
                    {game.game_drive_link ? (
                      <a href={game.game_drive_link} target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:text-blue-300 text-sm underline break-all">{game.game_drive_link}</a>
                    ) : <span className="text-gray-600 text-sm">--</span>}
                  </div>
                  <div>
                    <p className="text-sm text-gray-400 mb-1">Video Link</p>
                    {game.game_video_link ? (
                      <a href={game.game_video_link} target="_blank" rel="noopener noreferrer" className="text-purple-400 hover:text-purple-300 text-sm underline break-all">{game.game_video_link}</a>
                    ) : <span className="text-gray-600 text-sm">--</span>}
                  </div>
                </div>
                <div className="grid grid-cols-4 gap-4">
                  <div>
                    <p className="text-sm text-gray-400 mb-1">Version</p>
                    <p className="text-white text-sm font-mono">{game.version}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-400 mb-1">Game Code</p>
                    <p className="text-white text-sm font-mono">{game.game_code}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-400 mb-1">FPS</p>
                    <p className="text-white text-sm">{game.default_fps}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-400 mb-1">Grid Size</p>
                    <p className="text-white text-sm">{game.grid_max_size}x{game.grid_max_size}</p>
                  </div>
                </div>
                {game.tags && game.tags.length > 0 && (
                  <div>
                    <p className="text-sm text-gray-400 mb-2">Tags</p>
                    <div className="flex flex-wrap gap-2">
                      {game.tags.map(tag => (
                        <span key={tag} className="px-2 py-1 bg-gray-800 rounded text-xs text-gray-300">{tag}</span>
                      ))}
                    </div>
                  </div>
                )}
                <div>
                  <p className="text-sm text-gray-400 mb-1">Created</p>
                  <p className="text-white text-sm">{new Date(game.created_at).toLocaleString()}</p>
                </div>
              </div>
            )}
          </div>

          {/* Stats Card */}
          <div className="space-y-6">
            <div className="bg-gray-900 rounded-xl border border-gray-800 p-6">
              <h2 className="text-lg font-semibold text-white mb-4">Statistics</h2>
              <div className="space-y-4">
                <StatItem label="Total Plays" value={game.total_plays} />
                <StatItem label="Total Wins" value={game.total_wins} />
                <StatItem label="Win Rate" value={game.total_plays > 0 ? `${((game.total_wins / game.total_plays) * 100).toFixed(1)}%` : '0%'} />
                {analytics && (
                  <>
                    <StatItem label="Avg Time to Win" value={formatTime(analytics.avg_time_to_win || 0)} />
                    <StatItem label="Avg Actions to Win" value={analytics.avg_actions_to_win || 0} />
                  </>
                )}
              </div>
            </div>

            {/* Recent sessions mini-list on overview */}
            {sessions.length > 0 && (
              <div className="bg-gray-900 rounded-xl border border-gray-800 p-6">
                <h2 className="text-sm font-semibold text-white mb-3">Recent Plays</h2>
                <div className="space-y-2">
                  {sessions.slice(0, 5).map((s) => (
                    <div key={s.id} className="flex items-center justify-between py-1.5 border-b border-gray-800/50 last:border-0">
                      <div className="flex items-center gap-2">
                        <span className={`w-1.5 h-1.5 rounded-full ${s.state === 'WIN' ? 'bg-green-400' : s.state === 'GAME_OVER' ? 'bg-red-400' : 'bg-blue-400'}`} />
                        <span className="text-xs text-white">{s.player}</span>
                      </div>
                      <div className="flex items-center gap-3 text-[11px] text-gray-400">
                        <span>Lv {s.current_level}</span>
                        <span>{s.total_actions} moves</span>
                        <span className="font-mono">{formatTime(s.total_time)}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {activeTab === 'analytics' && (
        <div className="bg-gray-900 rounded-xl border border-gray-800 p-6">
          <h2 className="text-lg font-semibold text-white mb-4">Analytics (Last 30 Days)</h2>
          {analytics ? (
            <div>
              <div className="grid grid-cols-2 lg:grid-cols-5 gap-4 mb-6">
                <MiniStat label="Plays" value={analytics.total_plays || 0} />
                <MiniStat label="Wins" value={analytics.total_wins || 0} />
                <MiniStat label="Win Rate" value={`${(analytics.win_rate || 0).toFixed(1)}%`} />
                <MiniStat label="Avg Actions" value={(analytics.avg_actions_to_win || 0).toFixed(0)} />
                <MiniStat label="Avg Time" value={formatTime(analytics.avg_time_to_win || 0)} />
              </div>

              {analytics.daily_stats && analytics.daily_stats.length > 0 && (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-gray-800">
                        <th className="text-left py-2 text-gray-400 font-medium">Date</th>
                        <th className="text-right py-2 text-gray-400 font-medium">Plays</th>
                        <th className="text-right py-2 text-gray-400 font-medium">Wins</th>
                        <th className="text-right py-2 text-gray-400 font-medium">Avg Time</th>
                      </tr>
                    </thead>
                    <tbody>
                      {analytics.daily_stats.map((day) => (
                        <tr key={day.date} className="border-b border-gray-800/50">
                          <td className="py-2 text-white">{day.date}</td>
                          <td className="py-2 text-right text-gray-300">{day.plays}</td>
                          <td className="py-2 text-right text-gray-300">{day.wins}</td>
                          <td className="py-2 text-right text-gray-300">{day.avg_time ? formatTime(day.avg_time) : '-'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          ) : (
            <p className="text-gray-400 text-sm">No analytics data available yet</p>
          )}
        </div>
      )}

      {activeTab === 'source' && (
        <div className="bg-gray-900 rounded-xl border border-gray-800 p-6">
          <h2 className="text-lg font-semibold text-white mb-4">Source Code</h2>
          {source ? (
            <div>
              <h3 className="text-sm font-medium text-gray-300 mb-2">{game.game_code}.py</h3>
              <pre className="bg-gray-950 rounded-lg p-4 overflow-x-auto text-sm text-gray-300 font-mono leading-relaxed border border-gray-800 max-h-[600px] overflow-y-auto">
                {source.source_code || 'No source code available'}
              </pre>
              {source.metadata && (
                <>
                  <h3 className="text-sm font-medium text-gray-300 mt-6 mb-2">metadata.json</h3>
                  <pre className="bg-gray-950 rounded-lg p-4 overflow-x-auto text-sm text-gray-300 font-mono border border-gray-800">
                    {JSON.stringify(source.metadata, null, 2)}
                  </pre>
                </>
              )}
            </div>
          ) : (
            <div className="flex justify-center py-8">
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500"></div>
            </div>
          )}
        </div>
      )}

      {activeTab === 'sessions' && (
        <SessionsList
          sessions={sessions}
          formatTime={formatTime}
          gameId={gameId}
          onClear={() => {
            // Refetch sessions + analytics after clearing
            Promise.all([
              analyticsAPI.sessions(gameId, 20).then(r => r.data).catch(() => []),
              analyticsAPI.game(gameId).then(r => r.data).catch(() => null),
              gamesAPI.get(gameId).then(r => r.data),
            ]).then(([s, a, g]) => {
              setSessions(s);
              setAnalytics(a);
              setGame(g);
            });
          }}
        />
      )}

      {activeTab === 'videos' && (
        <div className="bg-gray-900 rounded-xl border border-gray-800 p-6">
          <h2 className="text-lg font-semibold text-white mb-4">Gameplay Recordings</h2>
          {videosLoading ? (
            <div className="flex justify-center py-8">
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500"></div>
            </div>
          ) : videos.length > 0 ? (
            <div className="space-y-3">
              {videos.map((v: any) => (
                <div key={v.filename} className="flex items-center gap-4 p-4 bg-gray-800/50 rounded-lg">
                  {/* Play preview */}
                  <div className="w-48 h-28 bg-black rounded-lg overflow-hidden shrink-0">
                    <video
                      src={`/api/games/${gameId}/videos/${v.filename}`}
                      className="w-full h-full object-contain"
                      preload="metadata"
                      controls
                    />
                  </div>
                  {/* Info */}
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-white truncate">{v.filename}</p>
                    <div className="flex items-center gap-4 mt-1 text-xs text-gray-400">
                      <span>Player: <span className="text-gray-300">{v.player}</span></span>
                      <span>Size: <span className="text-gray-300">{(v.size / (1024 * 1024)).toFixed(1)} MB</span></span>
                      <span>{new Date(v.created_at).toLocaleString()}</span>
                    </div>
                  </div>
                  {/* Actions */}
                  <div className="flex items-center gap-2 shrink-0">
                    <a
                      href={`/api/games/${gameId}/videos/${v.filename}`}
                      download={v.filename}
                      className="p-2 rounded-lg hover:bg-gray-700 text-gray-400 hover:text-blue-400 transition-colors"
                      title="Download"
                    >
                      <Download size={16} />
                    </a>
                    <button
                      onClick={async () => {
                        if (!confirm(`Delete "${v.filename}"?`)) return;
                        try {
                          await gamesAPI.deleteVideo(gameId, v.filename);
                          setVideos(prev => prev.filter(x => x.filename !== v.filename));
                        } catch (err) { console.error(err); }
                      }}
                      className="p-2 rounded-lg hover:bg-gray-700 text-gray-400 hover:text-red-400 transition-colors"
                      title="Delete"
                    >
                      <Trash2 size={16} />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-gray-400 text-sm">No recordings yet. Players can record gameplay from the play page.</p>
          )}
        </div>
      )}
    </div>
  );
}

function StatItem({ label, value }: any) {
  return (
    <div className="flex justify-between items-center py-2 border-b border-gray-800 last:border-0">
      <span className="text-sm text-gray-400">{label}</span>
      <span className="text-sm font-medium text-white">{value}</span>
    </div>
  );
}

function MiniStat({ label, value }: any) {
  return (
    <div className="bg-gray-800/50 rounded-lg p-3 text-center">
      <p className="text-lg font-bold text-white">{value}</p>
      <p className="text-xs text-gray-400">{label}</p>
    </div>
  );
}

function SessionsList({ sessions, formatTime, gameId, onClear }: any) {
  const [expandedId, setExpandedId] = useState(null);
  const [showClearConfirm, setShowClearConfirm] = useState(false);

  // Export state
  const [exportFilter, setExportFilter] = useState('all');
  const [exportDate, setExportDate] = useState('');
  const [exportDateFrom, setExportDateFrom] = useState('');
  const [exportDateTo, setExportDateTo] = useState('');
  const [exporting, setExporting] = useState(false);

  const handleExport = async () => {
    setExporting(true);
    try {
      const params: any = { filter: exportFilter };
      if (exportFilter === 'date') params.date = exportDate;
      if (exportFilter === 'range') { params.date_from = exportDateFrom; params.date_to = exportDateTo; }
      const res = await analyticsAPI.exportExcel(gameId, params);
      const url = window.URL.createObjectURL(new Blob([res.data]));
      const a = document.createElement('a');
      a.href = url;
      a.download = `${gameId}_sessions_${exportFilter}.xlsx`;
      a.click();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      alert(err.response?.data?.detail || 'Failed to export');
    }
    setExporting(false);
  };

  const handleClear = async () => {
    try {
      await gamesAPI.clearSessions(gameId);
      setShowClearConfirm(false);
      if (onClear) onClear();
    } catch (err) {
      alert(err.response?.data?.detail || 'Failed to clear sessions');
      setShowClearConfirm(false);
    }
  };

  const toggle = (id) => setExpandedId(prev => prev === id ? null : id);

  if (sessions.length === 0) {
    return (
      <div className="bg-gray-900 rounded-xl border border-gray-800 p-6">
        <h2 className="text-lg font-semibold text-white mb-4">Sessions</h2>
        <p className="text-gray-400 text-sm">No sessions yet</p>
      </div>
    );
  }

  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-white">Sessions ({sessions.length})</h2>
        <div className="flex items-center gap-2">
          {sessions.length > 0 && (
            <button
              onClick={() => setShowClearConfirm(true)}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-red-600/20 hover:bg-red-600/30 border border-red-500/30 text-red-400 rounded-lg text-xs transition-colors"
            >
              <Trash2 size={12} /> Clear All
            </button>
          )}
        </div>
      </div>

      {/* Export bar */}
      <div className="flex flex-wrap items-center gap-2 mb-4 p-3 bg-gray-800/40 rounded-lg border border-gray-700/50">
        <Download size={14} className="text-gray-400" />
        <span className="text-xs text-gray-400 mr-1">Export:</span>

        <select
          value={exportFilter}
          onChange={(e) => setExportFilter(e.target.value)}
          className="px-2 py-1 bg-gray-800 border border-gray-700 rounded text-xs text-white focus:outline-none focus:border-blue-500/50"
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
            className="px-2 py-1 bg-gray-800 border border-gray-700 rounded text-xs text-white focus:outline-none focus:border-blue-500/50"
          />
        )}

        {exportFilter === 'range' && (
          <>
            <input
              type="date"
              value={exportDateFrom}
              onChange={(e) => setExportDateFrom(e.target.value)}
              className="px-2 py-1 bg-gray-800 border border-gray-700 rounded text-xs text-white focus:outline-none focus:border-blue-500/50"
            />
            <span className="text-xs text-gray-500">to</span>
            <input
              type="date"
              value={exportDateTo}
              onChange={(e) => setExportDateTo(e.target.value)}
              className="px-2 py-1 bg-gray-800 border border-gray-700 rounded text-xs text-white focus:outline-none focus:border-blue-500/50"
            />
          </>
        )}

        <button
          onClick={handleExport}
          disabled={exporting || (exportFilter === 'date' && !exportDate) || (exportFilter === 'range' && (!exportDateFrom || !exportDateTo))}
          className="flex items-center gap-1 px-3 py-1 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded text-xs font-medium transition-colors"
        >
          {exporting ? (
            <><div className="animate-spin rounded-full h-3 w-3 border-b-2 border-white" /> Exporting...</>
          ) : (
            <><Download size={12} /> Download .xlsx</>
          )}
        </button>
      </div>

      <ConfirmModal
        open={showClearConfirm}
        onClose={() => setShowClearConfirm(false)}
        onConfirm={handleClear}
        title="Clear all sessions?"
        message={`This will permanently delete all ${sessions.length} session(s) and analytics data for this game. Play counts will be reset to 0.`}
        confirmText="Clear All"
        variant="danger"
      />

      <div className="space-y-1">
        {sessions.map((s) => {
          const isOpen = expandedId === s.id;
          const levelStats = s.level_stats || [];

          return (
            <div key={s.id} className="rounded-lg overflow-hidden">
              {/* Collapsed row */}
              <button
                onClick={() => toggle(s.id)}
                className={`w-full flex items-center gap-3 px-4 py-3 text-left transition-colors ${
                  isOpen ? 'bg-gray-800' : 'bg-gray-800/40 hover:bg-gray-800/70'
                }`}
              >
                {/* Expand icon */}
                <span className="text-gray-500">
                  {isOpen ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                </span>

                {/* Player */}
                <div className="flex items-center gap-2 min-w-[120px]">
                  <User size={14} className="text-gray-500" />
                  <span className="text-white text-sm font-medium truncate">{s.player}</span>
                </div>

                {/* State badge */}
                <span className={`px-2 py-0.5 rounded text-[11px] font-medium shrink-0 ${
                  s.state === 'WIN' ? 'bg-green-500/20 text-green-400' :
                  s.state === 'GAME_OVER' ? 'bg-red-500/20 text-red-400' :
                  'bg-yellow-500/20 text-yellow-400'
                }`}>
                  {s.state === 'WIN' ? 'WIN' : s.state === 'GAME_OVER' ? 'GAME OVER' : 'IN PROGRESS'}
                </span>

                {/* Quick stats */}
                <div className="flex items-center gap-4 ml-auto text-xs text-gray-400">
                  <span className="flex items-center gap-1 font-mono">
                    <Layers size={12} className="text-blue-400" />
                    Lv {s.state === 'WIN' ? s.current_level : s.current_level + 1}
                  </span>
                  <span className="flex items-center gap-1 font-mono">
                    <Clock size={12} className="text-blue-400" />
                    {formatTime(s.total_time)}
                  </span>
                  <span className="flex items-center gap-1 font-mono">
                    <Skull size={12} className="text-red-400" />
                    {s.game_overs ?? 0}
                  </span>
                  <span className="flex items-center gap-1 font-mono">
                    <Heart size={12} className="text-pink-400" />
                    {s.score != null ? s.score : '--'}
                  </span>
                  <span className="text-gray-600 hidden sm:inline">
                    {new Date(s.started_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}
                  </span>
                </div>
              </button>

              {/* Expanded details */}
              <AnimatePresence>
              {isOpen && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  transition={{ duration: 0.2 }}
                >
                <div className="bg-gray-800/60 border-t border-gray-700/50 px-6 py-4">
                  {/* Summary stats */}
                  <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 mb-4">
                    <div className="bg-gray-900/60 rounded-lg p-3">
                      <p className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">Total Time</p>
                      <p className="text-white font-mono text-sm font-bold">{formatTime(s.total_time)}</p>
                    </div>
                    <div className="bg-gray-900/60 rounded-lg p-3">
                      <p className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">Actions</p>
                      <p className="text-white font-mono text-sm font-bold">{s.total_actions}</p>
                    </div>
                    <div className="bg-gray-900/60 rounded-lg p-3">
                      <p className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">Level Reached</p>
                      <p className="text-white font-mono text-sm font-bold">Lv {s.state === 'WIN' ? s.current_level : s.current_level + 1}</p>
                    </div>
                    <div className="bg-gray-900/60 rounded-lg p-3">
                      <p className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">Game Overs</p>
                      <p className="text-red-400 font-mono text-sm font-bold">{s.game_overs ?? 0}</p>
                    </div>
                    <div className="bg-gray-900/60 rounded-lg p-3">
                      <p className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">Score / Lives</p>
                      <p className="text-white font-mono text-sm font-bold">{s.score != null ? s.score : '--'}</p>
                    </div>
                  </div>

                  {/* Per-level breakdown */}
                  {levelStats.length > 0 ? (
                    <div>
                      <p className="text-[10px] text-gray-500 uppercase tracking-wider mb-2">Per-Level Breakdown</p>
                      <div className="overflow-x-auto">
                        <table className="w-full text-xs">
                          <thead>
                            <tr className="border-b border-gray-700/50">
                              <th className="text-left py-2 px-3 text-gray-500 font-medium">Level</th>
                              <th className="text-right py-2 px-3 text-gray-500 font-medium">Time</th>
                              <th className="text-right py-2 px-3 text-gray-500 font-medium">Moves</th>
                              <th className="text-right py-2 px-3 text-gray-500 font-medium">Deaths</th>
                              <th className="text-right py-2 px-3 text-gray-500 font-medium">Lives</th>
                              <th className="text-right py-2 px-3 text-gray-500 font-medium">Resets</th>
                              <th className="text-right py-2 px-3 text-gray-500 font-medium">Status</th>
                            </tr>
                          </thead>
                          <tbody>
                            {levelStats.map((ls, i) => (
                              <tr key={i} className="border-b border-gray-800/50 last:border-0">
                                <td className="py-2 px-3">
                                  <span className={`font-mono font-medium ${ls.completed ? 'text-green-400' : 'text-gray-300'}`}>
                                    Lv {ls.level + 1}
                                  </span>
                                </td>
                                <td className="py-2 px-3 text-right font-mono text-gray-300">{formatTime(ls.time)}</td>
                                <td className="py-2 px-3 text-right font-mono text-gray-300">{ls.actions}</td>
                                <td className="py-2 px-3 text-right font-mono text-red-400">{ls.game_overs || 0}</td>
                                <td className="py-2 px-3 text-right font-mono text-gray-300">{ls.lives_used || 0}</td>
                                <td className="py-2 px-3 text-right font-mono text-gray-300">{ls.resets || 0}</td>
                                <td className="py-2 px-3 text-right">
                                  {ls.completed ? (
                                    <span className="px-1.5 py-0.5 rounded bg-green-500/15 text-green-400 text-[10px]">Completed</span>
                                  ) : (
                                    <span className="px-1.5 py-0.5 rounded bg-gray-700 text-gray-400 text-[10px]">Incomplete</span>
                                  )}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  ) : (
                    <p className="text-gray-500 text-xs">No level data recorded for this session</p>
                  )}

                   {/* Timestamps */}
                  <div className="flex items-center gap-4 mt-3 pt-3 border-t border-gray-700/50 text-[11px] text-gray-500">
                    <span>Started: {new Date(s.started_at).toLocaleString()}</span>
                    {s.ended_at && <span>Ended: {new Date(s.ended_at).toLocaleString()}</span>}
                  </div>
                </div>
                </motion.div>
              )}
              </AnimatePresence>
            </div>
          );
        })}
      </div>
    </div>
  );
}
