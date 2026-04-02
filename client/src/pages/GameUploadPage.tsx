import React, { useState, useCallback, useRef } from 'react';
import { gamesAPI } from '../api/client';
import GameUploadForm from '../components/GameUploadForm';
import { Upload, Package, CheckCircle, XCircle, SkipForward, FileArchive, X } from 'lucide-react';

type Tab = 'single' | 'bulk';

interface BulkResult {
  game_id: string;
  status: 'success' | 'skipped' | 'error';
  message: string;
}

interface BulkResponse {
  total: number;
  success: number;
  skipped: number;
  errors: number;
  results: BulkResult[];
}

export default function GameUploadPage() {
  const [tab, setTab] = useState<Tab>('single');

  const [submitting, setSubmitting] = useState(false);
  const [success, setSuccess] = useState('');
  const [error, setError] = useState('');

  const [zipFile, setZipFile] = useState<File | null>(null);
  const [bulkSubmitting, setBulkSubmitting] = useState(false);
  const [bulkError, setBulkError] = useState('');
  const [bulkResults, setBulkResults] = useState<BulkResponse | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleUpload = async (formData) => {
    setSubmitting(true);
    setSuccess('');
    setError('');
    try {
      await gamesAPI.upload(formData);
      setSuccess('Game uploaded successfully (inactive). Activate it from the Games page.');
      return true;
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Upload failed');
      return false;
    } finally {
      setSubmitting(false);
    }
  };

  const handleZipSelect = useCallback((file: File) => {
    if (!file.name.toLowerCase().endsWith('.zip')) {
      setBulkError('Please select a .zip file');
      return;
    }
    if (file.size > 100 * 1024 * 1024) {
      setBulkError('File size must be under 100MB');
      return;
    }
    setZipFile(file);
    setBulkError('');
    setBulkResults(null);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleZipSelect(file);
  }, [handleZipSelect]);

  const handleBulkUpload = async () => {
    if (!zipFile) return;
    setBulkSubmitting(true);
    setBulkError('');
    setBulkResults(null);
    try {
      const formData = new FormData();
      formData.append('zip_file', zipFile);
      const res = await gamesAPI.bulkUpload(formData);
      setBulkResults(res.data);
      setZipFile(null);
    } catch (err: any) {
      setBulkError(err.response?.data?.detail || 'Bulk upload failed');
    } finally {
      setBulkSubmitting(false);
    }
  };

  const tabs: { key: Tab; label: string; icon: React.ReactNode }[] = [
    { key: 'single', label: 'Single Game', icon: <Upload size={16} /> },
    { key: 'bulk', label: 'Bulk Upload', icon: <Package size={16} /> },
  ];

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Upload Game</h1>
        <p className="text-gray-500 dark:text-gray-400 mt-1">Upload games to the platform</p>
      </div>

      <div className="flex gap-1 p-1 mb-6 bg-gray-100 dark:bg-gray-800 rounded-lg w-fit">
        {tabs.map(t => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className={`flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-md transition-colors ${
              tab === t.key
                ? 'bg-white dark:bg-gray-900 text-gray-900 dark:text-white shadow-sm'
                : 'text-gray-500 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
            }`}
          >
            {t.icon}
            {t.label}
          </button>
        ))}
      </div>

      {tab === 'single' ? (
        <GameUploadForm
          mode="admin"
          onSubmit={handleUpload}
          submitting={submitting}
          successMessage={success}
          errorMessage={error}
        />
      ) : (
        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-xl p-6">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">Bulk Upload via ZIP</h2>
            <p className="text-gray-500 dark:text-gray-400 text-sm mb-4">
              Upload a ZIP file containing multiple game folders. Each folder must contain:
            </p>
            <div className="bg-gray-50 dark:bg-gray-950 rounded-lg p-4 font-mono text-sm text-gray-700 dark:text-gray-300">
              <div className="text-gray-400 dark:text-gray-500 mb-1">games.zip</div>
              <div className="pl-4">├── game_folder_1/</div>
              <div className="pl-8">├── game.py</div>
              <div className="pl-8">└── metadata.json</div>
              <div className="pl-4">├── game_folder_2/</div>
              <div className="pl-8">├── game.py</div>
              <div className="pl-8">└── metadata.json</div>
              <div className="pl-4">└── ...</div>
            </div>
            <p className="text-gray-400 dark:text-gray-500 text-xs mt-3">
              Games are uploaded as inactive. Existing games (by game_id) are skipped. Max 100MB.
            </p>
          </div>

          <div
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-colors ${
              dragOver
                ? 'border-blue-500 bg-blue-50/50 dark:bg-blue-500/10'
                : zipFile
                  ? 'border-green-500/50 bg-green-50/30 dark:bg-green-500/5'
                  : 'border-gray-300 dark:border-gray-700 hover:border-gray-400 dark:hover:border-gray-600 bg-gray-50/50 dark:bg-gray-800/30'
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".zip"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) handleZipSelect(file);
                e.target.value = '';
              }}
            />
            {zipFile ? (
              <div className="flex flex-col items-center gap-2">
                <FileArchive size={40} className="text-green-500" />
                <div className="flex items-center gap-2">
                  <span className="text-gray-900 dark:text-white font-medium">{zipFile.name}</span>
                  <span className="text-gray-400 dark:text-gray-500 text-sm">
                    ({(zipFile.size / 1024 / 1024).toFixed(1)} MB)
                  </span>
                  <button
                    onClick={(e) => { e.stopPropagation(); setZipFile(null); }}
                    className="ml-1 text-gray-400 dark:text-gray-500 hover:text-red-500"
                  >
                    <X size={16} />
                  </button>
                </div>
                <span className="text-gray-400 dark:text-gray-500 text-sm">Click or drop to replace</span>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-2">
                <Package size={40} className="text-gray-400 dark:text-gray-600" />
                <span className="text-gray-500 dark:text-gray-400 font-medium">Drop ZIP file here or click to browse</span>
                <span className="text-gray-400 dark:text-gray-500 text-xs">Only .zip files up to 100MB</span>
              </div>
            )}
          </div>

          {bulkError && (
            <div className="bg-red-500/10 border border-red-500/20 text-red-600 dark:text-red-400 rounded-lg px-4 py-3 text-sm">
              {bulkError}
            </div>
          )}

          <button
            onClick={handleBulkUpload}
            disabled={!zipFile || bulkSubmitting}
            className="flex items-center gap-2 px-6 py-2.5 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {bulkSubmitting ? (
              <>
                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                Processing ZIP...
              </>
            ) : (
              <>
                <Upload size={18} />
                Upload ZIP
              </>
            )}
          </button>

          {bulkResults && (
            <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-xl p-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Upload Results</h3>

              <div className="grid grid-cols-4 gap-4 mb-6">
                <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-3 text-center">
                  <div className="text-2xl font-bold text-gray-900 dark:text-white">{bulkResults.total}</div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">Total Found</div>
                </div>
                <div className="bg-green-50 dark:bg-green-500/10 rounded-lg p-3 text-center">
                  <div className="text-2xl font-bold text-green-600 dark:text-green-400">{bulkResults.success}</div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">Uploaded</div>
                </div>
                <div className="bg-yellow-50 dark:bg-yellow-500/10 rounded-lg p-3 text-center">
                  <div className="text-2xl font-bold text-yellow-600 dark:text-yellow-400">{bulkResults.skipped}</div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">Skipped</div>
                </div>
                <div className="bg-red-50 dark:bg-red-500/10 rounded-lg p-3 text-center">
                  <div className="text-2xl font-bold text-red-600 dark:text-red-400">{bulkResults.errors}</div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">Errors</div>
                </div>
              </div>

              <div className="border border-gray-200 dark:border-gray-800 rounded-lg divide-y divide-gray-200 dark:divide-gray-800">
                {bulkResults.results.map((r, i) => (
                  <div key={i} className="flex items-center gap-3 px-4 py-2.5 text-sm">
                    {r.status === 'success' && <CheckCircle size={16} className="text-green-500 shrink-0" />}
                    {r.status === 'skipped' && <SkipForward size={16} className="text-yellow-500 shrink-0" />}
                    {r.status === 'error' && <XCircle size={16} className="text-red-500 shrink-0" />}
                    <span className="font-mono text-gray-900 dark:text-white">{r.game_id}</span>
                    <span className="text-gray-400 dark:text-gray-500">—</span>
                    <span className={
                      r.status === 'success' ? 'text-green-600 dark:text-green-400' :
                      r.status === 'skipped' ? 'text-yellow-600 dark:text-yellow-400' :
                      'text-red-600 dark:text-red-400'
                    }>
                      {r.message}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
