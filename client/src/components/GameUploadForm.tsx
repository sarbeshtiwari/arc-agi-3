import React, { useState, useRef } from 'react';
import { FileCode, FileJson, Upload, AlertCircle, CheckCircle, X } from 'lucide-react';

function DropZone({ file, setFile, fileRef, accept, icon: Icon, label, hint }) {
  const [dragOver, setDragOver] = useState(false);
  return (
    <div
      className={`relative border-2 border-dashed rounded-xl p-4 text-center cursor-pointer transition-all ${
        file ? 'border-blue-500/40 bg-blue-500/5' :
        dragOver ? 'border-blue-400/50 bg-blue-500/5' :
        'border-gray-300 dark:border-gray-700 hover:border-gray-400 dark:hover:border-gray-600 bg-gray-100/30 dark:bg-gray-800/30'
      }`}
      onClick={() => fileRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => { e.preventDefault(); setDragOver(false); const f = e.dataTransfer.files[0]; if (f) setFile(f); }}
    >
      <input ref={fileRef} type="file" accept={accept} className="hidden" onChange={(e) => setFile(e.target.files[0] || null)} />
      <Icon size={20} className={file ? 'text-blue-400 mx-auto mb-1.5' : 'text-gray-500 mx-auto mb-1.5'} />
      {file ? (
        <>
          <p className="text-xs text-blue-400 font-medium truncate">{file.name}</p>
           <button onClick={(e) => { e.stopPropagation(); setFile(null); }} className="absolute top-2 right-2 p-0.5 rounded-full bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 text-gray-500 dark:text-gray-400">
            <X size={10} />
          </button>
        </>
      ) : (
        <>
           <p className="text-xs text-gray-500 dark:text-gray-400 font-medium">{label}</p>
           <p className="text-[10px] text-gray-400 dark:text-gray-600">{hint}</p>
        </>
      )}
    </div>
  );
}

function validatePyFile(content) {
  let hasComments = false;
  let hasDocstrings = false;
  const lines = content.split('\n');
  for (let i = 0; i < lines.length; i++) {
    const trimmed = lines[i].trim();
    if (i === 0 && trimmed.startsWith('#!')) continue;
    const hashIdx = trimmed.indexOf('#');
    if (hashIdx >= 0 && !hasComments) {
      let inStr = false;
      let strChar = null;
      for (let j = 0; j < hashIdx; j++) {
        if ((trimmed[j] === '"' || trimmed[j] === "'") && (j === 0 || trimmed[j - 1] !== '\\')) {
          if (!inStr) { inStr = true; strChar = trimmed[j]; }
          else if (trimmed[j] === strChar) { inStr = false; }
        }
      }
      if (!inStr) hasComments = true;
    }
    if ((trimmed.startsWith('"""') || trimmed.startsWith("'''")) && !hasDocstrings) {
      hasDocstrings = true;
    }
  }
  const errors = [];
  if (hasComments) errors.push('Game file contains comments (#). Please remove all comments.');
  if (hasDocstrings) errors.push('Game file contains docstrings ("""). Please remove all docstrings.');
  return errors;
  return errors;
}

function validateMetadata(content) {
  const errors = [];
  let metadata;
  try {
    metadata = JSON.parse(content);
  } catch (e) {
    return ['Invalid JSON format'];
  }
  if (typeof metadata !== 'object' || Array.isArray(metadata)) return ['Must be a JSON object'];

  const allowed = new Set(['game_id', 'default_fps', 'baseline_actions', 'tags', 'local_dir', 'total_levels', 'available_actions', 'levels']);
  const unknown = Object.keys(metadata).filter(k => !allowed.has(k));
  if (unknown.length > 0) errors.push(`Unknown fields: ${unknown.join(', ')}`);
  if (!metadata.game_id) errors.push('Missing game_id');
  if (metadata.default_fps !== undefined && (typeof metadata.default_fps !== 'number' || metadata.default_fps < 1)) errors.push('default_fps must be a positive number');
  if (metadata.baseline_actions !== undefined && (!Array.isArray(metadata.baseline_actions) || !metadata.baseline_actions.every(x => typeof x === 'number'))) errors.push('baseline_actions must be array of numbers');
  if (metadata.tags !== undefined && (!Array.isArray(metadata.tags) || !metadata.tags.every(x => typeof x === 'string'))) errors.push('tags must be array of strings');

  return errors;
}

export default function GameUploadForm({
  mode = 'admin',
  onSubmit,
  submitting = false,
  successMessage = '',
  errorMessage = '',
}) {
  const [gameName, setGameName] = useState('');
  const [description, setDescription] = useState('');
  const [gameRules, setGameRules] = useState('');
  const [ownerName, setOwnerName] = useState('');
  const [driveLink, setDriveLink] = useState('');
  const [videoLink, setVideoLink] = useState('');

  // Request-only fields
  const [requesterName, setRequesterName] = useState('');
  const [requesterEmail, setRequesterEmail] = useState('');
  const [message, setMessage] = useState('');

  const [gameFile, setGameFile] = useState(null);
  const [metadataFile, setMetadataFile] = useState(null);
  const gameFileRef = useRef(null);
  const metadataFileRef = useRef(null);

  const [validationErrors, setValidationErrors] = useState([]);

  const isRequest = mode === 'request';

  const runValidation = async () => {
    const errors = [];
    if (gameFile) {
      const text = await gameFile.text();
      const pyErrors = validatePyFile(text);
      errors.push(...pyErrors.map(e => `game.py: ${e}`));
    }
    if (metadataFile) {
      const text = await metadataFile.text();
      const metaErrors = validateMetadata(text);
      errors.push(...metaErrors.map(e => `metadata.json: ${e}`));
    }
    return errors;
  };

  const handleSubmit = async () => {
    setValidationErrors([]);
    const errors = await runValidation();
    if (errors.length > 0) {
      setValidationErrors(errors);
      return;
    }

    const formData = new FormData();
    formData.append('game_file', gameFile);
    formData.append('metadata_file', metadataFile);

    if (isRequest) {
      formData.append('requester_name', requesterName.trim());
      formData.append('requester_email', requesterEmail.trim());
      formData.append('message', message.trim());
    } else {
      formData.append('name', gameName.trim());
    }

    formData.append('description', description.trim());
    formData.append('game_rules', gameRules.trim());
    formData.append('game_owner_name', ownerName.trim());
    formData.append('game_drive_link', driveLink.trim());
    formData.append('game_video_link', videoLink.trim());

    const success = await onSubmit(formData);
    if (success) {
      setGameName(''); setDescription(''); setGameRules('');
      setOwnerName(''); setDriveLink(''); setVideoLink('');
      setRequesterName(''); setRequesterEmail(''); setMessage('');
      setGameFile(null); setMetadataFile(null);
      setValidationErrors([]);
    }
  };

  const canSubmit = gameFile && metadataFile && !submitting
    && (isRequest ? requesterName.trim() && ownerName.trim() && description.trim() && gameRules.trim() : true);

  return (
    <div className="max-w-8xl mx-auto">
      {/* Messages */}
      {successMessage && (
        <div className="mb-6 flex items-center gap-2 p-4 bg-green-500/10 border border-green-500/30 rounded-xl text-green-400 text-sm">
          <CheckCircle size={16} /> {successMessage}
        </div>
      )}
      {errorMessage && (
        <div className="mb-6 p-4 bg-red-500/10 border border-red-500/30 rounded-xl text-red-400 text-sm">{errorMessage}</div>
      )}
      {validationErrors.length > 0 && (
        <div className="mb-6 p-4 bg-red-500/10 border border-red-500/30 rounded-xl">
          <div className="flex items-center gap-2 mb-2">
            <AlertCircle size={14} className="text-red-400" />
            <p className="text-sm font-medium text-red-400">Validation Failed</p>
          </div>
          <ul className="space-y-0.5">
            {validationErrors.map((e, i) => (
              <li key={i} className="text-xs text-red-400/80">{e}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Left Column */}
        <div className="space-y-4">
          {/* Section 1: About */}
          <div className="bg-white dark:bg-gray-900 rounded-2xl border border-gray-200 dark:border-gray-800 p-5">
            <div className="flex items-center gap-2 mb-4">
              <div className="w-5 h-5 rounded-full bg-blue-600 flex items-center justify-center text-white text-[10px] font-bold">1</div>
              <h3 className="text-xs font-semibold text-gray-900 dark:text-white uppercase tracking-wider">{isRequest ? 'About You' : 'Game Info'}</h3>
            </div>
            <div className="grid grid-cols-2 gap-3">
              {isRequest ? (
                <>
                  <div>
                    <label className="block text-[11px] text-gray-400 dark:text-gray-500 mb-1">Your Name <span className="text-red-400">*</span></label>
                    <input type="text" value={requesterName} onChange={(e) => setRequesterName(e.target.value)}
                      placeholder="John Doe" maxLength={100} required
                      className="w-full px-3 py-2 bg-gray-100 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-gray-600 focus:outline-none focus:border-blue-500/50 text-sm" />
                  </div>
                  <div>
                    <label className="block text-[11px] text-gray-400 dark:text-gray-500 mb-1">Email <span className="text-gray-400 dark:text-gray-600">(optional)</span></label>
                    <input type="email" value={requesterEmail} onChange={(e) => setRequesterEmail(e.target.value)}
                      placeholder="you@example.com" maxLength={200}
                      className="w-full px-3 py-2 bg-gray-100 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-gray-600 focus:outline-none focus:border-blue-500/50 text-sm" />
                  </div>
                </>
              ) : (
                <div className="col-span-2">
                  <label className="block text-[11px] text-gray-400 dark:text-gray-500 mb-1">Game Name</label>
                  <input type="text" value={gameName} onChange={(e) => setGameName(e.target.value)}
                    placeholder="Human-readable game name"
                    className="w-full px-3 py-2 bg-gray-100 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-gray-600 focus:outline-none focus:border-blue-500/50 text-sm" />
                </div>
              )}
              <div className="col-span-2">
                <label className="block text-[11px] text-gray-400 dark:text-gray-500 mb-1">Game Owner / Creator {isRequest && <span className="text-red-400">*</span>}</label>
                <input type="text" value={ownerName} onChange={(e) => setOwnerName(e.target.value)}
                  placeholder="Who built this game?" maxLength={100}
                  className="w-full px-3 py-2 bg-gray-100 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-gray-600 focus:outline-none focus:border-blue-500/50 text-sm" />
              </div>
            </div>
          </div>

          {/* Section 3: Links */}
          <div className="bg-white dark:bg-gray-900 rounded-2xl border border-gray-200 dark:border-gray-800 p-5">
            <div className="flex items-center gap-2 mb-4">
              <div className="w-5 h-5 rounded-full bg-blue-600 flex items-center justify-center text-white text-[10px] font-bold">3</div>
              <h3 className="text-xs font-semibold text-gray-900 dark:text-white uppercase tracking-wider">Links <span className="text-gray-400 dark:text-gray-600 normal-case font-normal">(optional)</span></h3>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-[11px] text-gray-400 dark:text-gray-500 mb-1">Drive / Download</label>
                <input type="url" value={driveLink} onChange={(e) => setDriveLink(e.target.value)}
                  placeholder="https://drive.google.com/..."
                  className="w-full px-3 py-2 bg-gray-100 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-gray-600 focus:outline-none focus:border-blue-500/50 text-sm" />
              </div>
              <div>
                <label className="block text-[11px] text-gray-400 dark:text-gray-500 mb-1">Video Demo</label>
                <input type="url" value={videoLink} onChange={(e) => setVideoLink(e.target.value)}
                  placeholder="https://youtube.com/..."
                  className="w-full px-3 py-2 bg-gray-100 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-gray-600 focus:outline-none focus:border-blue-500/50 text-sm" />
              </div>
            </div>
          </div>

          {/* Section 4: Upload Files */}
          <div className="bg-white dark:bg-gray-900 rounded-2xl border border-gray-200 dark:border-gray-800 p-5">
            <div className="flex items-center gap-2 mb-4">
              <div className="w-5 h-5 rounded-full bg-blue-600 flex items-center justify-center text-white text-[10px] font-bold">4</div>
              <h3 className="text-xs font-semibold text-gray-900 dark:text-white uppercase tracking-wider">Upload Files</h3>
            </div>
            <div className="grid grid-cols-2 gap-3 mb-3">
              <DropZone file={gameFile} setFile={setGameFile} fileRef={gameFileRef} accept=".py" icon={FileCode} label="game.py" hint="No comments/docstrings" />
              <DropZone file={metadataFile} setFile={setMetadataFile} fileRef={metadataFileRef} accept=".json" icon={FileJson} label="metadata.json" hint="Strict schema only" />
            </div>
            <div className="p-2.5 bg-gray-100/40 dark:bg-gray-800/40 rounded-lg text-[10px] text-gray-400 dark:text-gray-500">
              <strong className="text-gray-500 dark:text-gray-400">game.py</strong> -- No comments (#) or docstrings (""") allowed<br />
              <strong className="text-gray-500 dark:text-gray-400">metadata.json</strong> -- Only: game_id, default_fps, baseline_actions, tags, local_dir, total_levels, available_actions, levels
            </div>
          </div>
        </div>

        {/* Right Column */}
        <div className="space-y-4">
          {/* Section 2: Game Details */}
          <div className="bg-white dark:bg-gray-900 rounded-2xl border border-gray-200 dark:border-gray-800 p-5">
            <div className="flex items-center gap-2 mb-4">
              <div className="w-5 h-5 rounded-full bg-blue-600 flex items-center justify-center text-white text-[10px] font-bold">2</div>
              <h3 className="text-xs font-semibold text-gray-900 dark:text-white uppercase tracking-wider">Game Details</h3>
            </div>
            <div className="space-y-3">
              <div>
                <label className="block text-[11px] text-gray-400 dark:text-gray-500 mb-1">Description {isRequest && <span className="text-red-400">*</span>}</label>
                <textarea value={description} onChange={(e) => setDescription(e.target.value)}
                  placeholder="What is the game about? Core mechanics?" rows={3} maxLength={1000}
                  className="w-full px-3 py-2 bg-gray-100 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-gray-600 focus:outline-none focus:border-blue-500/50 text-sm resize-none" />
              </div>
              <div>
                <label className="block text-[11px] text-gray-400 dark:text-gray-500 mb-1">Game Rules / How to Play {isRequest && <span className="text-red-400">*</span>}</label>
                <textarea value={gameRules} onChange={(e) => setGameRules(e.target.value)}
                  placeholder="Controls, objectives, win/lose conditions..." rows={4} maxLength={2000}
                  className="w-full px-3 py-2 bg-gray-100 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-gray-600 focus:outline-none focus:border-blue-500/50 text-sm resize-none" />
              </div>
              {isRequest && (
                <div>
                  <label className="block text-[11px] text-gray-400 dark:text-gray-500 mb-1">Message to Admin <span className="text-gray-400 dark:text-gray-600">(optional)</span></label>
                  <textarea value={message} onChange={(e) => setMessage(e.target.value)}
                    placeholder="Extra context, known issues, notes..." rows={3} maxLength={1000}
                    className="w-full px-3 py-2 bg-gray-100 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-gray-600 focus:outline-none focus:border-blue-500/50 text-sm resize-none" />
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Submit */}
      <button
        onClick={handleSubmit}
        disabled={!canSubmit}
        className="w-full mt-4 px-4 py-3.5 bg-emerald-600 hover:bg-emerald-700 disabled:bg-gray-100 dark:disabled:bg-gray-800 disabled:text-gray-400 dark:disabled:text-gray-600 disabled:cursor-not-allowed text-white font-semibold rounded-xl text-sm transition-colors flex items-center justify-center gap-2"
      >
        {submitting ? (
          <><div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white" /> {isRequest ? 'Submitting...' : 'Uploading...'}</>
        ) : (
          <><Upload size={16} /> {isRequest ? 'Submit for Review' : 'Upload Game'}</>
        )}
      </button>
    </div>
  );
}
