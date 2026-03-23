import React, { useState } from 'react';
import { gamesAPI } from '../api/client';
import GameUploadForm from '../components/GameUploadForm';

export default function GameUploadPage() {
  const [submitting, setSubmitting] = useState(false);
  const [success, setSuccess] = useState('');
  const [error, setError] = useState('');

  const handleUpload = async (formData) => {
    setSubmitting(true);
    setSuccess('');
    setError('');
    try {
      await gamesAPI.upload(formData);
      setSuccess('Game uploaded successfully (inactive). Activate it from the Games page.');
      return true;
    } catch (err) {
      setError(err.response?.data?.detail || 'Upload failed');
      return false;
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-white">Upload Game</h1>
        <p className="text-gray-400 mt-1">Upload a new game directly to the platform</p>
      </div>
      <GameUploadForm
        mode="admin"
        onSubmit={handleUpload}
        submitting={submitting}
        successMessage={success}
        errorMessage={error}
      />
    </div>
  );
}
