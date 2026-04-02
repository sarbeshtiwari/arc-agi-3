import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AlertTriangle, CheckCircle, Trash2, X } from 'lucide-react';

const VARIANTS = {
  danger: {
    icon: AlertTriangle,
    iconColor: 'text-red-400',
    iconBg: 'bg-red-500/10 border-red-500/30',
    confirmBtn: 'bg-red-600 hover:bg-red-700',
  },
  warning: {
    icon: AlertTriangle,
    iconColor: 'text-yellow-400',
    iconBg: 'bg-yellow-500/10 border-yellow-500/30',
    confirmBtn: 'bg-yellow-600 hover:bg-yellow-700',
  },
  success: {
    icon: CheckCircle,
    iconColor: 'text-green-400',
    iconBg: 'bg-green-500/10 border-green-500/30',
    confirmBtn: 'bg-green-600 hover:bg-green-700',
  },
};

export default function ConfirmModal({
  open,
  onClose,
  onConfirm,
  title = 'Are you sure?',
  message = '',
  confirmText = 'Confirm',
  cancelText = 'Cancel',
  variant = 'danger',
  loading = false,
}) {
  if (!open) return null;

  const v = VARIANTS[variant] || VARIANTS.danger;
  const Icon = v.icon;

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.15 }}
        >
          <motion.div
            className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-xl w-full max-w-sm shadow-2xl animate-in"
            onClick={(e) => e.stopPropagation()}
            initial={{ opacity: 0, scale: 0.95, y: 10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.2, type: 'spring', stiffness: 300, damping: 25 }}
          >
        {/* Header */}
        <div className="flex items-start gap-4 p-5 pb-3">
          <div className={`w-10 h-10 rounded-full ${v.iconBg} border flex items-center justify-center shrink-0`}>
            <Icon size={20} className={v.iconColor} />
          </div>
          <div className="flex-1 min-w-0">
            <h3 className="text-gray-900 dark:text-white font-semibold text-base">{title}</h3>
            {message && (
              <p className="text-gray-500 dark:text-gray-400 text-sm mt-1 leading-relaxed">{message}</p>
            )}
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-md text-gray-400 dark:text-gray-500 hover:text-gray-600 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors shrink-0"
          >
            <X size={16} />
          </button>
        </div>

        {/* Actions */}
        <div className="flex gap-2 p-5 pt-3">
          <button
            onClick={onClose}
            disabled={loading}
            className="flex-1 px-4 py-2.5 bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 disabled:opacity-50 text-gray-700 dark:text-gray-300 rounded-lg text-sm font-medium transition-colors"
          >
            {cancelText}
          </button>
          <button
            onClick={onConfirm}
            disabled={loading}
            className={`flex-1 px-4 py-2.5 ${v.confirmBtn} disabled:opacity-50 text-white rounded-lg text-sm font-medium transition-colors`}
          >
            {loading ? 'Please wait...' : confirmText}
          </button>
        </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
