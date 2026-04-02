import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { useAuth } from '../hooks/useAuth';
import { User, Lock, Eye, EyeOff, ArrowRight, ShieldCheck, Zap, BarChart3, Gamepad2, Users, Sun, Moon } from 'lucide-react';
import { useTheme } from '../hooks/useTheme';

export default function LoginPage() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [shake, setShake] = useState(false);
  const [focused, setFocused] = useState(null);
  const [success, setSuccess] = useState(false);
  const { login, user } = useAuth();
  const navigate = useNavigate();
  const usernameRef = useRef(null);
  const { theme, toggleTheme } = useTheme();

  useEffect(() => {
    if (user) navigate('/admin', { replace: true });
  }, [user, navigate]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      await login(username, password);
      setSuccess(true);
      setTimeout(() => navigate('/admin', { replace: true }), 500);
    } catch (err) {
      setError(err.response?.data?.detail || 'Invalid username or password');
      setShake(true);
      setTimeout(() => setShake(false), 600);
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-[#0a0a0f] flex">
      {/* Theme toggle */}
      <button
        onClick={toggleTheme}
        className="fixed top-4 right-4 z-50 w-9 h-9 rounded-lg border border-gray-200 dark:border-gray-700 bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm flex items-center justify-center text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
        title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
      >
        {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
      </button>

      {/* Left panel */}
      <div className="hidden lg:flex lg:w-[55%] relative overflow-hidden">
        {/* Animated background */}
        <div className="absolute inset-0">
          <div className="absolute inset-0 bg-gradient-to-br from-blue-600/10 via-transparent to-purple-600/5 dark:from-blue-600/10 dark:via-transparent dark:to-purple-600/5" />
          <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-blue-500/5 rounded-full blur-[100px] animate-float" />
          <div className="absolute bottom-1/3 right-1/4 w-80 h-80 bg-purple-500/5 rounded-full blur-[100px] animate-float-delayed" />
        </div>

        {/* Grid */}
        <div className="absolute inset-0 opacity-[0.025]" style={{
          backgroundImage: 'linear-gradient(rgba(255,255,255,0.15) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.15) 1px, transparent 1px)',
          backgroundSize: '50px 50px',
        }} />

        <motion.div className="relative z-10 flex flex-col justify-between p-14 w-full"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6 }}
        >
          {/* Top */}
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-blue-600 flex items-center justify-center shadow-lg shadow-blue-600/25">
              <span className="text-white text-sm font-black">A</span>
            </div>
            <span className="text-gray-500 dark:text-white/60 text-sm font-medium">ARC-AGI Platform</span>
          </div>

          {/* Center */}
          <div>
            <h1 className="text-[3.5rem] font-black text-gray-900 dark:text-white leading-[1.1] mb-5 tracking-tight">
              Manage your<br />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-cyan-400 to-purple-400">
                game universe
              </span>
            </h1>
            <p className="text-gray-500 dark:text-gray-400 text-lg leading-relaxed max-w-lg">
              Upload, test, and deploy grid-based puzzle games. Track player performance with real-time analytics.
            </p>

            {/* Feature cards */}
            <div className="grid grid-cols-2 gap-3 mt-10 max-w-md">
              {[
                { icon: Gamepad2, label: 'Game Library', desc: 'Upload & manage', color: 'blue' },
                { icon: BarChart3, label: 'Analytics', desc: 'Track sessions', color: 'green' },
                { icon: Zap, label: 'Live Testing', desc: 'Play instantly', color: 'yellow' },
                { icon: Users, label: 'Team Access', desc: 'Role-based control', color: 'purple' },
              ].map(({ icon: Icon, label, desc, color }) => (
                <div key={label} className="flex items-center gap-3 p-3 bg-gray-100 dark:bg-white/[0.03] border border-gray-200 dark:border-white/[0.05] rounded-xl hover:bg-gray-200 dark:hover:bg-white/[0.05] transition-colors">
                  <div className={`w-9 h-9 rounded-lg flex items-center justify-center shrink-0 ${
                    color === 'blue' ? 'bg-blue-500/10 text-blue-400' :
                    color === 'green' ? 'bg-green-500/10 text-green-400' :
                    color === 'yellow' ? 'bg-yellow-500/10 text-yellow-400' :
                    'bg-purple-500/10 text-purple-400'
                  }`}>
                    <Icon size={18} />
                  </div>
                  <div>
                    <p className="text-gray-900 dark:text-white text-xs font-semibold">{label}</p>
                    <p className="text-gray-500 dark:text-gray-500 text-[10px]">{desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Bottom */}
          <p className="text-gray-400 dark:text-gray-600 text-xs">Internal Admin Panel</p>
        </motion.div>
      </div>

      {/* Right panel */}
      <div className="flex-1 flex items-center justify-center p-6 relative">
        {/* Subtle glow */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] bg-blue-500/[0.02] rounded-full blur-[80px]" />

        <motion.div className="w-full max-w-[360px] relative z-10"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, ease: 'easeOut' }}
        >
          {/* Mobile logo */}
          <div className="lg:hidden text-center mb-10">
            <div className="w-12 h-12 rounded-xl bg-blue-600 flex items-center justify-center mx-auto mb-3 shadow-lg shadow-blue-600/20">
              <span className="text-white text-lg font-black">A</span>
            </div>
            <h1 className="text-lg font-bold text-gray-900 dark:text-white">ARC-AGI Internal</h1>
          </div>

          {/* Header */}
          <div className="mb-8">
            <h2 className="text-[26px] font-bold text-gray-900 dark:text-white tracking-tight">Welcome back</h2>
            <p className="text-gray-400 dark:text-gray-500 mt-1.5 text-sm">Enter your credentials to continue</p>
          </div>

          {/* Success state */}
          {success && (
            <div className="flex items-center gap-3 bg-green-500/10 border border-green-500/20 rounded-xl p-4 mb-6 animate-fadeIn">
              <div className="w-8 h-8 rounded-full bg-green-500/15 flex items-center justify-center shrink-0">
                <ShieldCheck size={16} className="text-green-400" />
              </div>
              <div>
                <p className="text-sm font-medium text-green-400">Welcome!</p>
                <p className="text-xs text-green-400/60">Redirecting to dashboard...</p>
              </div>
            </div>
          )}

          {/* Error */}
          {error && !success && (
            <div className={`flex items-center gap-3 bg-red-500/8 border border-red-500/15 rounded-xl p-4 mb-6 ${shake ? 'animate-shake' : ''}`}>
              <div className="w-8 h-8 rounded-full bg-red-500/10 flex items-center justify-center shrink-0">
                <Lock size={14} className="text-red-400" />
              </div>
              <div>
                <p className="text-sm font-medium text-red-400">Authentication failed</p>
                <p className="text-xs text-red-400/60">{error}</p>
              </div>
            </div>
          )}

          {!success && (
            <form onSubmit={handleSubmit} className="space-y-5">
              {/* Username */}
              <div>
                <label className="block text-[11px] font-semibold text-gray-500 dark:text-gray-400 mb-2 uppercase tracking-widest">
                  Username
                </label>
                <div className={`relative rounded-xl border transition-all duration-200 ${
                  focused === 'username'
                    ? 'border-blue-500/50 shadow-[0_0_0_3px_rgba(59,130,246,0.08)]'
                    : 'border-gray-300 dark:border-gray-800 hover:border-gray-400 dark:hover:border-gray-700'
                }`}>
                  <div className="absolute left-3.5 top-1/2 -translate-y-1/2">
                    <User size={16} className={focused === 'username' ? 'text-blue-400' : 'text-gray-400 dark:text-gray-600'} />
                  </div>
                  <input
                    ref={usernameRef}
                    type="text"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    onFocus={() => setFocused('username')}
                    onBlur={() => setFocused(null)}
                    className="w-full pl-11 pr-4 py-3.5 bg-transparent rounded-xl text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-gray-600 focus:outline-none text-sm"
                    placeholder="Enter username"
                    autoComplete="username"
                    autoFocus
                    required
                  />
                </div>
              </div>

              {/* Password */}
              <div>
                <label className="block text-[11px] font-semibold text-gray-500 dark:text-gray-400 mb-2 uppercase tracking-widest">
                  Password
                </label>
                <div className={`relative rounded-xl border transition-all duration-200 ${
                  focused === 'password'
                    ? 'border-blue-500/50 shadow-[0_0_0_3px_rgba(59,130,246,0.08)]'
                    : 'border-gray-300 dark:border-gray-800 hover:border-gray-400 dark:hover:border-gray-700'
                }`}>
                  <div className="absolute left-3.5 top-1/2 -translate-y-1/2">
                    <Lock size={16} className={focused === 'password' ? 'text-blue-400' : 'text-gray-400 dark:text-gray-600'} />
                  </div>
                  <input
                    type={showPassword ? 'text' : 'password'}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    onFocus={() => setFocused('password')}
                    onBlur={() => setFocused(null)}
                    className="w-full pl-11 pr-11 py-3.5 bg-transparent rounded-xl text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-gray-600 focus:outline-none text-sm"
                    placeholder="Enter password"
                    autoComplete="current-password"
                    required
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    tabIndex={-1}
                    className="absolute right-3 top-1/2 -translate-y-1/2 p-1 text-gray-400 dark:text-gray-600 hover:text-gray-600 dark:hover:text-gray-400 transition-colors"
                  >
                    {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                  </button>
                </div>
              </div>

              {/* Submit */}
              <button
                type="submit"
                disabled={loading || !username || !password}
                className={`w-full px-4 py-3.5 rounded-xl font-semibold text-sm flex items-center justify-center gap-2 transition-all duration-200 group ${
                  loading || !username || !password
                    ? 'bg-gray-200 dark:bg-gray-800/50 text-gray-400 dark:text-gray-600 cursor-not-allowed'
                    : 'bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-600/20 hover:shadow-blue-500/30 active:scale-[0.98]'
                }`}
              >
                {loading ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-2 border-white/20 border-t-white" />
                    Authenticating...
                  </>
                ) : (
                  <>
                    Sign In
                    <ArrowRight size={16} className="group-hover:translate-x-0.5 transition-transform" />
                  </>
                )}
              </button>
            </form>
          )}

          {/* Footer */}
          <div className="mt-10 pt-6 border-t border-gray-200 dark:border-gray-800/50">
            <div className="flex items-center justify-center gap-2 text-[11px] text-gray-400 dark:text-gray-600">
              <ShieldCheck size={12} />
              <span>Encrypted and secure admin access</span>
            </div>
          </div>
        </motion.div>
      </div>

      <style>{`
        @keyframes shake { 0%,100%{transform:translateX(0)} 15%,45%,75%{transform:translateX(-6px)} 30%,60%,90%{transform:translateX(6px)} }
        .animate-shake { animation: shake 0.5s ease-in-out; }
        @keyframes fadeIn { from{opacity:0;transform:translateY(-5px)} to{opacity:1;transform:translateY(0)} }
        .animate-fadeIn { animation: fadeIn 0.3s ease-out; }
        @keyframes float { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-20px)} }
        .animate-float { animation: float 8s ease-in-out infinite; }
        .animate-float-delayed { animation: float 8s ease-in-out 4s infinite; }
      `}</style>
    </div>
  );
}
