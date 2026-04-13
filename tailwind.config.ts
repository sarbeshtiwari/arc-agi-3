/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: [
    "./client/index.html",
    "./client/src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        pixel: ['"Press Start 2P"', 'monospace'],
      },
      colors: {
        arc: {
          white: '#FFFFFF',
          offwhite: '#CCCCCC',
          lightgrey: '#999999',
          grey: '#666666',
          darkgrey: '#333333',
          black: '#000000',
          magenta: '#E53AA3',
          pink: '#FF7BCC',
          red: '#F93C31',
          blue: '#1E93FF',
          lightblue: '#88D8F1',
          yellow: '#FFDC00',
          orange: '#FF851B',
          maroon: '#921231',
          green: '#4FCC30',
          purple: '#A356D6',
        },
        neon: {
          cyan: '#00FFFF',
          magenta: '#FF00FF',
          green: '#39FF14',
          yellow: '#FFE600',
          orange: '#FF6600',
          pink: '#FF69B4',
          blue: '#4D4DFF',
        },
        cabinet: {
          body: '#1a1a2e',
          panel: '#16213e',
          bezel: '#0f0f23',
          trim: '#2a2a4a',
          screen: '#0a0a1a',
        },
      },
      boxShadow: {
        'neon-cyan': '0 0 5px #00FFFF, 0 0 20px #00FFFF40, 0 0 40px #00FFFF20',
        'neon-magenta': '0 0 5px #FF00FF, 0 0 20px #FF00FF40, 0 0 40px #FF00FF20',
        'neon-green': '0 0 5px #39FF14, 0 0 20px #39FF1440, 0 0 40px #39FF1420',
        'neon-yellow': '0 0 5px #FFE600, 0 0 20px #FFE60040, 0 0 40px #FFE60020',
        'neon-blue': '0 0 5px #4D4DFF, 0 0 20px #4D4DFF40, 0 0 40px #4D4DFF20',
        'arcade-screen': 'inset 0 0 30px rgba(0,255,255,0.1), 0 0 15px rgba(0,0,0,0.8)',
        'cabinet': '0 10px 40px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.05)',
      },
      keyframes: {
        'scanline': {
          '0%': { transform: 'translateY(-100%)' },
          '100%': { transform: 'translateY(100%)' },
        },
        'crt-flicker': {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.98' },
          '25%, 75%': { opacity: '0.99' },
        },
        'neon-pulse': {
          '0%, 100%': { opacity: '1', filter: 'brightness(1)' },
          '50%': { opacity: '0.85', filter: 'brightness(1.2)' },
        },
        'coin-blink': {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.3' },
        },
        'marquee-glow': {
          '0%, 100%': { textShadow: '0 0 10px currentColor, 0 0 20px currentColor, 0 0 40px currentColor' },
          '50%': { textShadow: '0 0 5px currentColor, 0 0 10px currentColor, 0 0 20px currentColor' },
        },
      },
      animation: {
        'scanline': 'scanline 8s linear infinite',
        'crt-flicker': 'crt-flicker 0.15s infinite',
        'neon-pulse': 'neon-pulse 2s ease-in-out infinite',
        'coin-blink': 'coin-blink 1.5s ease-in-out infinite',
        'marquee-glow': 'marquee-glow 3s ease-in-out infinite',
      },
    },
  },
  plugins: [],
}
