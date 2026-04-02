/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: [
    "./client/index.html",
    "./client/src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
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
        }
      }
    },
  },
  plugins: [],
}
