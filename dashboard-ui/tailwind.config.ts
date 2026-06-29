import type { Config } from 'tailwindcss'

const config: Config = {
  darkMode: 'class',
  content: [
    './app/**/*.{ts,tsx}',
    './components/**/*.{ts,tsx}',
    './lib/**/*.{ts,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        background: '#08080a',
        surface:    '#1a1a1f',
        surface2:   '#212127',
        border:     '#2a2a30',
        accent:     '#f97316',
        positive:   '#22c55e',
        negative:   '#ef4444',
        warning:    '#eab308',
        muted:      '#71717a',
      },
      fontFamily: {
        sans: ['var(--font-geist-sans)', 'system-ui', 'sans-serif'],
        mono: ['var(--font-geist-mono)', 'ui-monospace', 'monospace'],
      },
      borderRadius: {
        '2xl': '1rem',
        '3xl': '1.25rem',
      },
      boxShadow: {
        card:        '0 1px 2px rgba(0,0,0,0.4), 0 2px 8px -2px rgba(0,0,0,0.35)',
        'card-hover':'0 12px 32px -8px rgba(0,0,0,0.6), 0 2px 8px -2px rgba(0,0,0,0.4)',
        'glow-orange':'0 0 24px -6px rgba(249,115,22,0.45)',
        'glow-green': '0 0 24px -6px rgba(34,197,94,0.45)',
        'glow-red':   '0 0 24px -6px rgba(239,68,68,0.45)',
      },
      backgroundImage: {
        'card-grad':  'linear-gradient(180deg, rgba(255,255,255,0.07), rgba(255,255,255,0.015) 45%, transparent)',
        'app-glow':   'radial-gradient(900px 500px at 70% -10%, rgba(249,115,22,0.06), transparent 70%)',
        'shimmer':    'linear-gradient(90deg, transparent, rgba(255,255,255,0.05), transparent)',
      },
      keyframes: {
        'fade-up': {
          '0%':   { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        shimmer: {
          '100%': { transform: 'translateX(100%)' },
        },
        'pulse-ring': {
          '0%':   { boxShadow: '0 0 0 0 rgba(34,197,94,0.5)' },
          '70%':  { boxShadow: '0 0 0 5px rgba(34,197,94,0)' },
          '100%': { boxShadow: '0 0 0 0 rgba(34,197,94,0)' },
        },
      },
      animation: {
        'fade-up':   'fade-up 0.5s cubic-bezier(0.16,1,0.3,1) both',
        shimmer:     'shimmer 1.6s infinite',
        'pulse-ring':'pulse-ring 2s infinite',
      },
    },
  },
  plugins: [],
}

export default config
