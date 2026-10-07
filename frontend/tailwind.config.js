/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        sky: {
          50:  '#F0F9FF',
          100: '#EEF9FF',
          200: '#DFF4FF',
          300: '#BAE6FD',
          400: '#5BB8E8',
          500: '#87CEEB',
          600: '#3B9FD4',
          700: '#2176AE',
          800: '#1A5F8A',
          900: '#164E72',
        },
        brand: {
          bg:      '#F8FCFF',
          white:   '#FFFFFF',
          primary: '#87CEEB',
          soft:    '#DFF4FF',
          light:   '#EEF9FF',
          accent:  '#5BB8E8',
          border:  '#D9EAF2',
          dark:    '#1E293B',
          muted:   '#64748B',
          success: '#10B981',
          warning: '#F59E0B',
          danger:  '#EF4444',
        },
      },
      borderRadius: {
        '2xl': '1rem',
        '3xl': '1.5rem',
      },
      boxShadow: {
        card:  '0 1px 3px rgba(135,206,235,0.12), 0 4px 16px rgba(135,206,235,0.08)',
        'card-hover': '0 4px 12px rgba(91,184,232,0.18), 0 8px 24px rgba(91,184,232,0.12)',
        soft:  '0 2px 8px rgba(0,0,0,0.06)',
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
      },
      animation: {
        'fade-in': 'fadeIn 0.3s ease-out',
        'slide-up': 'slideUp 0.3s ease-out',
        'pulse-soft': 'pulseSoft 2s ease-in-out infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(12px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        pulseSoft: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.6' },
        },
      },
    },
  },
  plugins: [],
}
