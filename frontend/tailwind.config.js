/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'space-blue': {
          50: '#e6f1ff',
          100: '#b3d9ff',
          200: '#80c1ff',
          300: '#4da9ff',
          400: '#1a91ff',
          500: '#0078e6',
          600: '#005eb3',
          700: '#004480',
          800: '#002a4d',
          900: '#00101a',
        },
        'cosmic-purple': {
          50: '#f0e6ff',
          100: '#d1b3ff',
          200: '#b380ff',
          300: '#944dff',
          400: '#751aff',
          500: '#5c00e6',
          600: '#4700b3',
          700: '#330080',
          800: '#1f004d',
          900: '#0a001a',
        },
        'nebula-pink': {
          50: '#ffe6f7',
          100: '#ffb3e6',
          200: '#ff80d5',
          300: '#ff4dc4',
          400: '#ff1ab3',
          500: '#e6009a',
          600: '#b30078',
          700: '#800056',
          800: '#4d0034',
          900: '#1a0012',
        }
      },
      fontFamily: {
        'space': ['Inter', 'system-ui', 'sans-serif'],
        'mono': ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      animation: {
        'float': 'float 6s ease-in-out infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
        'pulse-slow': 'pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'sparkle': 'sparkle 3s linear infinite',
        'drift': 'drift 10s linear infinite',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-20px)' },
        },
        glow: {
          '0%': { boxShadow: '0 0 20px #0078e6' },
          '100%': { boxShadow: '0 0 30px #0078e6, 0 0 40px #0078e6' },
        },
        sparkle: {
          '0%, 100%': { opacity: '0', transform: 'scale(0)' },
          '50%': { opacity: '1', transform: 'scale(1)' },
        },
        drift: {
          '0%': { transform: 'translateX(-100vw)' },
          '100%': { transform: 'translateX(100vw)' },
        }
      },
      backgroundImage: {
        'space-gradient': 'linear-gradient(135deg, #0a0e27 0%, #1a1f3a 25%, #2a344a 50%, #1a1f3a 75%, #0a0e27 100%)',
        'nebula-gradient': 'radial-gradient(ellipse at center, #4c1d95 0%, #1e1b4b 50%, #0f0f23 100%)',
        'cosmic-gradient': 'linear-gradient(45deg, #0078e6, #5c00e6, #e6009a)',
        'starfield': 'radial-gradient(2px 2px at 20px 30px, #eee, transparent), radial-gradient(2px 2px at 40px 70px, rgba(255,255,255,0.8), transparent), radial-gradient(1px 1px at 90px 40px, #fff, transparent), radial-gradient(1px 1px at 130px 80px, rgba(255,255,255,0.6), transparent)',
      }
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
    require('@tailwindcss/typography'),
  ],
}