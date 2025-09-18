import React from 'react';
import { motion } from 'framer-motion';

interface AuroraProps {
  className?: string;
  intensity?: 'low' | 'medium' | 'high';
}

const Aurora: React.FC<AuroraProps> = ({ 
  className = '', 
  intensity = 'medium' 
}) => {
  const intensitySettings = {
    low: { opacity: 0.2, scale: 1, blur: 'blur-3xl' },
    medium: { opacity: 0.4, scale: 1.1, blur: 'blur-2xl' },
    high: { opacity: 0.6, scale: 1.2, blur: 'blur-xl' },
  };

  const settings = intensitySettings[intensity];

  return (
    <div className={`fixed inset-0 pointer-events-none z-0 overflow-hidden ${className}`}>
      {/* Primary Aurora */}
      <motion.div
        className={`absolute -top-1/2 -left-1/2 w-full h-full ${settings.blur}`}
        style={{
          backgroundImage: `conic-gradient(from 0deg at 50% 50%, 
            rgba(0, 120, 230, ${settings.opacity}) 0deg,
            rgba(92, 0, 230, ${settings.opacity * 0.8}) 72deg,
            rgba(230, 0, 154, ${settings.opacity * 0.6}) 144deg,
            rgba(92, 0, 230, ${settings.opacity * 0.8}) 216deg,
            rgba(0, 120, 230, ${settings.opacity}) 288deg,
            rgba(0, 120, 230, ${settings.opacity}) 360deg
          )`,
        }}
        animate={{
          rotate: [0, 360],
          scale: [settings.scale, settings.scale * 1.1, settings.scale],
        }}
        transition={{
          rotate: { duration: 20, repeat: Infinity, ease: 'linear' },
          scale: { duration: 8, repeat: Infinity, ease: 'easeInOut' },
        }}
      />

      {/* Secondary Aurora */}
      <motion.div
        className={`absolute -top-1/4 -right-1/4 w-3/4 h-3/4 ${settings.blur}`}
        style={{
          backgroundImage: `radial-gradient(ellipse at center,
            rgba(92, 0, 230, ${settings.opacity * 0.7}) 0%,
            rgba(230, 0, 154, ${settings.opacity * 0.5}) 30%,
            rgba(0, 120, 230, ${settings.opacity * 0.3}) 60%,
            transparent 100%
          )`,
        }}
        animate={{
          rotate: [360, 0],
          scale: [settings.scale * 0.8, settings.scale * 1.2, settings.scale * 0.8],
          x: [0, 50, 0],
          y: [0, -30, 0],
        }}
        transition={{
          rotate: { duration: 15, repeat: Infinity, ease: 'linear' },
          scale: { duration: 6, repeat: Infinity, ease: 'easeInOut' },
          x: { duration: 12, repeat: Infinity, ease: 'easeInOut' },
          y: { duration: 10, repeat: Infinity, ease: 'easeInOut' },
        }}
      />

      {/* Tertiary Aurora */}
      <motion.div
        className={`absolute -bottom-1/4 -left-1/4 w-1/2 h-1/2 ${settings.blur}`}
        style={{
          backgroundImage: `conic-gradient(from 180deg at 50% 50%,
            rgba(230, 0, 154, ${settings.opacity * 0.6}) 0deg,
            rgba(0, 120, 230, ${settings.opacity * 0.4}) 120deg,
            rgba(92, 0, 230, ${settings.opacity * 0.5}) 240deg,
            rgba(230, 0, 154, ${settings.opacity * 0.6}) 360deg
          )`,
        }}
        animate={{
          rotate: [180, -180],
          scale: [settings.scale * 0.6, settings.scale * 1.4, settings.scale * 0.6],
        }}
        transition={{
          rotate: { duration: 25, repeat: Infinity, ease: 'linear' },
          scale: { duration: 14, repeat: Infinity, ease: 'easeInOut' },
        }}
      />
    </div>
  );
};

export default Aurora;