import React from 'react';
import { motion } from 'framer-motion';
import { clsx } from 'clsx';

interface CardProps {
  children: React.ReactNode;
  className?: string;
  variant?: 'default' | 'cosmic' | 'nebula' | 'transparent';
  size?: 'sm' | 'md' | 'lg' | 'xl';
  hover?: boolean;
  glow?: boolean;
  onClick?: () => void;
  as?: React.ElementType;
}

const Card: React.FC<CardProps> = ({
  children,
  className,
  variant = 'default',
  size = 'md',
  hover = false,
  glow = false,
  onClick,
  as: Component = 'div',
}) => {
  const baseClasses = 'rounded-xl transition-all duration-300';
  
  const variants = {
    default: 'glass-morphism bg-white/5 border border-white/10',
    cosmic: 'bg-gradient-to-br from-space-blue-500/10 via-cosmic-purple-500/10 to-nebula-pink-500/10 border border-space-blue-400/30',
    nebula: 'nebula-bg border border-cosmic-purple-400/30',
    transparent: 'bg-transparent border border-white/5',
  };
  
  const sizes = {
    sm: 'p-4',
    md: 'p-6',
    lg: 'p-8',
    xl: 'p-12',
  };
  
  const hoverClasses = hover ? 'hover:bg-white/10 hover:border-white/20 hover:shadow-lg hover:shadow-space-blue-500/10 hover:-translate-y-1 cursor-pointer' : '';
  const glowClasses = glow ? 'shadow-lg shadow-space-blue-500/20' : '';
  
  const cardClasses = clsx(
    baseClasses,
    variants[variant],
    sizes[size],
    hoverClasses,
    glowClasses,
    className
  );

  if (Component && Component !== 'div') {
    return (
      <Component className={cardClasses} onClick={onClick}>
        {children}
      </Component>
    );
  }

  return (
    <motion.div
      className={cardClasses}
      onClick={onClick}
      whileHover={hover ? { scale: 1.02 } : undefined}
      whileTap={onClick ? { scale: 0.98 } : undefined}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      {children}
    </motion.div>
  );
};

export default Card;