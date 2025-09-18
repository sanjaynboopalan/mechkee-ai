import React from 'react';
import { motion } from 'framer-motion';
import { LucideIcon } from 'lucide-react';
import { clsx } from 'clsx';

interface ButtonProps {
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost' | 'cosmic';
  size?: 'sm' | 'md' | 'lg' | 'xl';
  icon?: LucideIcon;
  iconPosition?: 'left' | 'right';
  loading?: boolean;
  glow?: boolean;
  children: React.ReactNode;
  className?: string;
  disabled?: boolean;
  onClick?: (e: React.MouseEvent<HTMLButtonElement>) => void;
  type?: 'button' | 'submit' | 'reset';
}

const Button: React.FC<ButtonProps> = ({
  variant = 'primary',
  size = 'md',
  icon: Icon,
  iconPosition = 'left',
  loading = false,
  glow = false,
  children,
  className,
  disabled,
  ...props
}) => {
  const baseClasses = 'inline-flex items-center justify-center font-medium rounded-lg transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-transparent disabled:opacity-50 disabled:cursor-not-allowed';
  
  const variants = {
    primary: 'bg-gradient-to-r from-space-blue-500 to-cosmic-purple-500 text-white hover:from-space-blue-600 hover:to-cosmic-purple-600 focus:ring-space-blue-400',
    secondary: 'bg-gradient-to-r from-cosmic-purple-500 to-nebula-pink-500 text-white hover:from-cosmic-purple-600 hover:to-nebula-pink-600 focus:ring-cosmic-purple-400',
    outline: 'border-2 border-space-blue-400 text-space-blue-400 hover:bg-space-blue-400 hover:text-white focus:ring-space-blue-400',
    ghost: 'text-space-blue-400 hover:bg-space-blue-400/10 focus:ring-space-blue-400',
    cosmic: 'bg-gradient-to-r from-space-blue-500 via-cosmic-purple-500 to-nebula-pink-500 text-white hover:shadow-lg hover:shadow-space-blue-500/25 focus:ring-cosmic-purple-400',
  };
  
  const sizes = {
    sm: 'px-3 py-1.5 text-sm',
    md: 'px-4 py-2 text-sm',
    lg: 'px-6 py-3 text-base',
    xl: 'px-8 py-4 text-lg',
  };
  
  const glowClasses = glow ? 'shadow-lg shadow-current/25 hover:shadow-xl hover:shadow-current/40' : '';
  
  const buttonClasses = clsx(
    baseClasses,
    variants[variant],
    sizes[size],
    glowClasses,
    className
  );

  return (
    <motion.button
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      className={buttonClasses}
      disabled={disabled || loading}
      {...props}
    >
      {loading && (
        <div className="flex items-center mr-2">
          <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent"></div>
        </div>
      )}
      
      {Icon && iconPosition === 'left' && !loading && (
        <Icon className="w-4 h-4 mr-2" />
      )}
      
      {children}
      
      {Icon && iconPosition === 'right' && !loading && (
        <Icon className="w-4 h-4 ml-2" />
      )}
    </motion.button>
  );
};

export default Button;