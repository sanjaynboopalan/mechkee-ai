import React from 'react';
import { motion } from 'framer-motion';
import { LucideIcon } from 'lucide-react';
import { clsx } from 'clsx';

interface InputProps {
  type?: 'text' | 'email' | 'password' | 'search' | 'url' | 'number';
  placeholder?: string;
  value?: string;
  onChange?: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onFocus?: (e: React.FocusEvent<HTMLInputElement>) => void;
  onBlur?: (e: React.FocusEvent<HTMLInputElement>) => void;
  onKeyDown?: (e: React.KeyboardEvent<HTMLInputElement>) => void;
  icon?: LucideIcon;
  iconPosition?: 'left' | 'right';
  size?: 'sm' | 'md' | 'lg';
  variant?: 'default' | 'cosmic' | 'transparent';
  disabled?: boolean;
  className?: string;
  autoFocus?: boolean;
  id?: string;
  name?: string;
  required?: boolean;
}

const Input: React.FC<InputProps> = ({
  type = 'text',
  placeholder,
  value,
  onChange,
  onFocus,
  onBlur,
  onKeyDown,
  icon: Icon,
  iconPosition = 'left',
  size = 'md',
  variant = 'default',
  disabled = false,
  className,
  autoFocus,
  id,
  name,
  required,
}) => {
  const baseClasses = 'w-full rounded-lg border transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-transparent placeholder-gray-400';
  
  const variants = {
    default: 'glass-morphism bg-white/5 border-white/20 text-white focus:border-space-blue-400 focus:ring-space-blue-400/20',
    cosmic: 'bg-gradient-to-r from-space-blue-500/10 to-cosmic-purple-500/10 border-space-blue-400/50 text-white focus:border-cosmic-purple-400 focus:ring-cosmic-purple-400/20',
    transparent: 'bg-transparent border-transparent text-white focus:border-space-blue-400 focus:ring-space-blue-400/20',
  };
  
  const sizes = {
    sm: 'px-3 py-2 text-sm',
    md: 'px-4 py-3 text-base',
    lg: 'px-6 py-4 text-lg',
  };
  
  const inputClasses = clsx(
    baseClasses,
    variants[variant],
    sizes[size],
    {
      'pl-10': Icon && iconPosition === 'left',
      'pr-10': Icon && iconPosition === 'right',
      'opacity-50 cursor-not-allowed': disabled,
    },
    className
  );

  return (
    <div className="relative">
      {Icon && iconPosition === 'left' && (
        <div className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400">
          <Icon className="w-5 h-5" />
        </div>
      )}
      
      <motion.input
        type={type}
        placeholder={placeholder}
        value={value}
        onChange={onChange}
        onFocus={onFocus}
        onBlur={onBlur}
        onKeyDown={onKeyDown}
        disabled={disabled}
        autoFocus={autoFocus}
        id={id}
        name={name}
        required={required}
        className={inputClasses}
        whileFocus={{ scale: 1.01 }}
        transition={{ duration: 0.2 }}
      />
      
      {Icon && iconPosition === 'right' && (
        <div className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400">
          <Icon className="w-5 h-5" />
        </div>
      )}
    </div>
  );
};

export default Input;