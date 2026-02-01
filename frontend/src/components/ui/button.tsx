'use client';

import React from 'react';
import clsx from 'clsx';

type ButtonVariant = 'primary' | 'secondary' | 'outline' | 'ghost' | 'destructive';
type ButtonSize = 'sm' | 'md' | 'lg' | 'icon';

const getButtonClasses = (variant: ButtonVariant = 'primary', size: ButtonSize = 'md') => {
  const baseClasses = 'inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-500 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50';
  
  const variantClasses = {
    primary: 'bg-primary-600 text-white hover:bg-primary-700 active:bg-primary-800 dark:bg-primary-600 dark:hover:bg-primary-700 dark:active:bg-primary-800',
    secondary: 'bg-secondary-100 text-secondary-900 hover:bg-secondary-200 active:bg-secondary-300 dark:bg-secondary-800 dark:text-secondary-100 dark:hover:bg-secondary-700 dark:active:bg-secondary-600',
    outline: 'border border-secondary-300 bg-transparent text-secondary-700 hover:bg-secondary-50 hover:text-secondary-900 active:bg-secondary-100 dark:border-secondary-600 dark:text-secondary-300 dark:hover:bg-secondary-800 dark:hover:text-secondary-100 dark:active:bg-secondary-700',
    ghost: 'text-secondary-700 hover:bg-secondary-100 hover:text-secondary-900 active:bg-secondary-200 dark:text-secondary-300 dark:hover:bg-secondary-800 dark:hover:text-secondary-100 dark:active:bg-secondary-700',
    destructive: 'bg-error-600 text-white hover:bg-error-700 active:bg-error-800 dark:bg-error-600 dark:hover:bg-error-700 dark:active:bg-error-800',
  };

  const sizeClasses = {
    sm: 'h-8 px-3 text-xs',
    md: 'h-10 px-4',
    lg: 'h-12 px-6 text-base',
    icon: 'h-10 w-10',
  };

  return clsx(baseClasses, variantClasses[variant], sizeClasses[size]);
};

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = 'primary', size = 'md', ...props }, ref) => {
    return (
      <button
        className={clsx(getButtonClasses(variant, size), className)}
        ref={ref}
        {...props}
      />
    );
  }
);
Button.displayName = 'Button';

export { Button };