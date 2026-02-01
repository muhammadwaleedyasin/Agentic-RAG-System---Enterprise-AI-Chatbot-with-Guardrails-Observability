'use client';

import { useState } from 'react';
import { useAuth } from '@/hooks/useAuth';
import { useRouter } from 'next/navigation';
import { Menu, Moon, Sun, User, LogOut, LogIn } from 'lucide-react';
import { useTheme } from '@/contexts/theme-context';

export function Header() {
  const { user, logout, isAuthenticated } = useAuth();
  const { theme, setTheme } = useTheme();
  const router = useRouter();
  const [userMenuOpen, setUserMenuOpen] = useState(false);

  const handleLogout = () => {
    logout();
    setUserMenuOpen(false);
  };

  const handleLogin = () => {
    router.push('/login');
  };

  const toggleTheme = () => {
    setTheme(theme === 'dark' ? 'light' : 'dark');
  };

  return (
    <header
      className="bg-white dark:bg-secondary-900 border-b border-secondary-200 dark:border-secondary-700 px-4 py-3"
      role="banner"
    >
      <div className="flex items-center justify-between">
        {/* Logo and Brand */}
        <div className="flex items-center gap-4">
          <button
            className="md:hidden p-2 rounded-md hover:bg-secondary-100 dark:hover:bg-secondary-800 focus:outline-none focus:ring-2 focus:ring-primary-500"
            aria-label="Toggle navigation menu"
            type="button"
          >
            <Menu className="h-5 w-5" />
          </button>
          
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-primary-600 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-sm">AR</span>
            </div>
            <h1 className="text-lg font-semibold text-secondary-900 dark:text-secondary-100">
              Agentic RAG
            </h1>
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-2">
          {/* Theme Toggle */}
          <button
            onClick={toggleTheme}
            className="p-2 rounded-md hover:bg-secondary-100 dark:hover:bg-secondary-800 focus:outline-none focus:ring-2 focus:ring-primary-500"
            aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
            type="button"
          >
            {theme === 'dark' ? (
              <Sun className="h-5 w-5" />
            ) : (
              <Moon className="h-5 w-5" />
            )}
          </button>

          {/* Authentication Actions */}
          {isAuthenticated && user ? (
            <div className="relative">
              <button
                onClick={() => setUserMenuOpen(!userMenuOpen)}
                className="flex items-center gap-2 p-2 rounded-md hover:bg-secondary-100 dark:hover:bg-secondary-800 focus:outline-none focus:ring-2 focus:ring-primary-500"
                aria-label="User menu"
                aria-expanded={userMenuOpen}
                aria-haspopup="true"
                type="button"
              >
                <User className="h-5 w-5" />
                <span className="hidden sm:inline text-sm font-medium">
                  {user.full_name || user.username}
                </span>
              </button>

              {userMenuOpen && (
                <div
                  className="absolute right-0 mt-2 w-48 bg-white dark:bg-secondary-800 rounded-md shadow-lg border border-secondary-200 dark:border-secondary-700 py-1 z-50"
                  role="menu"
                  aria-orientation="vertical"
                >
                  <div className="px-4 py-2 text-sm text-secondary-500 dark:text-secondary-400 border-b border-secondary-200 dark:border-secondary-700">
                    {user.email}
                  </div>
                  
                  <button
                    onClick={handleLogout}
                    className="flex items-center gap-2 w-full px-4 py-2 text-sm text-secondary-700 dark:text-secondary-300 hover:bg-secondary-100 dark:hover:bg-secondary-700 focus:outline-none focus:bg-secondary-100 dark:focus:bg-secondary-700"
                    role="menuitem"
                    type="button"
                  >
                    <LogOut className="h-4 w-4" />
                    Sign out
                  </button>
                </div>
              )}
            </div>
          ) : (
            <button
              onClick={handleLogin}
              className="flex items-center gap-2 px-3 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-md text-sm font-medium focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2"
              type="button"
              data-testid="header-sign-in"
            >
              <LogIn className="h-4 w-4" />
              <span className="hidden sm:inline">Sign In</span>
            </button>
          )}
        </div>
      </div>
    </header>
  );
}