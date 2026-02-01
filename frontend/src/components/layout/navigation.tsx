'use client';

import { useAuth } from '@/hooks/useAuth';
import { usePathname } from 'next/navigation';
import Link from 'next/link';
import { 
  MessageSquare, 
  FileText, 
  BarChart3, 
  Settings, 
  Home 
} from 'lucide-react';
import clsx from 'clsx';

interface NavItem {
  name: string;
  href: string;
  icon: React.ComponentType<{ className?: string }>;
  requireAuth?: boolean;
  requiredRole?: string;
}

const navItems: NavItem[] = [
  {
    name: 'Home',
    href: '/',
    icon: Home,
  },
  {
    name: 'Chat',
    href: '/chat',
    icon: MessageSquare,
    requireAuth: true,
  },
  {
    name: 'Documents',
    href: '/documents',
    icon: FileText,
    requireAuth: true,
  },
  {
    name: 'Dashboard',
    href: '/dashboard',
    icon: BarChart3,
    requireAuth: true,
  },
  {
    name: 'Admin',
    href: '/admin',
    icon: Settings,
    requireAuth: true,
    requiredRole: 'admin',
  },
];

export function Navigation() {
  const { isAuthenticated, hasRole } = useAuth();
  const pathname = usePathname();

  const filteredNavItems = navItems.filter(item => {
    if (item.requireAuth && !isAuthenticated) return false;
    if (item.requiredRole && !hasRole(item.requiredRole)) return false;
    return true;
  });

  return (
    <nav
      className="hidden md:flex md:flex-col w-64 bg-white dark:bg-secondary-900 border-r border-secondary-200 dark:border-secondary-700"
      role="navigation"
      aria-label="Main navigation"
    >
      <div className="flex-1 px-4 py-6 space-y-1">
        {filteredNavItems.map((item) => {
          const isActive = pathname === item.href;
          const Icon = item.icon;

          return (
            <Link
              key={item.name}
              href={item.href}
              className={clsx(
                'flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2',
                isActive
                  ? 'bg-primary-100 dark:bg-primary-900/50 text-primary-700 dark:text-primary-300'
                  : 'text-secondary-700 dark:text-secondary-300 hover:bg-secondary-100 dark:hover:bg-secondary-800 hover:text-secondary-900 dark:hover:text-secondary-100'
              )}
              aria-current={isActive ? 'page' : undefined}
            >
              <Icon className="h-5 w-5 flex-shrink-0" aria-hidden="true" />
              {item.name}
            </Link>
          );
        })}
      </div>

      {/* Footer info */}
      <div className="px-4 py-4 border-t border-secondary-200 dark:border-secondary-700">
        <div className="text-xs text-secondary-500 dark:text-secondary-400">
          <p>Agentic RAG System</p>
          <p>Version 1.0.0</p>
        </div>
      </div>
    </nav>
  );
}