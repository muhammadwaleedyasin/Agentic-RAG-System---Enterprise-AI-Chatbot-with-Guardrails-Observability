'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/hooks/useAuth';
import { Loader2 } from 'lucide-react';

interface AuthGuardProps {
  children: React.ReactNode;
  requireAuth?: boolean;
  requiredRole?: string;
  fallbackPath?: string;
}

export function AuthGuard({ 
  children, 
  requireAuth = true, 
  requiredRole, 
  fallbackPath = '/login' 
}: AuthGuardProps) {
  const { isAuthenticated, isLoading, hasRole } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!isLoading) {
      if (requireAuth && !isAuthenticated) {
        router.push(fallbackPath);
        return;
      }
      
      if (requiredRole && !hasRole(requiredRole)) {
        router.push('/');
        return;
      }
    }
  }, [isAuthenticated, isLoading, hasRole, requiredRole, requireAuth, router, fallbackPath]);

  // Show loading while checking authentication
  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4 text-primary-600" />
          <h2 className="text-lg font-medium text-secondary-900 dark:text-secondary-100 mb-2">
            Loading...
          </h2>
          <p className="text-secondary-600 dark:text-secondary-400">
            Checking authentication status
          </p>
        </div>
      </div>
    );
  }

  // Don't render if auth requirements aren't met (redirect is in progress)
  if (requireAuth && !isAuthenticated) {
    return null;
  }

  if (requiredRole && !hasRole(requiredRole)) {
    return null;
  }

  return <>{children}</>;
}