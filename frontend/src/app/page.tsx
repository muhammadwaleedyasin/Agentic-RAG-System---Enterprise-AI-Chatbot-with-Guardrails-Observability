'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/hooks/useAuth';
import { Loader2 } from 'lucide-react';

export default function HomePage() {
  const { isAuthenticated, isLoading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!isLoading) {
      if (isAuthenticated) {
        router.push('/chat');
      } else {
        router.push('/login');
      }
    }
  }, [isAuthenticated, isLoading, router]);

  // Loading state while checking authentication
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

  // This component will redirect, so we show a minimal loading state
  return (
    <div className="min-h-screen flex items-center justify-center">
      <div className="text-center">
        <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4 text-primary-600" />
        <h2 className="text-lg font-medium text-secondary-900 dark:text-secondary-100 mb-2">
          Redirecting...
        </h2>
        <p className="text-secondary-600 dark:text-secondary-400">
          Taking you to the right place
        </p>
      </div>
    </div>
  );
}