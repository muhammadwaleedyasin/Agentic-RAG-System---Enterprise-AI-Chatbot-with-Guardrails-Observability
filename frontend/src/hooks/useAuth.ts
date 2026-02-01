'use client';

import { useContext } from 'react';
import { useAuth as useAuthContext } from '@/contexts/auth-context';

// Re-export the useAuth hook from the context
export { useAuth } from '@/contexts/auth-context';

// Additional auth-related hooks can be added here
export function useRequireAuth() {
  const { isAuthenticated, isLoading } = useAuthContext();
  
  if (!isLoading && !isAuthenticated) {
    throw new Error('Authentication required');
  }
  
  return { isAuthenticated, isLoading };
}

export function useRequireRole(roleName: string) {
  const { hasRole, isAuthenticated, isLoading } = useAuthContext();
  
  if (!isLoading && (!isAuthenticated || !hasRole(roleName))) {
    throw new Error(`Role '${roleName}' required`);
  }
  
  return { hasRole: () => hasRole(roleName), isAuthenticated, isLoading };
}

export function useRequirePermission(resource: string, action: string) {
  const { hasPermission, isAuthenticated, isLoading } = useAuthContext();
  
  if (!isLoading && (!isAuthenticated || !hasPermission(resource, action))) {
    throw new Error(`Permission '${action}' on '${resource}' required`);
  }
  
  return {
    hasPermission: () => hasPermission(resource, action),
    isAuthenticated,
    isLoading
  };
}