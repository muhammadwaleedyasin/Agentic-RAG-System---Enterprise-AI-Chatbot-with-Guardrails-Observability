'use client';

import React, { createContext, useContext, useEffect, useState, useCallback } from 'react';
import { User, LoginRequest, LoginResponse } from '@/types/api';
import { getStoredToken, setStoredToken, removeStoredToken, validateToken } from '@/lib/auth';

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (credentials: LoginRequest) => Promise<void>;
  logout: () => void;
  refresh: () => Promise<void>;
  hasRole: (roleName: string) => boolean;
  hasPermission: (resource: string, action: string) => boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const isAuthenticated = !!user && !!getStoredToken();

  const fetchUser = async (token: string): Promise<User> => {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    const response = await fetch(`${apiUrl}/api/v1/auth/me`, {
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error('Failed to fetch user');
    }

    const data = await response.json();
    return data.data || data;
  };

  const login = async (credentials: LoginRequest): Promise<void> => {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    
    // Convert credentials to form-encoded format
    const formData = new URLSearchParams();
    formData.append('username', credentials.username);
    formData.append('password', credentials.password);
    
    console.log('Sending login request:', {
      url: `${apiUrl}/api/v1/auth/login`,
      credentials: credentials,
      formData: formData.toString()
    });
    
    const response = await fetch(`${apiUrl}/api/v1/auth/login`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: formData,
    });

    console.log('Login response status:', response.status);
    console.log('Login response headers:', Object.fromEntries(response.headers.entries()));

    if (!response.ok) {
      const errorText = await response.text();
      console.error('Login failed with error:', errorText);
      throw new Error(`Login failed: ${response.status} ${response.statusText}`);
    }

    const loginData: LoginResponse = await response.json();
    console.log('Login successful, received token:', loginData.access_token?.substring(0, 10) + '...');
    setStoredToken(loginData.access_token);

    // Fetch user data
    const userData = await fetchUser(loginData.access_token);
    setUser(userData);
  };

  const logout = (): void => {
    removeStoredToken();
    setUser(null);
  };

  const refresh = useCallback(async (): Promise<void> => {
    const token = getStoredToken();
    if (!token) {
      setUser(null);
      return;
    }

    try {
      const userData = await fetchUser(token);
      setUser(userData);
    } catch (error) {
      console.error('Failed to refresh user:', error);
      logout();
    }
  }, []);

  const hasRole = (roleName: string): boolean => {
    if (!user?.roles) return false;
    return user.roles.some(role => role.name === roleName);
  };

  const hasPermission = (resource: string, action: string): boolean => {
    if (!user?.roles) return false;
    return user.roles.some(role =>
      role.permissions.some(permission =>
        permission.resource === resource && permission.action === action
      )
    );
  };

  useEffect(() => {
    const initializeAuth = async () => {
      setIsLoading(true);
      const token = getStoredToken();
      
      if (token && validateToken(token)) {
        try {
          await refresh();
        } catch (error) {
          console.error('Auth initialization failed:', error);
          logout();
        }
      }
      
      setIsLoading(false);
    };

    initializeAuth();
  }, [refresh]);

  const value: AuthContextType = {
    user,
    isAuthenticated,
    isLoading,
    login,
    logout,
    refresh,
    hasRole,
    hasPermission,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextType {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}