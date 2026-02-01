'use client';

import { useState, useEffect, useCallback } from 'react';
import { useAuth } from '@/hooks/useAuth';
import { AuthGuard } from '@/components/auth/auth-guard';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Modal } from '@/components/ui/modal';
import { apiClient } from '@/lib/api';
import { User, AdminSettings, AuditLog, QueryParams } from '@/types/api';
import {
  Users,
  Settings,
  Shield,
  Activity,
  Loader2,
  RefreshCw,
  Search,
  ChevronLeft,
  ChevronRight,
  Edit,
  Trash2,
  AlertTriangle
} from 'lucide-react';
import clsx from 'clsx';

function AdminPageContent() {
  const { user, hasRole } = useAuth();
  const [activeTab, setActiveTab] = useState<'users' | 'settings' | 'audit'>('users');
  const [users, setUsers] = useState<User[]>([]);
  const [settings, setSettings] = useState<AdminSettings | null>(null);
  const [auditLogs, setAuditLogs] = useState<AuditLog[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [totalCount, setTotalCount] = useState(0);
  const pageSize = 20;

  const isAdmin = hasRole('admin') || hasRole('superuser');

  const loadData = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);

      const params: QueryParams = {
        page: currentPage,
        per_page: pageSize,
        search: searchQuery || undefined,
      };

      switch (activeTab) {
        case 'users':
          const usersData = await apiClient.getUsers(params);
          const userData = usersData as any;
          setUsers(userData.data || userData);
          setTotalCount(userData.total || userData.length || 0);
          break;
        case 'settings':
          const settingsData = await apiClient.getSettings();
          setSettings(settingsData as AdminSettings);
          break;
        case 'audit':
          const auditData = await apiClient.getAuditLogs(params);
          const auditArray = auditData as any;
          setAuditLogs(auditArray.data || auditArray);
          setTotalCount(auditArray.total || auditArray.length || 0);
          break;
      }
    } catch (err) {
      console.error('Failed to load admin data:', err);
      setError(err instanceof Error ? err.message : 'Failed to load data');
    } finally {
      setIsLoading(false);
    }
  }, [activeTab, currentPage, searchQuery]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const handleSearch = () => {
    setCurrentPage(1);
    loadData();
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString(undefined, {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const totalPages = Math.ceil(totalCount / pageSize);


  const tabs = [
    { id: 'users', label: 'Users', icon: Users },
    { id: 'settings', label: 'Settings', icon: Settings },
    { id: 'audit', label: 'Audit Logs', icon: Activity },
  ] as const;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-secondary-900 dark:text-secondary-100">
            Admin Panel
          </h1>
          <p className="text-secondary-600 dark:text-secondary-400 mt-1">
            Manage users, settings, and monitor system activity
          </p>
        </div>
        <Button
          onClick={loadData}
          variant="outline"
          disabled={isLoading}
          className="flex items-center gap-2"
        >
          <RefreshCw className={clsx('h-4 w-4', isLoading && 'animate-spin')} />
          Refresh
        </Button>
      </div>

      {/* Tabs */}
      <div className="border-b border-secondary-200 dark:border-secondary-700">
        <nav className="-mb-px flex space-x-8">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => {
                  setActiveTab(tab.id);
                  setCurrentPage(1);
                  setSearchQuery('');
                }}
                className={clsx(
                  'flex items-center gap-2 py-2 px-1 border-b-2 font-medium text-sm transition-colors',
                  activeTab === tab.id
                    ? 'border-primary-500 text-primary-600 dark:text-primary-400'
                    : 'border-transparent text-secondary-500 dark:text-secondary-400 hover:text-secondary-700 dark:hover:text-secondary-300 hover:border-secondary-300 dark:hover:border-secondary-600'
                )}
              >
                <Icon className="h-4 w-4" />
                {tab.label}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Error Alert */}
      {error && (
        <div className="bg-error-50 dark:bg-error-900/20 border border-error-200 dark:border-error-800 rounded-md p-4">
          <div className="flex">
            <AlertTriangle className="h-5 w-5 text-error-600 dark:text-error-400 mt-0.5" />
            <div className="ml-3">
              <h3 className="text-sm font-medium text-error-800 dark:text-error-200">
                Error loading data
              </h3>
              <div className="mt-2 text-sm text-error-700 dark:text-error-300">
                {error}
              </div>
              <div className="mt-3">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={loadData}
                >
                  Try Again
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Content */}
      <Card>
        <CardContent className="p-0">
          {/* Search for users and audit logs */}
          {(activeTab === 'users' || activeTab === 'audit') && (
            <div className="p-4 border-b border-secondary-200 dark:border-secondary-700">
              <div className="flex gap-2">
                <div className="relative flex-1">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-secondary-400" />
                  <Input
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                    placeholder={`Search ${activeTab}...`}
                    className="pl-10"
                  />
                </div>
                <Button onClick={handleSearch} variant="outline">
                  <Search className="h-4 w-4" />
                </Button>
              </div>
            </div>
          )}

          {/* Loading State */}
          {isLoading ? (
            <div className="flex items-center justify-center h-64">
              <div className="text-center">
                <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4 text-primary-600" />
                <p className="text-secondary-600 dark:text-secondary-400">
                  Loading {activeTab}...
                </p>
              </div>
            </div>
          ) : (
            <>
              {/* Users Tab */}
              {activeTab === 'users' && (
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-secondary-200 dark:divide-secondary-700">
                    <thead className="bg-secondary-50 dark:bg-secondary-800">
                      <tr>
                        <th className="px-6 py-3 text-left text-xs font-medium text-secondary-500 dark:text-secondary-400 uppercase tracking-wider">
                          User
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-secondary-500 dark:text-secondary-400 uppercase tracking-wider">
                          Status
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-secondary-500 dark:text-secondary-400 uppercase tracking-wider">
                          Roles
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-secondary-500 dark:text-secondary-400 uppercase tracking-wider">
                          Created
                        </th>
                        <th className="relative px-6 py-3">
                          <span className="sr-only">Actions</span>
                        </th>
                      </tr>
                    </thead>
                    <tbody className="bg-white dark:bg-secondary-900 divide-y divide-secondary-200 dark:divide-secondary-700">
                      {users.map((userItem) => (
                        <tr key={userItem.id} className="hover:bg-secondary-50 dark:hover:bg-secondary-800">
                          <td className="px-6 py-4 whitespace-nowrap">
                            <div>
                              <div className="text-sm font-medium text-secondary-900 dark:text-secondary-100">
                                {userItem.full_name || userItem.username}
                              </div>
                              <div className="text-sm text-secondary-500 dark:text-secondary-400">
                                {userItem.email}
                              </div>
                            </div>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <span className={clsx(
                              'inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium',
                              userItem.is_active
                                ? 'bg-success-100 text-success-800 dark:bg-success-900/20 dark:text-success-400'
                                : 'bg-error-100 text-error-800 dark:bg-error-900/20 dark:text-error-400'
                            )}>
                              {userItem.is_active ? 'Active' : 'Inactive'}
                            </span>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <div className="flex flex-wrap gap-1">
                              {userItem.roles.map((role, index) => (
                                <span
                                  key={index}
                                  className="inline-flex items-center px-2 py-1 rounded-md text-xs bg-primary-100 text-primary-800 dark:bg-primary-900/20 dark:text-primary-400"
                                >
                                  {role.name}
                                </span>
                              ))}
                            </div>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-secondary-500 dark:text-secondary-400">
                            {formatDate(userItem.created_at)}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                            <div className="flex items-center gap-2">
                              <Button variant="ghost" size="sm">
                                <Edit className="h-4 w-4" />
                              </Button>
                              <Button
                                variant="ghost"
                                size="sm"
                                className="text-error-600 hover:text-error-700 dark:text-error-400 dark:hover:text-error-300"
                              >
                                <Trash2 className="h-4 w-4" />
                              </Button>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {/* Settings Tab */}
              {activeTab === 'settings' && settings && (
                <div className="p-6 space-y-6">
                  <div>
                    <h3 className="text-lg font-medium text-secondary-900 dark:text-secondary-100 mb-4">
                      System Settings
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <label className="block text-sm font-medium text-secondary-700 dark:text-secondary-300 mb-2">
                          Site Name
                        </label>
                        <Input value={settings.site_name} readOnly />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-secondary-700 dark:text-secondary-300 mb-2">
                          Max Upload Size (MB)
                        </label>
                        <Input value={settings.max_upload_size} readOnly />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-secondary-700 dark:text-secondary-300 mb-2">
                          Session Timeout (minutes)
                        </label>
                        <Input value={settings.security.session_timeout} readOnly />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-secondary-700 dark:text-secondary-300 mb-2">
                          Max Login Attempts
                        </label>
                        <Input value={settings.security.max_login_attempts} readOnly />
                      </div>
                    </div>
                  </div>
                  
                  <div className="pt-4">
                    <Button disabled>
                      Save Settings (Read Only)
                    </Button>
                  </div>
                </div>
              )}

              {/* Audit Logs Tab */}
              {activeTab === 'audit' && (
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-secondary-200 dark:divide-secondary-700">
                    <thead className="bg-secondary-50 dark:bg-secondary-800">
                      <tr>
                        <th className="px-6 py-3 text-left text-xs font-medium text-secondary-500 dark:text-secondary-400 uppercase tracking-wider">
                          User
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-secondary-500 dark:text-secondary-400 uppercase tracking-wider">
                          Action
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-secondary-500 dark:text-secondary-400 uppercase tracking-wider">
                          Resource
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-secondary-500 dark:text-secondary-400 uppercase tracking-wider">
                          Timestamp
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-secondary-500 dark:text-secondary-400 uppercase tracking-wider">
                          IP Address
                        </th>
                      </tr>
                    </thead>
                    <tbody className="bg-white dark:bg-secondary-900 divide-y divide-secondary-200 dark:divide-secondary-700">
                      {auditLogs.map((log) => (
                        <tr key={log.id} className="hover:bg-secondary-50 dark:hover:bg-secondary-800">
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-secondary-900 dark:text-secondary-100">
                            {log.user_email}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-secondary-900 dark:text-secondary-100">
                            {log.action}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-secondary-600 dark:text-secondary-400">
                            {log.resource}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-secondary-600 dark:text-secondary-400">
                            {formatDate(log.created_at)}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-secondary-600 dark:text-secondary-400">
                            {log.ip_address}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {/* Pagination */}
              {(activeTab === 'users' || activeTab === 'audit') && totalPages > 1 && (
                <div className="flex items-center justify-between px-4 py-3 border-t border-secondary-200 dark:border-secondary-700">
                  <div className="flex-1 flex justify-between sm:hidden">
                    <Button
                      onClick={() => setCurrentPage(currentPage - 1)}
                      disabled={currentPage === 1}
                      variant="outline"
                    >
                      Previous
                    </Button>
                    <Button
                      onClick={() => setCurrentPage(currentPage + 1)}
                      disabled={currentPage === totalPages}
                      variant="outline"
                    >
                      Next
                    </Button>
                  </div>

                  <div className="hidden sm:flex-1 sm:flex sm:items-center sm:justify-between">
                    <div>
                      <p className="text-sm text-secondary-700 dark:text-secondary-300">
                        Showing{' '}
                        <span className="font-medium">{(currentPage - 1) * pageSize + 1}</span>
                        {' '}to{' '}
                        <span className="font-medium">
                          {Math.min(currentPage * pageSize, totalCount)}
                        </span>
                        {' '}of{' '}
                        <span className="font-medium">{totalCount}</span>
                        {' '}results
                      </p>
                    </div>
                    <div>
                      <nav className="relative z-0 inline-flex rounded-md shadow-sm -space-x-px">
                        <Button
                          onClick={() => setCurrentPage(currentPage - 1)}
                          disabled={currentPage === 1}
                          variant="outline"
                          className="rounded-r-none"
                        >
                          <ChevronLeft className="h-4 w-4" />
                        </Button>
                        
                        <Button
                          onClick={() => setCurrentPage(currentPage + 1)}
                          disabled={currentPage === totalPages}
                          variant="outline"
                          className="rounded-l-none"
                        >
                          <ChevronRight className="h-4 w-4" />
                        </Button>
                      </nav>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

export default function AdminPage() {
  return (
    <AuthGuard requireAuth={true} requiredRole="admin">
      <AdminPageContent />
    </AuthGuard>
  );
}