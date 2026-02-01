'use client';

import { useState, useEffect } from 'react';
import { useAuth } from '@/hooks/useAuth';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { AuthGuard } from '@/components/auth/auth-guard';
import { apiClient } from '@/lib/api';
import { AnalyticsData, SystemHealth } from '@/types/api';
import {
  MessageSquare,
  FileText,
  Users,
  Activity,
  Loader2,
  RefreshCw,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  Server
} from 'lucide-react';
import clsx from 'clsx';

function DashboardPageContent() {
  const { user } = useAuth();
  const [analytics, setAnalytics] = useState<AnalyticsData | null>(null);
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      const [analyticsData, healthData] = await Promise.all([
        apiClient.getAnalytics(),
        apiClient.getSystemHealth(),
      ]);
      
      setAnalytics(analyticsData as AnalyticsData);
      setHealth(healthData as SystemHealth);
    } catch (err) {
      console.error('Failed to load dashboard data:', err);
      setError(err instanceof Error ? err.message : 'Failed to load dashboard data');
    } finally {
      setIsLoading(false);
    }
  };

  const getHealthStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'text-success-600 dark:text-success-400';
      case 'degraded':
        return 'text-warning-600 dark:text-warning-400';
      case 'unhealthy':
        return 'text-error-600 dark:text-error-400';
      default:
        return 'text-secondary-600 dark:text-secondary-400';
    }
  };

  const getHealthIcon = (status: string) => {
    switch (status) {
      case 'healthy':
        return <CheckCircle className="h-4 w-4" />;
      case 'degraded':
        return <AlertTriangle className="h-4 w-4" />;
      case 'unhealthy':
        return <AlertTriangle className="h-4 w-4" />;
      default:
        return <Activity className="h-4 w-4" />;
    }
  };

  const formatUptime = (seconds: number) => {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    
    if (days > 0) {
      return `${days}d ${hours}h ${minutes}m`;
    } else if (hours > 0) {
      return `${hours}h ${minutes}m`;
    } else {
      return `${minutes}m`;
    }
  };

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };


  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-secondary-900 dark:text-secondary-100">
            Dashboard
          </h1>
          <p className="text-secondary-600 dark:text-secondary-400 mt-1">
            Welcome back, {user?.full_name || user?.username}
          </p>
        </div>
        <Button
          onClick={loadDashboardData}
          variant="outline"
          disabled={isLoading}
          className="flex items-center gap-2"
        >
          <RefreshCw className={clsx('h-4 w-4', isLoading && 'animate-spin')} />
          Refresh
        </Button>
      </div>

      {/* Error Alert */}
      {error && (
        <div className="bg-error-50 dark:bg-error-900/20 border border-error-200 dark:border-error-800 rounded-md p-4">
          <div className="flex">
            <AlertTriangle className="h-5 w-5 text-error-600 dark:text-error-400 mt-0.5" />
            <div className="ml-3">
              <h3 className="text-sm font-medium text-error-800 dark:text-error-200">
                Failed to load dashboard data
              </h3>
              <div className="mt-2 text-sm text-error-700 dark:text-error-300">
                {error}
              </div>
              <div className="mt-3">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={loadDashboardData}
                >
                  Try Again
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Loading State */}
      {isLoading ? (
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4 text-primary-600" />
            <p className="text-secondary-600 dark:text-secondary-400">
              Loading dashboard data...
            </p>
          </div>
        </div>
      ) : (
        <>
          {/* Analytics Cards */}
          {analytics && (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Total Conversations</CardTitle>
                  <MessageSquare className="h-4 w-4 text-secondary-600" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{analytics.conversations.total}</div>
                  <p className="text-xs text-secondary-600 dark:text-secondary-400">
                    {analytics.conversations.today} today
                    {analytics.conversations.week_growth > 0 && (
                      <span className="text-success-600 dark:text-success-400 ml-1">
                        (+{analytics.conversations.week_growth}% this week)
                      </span>
                    )}
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Messages</CardTitle>
                  <TrendingUp className="h-4 w-4 text-secondary-600" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{analytics.messages.total}</div>
                  <p className="text-xs text-secondary-600 dark:text-secondary-400">
                    {analytics.messages.today} today • Avg {analytics.messages.avg_per_conversation}/conversation
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Documents</CardTitle>
                  <FileText className="h-4 w-4 text-secondary-600" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{analytics.documents.total}</div>
                  <p className="text-xs text-secondary-600 dark:text-secondary-400">
                    {formatBytes(analytics.documents.total_size)} total
                    {analytics.documents.processing_queue > 0 && (
                      <span className="text-warning-600 dark:text-warning-400 ml-1">
                        • {analytics.documents.processing_queue} processing
                      </span>
                    )}
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Active Users</CardTitle>
                  <Users className="h-4 w-4 text-secondary-600" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{analytics.users.total}</div>
                  <p className="text-xs text-secondary-600 dark:text-secondary-400">
                    {analytics.users.active_today} today • {analytics.users.active_week} this week
                  </p>
                </CardContent>
              </Card>
            </div>
          )}

          {/* System Health */}
          {health && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Overall Health */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Server className="h-5 w-5" />
                    System Health
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">Overall Status</span>
                    <div className={clsx('flex items-center gap-2', getHealthStatusColor(health.status))}>
                      {getHealthIcon(health.status)}
                      <span className="capitalize">{health.status}</span>
                    </div>
                  </div>

                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">Version</span>
                    <span className="text-sm text-secondary-600 dark:text-secondary-400">
                      {health.version}
                    </span>
                  </div>

                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">Uptime</span>
                    <span className="text-sm text-secondary-600 dark:text-secondary-400">
                      {formatUptime(health.uptime)}
                    </span>
                  </div>

                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Memory Usage</span>
                      <span>{health.metrics.memory_usage}%</span>
                    </div>
                    <div className="w-full bg-secondary-200 dark:bg-secondary-700 rounded-full h-2">
                      <div
                        className={clsx(
                          'h-2 rounded-full',
                          health.metrics.memory_usage > 80 ? 'bg-error-500' :
                          health.metrics.memory_usage > 60 ? 'bg-warning-500' : 'bg-success-500'
                        )}
                        style={{ width: `${health.metrics.memory_usage}%` }}
                      />
                    </div>
                  </div>

                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>CPU Usage</span>
                      <span>{health.metrics.cpu_usage}%</span>
                    </div>
                    <div className="w-full bg-secondary-200 dark:bg-secondary-700 rounded-full h-2">
                      <div
                        className={clsx(
                          'h-2 rounded-full',
                          health.metrics.cpu_usage > 80 ? 'bg-error-500' :
                          health.metrics.cpu_usage > 60 ? 'bg-warning-500' : 'bg-success-500'
                        )}
                        style={{ width: `${health.metrics.cpu_usage}%` }}
                      />
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Services Status */}
              <Card>
                <CardHeader>
                  <CardTitle>Services</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {Object.entries(health.services).map(([service, status]) => (
                      <div key={service} className="flex items-center justify-between">
                        <span className="text-sm font-medium capitalize">
                          {service.replace('_', ' ')}
                        </span>
                        <div className={clsx('flex items-center gap-2', getHealthStatusColor(status.status))}>
                          {getHealthIcon(status.status)}
                          <span className="text-sm capitalize">{status.status}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default function DashboardPage() {
  return (
    <AuthGuard requireAuth={true}>
      <DashboardPageContent />
    </AuthGuard>
  );
}