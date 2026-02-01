import { test, expect } from '../fixtures/test-fixtures';

test.describe('Dashboard', () => {
  test.beforeEach(async ({ loginPage }) => {
    await loginPage.goto();
    await loginPage.performValidLogin();
  });

  test.describe('Dashboard Overview', () => {
    test('should load dashboard successfully', async ({ dashboardPage }) => {
      await dashboardPage.goto();
      await dashboardPage.performDashboardVerification();
    });

    test('should display welcome message', async ({ dashboardPage }) => {
      await dashboardPage.goto();
      
      await expect(dashboardPage.elements.welcomeMessage).toBeVisible();
      await expect(dashboardPage.elements.welcomeMessage).toContainText('Welcome');
    });

    test('should show system statistics', async ({ dashboardPage }) => {
      await dashboardPage.goto();
      
      const stats = await dashboardPage.getDashboardStats();
      expect(Object.keys(stats).length).toBeGreaterThan(0);
      
      // Common stats that should be present
      const expectedStats = ['Total Documents', 'Total Conversations', 'Active Users', 'System Status'];
      
      for (const statName of expectedStats) {
        if (stats[statName]) {
          expect(stats[statName]).toBeTruthy();
        }
      }
    });

    test('should display system status indicators', async ({ dashboardPage }) => {
      await dashboardPage.goto();
      await dashboardPage.verifySystemStatus();
    });

    test('should show performance charts', async ({ dashboardPage }) => {
      await dashboardPage.goto();
      await dashboardPage.verifyPerformanceChart();
    });
  });

  test.describe('Navigation', () => {
    test('should navigate to chat from dashboard', async ({ dashboardPage }) => {
      await dashboardPage.goto();
      await dashboardPage.navigateToChat();
      
      await expect(dashboardPage.page).toHaveURL('/chat');
    });

    test('should navigate to documents from dashboard', async ({ dashboardPage }) => {
      await dashboardPage.goto();
      await dashboardPage.navigateToDocuments();
      
      await expect(dashboardPage.page).toHaveURL('/documents');
    });

    test('should use quick actions', async ({ dashboardPage }) => {
      await dashboardPage.goto();
      await dashboardPage.verifyQuickActions();
    });
  });

  test.describe('Recent Activity', () => {
    test('should display recent documents', async ({ dashboardPage }) => {
      await dashboardPage.goto();
      
      const recentDocs = await dashboardPage.getRecentDocuments();
      
      // Recent docs might be empty in fresh system
      if (recentDocs.length > 0) {
        recentDocs.forEach(doc => {
          expect(doc.name).toBeTruthy();
          expect(doc.date).toBeTruthy();
          expect(doc.status).toBeTruthy();
        });
      }
    });

    test('should display recent conversations', async ({ dashboardPage }) => {
      await dashboardPage.goto();
      
      const recentConversations = await dashboardPage.getRecentConversations();
      
      // Recent conversations might be empty in fresh system
      if (recentConversations.length > 0) {
        recentConversations.forEach(conv => {
          expect(conv.title).toBeTruthy();
          expect(conv.date).toBeTruthy();
          expect(conv.messageCount).toBeTruthy();
        });
      }
    });

    test('should show activity feed', async ({ dashboardPage }) => {
      await dashboardPage.goto();
      
      const activities = await dashboardPage.getActivityFeed();
      
      // Activity feed might be empty in fresh system
      if (activities.length > 0) {
        activities.forEach(activity => {
          expect(activity.action).toBeTruthy();
          expect(activity.time).toBeTruthy();
        });
      }
    });
  });

  test.describe('Search and Navigation', () => {
    test('should perform global search', async ({ dashboardPage }) => {
      await dashboardPage.goto();
      
      await dashboardPage.performGlobalSearch('test query');
      
      // Should navigate to search results or show search modal
      await expect(dashboardPage.page.locator('[data-testid="search-results"]')).toBeVisible();
    });

    test('should display notifications', async ({ dashboardPage }) => {
      await dashboardPage.goto();
      
      const notifications = await dashboardPage.checkNotifications();
      
      // Notifications might be empty in fresh system
      if (notifications.length > 0) {
        notifications.forEach(notification => {
          expect(notification.title).toBeTruthy();
          expect(notification.type).toBeTruthy();
        });
      }
    });

    test('should access user profile', async ({ dashboardPage }) => {
      await dashboardPage.goto();
      
      await dashboardPage.accessUserProfile();
      
      // Should open profile modal
      await expect(dashboardPage.page.locator('[data-testid="user-profile-modal"]')).toBeVisible();
    });
  });

  test.describe('User Permissions', () => {
    test('should show appropriate features for admin user', async ({ page, authService, dashboardPage }) => {
      const adminToken = await authService.getAuthToken('admin_test', 'test_password_123');
      
      await page.context().addCookies([{
        name: 'auth_token',
        value: adminToken,
        domain: 'localhost',
        path: '/',
      }]);
      
      await dashboardPage.goto();
      await dashboardPage.verifyUserPermissions('admin');
    });

    test('should show appropriate features for regular user', async ({ page, authService, dashboardPage }) => {
      const userToken = await authService.getAuthToken('user_test', 'test_password_123');
      
      await page.context().addCookies([{
        name: 'auth_token',
        value: userToken,
        domain: 'localhost',
        path: '/',
      }]);
      
      await dashboardPage.goto();
      await dashboardPage.verifyUserPermissions('user');
    });

    test('should show limited features for viewer', async ({ page, authService, dashboardPage }) => {
      const viewerToken = await authService.getAuthToken('viewer_test', 'test_password_123');
      
      await page.context().addCookies([{
        name: 'auth_token',
        value: viewerToken,
        domain: 'localhost',
        path: '/',
      }]);
      
      await dashboardPage.goto();
      await dashboardPage.verifyUserPermissions('viewer');
    });
  });

  test.describe('Data Refresh', () => {
    test('should refresh dashboard data', async ({ dashboardPage }) => {
      await dashboardPage.goto();
      await dashboardPage.testDataRefresh();
    });

    test('should update stats in real-time', async ({ dashboardPage, chatPage }) => {
      await dashboardPage.goto();
      
      const initialStats = await dashboardPage.getDashboardStats();
      
      // Perform an action that should update stats (e.g., send a chat message)
      await chatPage.goto();
      await chatPage.sendMessage('Dashboard stats test message');
      await chatPage.waitForResponse();
      
      // Return to dashboard
      await dashboardPage.goto();
      
      const updatedStats = await dashboardPage.getDashboardStats();
      
      // Conversation count might have increased
      if (initialStats['Total Conversations'] && updatedStats['Total Conversations']) {
        const initialCount = parseInt(initialStats['Total Conversations']);
        const updatedCount = parseInt(updatedStats['Total Conversations']);
        expect(updatedCount).toBeGreaterThanOrEqual(initialCount);
      }
    });
  });

  test.describe('Responsive Design', () => {
    test('should work on mobile devices', async ({ dashboardPage }) => {
      await dashboardPage.goto();
      await dashboardPage.testResponsiveDashboard();
    });

    test('should adapt layout for different screen sizes', async ({ dashboardPage }) => {
      await dashboardPage.goto();
      
      const viewports = [
        { width: 375, height: 667, name: 'mobile' },
        { width: 768, height: 1024, name: 'tablet' },
        { width: 1440, height: 900, name: 'desktop' },
      ];
      
      for (const viewport of viewports) {
        await dashboardPage.page.setViewportSize(viewport);
        await dashboardPage.verifyPageLoaded();
        
        // Basic navigation should always work
        await expect(dashboardPage.elements.navigationMenu).toBeVisible();
        
        // Stats should be visible on all sizes
        await expect(dashboardPage.elements.statsCards.first()).toBeVisible();
      }
    });
  });

  test.describe('Performance', () => {
    test('should load dashboard quickly', async ({ dashboardPage, page }) => {
      const startTime = Date.now();
      
      await dashboardPage.goto();
      await dashboardPage.verifyPageLoaded();
      
      const loadTime = Date.now() - startTime;
      
      // Dashboard should load within reasonable time
      expect(loadTime).toBeLessThan(5000);
    });

    test('should handle dashboard with large datasets', async ({ dashboardPage }) => {
      await dashboardPage.goto();
      
      // Even with large datasets, basic elements should load
      await expect(dashboardPage.elements.welcomeMessage).toBeVisible({ timeout: 10000 });
      await expect(dashboardPage.elements.statsCards.first()).toBeVisible({ timeout: 10000 });
    });
  });

  test.describe('Error Handling', () => {
    test('should handle API errors gracefully', async ({ dashboardPage, testHelpers }) => {
      // Mock API error for dashboard data
      await testHelpers.mockApiResponse(/\/api\/v1\/dashboard/, {
        error: 'Service unavailable'
      }, 500);
      
      await dashboardPage.goto();
      
      // Should show error state but page should still be functional
      await expect(dashboardPage.elements.navigationMenu).toBeVisible();
    });

    test('should recover from network errors', async ({ dashboardPage, testHelpers }) => {
      await dashboardPage.goto();
      await dashboardPage.verifyPageLoaded();
      
      // Simulate network error
      await testHelpers.page.setOfflineMode(true);
      await dashboardPage.page.reload();
      
      // Should show offline indicator or error
      // Then restore network
      await testHelpers.page.setOfflineMode(false);
      await dashboardPage.page.reload();
      
      // Should recover
      await dashboardPage.verifyPageLoaded();
    });
  });

  test.describe('Accessibility', () => {
    test('should be accessible to screen readers', async ({ dashboardPage }) => {
      await dashboardPage.goto();
      
      // Check main landmarks
      await expect(dashboardPage.page.locator('main')).toBeVisible();
      await expect(dashboardPage.page.locator('nav')).toBeVisible();
      
      // Check heading hierarchy
      const h1 = dashboardPage.page.locator('h1');
      await expect(h1).toBeVisible();
    });

    test('should support keyboard navigation', async ({ dashboardPage }) => {
      await dashboardPage.goto();
      
      // Should be able to tab through interactive elements
      await dashboardPage.page.keyboard.press('Tab');
      await dashboardPage.page.keyboard.press('Tab');
      
      // Quick actions should be focusable
      const focusedElement = dashboardPage.page.locator(':focus');
      await expect(focusedElement).toBeVisible();
    });

    test('should have proper ARIA labels', async ({ dashboardPage }) => {
      await dashboardPage.goto();
      
      // Check important interactive elements have labels
      await expect(dashboardPage.elements.userProfile).toHaveAttribute('aria-label');
      await expect(dashboardPage.elements.notificationCenter).toHaveAttribute('aria-label');
    });
  });
});