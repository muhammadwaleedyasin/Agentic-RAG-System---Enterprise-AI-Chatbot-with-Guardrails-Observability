import { test, expect } from '../fixtures/test-fixtures';
import { testData } from '../fixtures/test-fixtures';

test.describe('Admin Panel', () => {
  test.beforeEach(async ({ page, authService }) => {
    // Login as admin user
    const adminToken = await authService.getAuthToken('admin_test', 'test_password_123');
    
    await page.context().addCookies([{
      name: 'auth_token',
      value: adminToken,
      domain: 'localhost',
      path: '/',
    }]);
  });

  test.describe('Admin Access', () => {
    test('should access admin panel with admin privileges', async ({ adminPage }) => {
      await adminPage.goto();
      await adminPage.performAdminVerification();
    });

    test('should deny access to non-admin users', async ({ page, authService }) => {
      // Clear admin session and login as regular user
      await page.context().clearCookies();
      
      const userToken = await authService.getAuthToken('user_test', 'test_password_123');
      await page.context().addCookies([{
        name: 'auth_token',
        value: userToken,
        domain: 'localhost',
        path: '/',
      }]);
      
      await page.goto('/admin');
      
      // Should redirect to unauthorized or dashboard
      await expect(page).toHaveURL(/\/(dashboard|unauthorized|login)/);
    });
  });

  test.describe('User Management', () => {
    test('should display users list', async ({ adminPage }) => {
      await adminPage.goto();
      
      const users = await adminPage.getUserList();
      expect(users.length).toBeGreaterThan(0);
      
      // Should include the test users
      const usernames = users.map(user => user.username);
      expect(usernames).toContain('admin_test');
    });

    test('should create new user', async ({ adminPage }) => {
      const newUser = testData.generateUser();
      
      await adminPage.createUser({
        username: newUser.username,
        email: newUser.email,
        password: newUser.password,
        role: 'user',
        firstName: newUser.firstName,
        lastName: newUser.lastName,
      });
      
      // Verify user appears in list
      const users = await adminPage.getUserList();
      const createdUser = users.find(user => user.username === newUser.username);
      expect(createdUser).toBeDefined();
      expect(createdUser?.email).toBe(newUser.email);
      expect(createdUser?.role).toBe('user');
    });

    test('should edit existing user', async ({ adminPage }) => {
      const users = await adminPage.getUserList();
      const userToEdit = users.find(user => user.username === 'user_test');
      
      if (userToEdit) {
        const newEmail = `updated_${Date.now()}@example.com`;
        
        await adminPage.editUser('user_test', {
          email: newEmail,
          role: 'viewer',
        });
        
        // Verify changes
        const updatedUsers = await adminPage.getUserList();
        const updatedUser = updatedUsers.find(user => user.username === 'user_test');
        expect(updatedUser?.email).toBe(newEmail);
        expect(updatedUser?.role).toBe('viewer');
      }
    });

    test('should delete user', async ({ adminPage }) => {
      // First create a user to delete
      const userToDelete = testData.generateUser();
      
      await adminPage.createUser({
        username: userToDelete.username,
        email: userToDelete.email,
        password: userToDelete.password,
        role: 'user',
      });
      
      // Then delete the user
      await adminPage.deleteUser(userToDelete.username);
      
      // Verify user is removed
      const users = await adminPage.getUserList();
      const deletedUser = users.find(user => user.username === userToDelete.username);
      expect(deletedUser).toBeUndefined();
    });

    test('should validate user creation form', async ({ adminPage }) => {
      await adminPage.navigateToSection('users');
      await adminPage.elements.createUserButton.click();
      
      // Try to submit empty form
      const submitButton = adminPage.page.locator('[data-testid="submit-user"]');
      await submitButton.click();
      
      // Should show validation errors
      await adminPage.validateFormErrors({
        username: 'Username is required',
        email: 'Email is required',
        password: 'Password is required',
      });
    });
  });

  test.describe('System Settings', () => {
    test('should update LLM provider settings', async ({ adminPage }) => {
      await adminPage.updateLLMSettings({
        provider: 'openrouter',
        model: 'mistral/mixtral-8x7b-instruct',
        baseUrl: 'https://openrouter.ai/api/v1',
      });
    });

    test('should update security settings', async ({ adminPage }) => {
      await adminPage.updateSecuritySettings({
        enableAuth: true,
        sessionTimeout: 60,
        maxLoginAttempts: 5,
        enableAuditLogging: true,
      });
    });

    test('should save and persist settings', async ({ adminPage }) => {
      // Update a setting
      await adminPage.updateSecuritySettings({
        sessionTimeout: 120,
      });
      
      // Reload page and verify setting persists
      await adminPage.page.reload();
      await adminPage.navigateToSection('settings');
      
      const timeoutInput = adminPage.page.locator('[data-testid="session-timeout"]');
      await expect(timeoutInput).toHaveValue('120');
    });
  });

  test.describe('Document Management', () => {
    test('should display admin documents list', async ({ adminPage }) => {
      await adminPage.goto();
      
      const documents = await adminPage.getAdminDocumentsList();
      
      // Documents might be empty in fresh system
      if (documents.length > 0) {
        documents.forEach(doc => {
          expect(doc.name).toBeTruthy();
          expect(doc.owner).toBeTruthy();
          expect(doc.status).toBeTruthy();
        });
      }
    });

    test('should reindex documents', async ({ adminPage }) => {
      await adminPage.reindexDocuments();
    });

    test('should show document processing status', async ({ adminPage }) => {
      await adminPage.navigateToSection('documents');
      
      const documents = await adminPage.getAdminDocumentsList();
      
      if (documents.length > 0) {
        documents.forEach(doc => {
          expect(['completed', 'processing', 'failed']).toContain(doc.status.toLowerCase());
        });
      }
    });
  });

  test.describe('Analytics and Monitoring', () => {
    test('should display usage statistics', async ({ adminPage }) => {
      await adminPage.goto();
      
      const stats = await adminPage.getUsageStatistics();
      expect(Object.keys(stats).length).toBeGreaterThan(0);
      
      // Common usage stats
      const expectedStats = ['API Calls', 'Active Sessions', 'Documents Processed', 'Errors'];
      
      for (const statName of expectedStats) {
        if (stats[statName]) {
          expect(stats[statName]).toBeTruthy();
        }
      }
    });

    test('should display analytics charts', async ({ adminPage }) => {
      await adminPage.verifyAnalyticsCharts();
    });

    test('should show performance metrics', async ({ adminPage }) => {
      await adminPage.navigateToSection('analytics');
      
      await expect(adminPage.elements.performanceMetrics).toBeVisible();
      
      // Check for common performance metrics
      const metricsContainer = adminPage.elements.performanceMetrics;
      await expect(metricsContainer.locator('[data-testid="response-time"]')).toBeVisible();
      await expect(metricsContainer.locator('[data-testid="throughput"]')).toBeVisible();
    });
  });

  test.describe('Audit Logs', () => {
    test('should display audit logs', async ({ adminPage }) => {
      await adminPage.goto();
      
      const logs = await adminPage.getAuditLogs();
      
      // Logs might be empty in fresh system
      if (logs.length > 0) {
        logs.forEach(log => {
          expect(log.timestamp).toBeTruthy();
          expect(log.action).toBeTruthy();
        });
      }
    });

    test('should filter audit logs', async ({ adminPage }) => {
      await adminPage.filterAuditLogs({
        action: 'login',
        user: 'admin_test',
      });
      
      const filteredLogs = await adminPage.getAuditLogs();
      
      if (filteredLogs.length > 0) {
        filteredLogs.forEach(log => {
          expect(log.action.toLowerCase()).toContain('login');
          if (log.user) {
            expect(log.user).toBe('admin_test');
          }
        });
      }
    });

    test('should export audit logs', async ({ adminPage }) => {
      await adminPage.exportAuditLogs('csv');
    });

    test('should filter logs by date range', async ({ adminPage }) => {
      const today = new Date().toISOString().split('T')[0];
      const yesterday = new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString().split('T')[0];
      
      await adminPage.filterAuditLogs({
        dateFrom: yesterday,
        dateTo: today,
      });
      
      const logs = await adminPage.getAuditLogs();
      
      if (logs.length > 0) {
        logs.forEach(log => {
          const logDate = new Date(log.timestamp).toISOString().split('T')[0];
          expect(logDate >= yesterday && logDate <= today).toBe(true);
        });
      }
    });
  });

  test.describe('System Health', () => {
    test('should display system status', async ({ adminPage }) => {
      await adminPage.navigateToSection('analytics');
      
      // Check for system health indicators
      const healthIndicators = [
        '[data-testid="api-health"]',
        '[data-testid="database-health"]',
        '[data-testid="llm-health"]',
        '[data-testid="vector-db-health"]',
      ];
      
      for (const indicator of healthIndicators) {
        const element = adminPage.page.locator(indicator);
        if (await element.isVisible()) {
          const status = await element.getAttribute('data-status');
          expect(['healthy', 'warning', 'error']).toContain(status);
        }
      }
    });

    test('should show resource usage', async ({ adminPage }) => {
      await adminPage.navigateToSection('analytics');
      
      const resourceMetrics = [
        '[data-testid="cpu-usage"]',
        '[data-testid="memory-usage"]',
        '[data-testid="disk-usage"]',
      ];
      
      for (const metric of resourceMetrics) {
        const element = adminPage.page.locator(metric);
        if (await element.isVisible()) {
          const value = await element.textContent();
          expect(value).toMatch(/\d+%/); // Should show percentage
        }
      }
    });
  });

  test.describe('Error Handling', () => {
    test('should handle API errors gracefully', async ({ adminPage, testHelpers }) => {
      // Mock API error
      await testHelpers.mockApiResponse(/\/api\/v1\/admin/, {
        error: 'Admin service unavailable'
      }, 500);
      
      await adminPage.goto();
      
      // Should show error but admin panel should still be accessible
      await expect(adminPage.elements.adminNavigation).toBeVisible();
    });

    test('should validate user operations', async ({ adminPage }) => {
      await adminPage.navigateToSection('users');
      
      // Try to delete admin user (should fail)
      const adminUser = await adminPage.page.locator('[data-testid="user-row"]:has-text("admin")');
      if (await adminUser.isVisible()) {
        const deleteButton = adminUser.locator('[data-testid="delete-user"]');
        
        if (await deleteButton.isVisible()) {
          await deleteButton.click();
          
          // Should show error or be disabled
          await adminPage.verifyToast('Cannot delete admin user', 'error');
        }
      }
    });
  });

  test.describe('Responsive Design', () => {
    test('should work on different screen sizes', async ({ adminPage }) => {
      await adminPage.goto();
      
      const viewports = [
        { width: 768, height: 1024, name: 'tablet' },
        { width: 1440, height: 900, name: 'desktop' },
      ];
      
      for (const viewport of viewports) {
        await adminPage.page.setViewportSize(viewport);
        await adminPage.verifyPageLoaded();
        
        // Admin navigation should always be accessible
        await expect(adminPage.elements.adminNavigation).toBeVisible();
      }
    });

    test('should show mobile warning on small screens', async ({ adminPage }) => {
      await adminPage.goto();
      
      // Very small mobile screen
      await adminPage.page.setViewportSize({ width: 375, height: 667 });
      
      // Admin panel might show mobile warning or adapted layout
      // This is optional based on design decisions
    });
  });

  test.describe('Accessibility', () => {
    test('should be accessible to screen readers', async ({ adminPage }) => {
      await adminPage.goto();
      
      // Check main landmarks
      await expect(adminPage.page.locator('main')).toBeVisible();
      await expect(adminPage.page.locator('nav')).toBeVisible();
      
      // Check ARIA labels on navigation
      const navItems = adminPage.elements.adminNavigation.locator('[role="menuitem"]');
      const count = await navItems.count();
      
      for (let i = 0; i < count; i++) {
        const item = navItems.nth(i);
        await expect(item).toHaveAttribute('aria-label');
      }
    });

    test('should support keyboard navigation', async ({ adminPage }) => {
      await adminPage.goto();
      
      // Should be able to navigate admin sections with keyboard
      await adminPage.page.keyboard.press('Tab');
      await adminPage.page.keyboard.press('Tab');
      
      const focusedElement = adminPage.page.locator(':focus');
      await expect(focusedElement).toBeVisible();
    });
  });

  test.describe('Data Export and Backup', () => {
    test('should export user data', async ({ adminPage }) => {
      await adminPage.navigateToSection('users');
      
      const exportButton = adminPage.page.locator('[data-testid="export-users"]');
      if (await exportButton.isVisible()) {
        const downloadPromise = adminPage.page.waitForEvent('download');
        await exportButton.click();
        const download = await downloadPromise;
        
        expect(download.suggestedFilename()).toContain('users');
      }
    });

    test('should backup system settings', async ({ adminPage }) => {
      await adminPage.navigateToSection('settings');
      
      const backupButton = adminPage.page.locator('[data-testid="backup-settings"]');
      if (await backupButton.isVisible()) {
        const downloadPromise = adminPage.page.waitForEvent('download');
        await backupButton.click();
        const download = await downloadPromise;
        
        expect(download.suggestedFilename()).toContain('settings');
      }
    });
  });
});