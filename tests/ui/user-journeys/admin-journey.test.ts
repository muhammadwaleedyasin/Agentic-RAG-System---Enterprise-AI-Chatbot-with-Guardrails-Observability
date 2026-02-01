import { test, expect } from '@playwright/test';

/**
 * Admin User Journey Tests
 * Tests admin workflows: login → user management → system stats → configuration
 */

test.describe('Admin User Journey', () => {
  test('should complete comprehensive admin workflow', async ({ page }) => {
    // Step 1: Admin Login
    await page.goto('/login');
    await page.fill('[data-testid="username-input"]', 'admin_test');
    await page.fill('[data-testid="password-input"]', 'AdminTest123!');
    await page.click('[data-testid="login-button"]');
    
    await expect(page).toHaveURL('/dashboard');
    
    // Step 2: Review Admin Dashboard
    // Should show admin-specific widgets
    await expect(page.locator('[data-testid="admin-dashboard"]')).toBeVisible();
    await expect(page.locator('[data-testid="system-health"]')).toBeVisible();
    await expect(page.locator('[data-testid="user-activity"]')).toBeVisible();
    await expect(page.locator('[data-testid="recent-alerts"]')).toBeVisible();
    
    // Check system health indicators
    await expect(page.locator('[data-testid="api-status"]')).toContainText('Healthy');
    await expect(page.locator('[data-testid="database-status"]')).toContainText('Connected');
    await expect(page.locator('[data-testid="vector-db-status"]')).toContainText('Online');
    
    // Review alerts if any
    const alertsCount = await page.locator('[data-testid="alert-item"]').count();
    if (alertsCount > 0) {
      await page.click('[data-testid="view-all-alerts"]');
      await expect(page.locator('[data-testid="alerts-panel"]')).toBeVisible();
      
      // Address critical alerts
      const criticalAlerts = page.locator('[data-testid="critical-alert"]');
      const criticalCount = await criticalAlerts.count();
      
      for (let i = 0; i < criticalCount; i++) {
        await criticalAlerts.nth(i).click();
        await expect(page.locator('[data-testid="alert-details"]')).toBeVisible();
        
        // Take action if needed
        if (await page.locator('[data-testid="resolve-alert"]').isVisible()) {
          await page.click('[data-testid="resolve-alert"]');
        }
      }
    }
    
    // Step 3: User Management Tasks
    await page.goto('/admin/users');
    
    // Review user activity
    await expect(page.locator('[data-testid="user-list"]')).toBeVisible();
    await expect(page.locator('[data-testid="active-users"]')).toBeVisible();
    
    // Check for users needing approval
    const pendingUsers = page.locator('[data-testid="pending-user"]');
    const pendingCount = await pendingUsers.count();
    
    if (pendingCount > 0) {
      // Approve pending users
      await pendingUsers.first().click();
      await expect(page.locator('[data-testid="user-details"]')).toBeVisible();
      
      // Review user information
      await expect(page.locator('[data-testid="user-email"]')).toBeVisible();
      await expect(page.locator('[data-testid="user-role-request"]')).toBeVisible();
      
      // Approve user
      await page.click('[data-testid="approve-user"]');
      await expect(page.locator('[data-testid="user-approved"]')).toBeVisible();
    }
    
    // Create new user account
    await page.click('[data-testid="create-user"]');
    await expect(page.locator('[data-testid="create-user-form"]')).toBeVisible();
    
    await page.fill('[data-testid="new-username"]', 'testuser_' + Date.now());
    await page.fill('[data-testid="new-email"]', 'testuser@company.com');
    await page.fill('[data-testid="new-password"]', 'TempPassword123!');
    await page.selectOption('[data-testid="new-user-role"]', 'user');
    
    await page.click('[data-testid="create-user-button"]');
    await expect(page.locator('[data-testid="user-created"]')).toBeVisible();
    
    // Modify user permissions
    const userList = page.locator('[data-testid="user-row"]');
    if (await userList.count() > 0) {
      await userList.first().click();
      await page.click('[data-testid="edit-permissions"]');
      
      // Update permissions
      await page.check('[data-testid="permission-upload"]');
      await page.check('[data-testid="permission-admin-read"]');
      await page.click('[data-testid="save-permissions"]');
      
      await expect(page.locator('[data-testid="permissions-updated"]')).toBeVisible();
    }
    
    // Step 4: System Statistics Review
    await page.goto('/admin/analytics');
    
    // Review comprehensive system metrics
    await expect(page.locator('[data-testid="analytics-dashboard"]')).toBeVisible();
    
    // Usage statistics
    await expect(page.locator('[data-testid="total-queries"]')).toBeVisible();
    await expect(page.locator('[data-testid="total-documents"]')).toBeVisible();
    await expect(page.locator('[data-testid="active-users-count"]')).toBeVisible();
    
    // Performance metrics
    await expect(page.locator('[data-testid="avg-response-time"]')).toBeVisible();
    await expect(page.locator('[data-testid="system-uptime"]')).toBeVisible();
    await expect(page.locator('[data-testid="error-rate"]')).toBeVisible();
    
    // Resource usage
    await expect(page.locator('[data-testid="cpu-usage-chart"]')).toBeVisible();
    await expect(page.locator('[data-testid="memory-usage-chart"]')).toBeVisible();
    await expect(page.locator('[data-testid="storage-usage"]')).toBeVisible();
    
    // Generate custom report
    await page.click('[data-testid="generate-report"]');
    await page.selectOption('[data-testid="report-type"]', 'monthly-summary');
    await page.fill('[data-testid="report-start-date"]', '2024-01-01');
    await page.fill('[data-testid="report-end-date"]', '2024-01-31');
    
    const downloadPromise = page.waitForEvent('download');
    await page.click('[data-testid="download-report"]');
    const download = await downloadPromise;
    
    expect(download.suggestedFilename()).toContain('monthly-summary');
    
    // Step 5: System Configuration
    await page.goto('/admin/settings');
    
    // General system settings
    await expect(page.locator('[data-testid="system-settings"]')).toBeVisible();
    
    // Update system limits
    await page.fill('[data-testid="max-upload-size"]', '100');
    await page.fill('[data-testid="max-query-length"]', '5000');
    await page.fill('[data-testid="session-timeout"]', '60');
    
    // Configure rate limiting
    await page.fill('[data-testid="rate-limit-queries"]', '100');
    await page.fill('[data-testid="rate-limit-uploads"]', '10');
    
    // Save general settings
    await page.click('[data-testid="save-general-settings"]');
    await expect(page.locator('[data-testid="settings-saved"]')).toBeVisible();
    
    // Step 6: LLM Provider Configuration
    await page.click('[data-testid="llm-settings-tab"]');
    
    // Configure primary LLM provider
    await page.selectOption('[data-testid="primary-provider"]', 'openai');
    await page.fill('[data-testid="openai-api-key"]', 'sk-test-key');
    await page.selectOption('[data-testid="openai-model"]', 'gpt-4');
    
    // Configure fallback provider
    await page.selectOption('[data-testid="fallback-provider"]', 'anthropic');
    await page.fill('[data-testid="anthropic-api-key"]', 'sk-ant-test-key');
    
    // Test provider connections
    await page.click('[data-testid="test-providers"]');
    await expect(page.locator('[data-testid="provider-test-results"]')).toBeVisible();
    
    // Save LLM settings
    await page.click('[data-testid="save-llm-settings"]');
    await expect(page.locator('[data-testid="llm-settings-saved"]')).toBeVisible();
    
    // Step 7: Security Configuration
    await page.click('[data-testid="security-settings-tab"]');
    
    // Configure authentication settings
    await page.check('[data-testid="enable-2fa"]');
    await page.fill('[data-testid="password-min-length"]', '12');
    await page.check('[data-testid="require-special-chars"]');
    
    // Configure session security
    await page.check('[data-testid="secure-cookies"]');
    await page.fill('[data-testid="session-encryption-key"]', 'secure-encryption-key-12345');
    
    // Configure IP restrictions
    await page.fill('[data-testid="allowed-ip-ranges"]', '192.168.1.0/24\n10.0.0.0/8');
    
    // Save security settings
    await page.click('[data-testid="save-security-settings"]');
    await expect(page.locator('[data-testid="security-settings-saved"]')).toBeVisible();
    
    // Step 8: Backup and Maintenance
    await page.goto('/admin/maintenance');
    
    // Create system backup
    await page.click('[data-testid="create-backup"]');
    await page.selectOption('[data-testid="backup-type"]', 'full');
    await page.fill('[data-testid="backup-name"]', 'admin-backup-' + new Date().toISOString().split('T')[0]);
    
    await page.click('[data-testid="start-backup"]');
    await expect(page.locator('[data-testid="backup-started"]')).toBeVisible();
    
    // Monitor backup progress
    await expect(page.locator('[data-testid="backup-progress"]')).toBeVisible();
    
    // Schedule automatic backups
    await page.click('[data-testid="backup-schedule"]');
    await page.selectOption('[data-testid="backup-frequency"]', 'daily');
    await page.fill('[data-testid="backup-time"]', '02:00');
    await page.check('[data-testid="enable-backup-schedule"]');
    
    await page.click('[data-testid="save-backup-schedule"]');
    await expect(page.locator('[data-testid="schedule-saved"]')).toBeVisible();
    
    // Step 9: Monitor Real-time System Status
    await page.goto('/admin/monitoring');
    
    // Real-time system monitoring
    await expect(page.locator('[data-testid="real-time-monitoring"]')).toBeVisible();
    await expect(page.locator('[data-testid="active-connections"]')).toBeVisible();
    await expect(page.locator('[data-testid="current-load"]')).toBeVisible();
    
    // Check logs
    await page.click('[data-testid="system-logs"]');
    await expect(page.locator('[data-testid="log-entries"]')).toBeVisible();
    
    // Filter for errors
    await page.selectOption('[data-testid="log-level-filter"]', 'ERROR');
    
    const errorLogs = page.locator('[data-testid="error-log-entry"]');
    const errorCount = await errorLogs.count();
    
    if (errorCount > 0) {
      // Investigate first error
      await errorLogs.first().click();
      await expect(page.locator('[data-testid="error-details"]')).toBeVisible();
      
      // Mark as investigated
      await page.click('[data-testid="mark-investigated"]');
    }
    
    // Step 10: Generate Admin Summary Report
    await page.goto('/admin/reports');
    
    // Create comprehensive admin report
    await page.click('[data-testid="create-admin-report"]');
    await page.check('[data-testid="include-user-stats"]');
    await page.check('[data-testid="include-system-health"]');
    await page.check('[data-testid="include-security-events"]');
    await page.check('[data-testid="include-performance-metrics"]');
    
    const reportDownloadPromise = page.waitForEvent('download');
    await page.click('[data-testid="generate-admin-report"]');
    const reportDownload = await reportDownloadPromise;
    
    expect(reportDownload.suggestedFilename()).toContain('admin-report');
    
    // Step 11: Review and Respond to Support Tickets
    await page.goto('/admin/support');
    
    // Check open support tickets
    const openTickets = page.locator('[data-testid="open-ticket"]');
    const ticketCount = await openTickets.count();
    
    if (ticketCount > 0) {
      // Address highest priority ticket
      await page.click('[data-testid="sort-by-priority"]');
      await openTickets.first().click();
      
      await expect(page.locator('[data-testid="ticket-details"]')).toBeVisible();
      
      // Respond to ticket
      await page.fill('[data-testid="ticket-response"]', 'Thank you for your inquiry. This issue has been resolved by updating the system configuration.');
      await page.selectOption('[data-testid="ticket-status"]', 'resolved');
      
      await page.click('[data-testid="update-ticket"]');
      await expect(page.locator('[data-testid="ticket-updated"]')).toBeVisible();
    }
    
    // Step 12: Plan System Updates
    await page.goto('/admin/updates');
    
    // Check for available updates
    await page.click('[data-testid="check-updates"]');
    await expect(page.locator('[data-testid="update-status"]')).toBeVisible();
    
    const updatesAvailable = await page.locator('[data-testid="available-update"]').count();
    
    if (updatesAvailable > 0) {
      // Schedule update for maintenance window
      await page.click('[data-testid="schedule-update"]');
      await page.fill('[data-testid="update-date"]', new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString().split('T')[0]);
      await page.fill('[data-testid="update-time"]', '02:00');
      
      await page.click('[data-testid="confirm-update-schedule"]');
      await expect(page.locator('[data-testid="update-scheduled"]')).toBeVisible();
    }
    
    // Step 13: Final System Check and Logout
    await page.goto('/admin/dashboard');
    
    // Final verification of system health
    await expect(page.locator('[data-testid="system-status"]')).toContainText('All Systems Operational');
    
    // Document session in admin log
    await page.click('[data-testid="admin-log"]');
    await page.fill('[data-testid="log-entry"]', 'Completed daily admin review - all systems healthy');
    await page.selectOption('[data-testid="log-category"]', 'routine-maintenance');
    await page.click('[data-testid="save-log-entry"]');
    
    // Secure logout
    await page.click('[data-testid="user-menu"]');
    await page.click('[data-testid="secure-logout"]');
    
    // Confirm secure logout (clears all admin session data)
    await page.click('[data-testid="confirm-secure-logout"]');
    
    await expect(page).toHaveURL('/login');
    await expect(page.locator('[data-testid="secure-logout-confirmation"]')).toBeVisible();
  });

  test('should handle emergency admin scenarios', async ({ page }) => {
    // Login as admin
    await page.goto('/login');
    await page.fill('[data-testid="username-input"]', 'admin_test');
    await page.fill('[data-testid="password-input"]', 'AdminTest123!');
    await page.click('[data-testid="login-button"]');
    
    // Simulate critical system alert
    await page.goto('/admin/monitoring');
    
    // Check for critical alerts
    const criticalAlert = page.locator('[data-testid="critical-system-alert"]');
    
    if (await criticalAlert.isVisible()) {
      // Emergency response protocol
      await criticalAlert.click();
      await expect(page.locator('[data-testid="emergency-response-panel"]')).toBeVisible();
      
      // Activate emergency mode
      await page.click('[data-testid="activate-emergency-mode"]');
      await page.fill('[data-testid="emergency-reason"]', 'Critical system alert - activating emergency protocols');
      await page.click('[data-testid="confirm-emergency-mode"]');
      
      // Emergency actions
      await expect(page.locator('[data-testid="emergency-mode-active"]')).toBeVisible();
      
      // Disable non-essential services
      await page.click('[data-testid="disable-non-essential"]');
      
      // Send emergency notifications
      await page.click('[data-testid="send-emergency-notifications"]');
      
      // Enable maintenance mode
      await page.click('[data-testid="enable-maintenance-mode"]');
      
      await expect(page.locator('[data-testid="emergency-actions-completed"]')).toBeVisible();
    }
  });

  test('should handle bulk user operations', async ({ page }) => {
    // Login as admin
    await page.goto('/login');
    await page.fill('[data-testid="username-input"]', 'admin_test');
    await page.fill('[data-testid="password-input"]', 'AdminTest123!');
    await page.click('[data-testid="login-button"]');
    
    await page.goto('/admin/users');
    
    // Bulk user import
    await page.click('[data-testid="bulk-import"]');
    
    // Upload CSV file with user data
    const csvFile = path.join(__dirname, '../fixtures/bulk-users.csv');
    await page.setInputFiles('[data-testid="import-file"]', csvFile);
    
    // Configure import settings
    await page.check('[data-testid="send-welcome-emails"]');
    await page.selectOption('[data-testid="default-role"]', 'user');
    
    await page.click('[data-testid="start-import"]');
    
    // Monitor import progress
    await expect(page.locator('[data-testid="import-progress"]')).toBeVisible();
    await expect(page.locator('[data-testid="import-completed"]')).toBeVisible();
    
    // Bulk user operations
    await page.check('[data-testid="select-all-users"]');
    
    // Bulk role update
    await page.click('[data-testid="bulk-actions"]');
    await page.selectOption('[data-testid="bulk-action-type"]', 'update-role');
    await page.selectOption('[data-testid="new-role"]', 'advanced-user');
    
    await page.click('[data-testid="apply-bulk-action"]');
    await expect(page.locator('[data-testid="bulk-update-completed"]')).toBeVisible();
  });
});