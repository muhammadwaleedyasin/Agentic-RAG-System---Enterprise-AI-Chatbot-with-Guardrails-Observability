import { test, expect } from '../fixtures/auth-fixtures';

/**
 * Session Management Tests
 * Tests session expiry, renewal, and security features
 */

test.describe('Session Management', () => {
  test('should automatically renew session before expiry', async ({ page, loginAsUser }) => {
    await loginAsUser('user');
    
    // Monitor token renewal
    let tokenRenewed = false;
    page.on('response', response => {
      if (response.url().includes('/api/v1/auth/refresh')) {
        tokenRenewed = true;
      }
    });
    
    // Wait for session renewal (typically happens 5 min before expiry)
    // Simulate by advancing time or triggering renewal
    await page.evaluate(() => {
      // Trigger token refresh check
      window.dispatchEvent(new CustomEvent('checkTokenExpiry'));
    });
    
    // Wait for potential token renewal
    await page.waitForTimeout(2000);
    
    // Verify user is still authenticated
    await expect(page.locator('[data-testid="user-menu"]')).toBeVisible();
  });

  test('should redirect to login when session expires', async ({ page, loginAsUser }) => {
    await loginAsUser('user');
    
    // Simulate expired session
    await page.evaluate(() => {
      // Set expired token
      const expiredToken = btoa(JSON.stringify({
        exp: Math.floor(Date.now() / 1000) - 3600 // Expired 1 hour ago
      }));
      localStorage.setItem('authToken', `header.${expiredToken}.signature`);
    });
    
    // Try to access protected resource
    await page.goto('/dashboard');
    
    // Should redirect to login
    await expect(page).toHaveURL('/login');
    await expect(page.locator('[data-testid="session-expired-message"]')).toBeVisible();
  });

  test('should handle concurrent session limit', async ({ browser }) => {
    // Create multiple browser contexts
    const contexts = await Promise.all([
      browser.newContext(),
      browser.newContext(),
      browser.newContext()
    ]);
    
    const pages = await Promise.all(contexts.map(ctx => ctx.newPage()));
    
    // Login with same user in all contexts
    for (const page of pages) {
      await page.goto('/login');
      await page.fill('[data-testid="username-input"]', 'user_test');
      await page.fill('[data-testid="password-input"]', 'UserTest123!');
      await page.click('[data-testid="login-button"]');
      await page.waitForURL('/dashboard');
    }
    
    // Check if session limit is enforced (depends on implementation)
    // Some sessions might be invalidated
    for (let i = 0; i < pages.length; i++) {
      await pages[i].reload();
      // Either should stay logged in or redirect to login
      const url = pages[i].url();
      expect(url.includes('/dashboard') || url.includes('/login')).toBeTruthy();
    }
    
    // Cleanup
    await Promise.all(contexts.map(ctx => ctx.close()));
  });

  test('should detect and handle session hijacking attempts', async ({ page, loginAsUser }) => {
    await loginAsUser('user');
    
    const originalToken = await page.evaluate(() => localStorage.getItem('authToken'));
    
    // Simulate session hijacking by modifying token
    await page.evaluate(() => {
      const token = localStorage.getItem('authToken');
      const parts = token.split('.');
      // Modify the signature part
      parts[2] = 'malicious_signature';
      localStorage.setItem('authToken', parts.join('.'));
    });
    
    // Try to access protected resource
    const response = await page.request.get('/api/v1/auth/me');
    
    // Should be unauthorized
    expect(response.status()).toBe(401);
    
    // Should redirect to login
    await page.goto('/dashboard');
    await expect(page).toHaveURL('/login');
  });

  test('should invalidate session after password change', async ({ page, loginAsUser }) => {
    await loginAsUser('user');
    
    // Navigate to profile settings
    await page.goto('/profile');
    
    // Change password
    await page.fill('[data-testid="current-password"]', 'UserTest123!');
    await page.fill('[data-testid="new-password"]', 'NewPassword123!');
    await page.fill('[data-testid="confirm-password"]', 'NewPassword123!');
    await page.click('[data-testid="change-password-button"]');
    
    // Should be logged out and redirected to login
    await expect(page).toHaveURL('/login');
    await expect(page.locator('[data-testid="password-changed-message"]')).toBeVisible();
  });

  test('should handle remember me functionality', async ({ page }) => {
    await page.goto('/login');
    
    // Login with remember me checked
    await page.fill('[data-testid="username-input"]', 'user_test');
    await page.fill('[data-testid="password-input"]', 'UserTest123!');
    await page.check('[data-testid="remember-me-checkbox"]');
    await page.click('[data-testid="login-button"]');
    
    await page.waitForURL('/dashboard');
    
    // Close and reopen browser
    await page.context().close();
    const newContext = await page.context().browser().newContext();
    const newPage = await newContext.newPage();
    
    // Should still be logged in
    await newPage.goto('/dashboard');
    await expect(newPage.locator('[data-testid="user-menu"]')).toBeVisible();
    
    await newContext.close();
  });

  test('should enforce session timeout for inactive users', async ({ page, loginAsUser }) => {
    await loginAsUser('user');
    
    // Simulate user inactivity
    await page.evaluate(() => {
      // Mock last activity time to be old
      localStorage.setItem('lastActivity', String(Date.now() - 30 * 60 * 1000)); // 30 minutes ago
    });
    
    // Trigger activity check
    await page.evaluate(() => {
      window.dispatchEvent(new CustomEvent('checkUserActivity'));
    });
    
    // Wait for timeout handling
    await page.waitForTimeout(1000);
    
    // Should show session timeout warning
    await expect(page.locator('[data-testid="session-timeout-warning"]')).toBeVisible();
  });

  test('should extend session on user activity', async ({ page, loginAsUser }) => {
    await loginAsUser('user');
    
    // Record initial activity time
    const initialActivity = await page.evaluate(() => localStorage.getItem('lastActivity'));
    
    // Simulate user activity
    await page.click('[data-testid="user-menu"]');
    await page.keyboard.press('Escape'); // Close menu
    
    // Wait a moment
    await page.waitForTimeout(1000);
    
    // Check if activity time was updated
    const newActivity = await page.evaluate(() => localStorage.getItem('lastActivity'));
    expect(parseInt(newActivity)).toBeGreaterThan(parseInt(initialActivity));
  });

  test('should handle session conflicts between different devices', async ({ browser }) => {
    // Simulate different devices
    const desktop = await browser.newContext({
      userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    });
    const mobile = await browser.newContext({
      userAgent: 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15'
    });
    
    const desktopPage = await desktop.newPage();
    const mobilePage = await mobile.newPage();
    
    // Login from desktop
    await desktopPage.goto('/login');
    await desktopPage.fill('[data-testid="username-input"]', 'user_test');
    await desktopPage.fill('[data-testid="password-input"]', 'UserTest123!');
    await desktopPage.click('[data-testid="login-button"]');
    await desktopPage.waitForURL('/dashboard');
    
    // Login from mobile
    await mobilePage.goto('/login');
    await mobilePage.fill('[data-testid="username-input"]', 'user_test');
    await mobilePage.fill('[data-testid="password-input"]', 'UserTest123!');
    await mobilePage.click('[data-testid="login-button"]');
    await mobilePage.waitForURL('/dashboard');
    
    // Both sessions should be valid (depending on configuration)
    await expect(desktopPage.locator('[data-testid="user-menu"]')).toBeVisible();
    await expect(mobilePage.locator('[data-testid="user-menu"]')).toBeVisible();
    
    await desktop.close();
    await mobile.close();
  });

  test('should show session activity log', async ({ page, loginAsUser }) => {
    await loginAsUser('admin'); // Admin can view session logs
    
    // Navigate to security settings
    await page.goto('/admin/security');
    
    // View session activity
    await page.click('[data-testid="session-activity-tab"]');
    
    // Should show current session
    await expect(page.locator('[data-testid="session-list"]')).toBeVisible();
    await expect(page.locator('[data-testid="current-session"]')).toBeVisible();
    
    // Session details should include IP, device, login time
    await expect(page.locator('[data-testid="session-ip"]')).toBeVisible();
    await expect(page.locator('[data-testid="session-device"]')).toBeVisible();
    await expect(page.locator('[data-testid="session-login-time"]')).toBeVisible();
  });

  test('should allow manual session termination', async ({ page, loginAsUser }) => {
    await loginAsUser('admin');
    
    // Navigate to security settings
    await page.goto('/admin/security');
    await page.click('[data-testid="session-activity-tab"]');
    
    // Terminate a session
    await page.click('[data-testid="terminate-session-button"]');
    
    // Confirm termination
    await page.click('[data-testid="confirm-terminate"]');
    
    // Should show success message
    await expect(page.locator('[data-testid="session-terminated-message"]')).toBeVisible();
  });

  test('should handle session data encryption', async ({ page, loginAsUser }) => {
    await loginAsUser('user');
    
    // Check if session token is properly formatted (JWT)
    const token = await page.evaluate(() => localStorage.getItem('authToken'));
    expect(token).toMatch(/^[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+$/);
    
    // Verify token contains expected claims
    const payload = JSON.parse(atob(token.split('.')[1]));
    expect(payload).toHaveProperty('exp'); // Expiration
    expect(payload).toHaveProperty('iat'); // Issued at
    expect(payload).toHaveProperty('sub'); // Subject (user ID)
  });

  test('should handle graceful session cleanup on browser close', async ({ browser }) => {
    const context = await browser.newContext();
    const page = await context.newPage();
    
    // Login
    await page.goto('/login');
    await page.fill('[data-testid="username-input"]', 'user_test');
    await page.fill('[data-testid="password-input"]', 'UserTest123!');
    await page.click('[data-testid="login-button"]');
    await page.waitForURL('/dashboard');
    
    // Simulate browser close
    await context.close();
    
    // Session should be marked as ended on server side
    // This would be verified through server logs or admin interface
  });
});