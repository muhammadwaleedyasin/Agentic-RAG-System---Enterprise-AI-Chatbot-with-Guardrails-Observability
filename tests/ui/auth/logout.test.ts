import { test, expect } from '../fixtures/auth-fixtures';

/**
 * Authentication Flow Tests - Logout
 * Tests user logout functionality and session cleanup
 */

test.describe('User Logout', () => {
  test('should successfully logout admin user', async ({ page, loginAsUser, logout }) => {
    // Login as admin
    await loginAsUser('admin');
    await expect(page).toHaveURL('/dashboard');
    
    // Logout
    await logout();
    
    // Verify redirect to login page
    await expect(page).toHaveURL('/login');
    
    // Verify authentication state is cleared
    await expect(page.locator('[data-testid="user-menu"]')).not.toBeVisible();
    
    // Verify cannot access protected routes
    await page.goto('/admin');
    await expect(page).toHaveURL('/login');
  });

  test('should successfully logout regular user', async ({ page, loginAsUser, logout }) => {
    // Login as user
    await loginAsUser('user');
    await expect(page).toHaveURL('/dashboard');
    
    // Logout
    await logout();
    
    // Verify logout completed
    await expect(page).toHaveURL('/login');
    await expect(page.locator('[data-testid="login-form"]')).toBeVisible();
  });

  test('should clear session tokens on logout', async ({ page, loginAsUser, logout }) => {
    // Login and store session info
    await loginAsUser('user');
    
    // Check that session token exists
    const tokenBefore = await page.evaluate(() => localStorage.getItem('authToken'));
    expect(tokenBefore).toBeTruthy();
    
    // Logout
    await logout();
    
    // Verify session token is cleared
    const tokenAfter = await page.evaluate(() => localStorage.getItem('authToken'));
    expect(tokenAfter).toBeFalsy();
  });

  test('should invalidate session on server side', async ({ page, loginAsUser, logout }) => {
    // Login
    await loginAsUser('user');
    
    // Store token for later verification
    const token = await page.evaluate(() => localStorage.getItem('authToken'));
    
    // Logout
    await logout();
    
    // Try to use the old token
    const response = await page.request.get('/api/v1/auth/me', {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    });
    
    // Should be unauthorized
    expect(response.status()).toBe(401);
  });

  test('should handle logout button click in user menu', async ({ page, loginAsUser }) => {
    await loginAsUser('user');
    
    // Click user menu
    await page.click('[data-testid="user-menu"]');
    
    // Verify dropdown is visible
    await expect(page.locator('[data-testid="user-dropdown"]')).toBeVisible();
    
    // Click logout
    await page.click('[data-testid="logout-button"]');
    
    // Verify logout
    await expect(page).toHaveURL('/login');
  });

  test('should handle logout via keyboard shortcut', async ({ page, loginAsUser }) => {
    await loginAsUser('user');
    
    // Use keyboard shortcut (Ctrl+Shift+L)
    await page.keyboard.press('Control+Shift+KeyL');
    
    // Should logout
    await expect(page).toHaveURL('/login');
  });

  test('should show logout confirmation dialog', async ({ page, loginAsUser }) => {
    await loginAsUser('user');
    
    // Click user menu
    await page.click('[data-testid="user-menu"]');
    await page.click('[data-testid="logout-button"]');
    
    // Should show confirmation dialog
    await expect(page.locator('[data-testid="logout-confirm-dialog"]')).toBeVisible();
    await expect(page.locator('[data-testid="logout-confirm-message"]'))
      .toContainText('Are you sure you want to log out?');
    
    // Cancel logout
    await page.click('[data-testid="logout-cancel-button"]');
    await expect(page.locator('[data-testid="logout-confirm-dialog"]')).not.toBeVisible();
    
    // Should still be logged in
    await expect(page.locator('[data-testid="user-menu"]')).toBeVisible();
  });

  test('should confirm logout and proceed', async ({ page, loginAsUser }) => {
    await loginAsUser('user');
    
    // Trigger logout with confirmation
    await page.click('[data-testid="user-menu"]');
    await page.click('[data-testid="logout-button"]');
    
    // Confirm logout
    await page.click('[data-testid="logout-confirm-button"]');
    
    // Should be logged out
    await expect(page).toHaveURL('/login');
  });

  test('should handle logout during active operations', async ({ page, loginAsUser }) => {
    await loginAsUser('user');
    
    // Navigate to chat and start a conversation
    await page.goto('/chat');
    await page.fill('[data-testid="message-input"]', 'Test message');
    await page.click('[data-testid="send-button"]');
    
    // Logout while message is being processed
    await page.click('[data-testid="user-menu"]');
    await page.click('[data-testid="logout-button"]');
    
    // Should logout successfully
    await expect(page).toHaveURL('/login');
  });

  test('should clear all user data from local storage', async ({ page, loginAsUser, logout }) => {
    await loginAsUser('user');
    
    // Store some user data
    await page.evaluate(() => {
      localStorage.setItem('userPreferences', JSON.stringify({ theme: 'dark' }));
      localStorage.setItem('chatHistory', JSON.stringify(['message1', 'message2']));
    });
    
    // Logout
    await logout();
    
    // Verify all user data is cleared
    const userPreferences = await page.evaluate(() => localStorage.getItem('userPreferences'));
    const chatHistory = await page.evaluate(() => localStorage.getItem('chatHistory'));
    
    expect(userPreferences).toBeFalsy();
    expect(chatHistory).toBeFalsy();
  });

  test('should handle logout when session is already expired', async ({ page, loginAsUser }) => {
    await loginAsUser('user');
    
    // Simulate expired session by manipulating token
    await page.evaluate(() => {
      localStorage.setItem('authToken', 'expired.token.here');
    });
    
    // Try to logout
    await page.click('[data-testid="user-menu"]');
    await page.click('[data-testid="logout-button"]');
    
    // Should handle gracefully and redirect to login
    await expect(page).toHaveURL('/login');
  });

  test('should handle network error during logout', async ({ page, loginAsUser }) => {
    await loginAsUser('user');
    
    // Simulate network error
    await page.route('/api/v1/auth/logout', route => route.abort());
    
    // Try to logout
    await page.click('[data-testid="user-menu"]');
    await page.click('[data-testid="logout-button"]');
    
    // Should still clear local state and redirect
    await expect(page).toHaveURL('/login');
    
    // Verify token is cleared locally
    const token = await page.evaluate(() => localStorage.getItem('authToken'));
    expect(token).toBeFalsy();
  });

  test('should log out from all tabs when logout is triggered', async ({ browser, loginAsUser }) => {
    // Create two browser contexts (simulate two tabs)
    const context1 = await browser.newContext();
    const context2 = await browser.newContext();
    
    const page1 = await context1.newPage();
    const page2 = await context2.newPage();
    
    // Login in both tabs
    await page1.goto('/login');
    await page1.fill('[data-testid="username-input"]', 'user_test');
    await page1.fill('[data-testid="password-input"]', 'UserTest123!');
    await page1.click('[data-testid="login-button"]');
    await page1.waitForURL('/dashboard');
    
    await page2.goto('/login');
    await page2.fill('[data-testid="username-input"]', 'user_test');
    await page2.fill('[data-testid="password-input"]', 'UserTest123!');
    await page2.click('[data-testid="login-button"]');
    await page2.waitForURL('/dashboard');
    
    // Logout from first tab
    await page1.click('[data-testid="user-menu"]');
    await page1.click('[data-testid="logout-button"]');
    
    // Both tabs should be logged out
    await expect(page1).toHaveURL('/login');
    
    // Check second tab
    await page2.reload();
    await expect(page2).toHaveURL('/login');
    
    await context1.close();
    await context2.close();
  });

  test('should show logout success message', async ({ page, loginAsUser, logout }) => {
    await loginAsUser('user');
    await logout();
    
    // Should show success message on login page
    await expect(page.locator('[data-testid="logout-success-message"]')).toBeVisible();
    await expect(page.locator('[data-testid="logout-success-message"]'))
      .toContainText('You have been successfully logged out');
  });
});