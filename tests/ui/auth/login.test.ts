import { test, expect } from '../fixtures/auth-fixtures';

/**
 * Authentication Flow Tests - Login
 * Tests user login functionality, session management, and security
 */

test.describe('User Login', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/login');
  });

  test('should display login form correctly', async ({ page }) => {
    // Verify login form elements
    await expect(page.locator('[data-testid="login-form"]')).toBeVisible();
    await expect(page.locator('[data-testid="username-input"]')).toBeVisible();
    await expect(page.locator('[data-testid="password-input"]')).toBeVisible();
    await expect(page.locator('[data-testid="login-button"]')).toBeVisible();
    
    // Verify form labels and placeholders
    await expect(page.locator('label[for="username"]')).toContainText('Username');
    await expect(page.locator('label[for="password"]')).toContainText('Password');
    await expect(page.locator('[data-testid="username-input"]')).toHaveAttribute('placeholder', 'Enter your username');
    await expect(page.locator('[data-testid="password-input"]')).toHaveAttribute('type', 'password');
  });

  test('should successfully login with valid admin credentials', async ({ page, loginAsUser, verifyAuthentication }) => {
    await loginAsUser('admin');
    await verifyAuthentication('admin');
    
    // Verify redirect to dashboard
    await expect(page).toHaveURL('/dashboard');
    
    // Verify admin-specific UI elements
    await expect(page.locator('[data-testid="admin-panel-link"]')).toBeVisible();
    await expect(page.locator('[data-testid="user-management-link"]')).toBeVisible();
  });

  test('should successfully login with valid user credentials', async ({ page, loginAsUser, verifyAuthentication }) => {
    await loginAsUser('user');
    await verifyAuthentication('user');
    
    // Verify redirect to dashboard
    await expect(page).toHaveURL('/dashboard');
    
    // Verify user-specific UI elements (no admin links)
    await expect(page.locator('[data-testid="admin-panel-link"]')).not.toBeVisible();
    await expect(page.locator('[data-testid="chat-link"]')).toBeVisible();
  });

  test('should successfully login with valid viewer credentials', async ({ page, loginAsUser, verifyAuthentication }) => {
    await loginAsUser('viewer');
    await verifyAuthentication('viewer');
    
    // Verify redirect to dashboard
    await expect(page).toHaveURL('/dashboard');
    
    // Verify viewer-specific UI elements (read-only access)
    await expect(page.locator('[data-testid="admin-panel-link"]')).not.toBeVisible();
    await expect(page.locator('[data-testid="document-upload"]')).not.toBeVisible();
  });

  test('should show error for invalid username', async ({ page }) => {
    await page.fill('[data-testid="username-input"]', 'invalid_user');
    await page.fill('[data-testid="password-input"]', 'SomePassword123!');
    await page.click('[data-testid="login-button"]');
    
    // Verify error message
    await expect(page.locator('[data-testid="error-message"]')).toBeVisible();
    await expect(page.locator('[data-testid="error-message"]')).toContainText('Invalid username or password');
    
    // Verify user stays on login page
    await expect(page).toHaveURL('/login');
  });

  test('should show error for invalid password', async ({ page }) => {
    await page.fill('[data-testid="username-input"]', 'user_test');
    await page.fill('[data-testid="password-input"]', 'WrongPassword');
    await page.click('[data-testid="login-button"]');
    
    // Verify error message
    await expect(page.locator('[data-testid="error-message"]')).toBeVisible();
    await expect(page.locator('[data-testid="error-message"]')).toContainText('Invalid username or password');
  });

  test('should show validation errors for empty fields', async ({ page }) => {
    await page.click('[data-testid="login-button"]');
    
    // Verify field validation errors
    await expect(page.locator('[data-testid="username-error"]')).toBeVisible();
    await expect(page.locator('[data-testid="username-error"]')).toContainText('Username is required');
    await expect(page.locator('[data-testid="password-error"]')).toBeVisible();
    await expect(page.locator('[data-testid="password-error"]')).toContainText('Password is required');
  });

  test('should handle special characters in credentials', async ({ page }) => {
    await page.fill('[data-testid="username-input"]', 'test@user.com');
    await page.fill('[data-testid="password-input"]', 'P@ssw0rd!#$%');
    await page.click('[data-testid="login-button"]');
    
    // Should handle special characters gracefully
    await expect(page.locator('[data-testid="error-message"]')).toBeVisible();
  });

  test('should prevent SQL injection attempts', async ({ page }) => {
    const maliciousInput = "'; DROP TABLE users; --";
    
    await page.fill('[data-testid="username-input"]', maliciousInput);
    await page.fill('[data-testid="password-input"]', maliciousInput);
    await page.click('[data-testid="login-button"]');
    
    // Should reject malicious input safely
    await expect(page.locator('[data-testid="error-message"]')).toBeVisible();
    await expect(page).toHaveURL('/login');
  });

  test('should handle network timeouts gracefully', async ({ page }) => {
    // Simulate slow network
    await page.route('/api/v1/auth/login', async route => {
      await new Promise(resolve => setTimeout(resolve, 30000));
      route.continue();
    });

    await page.fill('[data-testid="username-input"]', 'user_test');
    await page.fill('[data-testid="password-input"]', 'UserTest123!');
    await page.click('[data-testid="login-button"]');

    // Should show loading state
    await expect(page.locator('[data-testid="login-loading"]')).toBeVisible();
    
    // Should eventually show timeout error
    await expect(page.locator('[data-testid="error-message"]')).toBeVisible({ timeout: 35000 });
  });

  test('should disable login button during submission', async ({ page }) => {
    await page.fill('[data-testid="username-input"]', 'user_test');
    await page.fill('[data-testid="password-input"]', 'UserTest123!');
    
    // Click login and immediately check if button is disabled
    await page.click('[data-testid="login-button"]');
    await expect(page.locator('[data-testid="login-button"]')).toBeDisabled();
  });

  test('should maintain form state on validation errors', async ({ page }) => {
    const username = 'test_user';
    
    await page.fill('[data-testid="username-input"]', username);
    await page.fill('[data-testid="password-input"]', 'wrong');
    await page.click('[data-testid="login-button"]');
    
    // After error, username should still be filled
    await expect(page.locator('[data-testid="username-input"]')).toHaveValue(username);
    // Password should be cleared for security
    await expect(page.locator('[data-testid="password-input"]')).toHaveValue('');
  });

  test('should support keyboard navigation', async ({ page }) => {
    // Tab to username field
    await page.keyboard.press('Tab');
    await expect(page.locator('[data-testid="username-input"]')).toBeFocused();
    
    // Tab to password field
    await page.keyboard.press('Tab');
    await expect(page.locator('[data-testid="password-input"]')).toBeFocused();
    
    // Tab to login button
    await page.keyboard.press('Tab');
    await expect(page.locator('[data-testid="login-button"]')).toBeFocused();
    
    // Enter should submit form
    await page.fill('[data-testid="username-input"]', 'user_test');
    await page.fill('[data-testid="password-input"]', 'UserTest123!');
    await page.keyboard.press('Enter');
    
    // Should attempt login
    await expect(page.locator('[data-testid="login-loading"]')).toBeVisible();
  });

  test('should redirect authenticated users away from login page', async ({ page, loginAsUser }) => {
    // Login first
    await loginAsUser('user');
    
    // Try to access login page while authenticated
    await page.goto('/login');
    
    // Should redirect to dashboard
    await expect(page).toHaveURL('/dashboard');
  });
});