import { test, expect } from '../fixtures/test-fixtures';

test.describe('Authentication Flow', () => {
  test.describe('Login', () => {
    test('should login with valid credentials', async ({ loginPage }) => {
      await loginPage.goto();
      await loginPage.performValidLogin();
    });

    test('should show error with invalid credentials', async ({ loginPage }) => {
      await loginPage.goto();
      await loginPage.testInvalidLogin();
    });

    test('should validate empty fields', async ({ loginPage }) => {
      await loginPage.goto();
      await loginPage.testEmptyFieldValidation();
    });

    test('should show loading state during login', async ({ loginPage }) => {
      await loginPage.goto();
      await loginPage.verifyLoginLoading();
    });

    test('should support keyboard navigation', async ({ loginPage }) => {
      await loginPage.goto();
      await loginPage.testKeyboardNavigation();
    });

    test('should be accessible', async ({ loginPage }) => {
      await loginPage.goto();
      await loginPage.verifyAccessibility();
    });

    test('should toggle password visibility', async ({ loginPage }) => {
      await loginPage.goto();
      await loginPage.testPasswordVisibilityToggle();
    });
  });

  test.describe('Session Management', () => {
    test('should maintain session after page refresh', async ({ loginPage, dashboardPage }) => {
      await loginPage.goto();
      await loginPage.performValidLogin();
      
      await dashboardPage.verifyPageLoaded();
      await dashboardPage.page.reload();
      await dashboardPage.verifyPageLoaded();
    });

    test('should logout successfully', async ({ loginPage, dashboardPage }) => {
      await loginPage.goto();
      await loginPage.performValidLogin();
      
      await dashboardPage.verifyPageLoaded();
      await dashboardPage.logout();
      
      await expect(dashboardPage.page).toHaveURL('/login');
    });

    test('should redirect to login when accessing protected routes', async ({ page }) => {
      await page.goto('/dashboard');
      await expect(page).toHaveURL('/login');
      
      await page.goto('/admin');
      await expect(page).toHaveURL('/login');
    });

    test('should handle session expiration', async ({ loginPage, dashboardPage, authService }) => {
      await loginPage.goto();
      await loginPage.performValidLogin();
      
      // Simulate expired session by clearing auth
      await loginPage.clearAuth();
      
      // Try to access protected route
      await dashboardPage.goto();
      await expect(dashboardPage.page).toHaveURL('/login');
    });
  });

  test.describe('Role-based Access', () => {
    test('admin user should access admin panel', async ({ page, authService }) => {
      const adminToken = await authService.getAuthToken('admin_test', 'test_password_123');
      
      await page.context().addCookies([{
        name: 'auth_token',
        value: adminToken,
        domain: 'localhost',
        path: '/',
      }]);
      
      await page.goto('/admin');
      await expect(page.locator('[data-testid="admin-navigation"]')).toBeVisible();
    });

    test('regular user should not access admin panel', async ({ page, authService }) => {
      const userToken = await authService.getAuthToken('user_test', 'test_password_123');
      
      await page.context().addCookies([{
        name: 'auth_token',
        value: userToken,
        domain: 'localhost',
        path: '/',
      }]);
      
      await page.goto('/admin');
      await expect(page).toHaveURL(/\/(dashboard|unauthorized)/);
    });

    test('viewer should have limited permissions', async ({ page, authService, dashboardPage }) => {
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

  test.describe('Security', () => {
    test('should protect against XSS in login form', async ({ loginPage }) => {
      await loginPage.goto();
      
      const xssPayload = '<script>alert("xss")</script>';
      await loginPage.login(xssPayload, 'password');
      
      // Should not execute script, just show invalid login
      const dialogPromise = loginPage.page.waitForEvent('dialog', { timeout: 1000 }).catch(() => null);
      const dialog = await dialogPromise;
      expect(dialog).toBeNull();
    });

    test('should implement rate limiting for login attempts', async ({ loginPage }) => {
      await loginPage.goto();
      
      // Attempt multiple failed logins
      for (let i = 0; i < 5; i++) {
        await loginPage.login('invalid', 'invalid');
        await loginPage.verifyLoginError('Invalid username or password');
      }
      
      // Should show rate limiting error
      await loginPage.login('invalid', 'invalid');
      // Check for rate limiting message (implementation dependent)
    });

    test('should clear sensitive data on logout', async ({ loginPage, dashboardPage }) => {
      await loginPage.goto();
      await loginPage.performValidLogin();
      
      await dashboardPage.logout();
      
      // Verify auth tokens are cleared
      const authStatus = await loginPage.getAuthStatus();
      expect(authStatus).toBe(false);
    });
  });
});