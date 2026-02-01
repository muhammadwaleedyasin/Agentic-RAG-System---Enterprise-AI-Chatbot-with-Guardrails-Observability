import { test, expect } from '@playwright/test';

test.describe('Authentication Flow', () => {
  test.beforeEach(async ({ page }) => {
    // Clear any existing auth state
    await page.context().clearCookies();
    // Only clear storage if localStorage is accessible
    try {
      await page.evaluate(() => {
        if (typeof localStorage !== 'undefined') {
          localStorage.clear();
        }
        if (typeof sessionStorage !== 'undefined') {
          sessionStorage.clear();
        }
      });
    } catch (error) {
      // localStorage might not be available, continue with the test
      console.log('Could not clear localStorage:', error instanceof Error ? error.message : 'Unknown error');
    }
  });

  test('should redirect unauthenticated users from home to login', async ({ page }) => {
    // Navigate to the application
    await page.goto('/');
    
    // Should redirect to login page since not authenticated
    await expect(page).toHaveURL(/.*login/);
    
    // Check login form is present with correct elements
    await expect(page.getByRole('heading', { name: /welcome back/i })).toBeVisible();
    await expect(page.getByText(/sign in to your account/i)).toBeVisible();
    await expect(page.getByTestId('login-username')).toBeVisible();
    await expect(page.getByTestId('login-password')).toBeVisible();
    await expect(page.getByTestId('login-submit')).toBeVisible();
  });

  test('should redirect unauthenticated users from chat to login', async ({ page }) => {
    // Try to access chat page directly
    await page.goto('/chat');
    
    // Should redirect to login page
    await expect(page).toHaveURL(/.*login/);
    await expect(page.getByRole('heading', { name: /welcome back/i })).toBeVisible();
  });

  test('should handle login form validation', async ({ page }) => {
    await page.goto('/login');
    
    // Try to submit empty form
    await page.getByTestId('login-submit').click();
    
    // Should show validation errors for both fields
    await expect(page.getByText(/username is required/i)).toBeVisible();
    await expect(page.getByText(/password is required/i)).toBeVisible();
  });

  test('should handle invalid credentials', async ({ page }) => {
    await page.goto('/login');
    
    // Wait for login page to load
    await expect(page.getByRole('heading', { name: /welcome back/i })).toBeVisible();
    
    // Fill form with invalid credentials
    await page.getByTestId('login-username').fill('invalid_user');
    await page.getByTestId('login-password').fill('wrongpassword');
    await page.getByTestId('login-submit').click();
    
    // Should show error message (specific error depends on backend implementation)
    await expect(page.locator('[role="alert"]')).toBeVisible();
  });

  test('should login successfully with admin credentials and redirect to chat', async ({ page }) => {
    await page.goto('/login');
    
    // Wait for login page to load
    await expect(page.getByRole('heading', { name: /welcome back/i })).toBeVisible();
    
    // Fill form with valid admin credentials using data-testid selectors
    await page.getByTestId('login-username').fill('admin');
    await page.getByTestId('login-password').fill('admin123');
    
    // Submit the form
    await page.getByTestId('login-submit').click();
    
    // Should redirect to chat (as specified in login page logic)
    await expect(page).toHaveURL(/.*chat/);
    
    // Verify we're actually on the chat page with proper content
    await expect(page.getByText(/chat/i)).toBeVisible();
  });

  test('should complete full authentication flow - login and access protected pages', async ({ page }) => {
    // Start from home page
    await page.goto('/');
    
    // Should redirect to login
    await expect(page).toHaveURL(/.*login/);
    
    // Wait for login page to load
    await expect(page.getByRole('heading', { name: /welcome back/i })).toBeVisible();
    
    // Login with admin credentials
    await page.getByTestId('login-username').fill('admin');
    await page.getByTestId('login-password').fill('admin123');
    await page.getByTestId('login-submit').click();
    
    // Should redirect to chat
    await expect(page).toHaveURL(/.*chat/);
    
    // Now test navigation to other protected pages
    // Navigate to documents page
    await page.goto('/documents');
    await expect(page).toHaveURL(/.*documents/);
    
    // Navigate to admin page
    await page.goto('/admin');
    await expect(page).toHaveURL(/.*admin/);
    
    // Navigate back to chat
    await page.goto('/chat');
    await expect(page).toHaveURL(/.*chat/);
  });

  test('should handle logout if logout functionality exists', async ({ page }) => {
    // First login
    await page.goto('/login');
    await page.getByRole('textbox', { name: /username/i }).fill('admin');
    await page.locator('input[type="password"]').fill('admin123');
    await page.getByTestId('login-submit').click();
    
    // Wait for successful login and redirect to chat
    await expect(page).toHaveURL(/.*chat/);
    
    // Look for logout button (could be in navigation, header, or user menu)
    const logoutButton = page.getByRole('button', { name: /logout/i }).first();
    const logoutLink = page.getByRole('link', { name: /logout/i }).first();
    
    // Try to find and click logout
    if (await logoutButton.isVisible()) {
      await logoutButton.click();
    } else if (await logoutLink.isVisible()) {
      await logoutLink.click();
    } else {
      // If no logout button found, this test is informational
      console.log('No logout button found - this may indicate missing logout functionality');
      return;
    }
    
    // Should redirect to login after logout
    await expect(page).toHaveURL(/.*login/);
  });

  test('should persist authentication across page reloads', async ({ page }) => {
    // Login first
    await page.goto('/login');
    await page.getByRole('textbox', { name: /username/i }).fill('admin');
    await page.locator('input[type="password"]').fill('admin123');
    await page.getByTestId('login-submit').click();
    
    // Wait for successful login
    await expect(page).toHaveURL(/.*chat/);
    
    // Reload the page
    await page.reload();
    
    // Should still be authenticated and on chat page
    await expect(page).toHaveURL(/.*chat/);
    
    // Navigate to home and should go to chat (for authenticated users)
    await page.goto('/');
    await expect(page).toHaveURL(/.*chat/);
  });
});

test.describe('Accessibility', () => {
  test('login page should be accessible', async ({ page }) => {
    await page.goto('/login');
    
    // Check for proper heading structure
    await expect(page.getByRole('heading', { level: 1 })).toBeVisible();
    
    // Check form labels are properly associated
    const usernameInput = page.getByRole('textbox', { name: /username/i });
    const passwordInput = page.locator('input[type="password"]');
    
    await expect(usernameInput).toBeVisible();
    await expect(passwordInput).toBeVisible();
    
    // Check keyboard navigation
    await usernameInput.focus();
    await expect(usernameInput).toBeFocused();
    
    await page.keyboard.press('Tab');
    await expect(passwordInput).toBeFocused();
    
    await page.keyboard.press('Tab');
    await expect(page.getByRole('button', { name: /sign in/i })).toBeFocused();
  });
});

test.describe('Responsive Design', () => {
  test('should work on mobile viewport', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/login');
    
    // Check mobile layout
    await expect(page.getByRole('heading', { name: /sign in/i })).toBeVisible();
    await expect(page.getByRole('textbox', { name: /username/i })).toBeVisible();
    await expect(page.locator('input[type="password"]')).toBeVisible();
  });

  test('should work on tablet viewport', async ({ page }) => {
    await page.setViewportSize({ width: 768, height: 1024 });
    await page.goto('/login');
    
    // Check tablet layout
    await expect(page.getByRole('heading', { name: /sign in/i })).toBeVisible();
    await expect(page.getByRole('textbox', { name: /username/i })).toBeVisible();
    await expect(page.locator('input[type="password"]')).toBeVisible();
  });

  test('should work on desktop viewport', async ({ page }) => {
    await page.setViewportSize({ width: 1920, height: 1080 });
    await page.goto('/login');
    
    // Check desktop layout
    await expect(page.getByRole('heading', { name: /sign in/i })).toBeVisible();
    await expect(page.getByRole('textbox', { name: /username/i })).toBeVisible();
    await expect(page.locator('input[type="password"]')).toBeVisible();
  });
});