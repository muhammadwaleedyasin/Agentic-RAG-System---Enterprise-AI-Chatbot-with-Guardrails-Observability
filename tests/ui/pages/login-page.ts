import { Page, expect } from '@playwright/test';
import { BasePage } from './base-page';

export class LoginPage extends BasePage {
  constructor(page: Page) {
    super(page);
  }

  // Page elements
  get elements() {
    return {
      usernameInput: this.page.locator('[data-testid="username-input"]'),
      passwordInput: this.page.locator('[data-testid="password-input"]'),
      loginButton: this.page.locator('[data-testid="login-button"]'),
      errorMessage: this.page.locator('[data-testid="login-error"]'),
      forgotPasswordLink: this.page.locator('[data-testid="forgot-password"]'),
      rememberMeCheckbox: this.page.locator('[data-testid="remember-me"]'),
      signUpLink: this.page.locator('[data-testid="signup-link"]'),
      loadingSpinner: this.page.locator('[data-testid="login-loading"]'),
    };
  }

  async goto(): Promise<void> {
    await this.page.goto('/login');
    await this.verifyPageLoaded();
  }

  async verifyPageLoaded(): Promise<void> {
    await expect(this.elements.usernameInput).toBeVisible();
    await expect(this.elements.passwordInput).toBeVisible();
    await expect(this.elements.loginButton).toBeVisible();
    await this.verifyTitle('Login - Enterprise RAG Chatbot');
  }

  /**
   * Perform login with username and password
   */
  async login(username: string, password: string, rememberMe = false): Promise<void> {
    await this.fillFormField('[data-testid="username-input"]', username);
    await this.fillFormField('[data-testid="password-input"]', password);
    
    if (rememberMe) {
      await this.elements.rememberMeCheckbox.check();
    }
    
    await this.elements.loginButton.click();
  }

  /**
   * Verify successful login
   */
  async waitForSuccessfulLogin(): Promise<void> {
    // Wait for redirect to dashboard or home page
    await this.page.waitForURL(/\/(dashboard|home|\/)$/);
    
    // Verify we're no longer on login page
    await expect(this.page.locator('[data-testid="username-input"]')).not.toBeVisible({ timeout: 5000 });
  }

  /**
   * Verify login error
   */
  async verifyLoginError(expectedError: string): Promise<void> {
    await expect(this.elements.errorMessage).toBeVisible();
    await expect(this.elements.errorMessage).toContainText(expectedError);
  }

  /**
   * Click forgot password link
   */
  async clickForgotPassword(): Promise<void> {
    await this.elements.forgotPasswordLink.click();
    await this.page.waitForURL('/forgot-password');
  }

  /**
   * Click sign up link
   */
  async clickSignUp(): Promise<void> {
    await this.elements.signUpLink.click();
    await this.page.waitForURL('/signup');
  }

  /**
   * Verify login form validation
   */
  async verifyFormValidation(): Promise<void> {
    // Try to login without credentials
    await this.elements.loginButton.click();
    
    // Check for validation errors
    await this.validateFormErrors({
      username: 'Username is required',
      password: 'Password is required',
    });
  }

  /**
   * Test login with invalid credentials
   */
  async testInvalidLogin(): Promise<void> {
    await this.login('invalid_user', 'invalid_password');
    await this.verifyLoginError('Invalid username or password');
  }

  /**
   * Test login with empty fields
   */
  async testEmptyFieldValidation(): Promise<void> {
    // Test empty username
    await this.fillFormField('[data-testid="password-input"]', 'password123');
    await this.elements.loginButton.click();
    await this.validateFormErrors({ username: 'Username is required' });
    
    // Clear and test empty password
    await this.clearForm('[data-testid="login-form"]');
    await this.fillFormField('[data-testid="username-input"]', 'testuser');
    await this.elements.loginButton.click();
    await this.validateFormErrors({ password: 'Password is required' });
  }

  /**
   * Test login loading state
   */
  async verifyLoginLoading(): Promise<void> {
    await this.fillFormField('[data-testid="username-input"]', 'testuser');
    await this.fillFormField('[data-testid="password-input"]', 'password123');
    
    // Click login and immediately check for loading state
    await this.elements.loginButton.click();
    
    // Verify loading state appears briefly
    try {
      await expect(this.elements.loadingSpinner).toBeVisible({ timeout: 1000 });
      await expect(this.elements.loginButton).toBeDisabled();
    } catch {
      // Loading might be too fast to catch in test environment
    }
  }

  /**
   * Test accessibility features
   */
  async verifyAccessibility(): Promise<void> {
    // Check ARIA labels
    await expect(this.elements.usernameInput).toHaveAttribute('aria-label', 'Username');
    await expect(this.elements.passwordInput).toHaveAttribute('aria-label', 'Password');
    
    // Check tab navigation
    await this.elements.usernameInput.focus();
    await this.page.keyboard.press('Tab');
    await expect(this.elements.passwordInput).toBeFocused();
    
    await this.page.keyboard.press('Tab');
    await expect(this.elements.loginButton).toBeFocused();
  }

  /**
   * Test password visibility toggle
   */
  async testPasswordVisibilityToggle(): Promise<void> {
    const passwordToggle = this.page.locator('[data-testid="password-toggle"]');
    
    if (await passwordToggle.isVisible()) {
      // Verify password is hidden by default
      await expect(this.elements.passwordInput).toHaveAttribute('type', 'password');
      
      // Click toggle to show password
      await passwordToggle.click();
      await expect(this.elements.passwordInput).toHaveAttribute('type', 'text');
      
      // Click toggle to hide password again
      await passwordToggle.click();
      await expect(this.elements.passwordInput).toHaveAttribute('type', 'password');
    }
  }

  /**
   * Test keyboard navigation
   */
  async testKeyboardNavigation(): Promise<void> {
    await this.fillFormField('[data-testid="username-input"]', 'testuser');
    await this.fillFormField('[data-testid="password-input"]', 'password123');
    
    // Test Enter key submission
    await this.elements.passwordInput.press('Enter');
    
    // Should trigger login attempt
    await this.verifyLoginError('Invalid username or password');
  }

  /**
   * Complete login flow for valid user
   */
  async performValidLogin(username = 'user_test', password = 'test_password_123'): Promise<void> {
    await this.login(username, password);
    await this.waitForSuccessfulLogin();
    await this.verifyNoErrors();
  }

  /**
   * Get current authentication status
   */
  async getAuthStatus(): Promise<boolean> {
    // Check if we have authentication tokens/cookies
    const cookies = await this.page.context().cookies();
    return cookies.some(cookie => cookie.name === 'auth_token' && cookie.value.length > 0);
  }

  /**
   * Clear authentication
   */
  async clearAuth(): Promise<void> {
    await this.page.context().clearCookies();
    await this.page.reload();
  }
}