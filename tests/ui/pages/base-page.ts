import { Page, Locator, expect } from '@playwright/test';
import TestHelpers from '../utils/test-helpers';

export abstract class BasePage {
  protected helpers: TestHelpers;

  constructor(protected page: Page) {
    this.helpers = new TestHelpers(page);
  }

  /**
   * Navigate to the page
   */
  abstract goto(): Promise<void>;

  /**
   * Verify page is loaded
   */
  abstract verifyPageLoaded(): Promise<void>;

  /**
   * Common navigation elements
   */
  get navigation() {
    return {
      logo: this.page.locator('[data-testid="logo"]'),
      userMenu: this.page.locator('[data-testid="user-menu"]'),
      notifications: this.page.locator('[data-testid="notifications"]'),
      settings: this.page.locator('[data-testid="settings"]'),
      logout: this.page.locator('[data-testid="logout"]'),
    };
  }

  /**
   * Common loading states
   */
  get loadingStates() {
    return {
      pageLoader: this.page.locator('[data-testid="page-loader"]'),
      contentLoader: this.page.locator('[data-testid="content-loader"]'),
      spinner: this.page.locator('[data-testid="spinner"]'),
    };
  }

  /**
   * Common notification elements
   */
  get notifications() {
    return {
      success: this.page.locator('[data-testid="toast-success"]'),
      error: this.page.locator('[data-testid="toast-error"]'),
      warning: this.page.locator('[data-testid="toast-warning"]'),
      info: this.page.locator('[data-testid="toast-info"]'),
    };
  }

  /**
   * Wait for page to load completely
   */
  async waitForPageLoad(): Promise<void> {
    await this.page.waitForLoadState('networkidle');
    await this.helpers.waitForLoadingComplete();
  }

  /**
   * Navigate using the main menu
   */
  async navigateToSection(section: 'chat' | 'documents' | 'admin' | 'dashboard'): Promise<void> {
    const menuItem = this.page.locator(`[data-testid="nav-${section}"]`);
    await menuItem.click();
    await this.waitForPageLoad();
  }

  /**
   * Handle errors on the page
   */
  async getPageErrors(): Promise<string[]> {
    const errorElements = this.page.locator('[data-testid="error"]');
    const count = await errorElements.count();
    const errors: string[] = [];
    
    for (let i = 0; i < count; i++) {
      const errorText = await errorElements.nth(i).textContent();
      if (errorText) {
        errors.push(errorText);
      }
    }
    
    return errors;
  }

  /**
   * Verify no errors are present on the page
   */
  async verifyNoErrors(): Promise<void> {
    const errors = await this.getPageErrors();
    expect(errors).toHaveLength(0);
  }

  /**
   * Take a screenshot of the current page
   */
  async takeScreenshot(name?: string): Promise<void> {
    const screenshotName = name || `${this.constructor.name}-${Date.now()}`;
    await this.helpers.takeScreenshot(screenshotName);
  }

  /**
   * Wait for specific URL pattern
   */
  async waitForUrl(pattern: string | RegExp): Promise<void> {
    await this.page.waitForURL(pattern);
  }

  /**
   * Verify page title
   */
  async verifyTitle(expectedTitle: string): Promise<void> {
    await expect(this.page).toHaveTitle(expectedTitle);
  }

  /**
   * Verify page URL
   */
  async verifyUrl(expectedUrl: string | RegExp): Promise<void> {
    await expect(this.page).toHaveURL(expectedUrl);
  }

  /**
   * Handle confirmation dialogs
   */
  async confirmAction(accept = true): Promise<void> {
    const dialogPromise = this.page.waitForEvent('dialog');
    const dialog = await dialogPromise;
    
    if (accept) {
      await dialog.accept();
    } else {
      await dialog.dismiss();
    }
  }

  /**
   * Scroll to element and ensure it's visible
   */
  async scrollToElement(selector: string): Promise<void> {
    await this.helpers.scrollIntoView(selector);
  }

  /**
   * Fill and validate form field
   */
  async fillFormField(selector: string, value: string): Promise<void> {
    await this.helpers.fillField(selector, value);
  }

  /**
   * Click element with retry
   */
  async clickElement(selector: string): Promise<void> {
    await this.helpers.clickElement(selector);
  }

  /**
   * Wait for element to be visible
   */
  async waitForElement(selector: string): Promise<Locator> {
    return this.helpers.waitForElement(selector);
  }

  /**
   * Verify element contains text
   */
  async verifyElementText(selector: string, expectedText: string): Promise<void> {
    await this.helpers.waitForText(selector, expectedText);
  }

  /**
   * Verify element has attribute
   */
  async verifyElementAttribute(selector: string, attribute: string, value: string): Promise<void> {
    await this.helpers.waitForAttribute(selector, attribute, value);
  }

  /**
   * Get element count
   */
  async getElementCount(selector: string): Promise<number> {
    return this.helpers.getElementCount(selector);
  }

  /**
   * Wait for network request to complete
   */
  async waitForApiCall(urlPattern: string | RegExp): Promise<any> {
    return this.helpers.waitForApiResponse(urlPattern);
  }

  /**
   * Verify toast notification
   */
  async verifyToast(message: string, type: 'success' | 'error' | 'warning' | 'info' = 'success'): Promise<void> {
    await this.helpers.validateToast(message, type);
  }

  /**
   * Common form validation
   */
  async validateFormErrors(expectedErrors: Record<string, string>): Promise<void> {
    for (const [field, expectedError] of Object.entries(expectedErrors)) {
      const errorElement = this.page.locator(`[data-testid="error-${field}"]`);
      await expect(errorElement).toBeVisible();
      await expect(errorElement).toContainText(expectedError);
    }
  }

  /**
   * Clear all form fields
   */
  async clearForm(formSelector: string): Promise<void> {
    const form = this.page.locator(formSelector);
    const inputs = form.locator('input, textarea, select');
    const count = await inputs.count();
    
    for (let i = 0; i < count; i++) {
      const input = inputs.nth(i);
      const type = await input.getAttribute('type');
      
      if (type === 'checkbox' || type === 'radio') {
        await input.uncheck();
      } else {
        await input.clear();
      }
    }
  }

  /**
   * Upload file using drag and drop
   */
  async uploadFileViaDragDrop(fileSelector: string, dropZoneSelector: string): Promise<void> {
    const fileInput = this.page.locator(fileSelector);
    const dropZone = this.page.locator(dropZoneSelector);
    
    // Simulate drag and drop
    await fileInput.hover();
    await this.page.mouse.down();
    await dropZone.hover();
    await this.page.mouse.up();
  }

  /**
   * Verify responsive design
   */
  async verifyResponsiveLayout(breakpoint: 'mobile' | 'tablet' | 'desktop'): Promise<void> {
    const viewports = {
      mobile: { width: 375, height: 667 },
      tablet: { width: 768, height: 1024 },
      desktop: { width: 1920, height: 1080 },
    };
    
    await this.page.setViewportSize(viewports[breakpoint]);
    await this.waitForPageLoad();
    
    // Verify layout elements are properly arranged for the viewport
    const nav = this.navigation.logo;
    await expect(nav).toBeVisible();
  }
}