import { Page, Locator, expect } from '@playwright/test';

export class TestHelpers {
  constructor(private page: Page) {}

  /**
   * Wait for element to be visible and return it
   */
  async waitForElement(selector: string, timeout = 10000): Promise<Locator> {
    const element = this.page.locator(selector);
    await expect(element).toBeVisible({ timeout });
    return element;
  }

  /**
   * Wait for element to contain specific text
   */
  async waitForText(selector: string, text: string, timeout = 10000): Promise<void> {
    const element = this.page.locator(selector);
    await expect(element).toContainText(text, { timeout });
  }

  /**
   * Wait for API response with specific status
   */
  async waitForApiResponse(urlPattern: string | RegExp, status = 200, timeout = 30000): Promise<any> {
    const response = await this.page.waitForResponse(
      response => 
        response.url().match(urlPattern) !== null && 
        response.status() === status,
      { timeout }
    );
    return response.json();
  }

  /**
   * Fill form field with validation
   */
  async fillField(selector: string, value: string): Promise<void> {
    const field = await this.waitForElement(selector);
    await field.clear();
    await field.fill(value);
    await expect(field).toHaveValue(value);
  }

  /**
   * Click element with retry logic
   */
  async clickElement(selector: string, timeout = 10000): Promise<void> {
    const element = await this.waitForElement(selector, timeout);
    await element.click();
  }

  /**
   * Upload file to input element
   */
  async uploadFile(selector: string, filePath: string): Promise<void> {
    const fileInput = await this.waitForElement(selector);
    await fileInput.setInputFiles(filePath);
  }

  /**
   * Wait for loading to complete
   */
  async waitForLoadingComplete(loadingSelector = '[data-testid="loading"]', timeout = 30000): Promise<void> {
    try {
      // Wait for loading indicator to appear (optional)
      await this.page.locator(loadingSelector).waitFor({ state: 'visible', timeout: 5000 });
    } catch {
      // Loading indicator might not appear if operation is fast
    }
    
    // Wait for loading to disappear
    await this.page.locator(loadingSelector).waitFor({ state: 'hidden', timeout });
  }

  /**
   * Scroll element into view
   */
  async scrollIntoView(selector: string): Promise<void> {
    const element = await this.waitForElement(selector);
    await element.scrollIntoViewIfNeeded();
  }

  /**
   * Take screenshot with custom name
   */
  async takeScreenshot(name: string): Promise<void> {
    await this.page.screenshot({ 
      path: `test-results/screenshots/${name}-${Date.now()}.png`,
      fullPage: true 
    });
  }

  /**
   * Wait for WebSocket connection
   */
  async waitForWebSocket(urlPattern: string | RegExp, timeout = 10000): Promise<any> {
    return this.page.waitForEvent('websocket', { 
      predicate: ws => ws.url().match(urlPattern) !== null,
      timeout 
    });
  }

  /**
   * Validate toast notification
   */
  async validateToast(expectedMessage: string, type: 'success' | 'error' | 'info' = 'success'): Promise<void> {
    const toastSelector = `[data-testid="toast-${type}"]`;
    await this.waitForElement(toastSelector);
    await this.waitForText(toastSelector, expectedMessage);
  }

  /**
   * Handle browser dialog (alert, confirm, prompt)
   */
  async handleDialog(accept = true, promptText?: string): Promise<string> {
    return new Promise((resolve) => {
      this.page.once('dialog', async dialog => {
        const message = dialog.message();
        if (dialog.type() === 'prompt' && promptText) {
          await dialog.accept(promptText);
        } else if (accept) {
          await dialog.accept();
        } else {
          await dialog.dismiss();
        }
        resolve(message);
      });
    });
  }

  /**
   * Wait for network idle (no requests for specified time)
   */
  async waitForNetworkIdle(timeout = 500): Promise<void> {
    await this.page.waitForLoadState('networkidle', { timeout });
  }

  /**
   * Generate random test data
   */
  generateTestData() {
    const timestamp = Date.now();
    return {
      email: `test${timestamp}@example.com`,
      username: `testuser${timestamp}`,
      password: 'TestPassword123!',
      documentTitle: `Test Document ${timestamp}`,
      chatMessage: `Test message at ${new Date().toISOString()}`,
    };
  }

  /**
   * Validate API response structure
   */
  validateApiResponse(response: any, expectedFields: string[]): void {
    for (const field of expectedFields) {
      expect(response).toHaveProperty(field);
    }
  }

  /**
   * Mock API response
   */
  async mockApiResponse(url: string | RegExp, responseData: any, status = 200): Promise<void> {
    await this.page.route(url, route => {
      route.fulfill({
        status,
        contentType: 'application/json',
        body: JSON.stringify(responseData),
      });
    });
  }

  /**
   * Clear all mocked routes
   */
  async clearMocks(): Promise<void> {
    await this.page.unroute('**/*');
  }

  /**
   * Wait for element to have specific attribute value
   */
  async waitForAttribute(selector: string, attribute: string, value: string, timeout = 10000): Promise<void> {
    const element = this.page.locator(selector);
    await expect(element).toHaveAttribute(attribute, value, { timeout });
  }

  /**
   * Get element count
   */
  async getElementCount(selector: string): Promise<number> {
    return this.page.locator(selector).count();
  }

  /**
   * Validate pagination
   */
  async validatePagination(containerSelector: string, itemsPerPage: number): Promise<void> {
    const totalItems = await this.getElementCount(`${containerSelector} [data-testid="item"]`);
    expect(totalItems).toBeLessThanOrEqual(itemsPerPage);
    
    if (totalItems === itemsPerPage) {
      // Check if next button exists and is enabled
      const nextButton = this.page.locator('[data-testid="pagination-next"]');
      await expect(nextButton).toBeVisible();
      await expect(nextButton).toBeEnabled();
    }
  }
}

export default TestHelpers;