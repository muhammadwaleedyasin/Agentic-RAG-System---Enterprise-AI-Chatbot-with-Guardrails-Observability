import { test as base, Page } from '@playwright/test';
import { AuthService } from '../utils/auth-service';
import { TestDataService } from '../utils/test-data-service';
import TestHelpers from '../utils/test-helpers';
import { ChatPage } from '../pages/chat-page';
import { DocumentsPage } from '../pages/documents-page';
import { LoginPage } from '../pages/login-page';
import { DashboardPage } from '../pages/dashboard-page';
import { AdminPage } from '../pages/admin-page';

// Define custom fixtures
type TestFixtures = {
  authService: AuthService;
  testDataService: TestDataService;
  testHelpers: TestHelpers;
  chatPage: ChatPage;
  documentsPage: DocumentsPage;
  loginPage: LoginPage;
  dashboardPage: DashboardPage;
  adminPage: AdminPage;
  authenticatedPage: Page;
  adminPage: Page;
};

export const test = base.extend<TestFixtures>({
  // Auth service fixture
  authService: async ({}, use) => {
    const authService = new AuthService();
    await use(authService);
    await authService.dispose();
  },

  // Test data service fixture
  testDataService: async ({}, use) => {
    const testDataService = new TestDataService();
    await use(testDataService);
    await testDataService.dispose();
  },

  // Test helpers fixture
  testHelpers: async ({ page }, use) => {
    const testHelpers = new TestHelpers(page);
    await use(testHelpers);
  },

  // Page object fixtures
  chatPage: async ({ page }, use) => {
    const chatPage = new ChatPage(page);
    await use(chatPage);
  },

  documentsPage: async ({ page }, use) => {
    const documentsPage = new DocumentsPage(page);
    await use(documentsPage);
  },

  loginPage: async ({ page }, use) => {
    const loginPage = new LoginPage(page);
    await use(loginPage);
  },

  dashboardPage: async ({ page }, use) => {
    const dashboardPage = new DashboardPage(page);
    await use(dashboardPage);
  },

  adminPage: async ({ page }, use) => {
    const adminPage = new AdminPage(page);
    await use(adminPage);
  },

  // Authenticated page fixture (logged in as regular user)
  authenticatedPage: async ({ page, authService }, use) => {
    // Login as test user
    const token = await authService.getAuthToken('user_test', 'test_password_123');
    
    // Set authentication cookie/header
    await page.context().addCookies([{
      name: 'auth_token',
      value: token,
      domain: 'localhost',
      path: '/',
    }]);

    await page.goto('/');
    await use(page);
  },

  // Admin page fixture (logged in as admin)
  adminPage: async ({ page, authService }, use) => {
    // Login as admin user
    const token = await authService.getAuthToken('admin_test', 'test_password_123');
    
    // Set authentication cookie/header
    await page.context().addCookies([{
      name: 'auth_token',
      value: token,
      domain: 'localhost',
      path: '/',
    }]);

    await page.goto('/admin');
    await use(page);
  },
});

export { expect } from '@playwright/test';

// Custom matchers
export const customMatchers = {
  /**
   * Check if element has loading state
   */
  async toBeLoading(locator: any) {
    const isLoading = await locator.getAttribute('data-loading');
    return {
      pass: isLoading === 'true',
      message: () => `Expected element to ${this.isNot ? 'not ' : ''}be loading`,
    };
  },

  /**
   * Check if element has error state
   */
  async toHaveError(locator: any, expectedError?: string) {
    const errorText = await locator.getAttribute('data-error') || 
                     await locator.locator('[data-testid="error"]').textContent();
    
    if (expectedError) {
      return {
        pass: errorText?.includes(expectedError) || false,
        message: () => `Expected element to have error "${expectedError}", got "${errorText}"`,
      };
    }
    
    return {
      pass: !!errorText,
      message: () => `Expected element to ${this.isNot ? 'not ' : ''}have an error`,
    };
  },

  /**
   * Check if API response is valid
   */
  async toBeValidApiResponse(response: any, requiredFields: string[]) {
    const hasAllFields = requiredFields.every(field => response.hasOwnProperty(field));
    
    return {
      pass: hasAllFields && typeof response === 'object',
      message: () => `Expected response to be valid API response with fields: ${requiredFields.join(', ')}`,
    };
  },
};

// Test data generators
export const testData = {
  /**
   * Generate random user data
   */
  generateUser() {
    const timestamp = Date.now();
    return {
      username: `testuser${timestamp}`,
      email: `test${timestamp}@example.com`,
      password: 'TestPassword123!',
      firstName: 'Test',
      lastName: 'User',
    };
  },

  /**
   * Generate random document data
   */
  generateDocument() {
    const timestamp = Date.now();
    return {
      title: `Test Document ${timestamp}`,
      content: `This is test content generated at ${new Date().toISOString()}`,
      tags: ['test', 'automation'],
      description: `Test document for automated testing - ${timestamp}`,
    };
  },

  /**
   * Generate random chat message
   */
  generateChatMessage() {
    const messages = [
      'What is the company policy?',
      'How do I use this system?',
      'Can you help me find information about procedures?',
      'What documents are available?',
      'Tell me about the latest updates',
    ];
    
    return messages[Math.floor(Math.random() * messages.length)];
  },

  /**
   * Generate test file content
   */
  generateFileContent(type: 'txt' | 'json' | 'csv' = 'txt') {
    const timestamp = Date.now();
    
    switch (type) {
      case 'txt':
        return `Test document content created at ${timestamp}\nThis is a multi-line document for testing purposes.\nIt contains various information for RAG processing.`;
      
      case 'json':
        return JSON.stringify({
          title: `Test Data ${timestamp}`,
          description: 'Test JSON document',
          data: {
            items: [1, 2, 3],
            metadata: { created: timestamp }
          }
        }, null, 2);
      
      case 'csv':
        return `id,name,value,timestamp\n1,Item1,100,${timestamp}\n2,Item2,200,${timestamp + 1000}\n3,Item3,300,${timestamp + 2000}`;
      
      default:
        return `Test content - ${timestamp}`;
    }
  },
};

// Common test patterns
export const testPatterns = {
  /**
   * Standard login flow
   */
  async loginFlow(page: Page, username: string, password: string) {
    const loginPage = new LoginPage(page);
    await loginPage.goto();
    await loginPage.login(username, password);
    await loginPage.waitForSuccessfulLogin();
  },

  /**
   * Standard logout flow
   */
  async logoutFlow(page: Page) {
    const dashboardPage = new DashboardPage(page);
    await dashboardPage.logout();
  },

  /**
   * Upload and verify document
   */
  async uploadDocumentFlow(page: Page, filename: string, content: string) {
    const documentsPage = new DocumentsPage(page);
    await documentsPage.goto();
    
    // Create temporary file
    const testFile = await documentsPage.createTestFile(filename, content);
    
    // Upload document
    await documentsPage.uploadDocument(testFile);
    await documentsPage.waitForUploadSuccess();
    
    // Verify document appears in list
    await documentsPage.verifyDocumentInList(filename);
    
    return testFile;
  },

  /**
   * Chat interaction flow
   */
  async chatFlow(page: Page, message: string) {
    const chatPage = new ChatPage(page);
    await chatPage.goto();
    await chatPage.sendMessage(message);
    await chatPage.waitForResponse();
    
    return chatPage.getLastResponse();
  },
};