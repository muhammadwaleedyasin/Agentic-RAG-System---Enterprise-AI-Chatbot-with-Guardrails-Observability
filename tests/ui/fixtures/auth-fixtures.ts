import { test as base, expect } from '@playwright/test';

/**
 * Authentication fixtures for UI tests
 * Provides pre-authenticated user contexts and utilities
 */

export interface AuthUser {
  username: string;
  password: string;
  email: string;
  role: string;
  token?: string;
}

export const testUsers: Record<string, AuthUser> = {
  admin: {
    username: 'admin_test',
    password: 'AdminTest123!',
    email: 'admin@test.com',
    role: 'admin'
  },
  user: {
    username: 'user_test',
    password: 'UserTest123!',
    email: 'user@test.com',
    role: 'user'
  },
  viewer: {
    username: 'viewer_test',
    password: 'ViewerTest123!',
    email: 'viewer@test.com',
    role: 'viewer'
  }
};

type AuthFixtures = {
  authenticatedAdminPage: any;
  authenticatedUserPage: any;
  authenticatedViewerPage: any;
  loginAsUser: (userType: keyof typeof testUsers) => Promise<void>;
  logout: () => Promise<void>;
  verifyAuthentication: (expectedRole: string) => Promise<void>;
};

export const test = base.extend<AuthFixtures>({
  authenticatedAdminPage: async ({ browser }, use) => {
    const context = await browser.newContext();
    const page = await context.newPage();
    
    await loginUser(page, testUsers.admin);
    await use(page);
    
    await context.close();
  },
  
  authenticatedUserPage: async ({ browser }, use) => {
    const context = await browser.newContext();
    const page = await context.newPage();
    
    await loginUser(page, testUsers.user);
    await use(page);
    
    await context.close();
  },
  
  authenticatedViewerPage: async ({ browser }, use) => {
    const context = await browser.newContext();
    const page = await context.newPage();
    
    await loginUser(page, testUsers.viewer);
    await use(page);
    
    await context.close();
  },
  
  loginAsUser: async ({ page }, use) => {
    const loginFunction = async (userType: keyof typeof testUsers) => {
      await loginUser(page, testUsers[userType]);
    };
    
    await use(loginFunction);
  },
  
  logout: async ({ page }, use) => {
    const logoutFunction = async () => {
      await logoutUser(page);
    };
    
    await use(logoutFunction);
  },
  
  verifyAuthentication: async ({ page }, use) => {
    const verifyFunction = async (expectedRole: string) => {
      await verifyUserAuthentication(page, expectedRole);
    };
    
    await use(verifyFunction);
  }
});

async function loginUser(page: any, user: AuthUser) {
  console.log(`🔐 Logging in user: ${user.username}`);
  
  // Navigate to login page
  await page.goto('/login');
  
  // Fill login form
  await page.fill('[data-testid="username-input"]', user.username);
  await page.fill('[data-testid="password-input"]', user.password);
  
  // Submit login
  await page.click('[data-testid="login-button"]');
  
  // Wait for successful login
  await page.waitForURL('/dashboard', { timeout: 10000 });
  
  // Verify user info is displayed
  await expect(page.locator('[data-testid="user-menu"]')).toBeVisible();
  await expect(page.locator('[data-testid="user-role"]')).toContainText(user.role);
  
  console.log(`✅ Successfully logged in as ${user.username}`);
}

async function logoutUser(page: any) {
  console.log('🚪 Logging out user...');
  
  // Click user menu
  await page.click('[data-testid="user-menu"]');
  
  // Click logout
  await page.click('[data-testid="logout-button"]');
  
  // Wait for redirect to login page
  await page.waitForURL('/login', { timeout: 10000 });
  
  // Verify logout
  await expect(page.locator('[data-testid="login-form"]')).toBeVisible();
  
  console.log('✅ Successfully logged out');
}

async function verifyUserAuthentication(page: any, expectedRole: string) {
  // Check if user menu is visible
  await expect(page.locator('[data-testid="user-menu"]')).toBeVisible();
  
  // Check user role
  await expect(page.locator('[data-testid="user-role"]')).toContainText(expectedRole);
  
  // Verify access to protected endpoints
  const response = await page.request.get('/api/v1/auth/me');
  expect(response.ok()).toBeTruthy();
  
  const userData = await response.json();
  expect(userData.role).toBe(expectedRole);
}

export { expect };