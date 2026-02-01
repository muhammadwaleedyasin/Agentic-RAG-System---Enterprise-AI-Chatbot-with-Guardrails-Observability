import { defineConfig, devices } from '@playwright/test';

/**
 * Enterprise RAG Chatbot - Playwright Configuration
 * Comprehensive UI testing configuration for all user journeys
 */
export default defineConfig({
  testDir: './tests/ui',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: [
    ['html'],
    ['json', { outputFile: 'test-results/results.json' }],
    ['junit', { outputFile: 'test-results/results.xml' }]
  ],
  use: {
    baseURL: process.env.BASE_URL || 'http://localhost:8000',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
    headless: process.env.CI ? true : false
  },

  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'] },
    },
    {
      name: 'webkit',
      use: { ...devices['Desktop Safari'] },
    },
    {
      name: 'mobile-chrome',
      use: { ...devices['Pixel 5'] },
    },
    {
      name: 'mobile-safari',
      use: { ...devices['iPhone 12'] },
    },
  ],

  webServer: [
    {
      command: 'python -m uvicorn src.app.main:app --host 0.0.0.0 --port 8000',
      url: 'http://localhost:8000',
      reuseExistingServer: !process.env.CI,
      timeout: 120 * 1000,
    },
    {
      command: 'npm run start:frontend',
      url: 'http://localhost:3000',
      reuseExistingServer: !process.env.CI,
      timeout: 120 * 1000,
    }
  ],

  globalSetup: require.resolve('./tests/ui/utils/global-setup.ts'),
  globalTeardown: require.resolve('./tests/ui/utils/global-teardown.ts'),

  expect: {
    timeout: 30000,
    toHaveScreenshot: { threshold: 0.2, mode: 'pixel' },
    toMatchSnapshot: { threshold: 0.2 }
  },

  timeout: 60000,

  // Global test configuration
  globalTimeout: 600000,
  
  // Maximum time one test can run for
  testTimeout: 60000,
  
  // Maximum time for navigation
  navigationTimeout: 30000,
  
  // Maximum time for action  
  actionTimeout: 10000
});