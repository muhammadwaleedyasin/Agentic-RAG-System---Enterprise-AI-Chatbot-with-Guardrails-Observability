import { FullConfig } from '@playwright/test';

async function globalSetup(config: FullConfig) {
  // Global setup for Playwright tests
  console.log('Setting up Playwright tests...');
  
  // Any global setup needed before tests run
  // For example: seeding test database, starting services, etc.
  
  return Promise.resolve();
}

export default globalSetup;