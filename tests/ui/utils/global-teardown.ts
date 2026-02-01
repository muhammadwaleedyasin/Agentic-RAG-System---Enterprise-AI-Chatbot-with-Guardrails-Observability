import { FullConfig } from '@playwright/test';
import { TestDataManager } from './test-data-manager';
import { DatabaseSeeder } from './database-seeder';

/**
 * Global teardown for Playwright tests
 * Cleans up test environment and data
 */
async function globalTeardown(config: FullConfig) {
  console.log('🧹 Starting global teardown for Enterprise RAG Chatbot UI tests...');
  
  try {
    // Clean up test data
    const testDataManager = new TestDataManager();
    await testDataManager.cleanup();
    
    // Clean up database test data
    const databaseSeeder = new DatabaseSeeder();
    await databaseSeeder.cleanupTestData();
    
    // Clean up test files
    await cleanupTestFiles();
    
    console.log('✅ Global teardown completed successfully');
    
  } catch (error) {
    console.error('❌ Global teardown failed:', error);
    // Don't throw to prevent test failures
  }
}

async function cleanupTestFiles() {
  const fs = require('fs').promises;
  const path = require('path');
  
  const testDirectories = [
    './test-results',
    './playwright-report',
    './test-data',
    './screenshots',
    './videos'
  ];
  
  for (const dir of testDirectories) {
    try {
      await fs.rmdir(dir, { recursive: true });
      console.log(`🗑️ Cleaned up directory: ${dir}`);
    } catch (error) {
      // Directory might not exist, which is fine
    }
  }
}

export default globalTeardown;