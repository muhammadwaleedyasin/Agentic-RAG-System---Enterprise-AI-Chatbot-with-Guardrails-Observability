import { chromium, FullConfig } from '@playwright/test';
import { TestDataManager } from './test-data-manager';
import { DatabaseSeeder } from './database-seeder';

/**
 * Global setup for Playwright tests
 * Prepares test environment and data
 */
async function globalSetup(config: FullConfig) {
  console.log('🚀 Starting global setup for Enterprise RAG Chatbot UI tests...');
  
  const browser = await chromium.launch();
  const context = await browser.newContext();
  const page = await context.newPage();
  
  try {
    // Initialize test data manager
    const testDataManager = new TestDataManager();
    await testDataManager.initialize();
    
    // Seed database with test data
    const databaseSeeder = new DatabaseSeeder();
    await databaseSeeder.seedTestData();
    
    // Verify backend is running
    const baseURL = config.projects[0].use?.baseURL || 'http://localhost:8000';
    console.log(`📡 Checking backend health at ${baseURL}...`);
    
    let retries = 10;
    while (retries > 0) {
      try {
        await page.goto(`${baseURL}/api/v1/health`);
        const response = await page.textContent('body');
        if (response?.includes('healthy')) {
          console.log('✅ Backend is healthy');
          break;
        }
      } catch (error) {
        console.log(`⏳ Waiting for backend... (${retries} retries left)`);
        await page.waitForTimeout(5000);
        retries--;
      }
    }
    
    if (retries === 0) {
      throw new Error('❌ Backend failed to start or become healthy');
    }
    
    // Create admin user for tests
    await createTestUsers(page, baseURL);
    
    // Verify WebSocket connectivity
    await verifyWebSocketConnectivity(page, baseURL);
    
    console.log('✅ Global setup completed successfully');
    
  } catch (error) {
    console.error('❌ Global setup failed:', error);
    throw error;
  } finally {
    await browser.close();
  }
}

async function createTestUsers(page: any, baseURL: string) {
  console.log('👤 Creating test users...');
  
  const testUsers = [
    {
      username: 'admin_test',
      password: 'AdminTest123!',
      email: 'admin@test.com',
      role: 'admin'
    },
    {
      username: 'user_test',
      password: 'UserTest123!',
      email: 'user@test.com',
      role: 'user'
    },
    {
      username: 'viewer_test',
      password: 'ViewerTest123!',
      email: 'viewer@test.com',
      role: 'viewer'
    }
  ];
  
  for (const user of testUsers) {
    try {
      // Create user via API
      await page.evaluate(async (userData) => {
        const response = await fetch('/api/v1/admin/users', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(userData)
        });
        return response.ok;
      }, user);
      
      console.log(`✅ Created test user: ${user.username}`);
    } catch (error) {
      console.log(`⚠️ User ${user.username} may already exist`);
    }
  }
}

async function verifyWebSocketConnectivity(page: any, baseURL: string) {
  console.log('🔌 Verifying WebSocket connectivity...');
  
  try {
    await page.evaluate((baseUrl) => {
      return new Promise((resolve, reject) => {
        const wsUrl = baseUrl.replace('http', 'ws') + '/ws/test-connection';
        const ws = new WebSocket(wsUrl);
        
        ws.onopen = () => {
          console.log('WebSocket connected successfully');
          ws.close();
          resolve(true);
        };
        
        ws.onerror = (error) => {
          console.error('WebSocket connection failed:', error);
          reject(error);
        };
        
        // Timeout after 10 seconds
        setTimeout(() => {
          ws.close();
          reject(new Error('WebSocket connection timeout'));
        }, 10000);
      });
    }, baseURL);
    
    console.log('✅ WebSocket connectivity verified');
  } catch (error) {
    console.warn('⚠️ WebSocket connectivity could not be verified:', error);
  }
}

export default globalSetup;