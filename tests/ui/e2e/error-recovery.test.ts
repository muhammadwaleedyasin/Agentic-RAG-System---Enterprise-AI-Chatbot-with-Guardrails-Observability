import { test, expect } from '@playwright/test';

/**
 * Error Scenarios and Recovery Tests
 * Tests system behavior during failures, network issues, and recovery workflows
 */

test.describe('Error Scenarios and Recovery', () => {
  test.beforeEach(async ({ page }) => {
    // Login as user
    await page.goto('/login');
    await page.fill('[data-testid="username-input"]', 'user_test');
    await page.fill('[data-testid="password-input"]', 'UserTest123!');
    await page.click('[data-testid="login-button"]');
    await page.waitForURL('/dashboard');
  });

  test('should handle network connectivity issues gracefully', async ({ page }) => {
    await page.goto('/chat');
    
    // Start a conversation
    await page.fill('[data-testid="message-input"]', 'Initial message before network failure');
    await page.click('[data-testid="send-button"]');
    await expect(page.locator('[data-testid="assistant-message"]')).toBeVisible();
    
    // Simulate network failure
    await page.route('**/*', route => route.abort());
    
    // Try to send message during network failure
    await page.fill('[data-testid="message-input"]', 'Message during network failure');
    await page.click('[data-testid="send-button"]');
    
    // Should show network error state
    await expect(page.locator('[data-testid="network-error"]')).toBeVisible();
    await expect(page.locator('[data-testid="offline-indicator"]')).toBeVisible();
    
    // Should queue message locally
    await expect(page.locator('[data-testid="message-queued"]')).toBeVisible();
    
    // Should provide retry mechanism
    await expect(page.locator('[data-testid="retry-connection"]')).toBeVisible();
    
    // Restore network
    await page.unroute('**/*');
    
    // Auto-retry or manual retry
    await page.click('[data-testid="retry-connection"]');
    
    // Should show reconnection indicator
    await expect(page.locator('[data-testid="reconnecting"]')).toBeVisible();
    await expect(page.locator('[data-testid="connection-restored"]')).toBeVisible();
    
    // Queued message should be sent
    await expect(page.locator('[data-testid="message-sent"]')).toBeVisible();
  });

  test('should handle API server errors gracefully', async ({ page }) => {
    await page.goto('/chat');
    
    // Simulate 500 server error
    await page.route('/api/v1/chat', route => {
      route.fulfill({ status: 500, body: 'Internal Server Error' });
    });
    
    await page.fill('[data-testid="message-input"]', 'Message that will cause server error');
    await page.click('[data-testid="send-button"]');
    
    // Should show server error
    await expect(page.locator('[data-testid="server-error"]')).toBeVisible();
    await expect(page.locator('[data-testid="error-message"]'))
      .toContainText('Server temporarily unavailable');
    
    // Should provide helpful options
    await expect(page.locator('[data-testid="retry-button"]')).toBeVisible();
    await expect(page.locator('[data-testid="contact-support"]')).toBeVisible();
    
    // Remove server error
    await page.unroute('/api/v1/chat');
    
    // Retry should work
    await page.click('[data-testid="retry-button"]');
    await expect(page.locator('[data-testid="assistant-message"]')).toBeVisible();
  });

  test('should handle authentication token expiry', async ({ page }) => {
    await page.goto('/chat');
    
    // Simulate expired token
    await page.evaluate(() => {
      localStorage.setItem('authToken', 'expired.token.here');
    });
    
    // Try to make authenticated request
    await page.fill('[data-testid="message-input"]', 'Message with expired token');
    await page.click('[data-testid="send-button"]');
    
    // Should detect expired session
    await expect(page.locator('[data-testid="session-expired"]')).toBeVisible();
    
    // Should redirect to login
    await expect(page).toHaveURL('/login');
    
    // Should show session expiry message
    await expect(page.locator('[data-testid="session-expired-message"]')).toBeVisible();
    
    // Re-login should restore state
    await page.fill('[data-testid="username-input"]', 'user_test');
    await page.fill('[data-testid="password-input"]', 'UserTest123!');
    await page.click('[data-testid="login-button"]');
    
    // Should redirect back to intended page
    await expect(page).toHaveURL('/chat');
  });

  test('should handle document upload failures', async ({ page }) => {
    await page.goto('/documents');
    
    const filePath = path.join(__dirname, '../fixtures/test-document.pdf');
    await page.setInputFiles('[data-testid="file-input"]', filePath);
    
    // Simulate upload failure
    await page.route('/api/v1/documents/upload', route => {
      route.fulfill({ status: 413, body: 'File too large' });
    });
    
    await page.click('[data-testid="start-upload"]');
    
    // Should show specific error
    await expect(page.locator('[data-testid="upload-error"]')).toBeVisible();
    await expect(page.locator('[data-testid="error-message"]'))
      .toContainText('File too large');
    
    // Should provide helpful guidance
    await expect(page.locator('[data-testid="error-help"]')).toBeVisible();
    await expect(page.locator('[data-testid="size-limit-info"]')).toBeVisible();
    
    // Should allow file removal and retry
    await expect(page.locator('[data-testid="remove-file"]')).toBeVisible();
    await expect(page.locator('[data-testid="retry-upload"]')).toBeVisible();
  });

  test('should handle LLM provider failures', async ({ page }) => {
    await page.goto('/chat');
    
    // Simulate LLM provider error
    await page.route('/api/v1/chat', route => {
      route.fulfill({ 
        status: 503, 
        body: JSON.stringify({ 
          error: 'LLM provider unavailable',
          fallback_available: true 
        })
      });
    });
    
    await page.fill('[data-testid="message-input"]', 'Message that will trigger LLM error');
    await page.click('[data-testid="send-button"]');
    
    // Should show provider error with fallback option
    await expect(page.locator('[data-testid="llm-provider-error"]')).toBeVisible();
    await expect(page.locator('[data-testid="fallback-option"]')).toBeVisible();
    
    // Should offer to use fallback provider
    await page.click('[data-testid="use-fallback"]');
    
    // Simulate successful fallback
    await page.unroute('/api/v1/chat');
    
    await expect(page.locator('[data-testid="using-fallback-provider"]')).toBeVisible();
    await expect(page.locator('[data-testid="assistant-message"]')).toBeVisible();
  });

  test('should handle database connectivity issues', async ({ page }) => {
    await page.goto('/documents');
    
    // Simulate database error
    await page.route('/api/v1/documents', route => {
      route.fulfill({ 
        status: 503, 
        body: JSON.stringify({ error: 'Database connection failed' })
      });
    });
    
    // Try to load documents
    await page.reload();
    
    // Should show database error
    await expect(page.locator('[data-testid="database-error"]')).toBeVisible();
    await expect(page.locator('[data-testid="error-message"]'))
      .toContainText('Unable to load documents');
    
    // Should provide retry mechanism
    await expect(page.locator('[data-testid="retry-load"]')).toBeVisible();
    
    // Should suggest alternative actions
    await expect(page.locator('[data-testid="cached-data-option"]')).toBeVisible();
  });

  test('should handle WebSocket connection failures', async ({ page }) => {
    await page.goto('/chat');
    
    // Monitor WebSocket connection
    let wsDisconnected = false;
    page.on('websocket', ws => {
      ws.on('close', () => {
        wsDisconnected = true;
      });
    });
    
    // Force WebSocket disconnection
    await page.evaluate(() => {
      if (window.wsConnection) {
        window.wsConnection.close();
      }
    });
    
    // Send message during WebSocket failure
    await page.fill('[data-testid="message-input"]', 'Message during WebSocket failure');
    await page.click('[data-testid="send-button"]');
    
    // Should detect WebSocket failure
    await expect(page.locator('[data-testid="websocket-error"]')).toBeVisible();
    
    // Should fall back to HTTP polling
    await expect(page.locator('[data-testid="using-fallback-connection"]')).toBeVisible();
    
    // Message should still be sent via HTTP
    await expect(page.locator('[data-testid="message-sent"]')).toBeVisible();
    
    // Should attempt to reconnect WebSocket
    await expect(page.locator('[data-testid="reconnecting-websocket"]')).toBeVisible();
  });

  test('should handle search service failures', async ({ page }) => {
    await page.goto('/documents');
    
    // Simulate search service failure
    await page.route('/api/v1/search**', route => {
      route.fulfill({ status: 503, body: 'Search service unavailable' });
    });
    
    await page.fill('[data-testid="document-search"]', 'test search query');
    await page.keyboard.press('Enter');
    
    // Should show search error
    await expect(page.locator('[data-testid="search-error"]')).toBeVisible();
    
    // Should provide alternative browsing options
    await expect(page.locator('[data-testid="browse-categories"]')).toBeVisible();
    await expect(page.locator('[data-testid="recent-documents"]')).toBeVisible();
    
    // Should allow basic filtering without search
    await expect(page.locator('[data-testid="basic-filters"]')).toBeVisible();
  });

  test('should handle memory/storage quota exceeded', async ({ page }) => {
    await page.goto('/documents');
    
    // Simulate storage quota exceeded
    await page.route('/api/v1/documents/upload', route => {
      route.fulfill({ 
        status: 507, 
        body: JSON.stringify({ 
          error: 'Storage quota exceeded',
          current_usage: '95%',
          limit: '10GB'
        })
      });
    });
    
    const filePath = path.join(__dirname, '../fixtures/test-document.pdf');
    await page.setInputFiles('[data-testid="file-input"]', filePath);
    await page.click('[data-testid="start-upload"]');
    
    // Should show quota error
    await expect(page.locator('[data-testid="quota-exceeded-error"]')).toBeVisible();
    
    // Should show current usage
    await expect(page.locator('[data-testid="storage-usage"]')).toBeVisible();
    
    // Should provide cleanup options
    await expect(page.locator('[data-testid="cleanup-suggestions"]')).toBeVisible();
    await expect(page.locator('[data-testid="delete-old-documents"]')).toBeVisible();
    
    // Should offer upgrade options
    await expect(page.locator('[data-testid="upgrade-storage"]')).toBeVisible();
  });

  test('should handle concurrent user session conflicts', async ({ page, browser }) => {
    // Create second browser context for same user
    const context2 = await browser.newContext();
    const page2 = await context2.newPage();
    
    // Login same user in both contexts
    await page2.goto('/login');
    await page2.fill('[data-testid="username-input"]', 'user_test');
    await page2.fill('[data-testid="password-input"]', 'UserTest123!');
    await page2.click('[data-testid="login-button"]');
    
    // Simulate session conflict
    await page.goto('/chat');
    await page2.goto('/chat');
    
    // Start conversation in first session
    await page.fill('[data-testid="message-input"]', 'Message from session 1');
    await page.click('[data-testid="send-button"]');
    
    // Start conversation in second session
    await page2.fill('[data-testid="message-input"]', 'Message from session 2');
    await page2.click('[data-testid="send-button"]');
    
    // Should handle session conflict gracefully
    // Either by merging sessions or notifying of conflict
    const hasConflictWarning = await page.locator('[data-testid="session-conflict"]').isVisible();
    
    if (hasConflictWarning) {
      await expect(page.locator('[data-testid="session-conflict-message"]')).toBeVisible();
      await expect(page.locator('[data-testid="resolve-conflict"]')).toBeVisible();
    }
    
    await context2.close();
  });

  test('should handle malformed API responses', async ({ page }) => {
    await page.goto('/chat');
    
    // Simulate malformed JSON response
    await page.route('/api/v1/chat', route => {
      route.fulfill({ 
        status: 200, 
        body: 'Invalid JSON response {'
      });
    });
    
    await page.fill('[data-testid="message-input"]', 'Message with malformed response');
    await page.click('[data-testid="send-button"]');
    
    // Should handle parsing error gracefully
    await expect(page.locator('[data-testid="response-error"]')).toBeVisible();
    await expect(page.locator('[data-testid="error-message"]'))
      .toContainText('Unable to process response');
    
    // Should provide retry option
    await expect(page.locator('[data-testid="retry-button"]')).toBeVisible();
  });

  test('should handle browser compatibility issues', async ({ page }) => {
    // Simulate unsupported browser features
    await page.addInitScript(() => {
      // Remove WebSocket support
      delete window.WebSocket;
      
      // Remove FileReader support
      delete window.FileReader;
    });
    
    await page.goto('/chat');
    
    // Should detect missing features
    await expect(page.locator('[data-testid="compatibility-warning"]')).toBeVisible();
    
    // Should provide fallback options
    await expect(page.locator('[data-testid="fallback-upload"]')).toBeVisible();
    await expect(page.locator('[data-testid="fallback-messaging"]')).toBeVisible();
    
    // Should suggest browser upgrade
    await expect(page.locator('[data-testid="browser-upgrade-suggestion"]')).toBeVisible();
  });

  test('should provide comprehensive error reporting', async ({ page }) => {
    // Enable error reporting
    await page.goto('/settings');
    await page.check('[data-testid="enable-error-reporting"]');
    await page.click('[data-testid="save-settings"]');
    
    await page.goto('/chat');
    
    // Simulate various errors
    await page.route('/api/v1/chat', route => {
      route.fulfill({ status: 500, body: 'Test error for reporting' });
    });
    
    await page.fill('[data-testid="message-input"]', 'Message that will trigger error reporting');
    await page.click('[data-testid="send-button"]');
    
    // Should show error dialog with reporting option
    await expect(page.locator('[data-testid="error-report-dialog"]')).toBeVisible();
    
    // Should allow user to describe the issue
    await page.fill('[data-testid="error-description"]', 'Chat stopped working after clicking send');
    
    // Should collect system information
    await expect(page.locator('[data-testid="system-info"]')).toBeVisible();
    
    // Send error report
    await page.click('[data-testid="send-error-report"]');
    
    // Should confirm report was sent
    await expect(page.locator('[data-testid="error-report-sent"]')).toBeVisible();
  });
});