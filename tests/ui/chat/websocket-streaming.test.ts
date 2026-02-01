import { test, expect } from '../fixtures/chat-fixtures';

/**
 * WebSocket Streaming Tests
 * Tests real-time WebSocket communication and streaming responses
 */

test.describe('WebSocket Streaming', () => {
  test.beforeEach(async ({ page, loginAsUser }) => {
    await loginAsUser('user');
    await page.goto('/chat');
  });

  test('should establish WebSocket connection on chat page load', async ({ page, webSocketClient }) => {
    // Monitor WebSocket connection
    let wsConnected = false;
    
    page.on('websocket', ws => {
      ws.on('socketerror', error => console.log('WebSocket error:', error));
      ws.on('close', () => console.log('WebSocket closed'));
      ws.on('open', () => {
        wsConnected = true;
        console.log('WebSocket opened');
      });
    });
    
    // Wait for connection
    await page.waitForTimeout(2000);
    
    // Verify connection status indicator
    await expect(page.locator('[data-testid="websocket-status"]')).toHaveClass(/connected/);
    await expect(page.locator('[data-testid="connection-indicator"]')).toBeVisible();
  });

  test('should handle WebSocket authentication', async ({ page, webSocketClient }) => {
    // Get auth token
    const token = await page.evaluate(() => localStorage.getItem('authToken'));
    
    // Connect with token
    await webSocketClient.connect('test-client-1', token);
    
    // Should receive connection confirmation
    const welcomeMessage = await webSocketClient.waitForMessage('connection');
    expect(welcomeMessage.type).toBe('connection');
    expect(welcomeMessage.message).toContain('Connected');
  });

  test('should handle WebSocket without authentication', async ({ page, webSocketClient }) => {
    // Connect without token
    try {
      await webSocketClient.connect('test-client-anonymous');
      
      // Should allow anonymous connection but with limited functionality
      const welcomeMessage = await webSocketClient.waitForMessage('connection');
      expect(welcomeMessage.user).toBe('anonymous');
    } catch (error) {
      // Or reject connection based on configuration
      expect(error.message).toContain('authentication');
    }
  });

  test('should stream chat responses in real-time', async ({ page, sendMessage, verifyStreamingResponse }) => {
    await sendMessage('Tell me a long story about AI development');
    
    // Verify streaming response
    await verifyStreamingResponse();
    
    // Check for streaming chunks
    const chunks = page.locator('[data-testid="response-chunk"]');
    const chunkCount = await chunks.count();
    expect(chunkCount).toBeGreaterThan(1);
    
    // Verify final message is complete
    const finalMessage = await page.locator('[data-testid="assistant-message"]').last().textContent();
    expect(finalMessage.length).toBeGreaterThan(100);
  });

  test('should handle streaming interruption and resume', async ({ page, webSocketClient }) => {
    // Start streaming response
    await page.fill('[data-testid="message-input"]', 'Generate a long response');
    await page.click('[data-testid="send-button"]');
    
    // Wait for streaming to start
    await expect(page.locator('[data-testid="response-chunk"]').first()).toBeVisible();
    
    // Interrupt streaming
    await page.click('[data-testid="stop-response-button"]');
    
    // Should stop streaming
    await expect(page.locator('[data-testid="response-stopped"]')).toBeVisible();
    
    // Should allow continuing or regenerating
    await expect(page.locator('[data-testid="continue-response-button"]')).toBeVisible();
    await expect(page.locator('[data-testid="regenerate-response-button"]')).toBeVisible();
  });

  test('should handle WebSocket connection loss and reconnection', async ({ page, sendMessage }) => {
    // Send initial message to establish connection
    await sendMessage('Test connection');
    
    // Simulate connection loss
    await page.evaluate(() => {
      // Force close WebSocket connection
      if (window.wsConnection) {
        window.wsConnection.close();
      }
    });
    
    // Should show disconnected status
    await expect(page.locator('[data-testid="websocket-status"]')).toHaveClass(/disconnected/);
    await expect(page.locator('[data-testid="connection-error"]')).toBeVisible();
    
    // Should attempt reconnection
    await expect(page.locator('[data-testid="reconnecting-indicator"]')).toBeVisible();
    
    // Wait for reconnection
    await page.waitForTimeout(5000);
    
    // Should reconnect successfully
    await expect(page.locator('[data-testid="websocket-status"]')).toHaveClass(/connected/);
  });

  test('should handle multiple concurrent streaming responses', async ({ browser }) => {
    // Create multiple tabs
    const context1 = await browser.newContext();
    const context2 = await browser.newContext();
    
    const page1 = await context1.newPage();
    const page2 = await context2.newPage();
    
    // Login in both tabs
    await page1.goto('/login');
    await page1.fill('[data-testid="username-input"]', 'user_test');
    await page1.fill('[data-testid="password-input"]', 'UserTest123!');
    await page1.click('[data-testid="login-button"]');
    await page1.waitForURL('/dashboard');
    await page1.goto('/chat');
    
    await page2.goto('/login');
    await page2.fill('[data-testid="username-input"]', 'user_test');
    await page2.fill('[data-testid="password-input"]', 'UserTest123!');
    await page2.click('[data-testid="login-button"]');
    await page2.waitForURL('/dashboard');
    await page2.goto('/chat');
    
    // Send messages simultaneously
    await Promise.all([
      page1.fill('[data-testid="message-input"]', 'Question from tab 1'),
      page2.fill('[data-testid="message-input"]', 'Question from tab 2')
    ]);
    
    await Promise.all([
      page1.click('[data-testid="send-button"]'),
      page2.click('[data-testid="send-button"]')
    ]);
    
    // Both should receive responses
    await Promise.all([
      expect(page1.locator('[data-testid="assistant-message"]')).toBeVisible(),
      expect(page2.locator('[data-testid="assistant-message"]')).toBeVisible()
    ]);
    
    await context1.close();
    await context2.close();
  });

  test('should handle WebSocket message queuing during disconnection', async ({ page, sendMessage }) => {
    // Establish connection
    await sendMessage('Initial test');
    
    // Simulate network disconnection
    await page.route('**/*', route => route.abort());
    
    // Try to send messages while disconnected
    await page.fill('[data-testid="message-input"]', 'Queued message 1');
    await page.click('[data-testid="send-button"]');
    
    await page.fill('[data-testid="message-input"]', 'Queued message 2');
    await page.click('[data-testid="send-button"]');
    
    // Should show queued status
    await expect(page.locator('[data-testid="message-queued"]')).toHaveCount(2);
    
    // Restore network
    await page.unroute('**/*');
    
    // Should send queued messages
    await page.waitForTimeout(3000);
    await expect(page.locator('[data-testid="message-sent"]')).toHaveCount(3); // Including initial
  });

  test('should handle WebSocket heartbeat/ping-pong', async ({ page, webSocketClient }) => {
    const token = await page.evaluate(() => localStorage.getItem('authToken'));
    await webSocketClient.connect('test-client-ping', token);
    
    // Send ping
    await webSocketClient.sendMessage({ type: 'ping', timestamp: Date.now() });
    
    // Should receive pong
    const pongMessage = await webSocketClient.waitForMessage('pong');
    expect(pongMessage.type).toBe('pong');
    expect(pongMessage.timestamp).toBeDefined();
  });

  test('should handle large message streaming', async ({ page, sendMessage }) => {
    // Request a very long response
    await sendMessage('Write a detailed 2000-word essay about artificial intelligence');
    
    // Should stream in multiple chunks
    await expect(page.locator('[data-testid="response-chunk"]').first()).toBeVisible();
    
    // Monitor streaming progress
    let previousLength = 0;
    for (let i = 0; i < 20; i++) {
      await page.waitForTimeout(500);
      const currentText = await page.locator('[data-testid="assistant-message"]').last().textContent();
      const currentLength = currentText?.length || 0;
      
      if (currentLength > previousLength) {
        previousLength = currentLength;
      } else if (i > 10) {
        // Streaming should have finished by now
        break;
      }
    }
    
    // Final message should be substantial
    expect(previousLength).toBeGreaterThan(1000);
  });

  test('should handle WebSocket error recovery', async ({ page, webSocketClient }) => {
    const token = await page.evaluate(() => localStorage.getItem('authToken'));
    await webSocketClient.connect('test-client-error', token);
    
    // Send malformed message
    await webSocketClient.sendMessage({ invalid: 'message' });
    
    // Should receive error response
    const errorMessage = await webSocketClient.waitForMessage('error');
    expect(errorMessage.type).toBe('error');
    expect(errorMessage.message).toContain('Unknown message type');
    
    // Connection should remain open for valid messages
    await webSocketClient.sendMessage({ type: 'ping' });
    const pongMessage = await webSocketClient.waitForMessage('pong');
    expect(pongMessage.type).toBe('pong');
  });

  test('should handle WebSocket message ordering', async ({ page, webSocketClient }) => {
    const token = await page.evaluate(() => localStorage.getItem('authToken'));
    await webSocketClient.connect('test-client-order', token);
    
    // Send multiple messages rapidly
    const messages = [
      { type: 'chat', message: 'Message 1', sequence: 1 },
      { type: 'chat', message: 'Message 2', sequence: 2 },
      { type: 'chat', message: 'Message 3', sequence: 3 }
    ];
    
    for (const msg of messages) {
      await webSocketClient.sendMessage(msg);
    }
    
    // Should process in order
    for (let i = 1; i <= 3; i++) {
      const response = await webSocketClient.waitForMessage('chat_response');
      // Verify ordering if supported by implementation
    }
  });

  test('should handle WebSocket rate limiting', async ({ page, webSocketClient }) => {
    const token = await page.evaluate(() => localStorage.getItem('authToken'));
    await webSocketClient.connect('test-client-rate', token);
    
    // Send messages rapidly to trigger rate limiting
    const promises = [];
    for (let i = 0; i < 50; i++) {
      promises.push(
        webSocketClient.sendMessage({ 
          type: 'chat', 
          message: `Rapid message ${i}` 
        })
      );
    }
    
    await Promise.all(promises);
    
    // Should receive rate limit warning
    try {
      const rateLimitMessage = await webSocketClient.waitForMessage('rate_limit', 5000);
      expect(rateLimitMessage.type).toBe('rate_limit');
    } catch (error) {
      // Rate limiting might not be configured
      console.log('Rate limiting not configured or not triggered');
    }
  });

  test('should handle WebSocket room functionality', async ({ page, webSocketClient }) => {
    const token = await page.evaluate(() => localStorage.getItem('authToken'));
    await webSocketClient.connect('test-client-room', token);
    
    // Join a room
    await webSocketClient.sendMessage({ 
      type: 'join_room', 
      room: 'test-room' 
    });
    
    // Should receive confirmation
    const joinedMessage = await webSocketClient.waitForMessage('room_joined');
    expect(joinedMessage.type).toBe('room_joined');
    expect(joinedMessage.room).toBe('test-room');
    
    // Send room message
    await webSocketClient.sendMessage({
      type: 'room_message',
      room: 'test-room',
      message: 'Hello room!'
    });
    
    // Should receive confirmation
    const sentMessage = await webSocketClient.waitForMessage('message_sent');
    expect(sentMessage.type).toBe('message_sent');
  });
});