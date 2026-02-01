import { test, expect } from '../fixtures/test-fixtures';

test.describe('Chat Functionality', () => {
  test.beforeEach(async ({ loginPage }) => {
    await loginPage.goto();
    await loginPage.performValidLogin();
  });

  test.describe('Basic Chat Operations', () => {
    test('should send and receive messages', async ({ chatPage }) => {
      await chatPage.goto();
      
      const testMessage = 'What is the company policy?';
      const response = await chatPage.performChatInteraction(testMessage);
      
      expect(response.length).toBeGreaterThan(0);
      expect(response).not.toContain('error');
    });

    test('should stream responses in real-time', async ({ chatPage }) => {
      await chatPage.goto();
      await chatPage.verifyMessageStreaming('Tell me about the system features');
    });

    test('should handle empty messages', async ({ chatPage }) => {
      await chatPage.goto();
      await chatPage.testErrorHandling();
    });

    test('should send messages with Enter key', async ({ chatPage }) => {
      await chatPage.goto();
      
      await chatPage.sendMessageWithEnter('Test message via Enter key');
      await chatPage.waitForResponse();
      
      const response = await chatPage.getLastResponse();
      expect(response.length).toBeGreaterThan(0);
    });

    test('should maintain conversation history', async ({ chatPage }) => {
      await chatPage.goto();
      
      await chatPage.sendMessage('First message');
      await chatPage.waitForResponse();
      
      await chatPage.sendMessage('Second message');
      await chatPage.waitForResponse();
      
      const messages = await chatPage.getAllMessages();
      expect(messages.length).toBeGreaterThanOrEqual(4); // 2 user + 2 assistant
      
      await chatPage.verifyConversationHistory();
    });
  });

  test.describe('RAG Integration', () => {
    test('should use RAG mode by default', async ({ chatPage }) => {
      await chatPage.goto();
      await chatPage.toggleRAG(true);
      
      const response = await chatPage.performChatInteraction('What documents are available?', true);
      await chatPage.verifySources(1);
    });

    test('should display sources for RAG responses', async ({ chatPage }) => {
      await chatPage.goto();
      await chatPage.toggleRAG(true);
      
      await chatPage.sendMessage('What is in the company policy document?');
      await chatPage.waitForResponse();
      
      const sources = await chatPage.getSources();
      expect(sources.length).toBeGreaterThan(0);
      
      sources.forEach(source => {
        expect(source.title).toBeTruthy();
      });
    });

    test('should work without RAG when disabled', async ({ chatPage }) => {
      await chatPage.goto();
      await chatPage.toggleRAG(false);
      
      await chatPage.sendMessage('Hello, how are you?');
      await chatPage.waitForResponse();
      
      const response = await chatPage.getLastResponse();
      expect(response.length).toBeGreaterThan(0);
      
      // Sources panel should not be visible
      await expect(chatPage.elements.sourcesPanel).not.toBeVisible();
    });

    test('should handle queries with no relevant documents', async ({ chatPage }) => {
      await chatPage.goto();
      await chatPage.toggleRAG(true);
      
      await chatPage.sendMessage('What is the weather like on Mars?');
      await chatPage.waitForResponse();
      
      const response = await chatPage.getLastResponse();
      expect(response).toContain('I don\'t have information' || 'cannot find relevant documents');
    });
  });

  test.describe('Conversation Management', () => {
    test('should start new conversation', async ({ chatPage }) => {
      await chatPage.goto();
      
      await chatPage.sendMessage('First conversation message');
      await chatPage.waitForResponse();
      
      await chatPage.startNewConversation();
      
      const messages = await chatPage.getAllMessages();
      expect(messages.length).toBe(0);
    });

    test('should clear current chat', async ({ chatPage }) => {
      await chatPage.goto();
      
      await chatPage.sendMessage('Message to be cleared');
      await chatPage.waitForResponse();
      
      await chatPage.clearChat();
      
      const messages = await chatPage.getAllMessages();
      expect(messages.length).toBe(0);
    });

    test('should handle multiple message types', async ({ chatPage }) => {
      await chatPage.goto();
      await chatPage.testDifferentMessageTypes();
    });
  });

  test.describe('WebSocket Connection', () => {
    test('should establish WebSocket connection', async ({ chatPage }) => {
      await chatPage.goto();
      await chatPage.verifyConnectionStatus('connected');
    });

    test('should handle WebSocket disconnection gracefully', async ({ chatPage, testHelpers }) => {
      await chatPage.goto();
      
      // Simulate network disconnection
      await testHelpers.page.setOfflineMode(true);
      await chatPage.verifyConnectionStatus('disconnected');
      
      // Restore connection
      await testHelpers.page.setOfflineMode(false);
      await chatPage.verifyConnectionStatus('connected');
    });

    test('should queue messages when disconnected', async ({ chatPage, testHelpers }) => {
      await chatPage.goto();
      
      // Disconnect
      await testHelpers.page.setOfflineMode(true);
      
      // Try to send message
      await chatPage.sendMessage('Message while offline');
      
      // Reconnect
      await testHelpers.page.setOfflineMode(false);
      
      // Message should be sent when reconnected
      await chatPage.waitForResponse();
      const response = await chatPage.getLastResponse();
      expect(response.length).toBeGreaterThan(0);
    });
  });

  test.describe('Accessibility', () => {
    test('should be accessible to screen readers', async ({ chatPage }) => {
      await chatPage.goto();
      await chatPage.verifyAccessibility();
    });

    test('should support keyboard navigation', async ({ chatPage }) => {
      await chatPage.goto();
      
      // Focus should be on chat input by default
      await expect(chatPage.elements.chatInput).toBeFocused();
      
      // Tab navigation should work
      await chatPage.page.keyboard.press('Tab');
      await expect(chatPage.elements.sendButton).toBeFocused();
    });
  });

  test.describe('Responsive Design', () => {
    test('should work on mobile devices', async ({ chatPage }) => {
      await chatPage.goto();
      await chatPage.testResponsiveDesign();
    });

    test('should adapt to different screen sizes', async ({ chatPage }) => {
      await chatPage.goto();
      
      // Test different viewport sizes
      const viewports = [
        { width: 320, height: 568 }, // iPhone SE
        { width: 768, height: 1024 }, // iPad
        { width: 1920, height: 1080 }, // Desktop
      ];
      
      for (const viewport of viewports) {
        await chatPage.page.setViewportSize(viewport);
        await chatPage.verifyPageLoaded();
        
        // Basic functionality should work
        await chatPage.sendMessage(`Test at ${viewport.width}x${viewport.height}`);
        await chatPage.waitForResponse();
        
        const response = await chatPage.getLastResponse();
        expect(response.length).toBeGreaterThan(0);
      }
    });
  });

  test.describe('Performance', () => {
    test('should handle rapid message sending', async ({ chatPage }) => {
      await chatPage.goto();
      
      const messages = [
        'First rapid message',
        'Second rapid message',
        'Third rapid message',
      ];
      
      // Send messages rapidly
      for (const message of messages) {
        await chatPage.sendMessage(message);
      }
      
      // Wait for all responses
      await chatPage.page.waitForTimeout(10000);
      
      const allMessages = await chatPage.getAllMessages();
      const assistantMessages = allMessages.filter(msg => msg.role === 'assistant');
      
      // Should have responses for all messages
      expect(assistantMessages.length).toBe(messages.length);
    });

    test('should handle long conversations', async ({ chatPage }) => {
      await chatPage.goto();
      
      // Send multiple messages to create long conversation
      for (let i = 1; i <= 10; i++) {
        await chatPage.sendMessage(`Message number ${i}`);
        await chatPage.waitForResponse();
      }
      
      const messages = await chatPage.getAllMessages();
      expect(messages.length).toBe(20); // 10 user + 10 assistant
      
      // UI should still be responsive
      await chatPage.sendMessage('Final test message');
      await chatPage.waitForResponse();
      
      const response = await chatPage.getLastResponse();
      expect(response.length).toBeGreaterThan(0);
    });
  });

  test.describe('Error Handling', () => {
    test('should handle API errors gracefully', async ({ chatPage, testHelpers }) => {
      await chatPage.goto();
      
      // Mock API error
      await testHelpers.mockApiResponse(/\/api\/v1\/chat/, {
        error: 'Service temporarily unavailable'
      }, 500);
      
      await chatPage.sendMessage('This should trigger an error');
      
      // Should show error message
      await expect(chatPage.elements.errorMessage).toBeVisible();
      await expect(chatPage.elements.errorMessage).toContainText('Service temporarily unavailable');
      
      // Clear mock and retry
      await testHelpers.clearMocks();
      await chatPage.sendMessage('This should work now');
      await chatPage.waitForResponse();
      
      const response = await chatPage.getLastResponse();
      expect(response.length).toBeGreaterThan(0);
    });

    test('should handle network timeouts', async ({ chatPage, testHelpers }) => {
      await chatPage.goto();
      
      // Mock slow response
      await testHelpers.mockApiResponse(/\/api\/v1\/chat/, {
        answer: 'Delayed response'
      }, 200);
      
      await chatPage.sendMessage('Test timeout handling');
      
      // Should show timeout error after configured timeout
      await expect(chatPage.elements.errorMessage).toBeVisible({ timeout: 35000 });
    });
  });
});