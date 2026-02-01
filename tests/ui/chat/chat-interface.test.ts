import { test, expect } from '../fixtures/chat-fixtures';

/**
 * Chat Interface Tests
 * Tests real-time messaging, streaming responses, and chat UI functionality
 */

test.describe('Chat Interface', () => {
  test.beforeEach(async ({ page, loginAsUser }) => {
    await loginAsUser('user');
    await page.goto('/chat');
  });

  test('should display chat interface correctly', async ({ page }) => {
    // Verify main chat components
    await expect(page.locator('[data-testid="chat-interface"]')).toBeVisible();
    await expect(page.locator('[data-testid="chat-messages"]')).toBeVisible();
    await expect(page.locator('[data-testid="message-input"]')).toBeVisible();
    await expect(page.locator('[data-testid="send-button"]')).toBeVisible();
    
    // Verify chat controls
    await expect(page.locator('[data-testid="new-conversation-button"]')).toBeVisible();
    await expect(page.locator('[data-testid="use-rag-toggle"]')).toBeVisible();
    await expect(page.locator('[data-testid="file-upload"]')).toBeVisible();
    
    // Verify sidebar
    await expect(page.locator('[data-testid="conversation-sidebar"]')).toBeVisible();
    await expect(page.locator('[data-testid="conversation-list"]')).toBeVisible();
  });

  test('should send message successfully', async ({ chatInterface, sendMessage, waitForResponse }) => {
    const testMessage = 'What is the company policy on remote work?';
    
    await sendMessage(testMessage);
    
    // Verify message appears in chat
    await expect(page.locator('[data-testid="user-message"]').last()).toContainText(testMessage);
    
    // Wait for and verify response
    const response = await waitForResponse();
    expect(response.length).toBeGreaterThan(0);
    
    // Verify response appears in chat
    await expect(page.locator('[data-testid="assistant-message"]').last()).toContainText(response);
  });

  test('should handle empty message validation', async ({ page }) => {
    // Try to send empty message
    await page.click('[data-testid="send-button"]');
    
    // Should show validation error
    await expect(page.locator('[data-testid="message-required-error"]')).toBeVisible();
    await expect(page.locator('[data-testid="message-required-error"]'))
      .toContainText('Message cannot be empty');
    
    // Message should not be sent
    await expect(page.locator('[data-testid="user-message"]')).toHaveCount(0);
  });

  test('should handle message length validation', async ({ page }) => {
    const longMessage = 'a'.repeat(10001); // Assuming 10k char limit
    
    await page.fill('[data-testid="message-input"]', longMessage);
    await page.click('[data-testid="send-button"]');
    
    // Should show length validation error
    await expect(page.locator('[data-testid="message-length-error"]')).toBeVisible();
    await expect(page.locator('[data-testid="message-length-error"]'))
      .toContainText('Message too long');
  });

  test('should disable send button during message processing', async ({ page, sendMessage }) => {
    await page.fill('[data-testid="message-input"]', 'Test message');
    await page.click('[data-testid="send-button"]');
    
    // Button should be disabled while processing
    await expect(page.locator('[data-testid="send-button"]')).toBeDisabled();
    
    // Should show loading state
    await expect(page.locator('[data-testid="sending-indicator"]')).toBeVisible();
  });

  test('should support keyboard shortcuts', async ({ page }) => {
    // Focus message input
    await page.focus('[data-testid="message-input"]');
    
    // Type message
    await page.keyboard.type('Test message');
    
    // Send with Enter
    await page.keyboard.press('Enter');
    
    // Message should be sent
    await expect(page.locator('[data-testid="user-message"]').last())
      .toContainText('Test message');
    
    // Shift+Enter should create new line
    await page.focus('[data-testid="message-input"]');
    await page.keyboard.type('Line 1');
    await page.keyboard.press('Shift+Enter');
    await page.keyboard.type('Line 2');
    
    const inputValue = await page.locator('[data-testid="message-input"]').inputValue();
    expect(inputValue).toContain('\n');
  });

  test('should toggle RAG mode', async ({ page, sendMessage }) => {
    // Verify RAG is enabled by default
    await expect(page.locator('[data-testid="use-rag-toggle"]')).toBeChecked();
    
    // Disable RAG
    await page.uncheck('[data-testid="use-rag-toggle"]');
    
    // Send message without RAG
    await sendMessage('Test without RAG');
    
    // Should show direct LLM response indicator
    await expect(page.locator('[data-testid="direct-llm-indicator"]')).toBeVisible();
    
    // Re-enable RAG
    await page.check('[data-testid="use-rag-toggle"]');
    
    // Send message with RAG
    await sendMessage('Test with RAG');
    
    // Should show RAG response with sources
    await expect(page.locator('[data-testid="source-citations"]')).toBeVisible();
  });

  test('should display typing indicator', async ({ page, sendMessage }) => {
    await sendMessage('Tell me about the company');
    
    // Should show typing indicator
    await expect(page.locator('[data-testid="typing-indicator"]')).toBeVisible();
    await expect(page.locator('[data-testid="typing-indicator"]'))
      .toContainText('AI is thinking...');
    
    // Typing indicator should disappear when response starts
    await page.waitForSelector('[data-testid="assistant-message"]');
    await expect(page.locator('[data-testid="typing-indicator"]')).not.toBeVisible();
  });

  test('should handle message timestamps', async ({ page, sendMessage }) => {
    await sendMessage('Test message with timestamp');
    
    // User message should have timestamp
    await expect(page.locator('[data-testid="user-message-timestamp"]').last()).toBeVisible();
    
    // Wait for response
    await page.waitForSelector('[data-testid="assistant-message"]');
    
    // Assistant message should have timestamp
    await expect(page.locator('[data-testid="assistant-message-timestamp"]').last()).toBeVisible();
    
    // Timestamps should be different
    const userTimestamp = await page.locator('[data-testid="user-message-timestamp"]').last().textContent();
    const assistantTimestamp = await page.locator('[data-testid="assistant-message-timestamp"]').last().textContent();
    
    expect(userTimestamp).not.toBe(assistantTimestamp);
  });

  test('should display message status indicators', async ({ page, sendMessage }) => {
    await sendMessage('Test message status');
    
    // Should show sending status
    await expect(page.locator('[data-testid="message-sending"]').last()).toBeVisible();
    
    // Should show sent status
    await expect(page.locator('[data-testid="message-sent"]').last()).toBeVisible();
    
    // Wait for response and check delivered status
    await page.waitForSelector('[data-testid="assistant-message"]');
    await expect(page.locator('[data-testid="message-delivered"]').last()).toBeVisible();
  });

  test('should handle message retry on failure', async ({ page }) => {
    // Simulate network failure
    await page.route('/api/v1/chat', route => route.abort());
    
    await page.fill('[data-testid="message-input"]', 'Test message');
    await page.click('[data-testid="send-button"]');
    
    // Should show error status
    await expect(page.locator('[data-testid="message-error"]').last()).toBeVisible();
    
    // Should show retry button
    await expect(page.locator('[data-testid="retry-message-button"]').last()).toBeVisible();
    
    // Remove network block
    await page.unroute('/api/v1/chat');
    
    // Retry message
    await page.click('[data-testid="retry-message-button"]');
    
    // Should succeed
    await expect(page.locator('[data-testid="message-sent"]').last()).toBeVisible();
  });

  test('should support message copy functionality', async ({ page, sendMessage, waitForResponse }) => {
    await sendMessage('Copy this message');
    const response = await waitForResponse();
    
    // Copy user message
    await page.hover('[data-testid="user-message"]');
    await page.click('[data-testid="copy-message-button"]');
    
    // Should show copy confirmation
    await expect(page.locator('[data-testid="copy-success"]')).toBeVisible();
    
    // Copy assistant message
    await page.hover('[data-testid="assistant-message"]');
    await page.click('[data-testid="copy-message-button"]');
    
    await expect(page.locator('[data-testid="copy-success"]')).toBeVisible();
  });

  test('should handle message editing', async ({ page, sendMessage }) => {
    await sendMessage('Original message');
    
    // Edit message
    await page.hover('[data-testid="user-message"]');
    await page.click('[data-testid="edit-message-button"]');
    
    // Should show edit form
    await expect(page.locator('[data-testid="edit-message-input"]')).toBeVisible();
    
    // Update message
    await page.fill('[data-testid="edit-message-input"]', 'Edited message');
    await page.click('[data-testid="save-edit-button"]');
    
    // Should show updated message
    await expect(page.locator('[data-testid="user-message"]').last())
      .toContainText('Edited message');
    
    // Should show edit indicator
    await expect(page.locator('[data-testid="message-edited-indicator"]')).toBeVisible();
  });

  test('should handle message deletion', async ({ page, sendMessage }) => {
    await sendMessage('Message to delete');
    
    // Delete message
    await page.hover('[data-testid="user-message"]');
    await page.click('[data-testid="delete-message-button"]');
    
    // Should show confirmation
    await expect(page.locator('[data-testid="delete-confirmation"]')).toBeVisible();
    await page.click('[data-testid="confirm-delete"]');
    
    // Message should be removed
    await expect(page.locator('[data-testid="user-message"]')).toHaveCount(0);
  });

  test('should auto-scroll to latest message', async ({ page, sendMessage, waitForResponse }) => {
    // Send multiple messages to create scroll
    for (let i = 1; i <= 10; i++) {
      await sendMessage(`Message ${i}`);
      await page.waitForTimeout(500);
    }
    
    // Chat should auto-scroll to bottom
    const chatContainer = page.locator('[data-testid="chat-messages"]');
    const isScrolledToBottom = await chatContainer.evaluate(el => {
      return el.scrollTop + el.clientHeight >= el.scrollHeight - 10;
    });
    
    expect(isScrolledToBottom).toBeTruthy();
  });

  test('should handle chat clear functionality', async ({ page, sendMessage }) => {
    // Send some messages
    await sendMessage('Message 1');
    await sendMessage('Message 2');
    
    // Clear chat
    await page.click('[data-testid="clear-chat-button"]');
    
    // Should show confirmation
    await expect(page.locator('[data-testid="clear-chat-confirmation"]')).toBeVisible();
    await page.click('[data-testid="confirm-clear-chat"]');
    
    // Chat should be empty
    await expect(page.locator('[data-testid="chat-messages"]')).toBeEmpty();
    
    // Should show empty state
    await expect(page.locator('[data-testid="empty-chat-state"]')).toBeVisible();
  });

  test('should handle export chat functionality', async ({ page, sendMessage, waitForResponse }) => {
    // Create chat history
    await sendMessage('Question 1');
    await waitForResponse();
    await sendMessage('Question 2');
    await waitForResponse();
    
    // Export chat
    const downloadPromise = page.waitForEvent('download');
    await page.click('[data-testid="export-chat-button"]');
    const download = await downloadPromise;
    
    // Verify download
    expect(download.suggestedFilename()).toContain('chat-export');
    expect(download.suggestedFilename()).toMatch(/\.(txt|json|pdf)$/);
  });
});