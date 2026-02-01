import { test, expect } from '../fixtures/chat-fixtures';

/**
 * Conversation Management Tests
 * Tests conversation creation, switching, history, and management
 */

test.describe('Conversation Management', () => {
  test.beforeEach(async ({ page, loginAsUser }) => {
    await loginAsUser('user');
    await page.goto('/chat');
  });

  test('should create new conversation', async ({ page, chatInterface }) => {
    // Start new conversation
    await chatInterface.startNewConversation();
    
    // Verify new conversation is created
    await expect(page.locator('[data-testid="conversation-title"]'))
      .toContainText('New Conversation');
    
    // Verify conversation appears in sidebar
    await expect(page.locator('[data-testid="conversation-list"] li').first())
      .toContainText('New Conversation');
    
    // Verify conversation ID is generated
    const conversationId = await page.getAttribute('[data-testid="conversation-id"]', 'data-id');
    expect(conversationId).toBeTruthy();
    expect(conversationId).toMatch(/^conv_[a-zA-Z0-9]+$/);
  });

  test('should rename conversation', async ({ page, sendMessage }) => {
    await sendMessage('Initial message for conversation');
    
    // Click on conversation title to edit
    await page.click('[data-testid="conversation-title"]');
    
    // Should show edit input
    await expect(page.locator('[data-testid="conversation-title-input"]')).toBeVisible();
    
    // Rename conversation
    await page.fill('[data-testid="conversation-title-input"]', 'My Custom Conversation');
    await page.keyboard.press('Enter');
    
    // Verify title is updated
    await expect(page.locator('[data-testid="conversation-title"]'))
      .toContainText('My Custom Conversation');
    
    // Verify update in sidebar
    await expect(page.locator('[data-testid="conversation-list"] li').first())
      .toContainText('My Custom Conversation');
  });

  test('should auto-generate conversation title from first message', async ({ page, sendMessage, waitForResponse }) => {
    await sendMessage('What are the company holidays for 2024?');
    await waitForResponse();
    
    // Should auto-generate title
    await expect(page.locator('[data-testid="conversation-title"]'))
      .not.toContainText('New Conversation');
    
    const title = await page.locator('[data-testid="conversation-title"]').textContent();
    expect(title).toContain('holidays'); // Should relate to message content
  });

  test('should switch between conversations', async ({ page, chatInterface, sendMessage }) => {
    // Create first conversation
    await sendMessage('Message in conversation 1');
    const conv1Id = await page.getAttribute('[data-testid="conversation-id"]', 'data-id');
    
    // Create second conversation
    await chatInterface.startNewConversation();
    await sendMessage('Message in conversation 2');
    const conv2Id = await page.getAttribute('[data-testid="conversation-id"]', 'data-id');
    
    // Switch back to first conversation
    await chatInterface.selectConversation(conv1Id);
    
    // Verify correct conversation is loaded
    await expect(page.locator('[data-testid="user-message"]'))
      .toContainText('Message in conversation 1');
    
    // Should not see second conversation's message
    await expect(page.locator('[data-testid="chat-messages"]'))
      .not.toContainText('Message in conversation 2');
  });

  test('should preserve conversation state when switching', async ({ page, chatInterface, sendMessage }) => {
    // Create conversation with multiple messages
    await sendMessage('First message');
    await sendMessage('Second message');
    
    const conv1Id = await page.getAttribute('[data-testid="conversation-id"]', 'data-id');
    
    // Create second conversation
    await chatInterface.startNewConversation();
    await sendMessage('Different message');
    
    // Switch back to first conversation
    await chatInterface.selectConversation(conv1Id);
    
    // Should preserve all messages
    await expect(page.locator('[data-testid="user-message"]')).toHaveCount(2);
    await expect(page.locator('[data-testid="user-message"]').first())
      .toContainText('First message');
    await expect(page.locator('[data-testid="user-message"]').last())
      .toContainText('Second message');
  });

  test('should delete conversation', async ({ page, sendMessage }) => {
    await sendMessage('Message to be deleted');
    
    // Open conversation menu
    await page.click('[data-testid="conversation-menu"]');
    
    // Delete conversation
    await page.click('[data-testid="delete-conversation"]');
    
    // Should show confirmation dialog
    await expect(page.locator('[data-testid="delete-confirmation"]')).toBeVisible();
    await expect(page.locator('[data-testid="delete-confirmation"]'))
      .toContainText('Are you sure you want to delete this conversation?');
    
    // Confirm deletion
    await page.click('[data-testid="confirm-delete"]');
    
    // Should remove conversation and create new one
    await expect(page.locator('[data-testid="conversation-title"]'))
      .toContainText('New Conversation');
    
    // Should not appear in conversation list
    await expect(page.locator('[data-testid="conversation-list"]'))
      .not.toContainText('Message to be deleted');
  });

  test('should handle conversation list pagination', async ({ page, chatInterface, sendMessage }) => {
    // Create many conversations
    for (let i = 1; i <= 25; i++) {
      await chatInterface.startNewConversation();
      await sendMessage(`Conversation ${i} message`);
    }
    
    // Should show pagination controls
    await expect(page.locator('[data-testid="conversation-pagination"]')).toBeVisible();
    await expect(page.locator('[data-testid="next-page"]')).toBeVisible();
    
    // Navigate to next page
    await page.click('[data-testid="next-page"]');
    
    // Should show different conversations
    const firstPageConversations = await page.locator('[data-testid="conversation-list"] li').count();
    expect(firstPageConversations).toBeLessThanOrEqual(20); // Assuming 20 per page
  });

  test('should search conversations', async ({ page, chatInterface, sendMessage }) => {
    // Create conversations with distinct content
    await sendMessage('Question about company policy');
    await chatInterface.startNewConversation();
    await sendMessage('Technical documentation inquiry');
    await chatInterface.startNewConversation();
    await sendMessage('HR related question');
    
    // Search conversations
    await page.fill('[data-testid="conversation-search"]', 'policy');
    
    // Should filter conversations
    await expect(page.locator('[data-testid="conversation-list"] li')).toHaveCount(1);
    await expect(page.locator('[data-testid="conversation-list"] li').first())
      .toContainText('company policy');
    
    // Clear search
    await page.fill('[data-testid="conversation-search"]', '');
    
    // Should show all conversations
    await expect(page.locator('[data-testid="conversation-list"] li')).toHaveCount(3);
  });

  test('should show conversation metadata', async ({ page, sendMessage }) => {
    await sendMessage('Test message for metadata');
    
    // Open conversation details
    await page.click('[data-testid="conversation-info"]');
    
    // Should show metadata
    await expect(page.locator('[data-testid="conversation-created"]')).toBeVisible();
    await expect(page.locator('[data-testid="conversation-updated"]')).toBeVisible();
    await expect(page.locator('[data-testid="conversation-message-count"]')).toBeVisible();
    await expect(page.locator('[data-testid="conversation-message-count"]'))
      .toContainText('1 message');
    
    // Add another message
    await sendMessage('Second message');
    
    // Message count should update
    await expect(page.locator('[data-testid="conversation-message-count"]'))
      .toContainText('2 messages');
  });

  test('should export conversation', async ({ page, sendMessage, waitForResponse }) => {
    // Create conversation with content
    await sendMessage('What is the company mission?');
    await waitForResponse();
    await sendMessage('Tell me about the team structure');
    await waitForResponse();
    
    // Export conversation
    await page.click('[data-testid="conversation-menu"]');
    await page.click('[data-testid="export-conversation"]');
    
    // Should show export options
    await expect(page.locator('[data-testid="export-format-options"]')).toBeVisible();
    
    // Select format and export
    await page.click('[data-testid="export-json"]');
    
    const downloadPromise = page.waitForEvent('download');
    await page.click('[data-testid="confirm-export"]');
    const download = await downloadPromise;
    
    // Verify download
    expect(download.suggestedFilename()).toContain('conversation');
    expect(download.suggestedFilename()).toContain('.json');
  });

  test('should handle conversation sharing', async ({ page, sendMessage }) => {
    await sendMessage('Shareable conversation content');
    
    // Share conversation
    await page.click('[data-testid="conversation-menu"]');
    await page.click('[data-testid="share-conversation"]');
    
    // Should show sharing options
    await expect(page.locator('[data-testid="share-options"]')).toBeVisible();
    
    // Generate share link
    await page.click('[data-testid="generate-share-link"]');
    
    // Should show share link
    await expect(page.locator('[data-testid="share-link"]')).toBeVisible();
    
    const shareLink = await page.locator('[data-testid="share-link-input"]').inputValue();
    expect(shareLink).toContain('/shared/');
    expect(shareLink).toMatch(/\/shared\/[a-zA-Z0-9]+$/);
  });

  test('should handle conversation archiving', async ({ page, sendMessage }) => {
    await sendMessage('Message in conversation to archive');
    
    // Archive conversation
    await page.click('[data-testid="conversation-menu"]');
    await page.click('[data-testid="archive-conversation"]');
    
    // Should move to archived section
    await expect(page.locator('[data-testid="archived-indicator"]')).toBeVisible();
    
    // Should appear in archived conversations
    await page.click('[data-testid="show-archived"]');
    await expect(page.locator('[data-testid="archived-conversations"]'))
      .toContainText('Message in conversation to archive');
    
    // Should be able to unarchive
    await page.click('[data-testid="unarchive-conversation"]');
    await expect(page.locator('[data-testid="archived-indicator"]')).not.toBeVisible();
  });

  test('should handle conversation favorites', async ({ page, sendMessage }) => {
    await sendMessage('Important conversation to favorite');
    
    // Add to favorites
    await page.click('[data-testid="favorite-conversation"]');
    
    // Should show favorite indicator
    await expect(page.locator('[data-testid="favorite-indicator"]')).toBeVisible();
    
    // Should appear in favorites list
    await page.click('[data-testid="show-favorites"]');
    await expect(page.locator('[data-testid="favorite-conversations"]'))
      .toContainText('Important conversation to favorite');
    
    // Should be able to unfavorite
    await page.click('[data-testid="unfavorite-conversation"]');
    await expect(page.locator('[data-testid="favorite-indicator"]')).not.toBeVisible();
  });

  test('should restore conversation from backup', async ({ page }) => {
    // Simulate conversation restore
    await page.click('[data-testid="conversation-menu"]');
    await page.click('[data-testid="restore-conversation"]');
    
    // Should show file upload for restoration
    await expect(page.locator('[data-testid="restore-upload"]')).toBeVisible();
    
    // Upload backup file (mock)
    await page.setInputFiles('[data-testid="restore-file-input"]', {
      name: 'conversation-backup.json',
      mimeType: 'application/json',
      buffer: Buffer.from(JSON.stringify({
        conversation_id: 'restored_conv_001',
        messages: [
          { role: 'user', content: 'Restored message', timestamp: new Date().toISOString() },
          { role: 'assistant', content: 'Restored response', timestamp: new Date().toISOString() }
        ]
      }))
    });
    
    // Confirm restoration
    await page.click('[data-testid="confirm-restore"]');
    
    // Should load restored conversation
    await expect(page.locator('[data-testid="user-message"]'))
      .toContainText('Restored message');
    await expect(page.locator('[data-testid="assistant-message"]'))
      .toContainText('Restored response');
  });
});