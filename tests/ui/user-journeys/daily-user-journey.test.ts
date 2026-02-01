import { test, expect } from '@playwright/test';
import { test as authTest } from '../fixtures/auth-fixtures';
import { test as chatTest } from '../fixtures/chat-fixtures';

/**
 * Daily User Journey Tests
 * Tests typical daily workflow: login → browse conversations → search documents → new chat → logout
 */

test.describe('Daily User Journey', () => {
  test('should complete typical daily user workflow', async ({ page }) => {
    // Step 1: Login
    await page.goto('/login');
    await page.fill('[data-testid="username-input"]', 'user_test');
    await page.fill('[data-testid="password-input"]', 'UserTest123!');
    await page.click('[data-testid="login-button"]');
    
    await expect(page).toHaveURL('/dashboard');
    
    // Step 2: Check Dashboard for Updates
    // Should show recent activity
    await expect(page.locator('[data-testid="recent-activity"]')).toBeVisible();
    await expect(page.locator('[data-testid="recent-conversations"]')).toBeVisible();
    await expect(page.locator('[data-testid="recent-documents"]')).toBeVisible();
    
    // Check notifications
    if (await page.locator('[data-testid="notifications-indicator"]').isVisible()) {
      await page.click('[data-testid="notifications-indicator"]');
      await expect(page.locator('[data-testid="notifications-panel"]')).toBeVisible();
      
      // Mark notifications as read
      await page.click('[data-testid="mark-all-read"]');
    }
    
    // Step 3: Browse Previous Conversations
    await page.goto('/chat');
    
    // View conversation history
    await expect(page.locator('[data-testid="conversation-list"]')).toBeVisible();
    
    // Open recent conversation
    const recentConversation = page.locator('[data-testid="conversation-item"]').first();
    if (await recentConversation.isVisible()) {
      await recentConversation.click();
      
      // Review previous messages
      await expect(page.locator('[data-testid="chat-messages"]')).toBeVisible();
      
      // Continue conversation
      await page.fill('[data-testid="message-input"]', 'Can you expand on your previous answer?');
      await page.click('[data-testid="send-button"]');
      
      await expect(page.locator('[data-testid="assistant-message"]')).toBeVisible();
    }
    
    // Step 4: Search Documents
    await page.goto('/documents');
    
    // Search for specific document
    const searchTerm = 'quarterly report';
    await page.fill('[data-testid="document-search"]', searchTerm);
    await page.keyboard.press('Enter');
    
    // Browse search results
    if (await page.locator('[data-testid="document-item"]').count() > 0) {
      // View document details
      await page.click('[data-testid="document-item"]');
      await expect(page.locator('[data-testid="document-details"]')).toBeVisible();
      
      // Preview document
      await page.click('[data-testid="preview-tab"]');
      await expect(page.locator('[data-testid="document-preview"]')).toBeVisible();
    }
    
    // Step 5: Upload New Document (if needed)
    // Check if user has documents to upload
    const hasNewDocuments = await page.evaluate(() => {
      // Simulate check for new documents in user's workflow
      return Math.random() > 0.5; // 50% chance of having new documents
    });
    
    if (hasNewDocuments) {
      const filePath = path.join(__dirname, '../fixtures/daily-report.pdf');
      await page.setInputFiles('[data-testid="file-input"]', filePath);
      
      // Quick metadata entry
      await page.fill('[data-testid="document-title"]', 'Daily Report - ' + new Date().toLocaleDateString());
      await page.selectOption('[data-testid="document-category"]', 'reports');
      
      await page.click('[data-testid="start-upload"]');
      await expect(page.locator('[data-testid="upload-success"]')).toBeVisible();
    }
    
    // Step 6: Start New Research Session
    await page.goto('/chat');
    
    // Create new conversation for today's work
    await page.click('[data-testid="new-conversation-button"]');
    
    // Conduct research session
    const researchQuestions = [
      'What are the key trends in our Q3 performance data?',
      'Can you summarize the main compliance requirements we need to address?',
      'What are the action items from recent board meetings?'
    ];
    
    for (const question of researchQuestions) {
      await page.fill('[data-testid="message-input"]', question);
      await page.click('[data-testid="send-button"]');
      
      // Wait for response
      await expect(page.locator('[data-testid="assistant-message"]')).toBeVisible();
      
      // Check if sources were cited
      if (await page.locator('[data-testid="source-citations"]').isVisible()) {
        // Review cited sources
        await expect(page.locator('[data-testid="source-citation"]')).toBeVisible();
      }
      
      // Small delay between questions
      await page.waitForTimeout(2000);
    }
    
    // Step 7: Export Important Information
    // Export current conversation for records
    await page.click('[data-testid="conversation-menu"]');
    await page.click('[data-testid="export-conversation"]');
    
    const downloadPromise = page.waitForEvent('download');
    await page.click('[data-testid="export-pdf"]');
    const download = await downloadPromise;
    
    expect(download.suggestedFilename()).toContain('conversation');
    
    // Step 8: Save Important Findings
    // Bookmark useful responses
    const assistantMessages = page.locator('[data-testid="assistant-message"]');
    const messageCount = await assistantMessages.count();
    
    if (messageCount > 0) {
      // Bookmark first message
      await assistantMessages.first().hover();
      await page.click('[data-testid="bookmark-message"]');
      
      await expect(page.locator('[data-testid="message-bookmarked"]')).toBeVisible();
    }
    
    // Step 9: Check Personal Analytics
    await page.click('[data-testid="user-menu"]');
    await page.click('[data-testid="analytics-link"]');
    
    // Review daily usage statistics
    await expect(page.locator('[data-testid="daily-stats"]')).toBeVisible();
    await expect(page.locator('[data-testid="questions-asked"]')).toBeVisible();
    await expect(page.locator('[data-testid="documents-accessed"]')).toBeVisible();
    
    // Step 10: Clean Up and Organize
    await page.goto('/chat');
    
    // Organize conversations
    await page.click('[data-testid="organize-conversations"]');
    
    // Move old conversations to archive if needed
    const oldConversations = page.locator('[data-testid="old-conversation"]');
    const oldCount = await oldConversations.count();
    
    if (oldCount > 10) { // Archive if more than 10 conversations
      await page.check('[data-testid="select-old-conversations"]');
      await page.click('[data-testid="archive-selected"]');
      
      await expect(page.locator('[data-testid="conversations-archived"]')).toBeVisible();
    }
    
    // Step 11: Set Reminders/Tasks for Tomorrow
    await page.click('[data-testid="user-menu"]');
    await page.click('[data-testid="tasks-link"]');
    
    // Add tomorrow's tasks
    await page.click('[data-testid="add-task"]');
    await page.fill('[data-testid="task-title"]', 'Review quarterly compliance report');
    await page.fill('[data-testid="task-due-date"]', new Date(Date.now() + 86400000).toISOString().split('T')[0]);
    await page.click('[data-testid="save-task"]');
    
    // Step 12: Logout
    await page.click('[data-testid="user-menu"]');
    await page.click('[data-testid="logout-button"]');
    
    // Confirm logout
    await page.click('[data-testid="confirm-logout"]');
    
    // Should redirect to login page
    await expect(page).toHaveURL('/login');
    await expect(page.locator('[data-testid="logout-success-message"]')).toBeVisible();
  });

  test('should handle power user shortcuts and efficiency features', async ({ page }) => {
    // Login as experienced user
    await page.goto('/login');
    await page.fill('[data-testid="username-input"]', 'power_user');
    await page.fill('[data-testid="password-input"]', 'PowerUser123!');
    await page.click('[data-testid="login-button"]');
    
    await page.goto('/chat');
    
    // Use keyboard shortcuts
    // Quick new conversation (Ctrl+N)
    await page.keyboard.press('Control+KeyN');
    await expect(page.locator('[data-testid="conversation-title"]')).toContainText('New Conversation');
    
    // Quick search (Ctrl+K)
    await page.keyboard.press('Control+KeyK');
    await expect(page.locator('[data-testid="global-search"]')).toBeVisible();
    
    // Type search query
    await page.keyboard.type('financial data');
    await page.keyboard.press('Enter');
    
    // Should show search results across all content
    await expect(page.locator('[data-testid="search-results"]')).toBeVisible();
    
    // Use command palette (Ctrl+Shift+P)
    await page.keyboard.press('Control+Shift+KeyP');
    await expect(page.locator('[data-testid="command-palette"]')).toBeVisible();
    
    // Execute command
    await page.keyboard.type('export conversation');
    await page.keyboard.press('Enter');
    
    // Should trigger export
    await expect(page.locator('[data-testid="export-options"]')).toBeVisible();
  });

  test('should handle interruptions and resume workflow', async ({ page }) => {
    // Login
    await page.goto('/login');
    await page.fill('[data-testid="username-input"]', 'user_test');
    await page.fill('[data-testid="password-input"]', 'UserTest123!');
    await page.click('[data-testid="login-button"]');
    
    await page.goto('/chat');
    
    // Start a research session
    await page.fill('[data-testid="message-input"]', 'I need to analyze market trends for Q4 planning');
    await page.click('[data-testid="send-button"]');
    
    // Simulate interruption - navigate away
    await page.goto('/documents');
    
    // Upload urgent document
    const urgentFile = path.join(__dirname, '../fixtures/urgent-memo.pdf');
    await page.setInputFiles('[data-testid="file-input"]', urgentFile);
    await page.fill('[data-testid="document-title"]', 'Urgent: Market Analysis Update');
    await page.click('[data-testid="start-upload"]');
    
    // Return to chat
    await page.goto('/chat');
    
    // Should preserve context and offer to continue
    await expect(page.locator('[data-testid="resume-session"]')).toBeVisible();
    await page.click('[data-testid="resume-session"]');
    
    // Continue with new information
    await page.fill('[data-testid="message-input"]', 'I just uploaded new market data. Can you incorporate that into your analysis?');
    await page.click('[data-testid="send-button"]');
    
    // Should reference both previous context and new document
    await expect(page.locator('[data-testid="assistant-message"]')).toBeVisible();
    await expect(page.locator('[data-testid="source-citations"]')).toBeVisible();
  });

  test('should handle collaborative workflow', async ({ page, browser }) => {
    // Simulate collaborative work with multiple browser contexts
    const context1 = await browser.newContext();
    const context2 = await browser.newContext();
    
    const user1Page = await context1.newPage();
    const user2Page = await context2.newPage();
    
    // User 1 login
    await user1Page.goto('/login');
    await user1Page.fill('[data-testid="username-input"]', 'user1_collab');
    await user1Page.fill('[data-testid="password-input"]', 'User123!');
    await user1Page.click('[data-testid="login-button"]');
    
    // User 2 login
    await user2Page.goto('/login');
    await user2Page.fill('[data-testid="username-input"]', 'user2_collab');
    await user2Page.fill('[data-testid="password-password"]', 'User123!');
    await user2Page.click('[data-testid="login-button"]');
    
    // User 1 uploads shared document
    await user1Page.goto('/documents');
    const sharedFile = path.join(__dirname, '../fixtures/project-specs.pdf');
    await user1Page.setInputFiles('[data-testid="file-input"]', sharedFile);
    
    // Mark as shared
    await user1Page.check('[data-testid="share-document"]');
    await user1Page.fill('[data-testid="shared-with"]', 'user2_collab');
    await user1Page.click('[data-testid="start-upload"]');
    
    // User 2 should see notification about shared document
    await user2Page.goto('/documents');
    await expect(user2Page.locator('[data-testid="shared-documents"]')).toBeVisible();
    await expect(user2Page.locator('[data-testid="new-shared-indicator"]')).toBeVisible();
    
    // User 2 uses shared document in chat
    await user2Page.goto('/chat');
    await user2Page.fill('[data-testid="message-input"]', 'Can you review the project specifications that were just shared?');
    await user2Page.click('[data-testid="send-button"]');
    
    // Should reference shared document
    await expect(user2Page.locator('[data-testid="source-citations"]')).toBeVisible();
    
    await context1.close();
    await context2.close();
  });

  test('should handle mobile workflow', async ({ browser }) => {
    // Create mobile context
    const mobileContext = await browser.newContext({
      ...devices['iPhone 12']
    });
    
    const mobilePage = await mobileContext.newPage();
    
    // Mobile login
    await mobilePage.goto('/login');
    await mobilePage.fill('[data-testid="username-input"]', 'mobile_user');
    await mobilePage.fill('[data-testid="password-input"]', 'Mobile123!');
    await mobilePage.click('[data-testid="login-button"]');
    
    // Mobile dashboard should be optimized
    await expect(mobilePage.locator('[data-testid="mobile-dashboard"]')).toBeVisible();
    
    // Quick actions should be prominent
    await expect(mobilePage.locator('[data-testid="mobile-quick-chat"]')).toBeVisible();
    await expect(mobilePage.locator('[data-testid="mobile-quick-search"]')).toBeVisible();
    
    // Mobile chat experience
    await mobilePage.click('[data-testid="mobile-quick-chat"]');
    
    // Should have mobile-optimized interface
    await expect(mobilePage.locator('[data-testid="mobile-chat-interface"]')).toBeVisible();
    
    // Voice input should be available
    await expect(mobilePage.locator('[data-testid="voice-input"]')).toBeVisible();
    
    // Send quick voice-to-text message
    await mobilePage.click('[data-testid="voice-input"]');
    // Simulate voice input
    await mobilePage.fill('[data-testid="message-input"]', 'Quick status update on project alpha');
    await mobilePage.click('[data-testid="send-button"]');
    
    await expect(mobilePage.locator('[data-testid="assistant-message"]')).toBeVisible();
    
    await mobileContext.close();
  });

  test('should handle offline and reconnection scenarios', async ({ page }) => {
    // Login
    await page.goto('/login');
    await page.fill('[data-testid="username-input"]', 'user_test');
    await page.fill('[data-testid="password-input"]', 'UserTest123!');
    await page.click('[data-testid="login-button"]');
    
    await page.goto('/chat');
    
    // Start conversation
    await page.fill('[data-testid="message-input"]', 'Testing offline functionality');
    await page.click('[data-testid="send-button"]');
    
    // Simulate going offline
    await page.route('**/*', route => route.abort());
    
    // Try to send message while offline
    await page.fill('[data-testid="message-input"]', 'This message should be queued');
    await page.click('[data-testid="send-button"]');
    
    // Should show offline indicator
    await expect(page.locator('[data-testid="offline-indicator"]')).toBeVisible();
    
    // Should queue message
    await expect(page.locator('[data-testid="message-queued"]')).toBeVisible();
    
    // Restore connection
    await page.unroute('**/*');
    
    // Should automatically reconnect and send queued messages
    await expect(page.locator('[data-testid="reconnected-indicator"]')).toBeVisible();
    await expect(page.locator('[data-testid="message-sent"]')).toBeVisible();
  });
});