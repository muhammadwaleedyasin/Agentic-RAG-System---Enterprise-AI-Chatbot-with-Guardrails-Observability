import { test, expect } from '@playwright/test';
import { test as authTest } from '../fixtures/auth-fixtures';
import { test as chatTest } from '../fixtures/chat-fixtures';

/**
 * New User Journey Tests
 * Tests complete workflow for new users: registration → login → first chat → document upload
 */

test.describe('New User Journey', () => {
  test('should complete full new user onboarding journey', async ({ page }) => {
    // Step 1: Account Creation (if registration is enabled)
    await page.goto('/');
    
    // Check if registration is available
    const hasRegistration = await page.locator('[data-testid="register-link"]').isVisible();
    
    if (hasRegistration) {
      await page.click('[data-testid="register-link"]');
      
      // Fill registration form
      await page.fill('[data-testid="register-username"]', 'newuser_test');
      await page.fill('[data-testid="register-email"]', 'newuser@test.com');
      await page.fill('[data-testid="register-password"]', 'NewUser123!');
      await page.fill('[data-testid="register-confirm-password"]', 'NewUser123!');
      
      // Accept terms
      await page.check('[data-testid="accept-terms"]');
      
      // Submit registration
      await page.click('[data-testid="register-button"]');
      
      // Should show success or email verification
      await expect(page.locator('[data-testid="registration-success"]')).toBeVisible();
    }
    
    // Step 2: Login
    await page.goto('/login');
    await page.fill('[data-testid="username-input"]', 'newuser_test');
    await page.fill('[data-testid="password-input"]', 'NewUser123!');
    await page.click('[data-testid="login-button"]');
    
    // Should be redirected to dashboard or onboarding
    await expect(page).toHaveURL(/\/(dashboard|onboarding)/);
    
    // Step 3: Welcome/Onboarding Flow
    if (page.url().includes('/onboarding')) {
      // Complete onboarding steps
      await expect(page.locator('[data-testid="welcome-message"]')).toBeVisible();
      
      // Step 1: Profile setup
      await page.fill('[data-testid="display-name"]', 'New User');
      await page.selectOption('[data-testid="role-preference"]', 'researcher');
      await page.click('[data-testid="next-step"]');
      
      // Step 2: Feature tour
      await expect(page.locator('[data-testid="feature-tour"]')).toBeVisible();
      await page.click('[data-testid="start-tour"]');
      
      // Navigate through tour steps
      for (let i = 0; i < 5; i++) {
        await expect(page.locator('[data-testid="tour-step"]')).toBeVisible();
        await page.click('[data-testid="next-tour-step"]');
      }
      
      // Complete onboarding
      await page.click('[data-testid="finish-onboarding"]');
      
      // Should redirect to dashboard
      await expect(page).toHaveURL('/dashboard');
    }
    
    // Step 4: First Chat Interaction
    await page.goto('/chat');
    
    // Should show welcome state for new users
    await expect(page.locator('[data-testid="new-user-welcome"]')).toBeVisible();
    await expect(page.locator('[data-testid="getting-started-tips"]')).toBeVisible();
    
    // Send first message
    const firstMessage = 'Hello! This is my first question. Can you help me understand how this system works?';
    await page.fill('[data-testid="message-input"]', firstMessage);
    await page.click('[data-testid="send-button"]');
    
    // Should receive helpful onboarding response
    await expect(page.locator('[data-testid="assistant-message"]')).toBeVisible();
    
    const response = await page.locator('[data-testid="assistant-message"]').textContent();
    expect(response).toContain('welcome' || 'help' || 'getting started');
    
    // Should show helpful tips
    await expect(page.locator('[data-testid="chat-tips"]')).toBeVisible();
    
    // Step 5: First Document Upload
    await page.goto('/documents');
    
    // Should show empty state with helpful guidance
    await expect(page.locator('[data-testid="empty-document-state"]')).toBeVisible();
    await expect(page.locator('[data-testid="upload-guidance"]')).toBeVisible();
    
    // Upload first document
    const filePath = path.join(__dirname, '../fixtures/welcome-document.pdf');
    await page.setInputFiles('[data-testid="file-input"]', filePath);
    
    // Should show first-time upload help
    await expect(page.locator('[data-testid="first-upload-help"]')).toBeVisible();
    
    // Fill document metadata with guidance
    await page.fill('[data-testid="document-title"]', 'My First Document');
    await page.fill('[data-testid="document-description"]', 'This is my first uploaded document');
    await page.selectOption('[data-testid="document-category"]', 'general');
    
    // Start upload
    await page.click('[data-testid="start-upload"]');
    
    // Should show progress and success
    await expect(page.locator('[data-testid="upload-success"]')).toBeVisible();
    
    // Step 6: Use Uploaded Document in Chat
    await page.goto('/chat');
    
    // Ask question about uploaded document
    const documentQuestion = 'Can you tell me about the document I just uploaded?';
    await page.fill('[data-testid="message-input"]', documentQuestion);
    await page.click('[data-testid="send-button"]');
    
    // Should get response referencing the document
    await expect(page.locator('[data-testid="assistant-message"]')).toBeVisible();
    await expect(page.locator('[data-testid="source-citations"]')).toBeVisible();
    
    // Should show that document was found and used
    await expect(page.locator('[data-testid="source-citation"]'))
      .toContainText('My First Document');
    
    // Step 7: Complete Tutorial/Achievements
    // Should show progress indicators
    await expect(page.locator('[data-testid="user-progress"]')).toBeVisible();
    
    // Navigate to profile to see achievements
    await page.click('[data-testid="user-menu"]');
    await page.click('[data-testid="profile-link"]');
    
    // Should show completed onboarding achievements
    await expect(page.locator('[data-testid="achievements"]')).toBeVisible();
    await expect(page.locator('[data-testid="first-chat-achievement"]')).toBeVisible();
    await expect(page.locator('[data-testid="first-upload-achievement"]')).toBeVisible();
    
    // Should show next steps
    await expect(page.locator('[data-testid="next-steps"]')).toBeVisible();
  });

  test('should handle new user without documents gracefully', async ({ page }) => {
    // Login as new user with no documents
    await page.goto('/login');
    await page.fill('[data-testid="username-input"]', 'newuser_nodocs');
    await page.fill('[data-testid="password-input"]', 'NewUser123!');
    await page.click('[data-testid="login-button"]');
    
    await page.goto('/chat');
    
    // Ask question when no documents exist
    const question = 'What documents do you have access to?';
    await page.fill('[data-testid="message-input"]', question);
    await page.click('[data-testid="send-button"]');
    
    // Should get helpful response about uploading documents
    await expect(page.locator('[data-testid="assistant-message"]')).toBeVisible();
    
    const response = await page.locator('[data-testid="assistant-message"]').textContent();
    expect(response).toContain('no documents' || 'upload' || 'get started');
    
    // Should show helpful suggestions
    await expect(page.locator('[data-testid="upload-suggestion"]')).toBeVisible();
  });

  test('should guide new user through RAG vs Direct LLM', async ({ page }) => {
    // Login as new user
    await page.goto('/login');
    await page.fill('[data-testid="username-input"]', 'newuser_test');
    await page.fill('[data-testid="password-input"]', 'NewUser123!');
    await page.click('[data-testid="login-button"]');
    
    await page.goto('/chat');
    
    // Should explain RAG toggle
    await expect(page.locator('[data-testid="rag-explanation"]')).toBeVisible();
    
    // Test with RAG enabled (default)
    await page.fill('[data-testid="message-input"]', 'What is artificial intelligence?');
    await page.click('[data-testid="send-button"]');
    
    await expect(page.locator('[data-testid="assistant-message"]')).toBeVisible();
    
    // Should show explanation of RAG mode
    await expect(page.locator('[data-testid="rag-mode-explanation"]')).toBeVisible();
    
    // Test with RAG disabled
    await page.uncheck('[data-testid="use-rag-toggle"]');
    
    // Should show explanation of direct mode
    await expect(page.locator('[data-testid="direct-mode-explanation"]')).toBeVisible();
    
    await page.fill('[data-testid="message-input"]', 'Tell me a joke');
    await page.click('[data-testid="send-button"]');
    
    // Should show different response style indicator
    await expect(page.locator('[data-testid="direct-llm-indicator"]')).toBeVisible();
  });

  test('should show new user dashboard with helpful widgets', async ({ page }) => {
    // Login as new user
    await page.goto('/login');
    await page.fill('[data-testid="username-input"]', 'newuser_test');
    await page.fill('[data-testid="password-input"]', 'NewUser123!');
    await page.click('[data-testid="login-button"]');
    
    await page.goto('/dashboard');
    
    // Should show new user specific widgets
    await expect(page.locator('[data-testid="welcome-widget"]')).toBeVisible();
    await expect(page.locator('[data-testid="getting-started-widget"]')).toBeVisible();
    await expect(page.locator('[data-testid="quick-actions-widget"]')).toBeVisible();
    
    // Should have helpful quick actions
    await expect(page.locator('[data-testid="quick-upload"]')).toBeVisible();
    await expect(page.locator('[data-testid="quick-chat"]')).toBeVisible();
    await expect(page.locator('[data-testid="view-tutorials"]')).toBeVisible();
    
    // Should show progress tracking
    await expect(page.locator('[data-testid="onboarding-progress"]')).toBeVisible();
    
    // Should show helpful tips
    await expect(page.locator('[data-testid="daily-tip"]')).toBeVisible();
  });

  test('should handle new user errors gracefully', async ({ page }) => {
    // Login as new user
    await page.goto('/login');
    await page.fill('[data-testid="username-input"]', 'newuser_test');
    await page.fill('[data-testid="password-input"]', 'NewUser123!');
    await page.click('[data-testid="login-button"]');
    
    // Try to access advanced features
    await page.goto('/admin');
    
    // Should show permission error with helpful guidance
    await expect(page.locator('[data-testid="permission-error"]')).toBeVisible();
    await expect(page.locator('[data-testid="new-user-guidance"]')).toBeVisible();
    
    // Should suggest appropriate actions
    await expect(page.locator('[data-testid="suggested-actions"]')).toBeVisible();
    
    // Try invalid operations in chat
    await page.goto('/chat');
    
    // Try to upload invalid file type through chat
    await page.setInputFiles('[data-testid="chat-file-upload"]', 
      path.join(__dirname, '../fixtures/invalid-file.exe'));
    
    // Should show helpful error with guidance
    await expect(page.locator('[data-testid="file-type-error"]')).toBeVisible();
    await expect(page.locator('[data-testid="file-help-link"]')).toBeVisible();
  });

  test('should provide contextual help throughout journey', async ({ page }) => {
    // Login as new user
    await page.goto('/login');
    await page.fill('[data-testid="username-input"]', 'newuser_test');
    await page.fill('[data-testid="password-input"]', 'NewUser123!');
    await page.click('[data-testid="login-button"]');
    
    // Should show help indicators on each page
    await page.goto('/dashboard');
    await expect(page.locator('[data-testid="help-indicator"]')).toBeVisible();
    
    await page.goto('/chat');
    await expect(page.locator('[data-testid="chat-help"]')).toBeVisible();
    
    await page.goto('/documents');
    await expect(page.locator('[data-testid="document-help"]')).toBeVisible();
    
    // Help should be contextual and progressive
    // Early help should be more detailed
    await page.click('[data-testid="help-indicator"]');
    await expect(page.locator('[data-testid="beginner-help"]')).toBeVisible();
    
    // Should track user progress and reduce help over time
    // Simulate some activity
    await page.goto('/chat');
    await page.fill('[data-testid="message-input"]', 'Test message');
    await page.click('[data-testid="send-button"]');
    
    // Help should become less intrusive
    await expect(page.locator('[data-testid="reduced-help"]')).toBeVisible();
  });

  test('should complete new user feedback flow', async ({ page }) => {
    // Login as new user
    await page.goto('/login');
    await page.fill('[data-testid="username-input"]', 'newuser_test');
    await page.fill('[data-testid="password-input"]', 'NewUser123!');
    await page.click('[data-testid="login-button"]');
    
    // Complete basic actions
    await page.goto('/chat');
    await page.fill('[data-testid="message-input"]', 'Test question');
    await page.click('[data-testid="send-button"]');
    
    // Should eventually show feedback prompt
    await expect(page.locator('[data-testid="new-user-feedback"]')).toBeVisible();
    
    // Fill feedback form
    await page.selectOption('[data-testid="ease-of-use"]', '5');
    await page.selectOption('[data-testid="feature-satisfaction"]', '4');
    await page.fill('[data-testid="feedback-comments"]', 'Great experience so far!');
    
    // Submit feedback
    await page.click('[data-testid="submit-feedback"]');
    
    // Should show thank you message
    await expect(page.locator('[data-testid="feedback-thanks"]')).toBeVisible();
    
    // Should offer additional help or resources
    await expect(page.locator('[data-testid="additional-resources"]')).toBeVisible();
  });
});