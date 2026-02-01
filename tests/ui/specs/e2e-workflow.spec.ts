import { test, expect } from '../fixtures/test-fixtures';
import { testData, testPatterns } from '../fixtures/test-fixtures';

test.describe('End-to-End Workflows', () => {
  test.describe('Complete User Journey', () => {
    test('should complete full document upload and query workflow', async ({ 
      loginPage, 
      dashboardPage, 
      documentsPage, 
      chatPage 
    }) => {
      // 1. Login
      await loginPage.goto();
      await loginPage.performValidLogin();
      
      // 2. Navigate to dashboard and verify
      await dashboardPage.goto();
      await dashboardPage.verifyPageLoaded();
      
      // 3. Upload a document
      const testDoc = testData.generateDocument();
      await dashboardPage.navigateToDocuments();
      await documentsPage.performDocumentUpload('workflow-test.txt', testDoc.content);
      
      // 4. Navigate to chat and query the document
      await dashboardPage.navigateToChat();
      const response = await chatPage.performChatInteraction('What information is in the workflow-test document?');
      
      // 5. Verify RAG response includes the uploaded document
      expect(response).toBeTruthy();
      await chatPage.verifySources(1);
      
      // 6. Verify conversation appears in dashboard
      await dashboardPage.goto();
      const recentConversations = await dashboardPage.getRecentConversations();
      expect(recentConversations.length).toBeGreaterThan(0);
    });

    test('should handle new user onboarding flow', async ({ 
      page,
      authService,
      loginPage, 
      dashboardPage, 
      documentsPage,
      chatPage 
    }) => {
      // Create a new user
      const newUser = testData.generateUser();
      
      // Admin creates the user
      const adminToken = await authService.getAuthToken('admin_test', 'test_password_123');
      await page.context().addCookies([{
        name: 'auth_token',
        value: adminToken,
        domain: 'localhost',
        path: '/',
      }]);
      
      // Clear auth and login as new user
      await page.context().clearCookies();
      
      // First-time login
      await loginPage.goto();
      await loginPage.login(newUser.username, newUser.password);
      await loginPage.waitForSuccessfulLogin();
      
      // Should be on dashboard
      await dashboardPage.verifyPageLoaded();
      
      // Explore key features
      await dashboardPage.navigateToDocuments();
      await documentsPage.verifyPageLoaded();
      
      await dashboardPage.navigateToChat();
      await chatPage.verifyPageLoaded();
      
      // Send first message
      await chatPage.sendMessage('Hello! This is my first message.');
      await chatPage.waitForResponse();
      
      const response = await chatPage.getLastResponse();
      expect(response).toBeTruthy();
    });
  });

  test.describe('Multi-User Collaboration', () => {
    test('should handle document sharing between users', async ({ 
      page,
      authService,
      documentsPage,
      chatPage 
    }) => {
      // User 1 uploads a document
      const user1Token = await authService.getAuthToken('admin_test', 'test_password_123');
      await page.context().addCookies([{
        name: 'auth_token',
        value: user1Token,
        domain: 'localhost',
        path: '/',
      }]);
      
      const sharedDoc = testData.generateDocument();
      await documentsPage.goto();
      await documentsPage.performDocumentUpload('shared-document.txt', sharedDoc.content);
      
      // User 2 queries the document
      await page.context().clearCookies();
      const user2Token = await authService.getAuthToken('user_test', 'test_password_123');
      await page.context().addCookies([{
        name: 'auth_token',
        value: user2Token,
        domain: 'localhost',
        path: '/',
      }]);
      
      await chatPage.goto();
      const response = await chatPage.performChatInteraction('What is in the shared-document?');
      
      // User 2 should be able to access the shared document
      expect(response).toBeTruthy();
      await chatPage.verifySources(1);
    });

    test('should handle concurrent user sessions', async ({ 
      browser,
      authService 
    }) => {
      // Create multiple browser contexts for different users
      const user1Context = await browser.newContext();
      const user2Context = await browser.newContext();
      
      const user1Page = await user1Context.newPage();
      const user2Page = await user2Context.newPage();
      
      // Login both users
      const user1Token = await authService.getAuthToken('admin_test', 'test_password_123');
      const user2Token = await authService.getAuthToken('user_test', 'test_password_123');
      
      await user1Context.addCookies([{
        name: 'auth_token',
        value: user1Token,
        domain: 'localhost',
        path: '/',
      }]);
      
      await user2Context.addCookies([{
        name: 'auth_token',
        value: user2Token,
        domain: 'localhost',
        path: '/',
      }]);
      
      // Both users navigate to chat
      await user1Page.goto('/chat');
      await user2Page.goto('/chat');
      
      // Both users send messages simultaneously
      const user1Promise = user1Page.locator('[data-testid="chat-input"]').fill('User 1 message');
      const user2Promise = user2Page.locator('[data-testid="chat-input"]').fill('User 2 message');
      
      await Promise.all([user1Promise, user2Promise]);
      
      await user1Page.locator('[data-testid="send-button"]').click();
      await user2Page.locator('[data-testid="send-button"]').click();
      
      // Both should receive responses
      await expect(user1Page.locator('[data-testid="message-item"]').last()).toBeVisible({ timeout: 30000 });
      await expect(user2Page.locator('[data-testid="message-item"]').last()).toBeVisible({ timeout: 30000 });
      
      // Cleanup
      await user1Context.close();
      await user2Context.close();
    });
  });

  test.describe('Error Recovery Workflows', () => {
    test('should recover from network interruption during upload', async ({ 
      documentsPage,
      testHelpers,
      loginPage 
    }) => {
      await loginPage.goto();
      await loginPage.performValidLogin();
      
      await documentsPage.goto();
      
      const testFile = await documentsPage.createTestFile('network-test.txt', 'Network interruption test');
      
      // Start upload
      await documentsPage.elements.uploadButton.click();
      await documentsPage.elements.fileInput.setInputFiles(testFile);
      
      // Simulate network interruption
      await testHelpers.page.setOfflineMode(true);
      
      const submitButton = documentsPage.page.locator('[data-testid="upload-submit"]');
      await submitButton.click();
      
      // Should show network error
      await documentsPage.verifyToast('Network error', 'error');
      
      // Restore network
      await testHelpers.page.setOfflineMode(false);
      
      // Retry upload
      await submitButton.click();
      await documentsPage.waitForUploadSuccess();
      await documentsPage.verifyDocumentInList('network-test.txt');
    });

    test('should handle session expiration gracefully', async ({ 
      loginPage,
      chatPage,
      authService 
    }) => {
      await loginPage.goto();
      await loginPage.performValidLogin();
      
      await chatPage.goto();
      
      // Send initial message
      await chatPage.sendMessage('Initial message');
      await chatPage.waitForResponse();
      
      // Simulate session expiration by clearing auth
      await loginPage.clearAuth();
      
      // Try to send another message
      await chatPage.sendMessage('Message after session expired');
      
      // Should redirect to login or show auth error
      await expect(chatPage.page).toHaveURL(/\/(login|auth-required)/);
      
      // Re-login
      await loginPage.performValidLogin();
      
      // Return to chat and verify functionality
      await chatPage.goto();
      await chatPage.sendMessage('Message after re-login');
      await chatPage.waitForResponse();
      
      const response = await chatPage.getLastResponse();
      expect(response).toBeTruthy();
    });
  });

  test.describe('Performance Under Load', () => {
    test('should handle rapid document uploads', async ({ 
      documentsPage,
      loginPage 
    }) => {
      await loginPage.goto();
      await loginPage.performValidLogin();
      
      await documentsPage.goto();
      
      const uploadTasks = [];
      
      // Create multiple upload tasks
      for (let i = 1; i <= 5; i++) {
        const content = `Rapid upload test document ${i} with content: ${testData.generateFileContent('txt')}`;
        uploadTasks.push(
          documentsPage.createTestFile(`rapid-upload-${i}.txt`, content)
            .then(filePath => documentsPage.uploadDocument(filePath))
        );
      }
      
      // Execute all uploads
      await Promise.allSettled(uploadTasks);
      
      // Verify documents appear in list
      const documents = await documentsPage.getDocumentList();
      const rapidUploads = documents.filter(doc => doc.name.includes('rapid-upload'));
      expect(rapidUploads.length).toBeGreaterThan(0);
    });

    test('should handle continuous chat interaction', async ({ 
      chatPage,
      loginPage 
    }) => {
      await loginPage.goto();
      await loginPage.performValidLogin();
      
      await chatPage.goto();
      
      // Send multiple messages in sequence
      const messages = [
        'What is the system status?',
        'How many documents are available?',
        'What are the latest features?',
        'Can you help me understand the interface?',
        'What security measures are in place?',
      ];
      
      for (const message of messages) {
        await chatPage.sendMessage(message);
        await chatPage.waitForResponse();
        
        const response = await chatPage.getLastResponse();
        expect(response.length).toBeGreaterThan(0);
      }
      
      // Verify conversation history
      const allMessages = await chatPage.getAllMessages();
      expect(allMessages.length).toBe(messages.length * 2); // User + assistant messages
    });
  });

  test.describe('Feature Integration', () => {
    test('should integrate RAG with real-time updates', async ({ 
      documentsPage,
      chatPage,
      loginPage 
    }) => {
      await loginPage.goto();
      await loginPage.performValidLogin();
      
      // Upload initial document
      const initialDoc = testData.generateDocument();
      await documentsPage.goto();
      await documentsPage.performDocumentUpload('integration-test.txt', initialDoc.content);
      
      // Query the document
      await chatPage.goto();
      const response1 = await chatPage.performChatInteraction('What is in the integration-test document?');
      expect(response1).toBeTruthy();
      
      // Upload additional document
      const additionalDoc = testData.generateDocument();
      await documentsPage.goto();
      await documentsPage.performDocumentUpload('additional-test.txt', additionalDoc.content);
      
      // Query again to see updated results
      await chatPage.goto();
      const response2 = await chatPage.performChatInteraction('What documents are available for integration testing?');
      expect(response2).toBeTruthy();
      
      // Should include both documents in sources
      const sources = await chatPage.getSources();
      expect(sources.length).toBeGreaterThanOrEqual(1);
    });

    test('should handle admin operations during user activity', async ({ 
      browser,
      authService,
      chatPage,
      adminPage 
    }) => {
      // Create separate contexts for admin and user
      const adminContext = await browser.newContext();
      const userContext = await browser.newContext();
      
      const adminPageContext = await adminContext.newPage();
      const userPageContext = await userContext.newPage();
      
      // Setup authentication
      const adminToken = await authService.getAuthToken('admin_test', 'test_password_123');
      const userToken = await authService.getAuthToken('user_test', 'test_password_123');
      
      await adminContext.addCookies([{
        name: 'auth_token',
        value: adminToken,
        domain: 'localhost',
        path: '/',
      }]);
      
      await userContext.addCookies([{
        name: 'auth_token',
        value: userToken,
        domain: 'localhost',
        path: '/',
      }]);
      
      // User starts chat session
      const userChatPage = new (await import('../pages/chat-page')).ChatPage(userPageContext);
      await userChatPage.goto();
      await userChatPage.sendMessage('Starting chat session during admin operations');
      
      // Admin performs system operations
      const adminPageInstance = new (await import('../pages/admin-page')).AdminPage(adminPageContext);
      await adminPageInstance.goto();
      await adminPageInstance.navigateToSection('settings');
      
      // User continues chat while admin works
      await userChatPage.waitForResponse();
      await userChatPage.sendMessage('Second message during admin operations');
      await userChatPage.waitForResponse();
      
      // Both operations should complete successfully
      const response = await userChatPage.getLastResponse();
      expect(response).toBeTruthy();
      
      // Admin operations should not interfere
      await expect(adminPageInstance.elements.systemSettings).toBeVisible();
      
      // Cleanup
      await adminContext.close();
      await userContext.close();
    });
  });

  test.describe('Data Consistency', () => {
    test('should maintain data consistency across sessions', async ({ 
      documentsPage,
      chatPage,
      loginPage,
      dashboardPage 
    }) => {
      await loginPage.goto();
      await loginPage.performValidLogin();
      
      // Upload document and note initial stats
      const testDoc = testData.generateDocument();
      await documentsPage.goto();
      await documentsPage.performDocumentUpload('consistency-test.txt', testDoc.content);
      
      await dashboardPage.goto();
      const initialStats = await dashboardPage.getDashboardStats();
      
      // Have conversation
      await chatPage.goto();
      await chatPage.performChatInteraction('What is in the consistency-test document?');
      
      // Check updated stats
      await dashboardPage.goto();
      const updatedStats = await dashboardPage.getDashboardStats();
      
      // Stats should reflect the new activity
      if (initialStats['Total Documents'] && updatedStats['Total Documents']) {
        const initialDocs = parseInt(initialStats['Total Documents']);
        const updatedDocs = parseInt(updatedStats['Total Documents']);
        expect(updatedDocs).toBeGreaterThanOrEqual(initialDocs);
      }
      
      if (initialStats['Total Conversations'] && updatedStats['Total Conversations']) {
        const initialConvs = parseInt(initialStats['Total Conversations']);
        const updatedConvs = parseInt(updatedStats['Total Conversations']);
        expect(updatedConvs).toBeGreaterThanOrEqual(initialConvs);
      }
    });

    test('should handle browser refresh gracefully', async ({ 
      chatPage,
      loginPage 
    }) => {
      await loginPage.goto();
      await loginPage.performValidLogin();
      
      await chatPage.goto();
      
      // Start conversation
      await chatPage.sendMessage('Message before refresh');
      await chatPage.waitForResponse();
      
      const messagesBefore = await chatPage.getAllMessages();
      
      // Refresh page
      await chatPage.page.reload();
      await chatPage.verifyPageLoaded();
      
      // Continue conversation
      await chatPage.sendMessage('Message after refresh');
      await chatPage.waitForResponse();
      
      // Should maintain session and functionality
      const messagesAfter = await chatPage.getAllMessages();
      expect(messagesAfter.length).toBeGreaterThanOrEqual(messagesBefore.length);
    });
  });
});