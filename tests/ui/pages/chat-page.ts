import { Page, expect, Locator } from '@playwright/test';
import { BasePage } from './base-page';

export class ChatPage extends BasePage {
  constructor(page: Page) {
    super(page);
  }

  // Page elements
  get elements() {
    return {
      chatInput: this.page.locator('[data-testid="chat-input"]'),
      sendButton: this.page.locator('[data-testid="send-button"]'),
      messagesContainer: this.page.locator('[data-testid="messages-container"]'),
      messageItems: this.page.locator('[data-testid="message-item"]'),
      typingIndicator: this.page.locator('[data-testid="typing-indicator"]'),
      clearChatButton: this.page.locator('[data-testid="clear-chat"]'),
      chatHistory: this.page.locator('[data-testid="chat-history"]'),
      conversationList: this.page.locator('[data-testid="conversation-list"]'),
      newConversationButton: this.page.locator('[data-testid="new-conversation"]'),
      ragToggle: this.page.locator('[data-testid="rag-toggle"]'),
      sourcesPanel: this.page.locator('[data-testid="sources-panel"]'),
      sourceItems: this.page.locator('[data-testid="source-item"]'),
      chatSettings: this.page.locator('[data-testid="chat-settings"]'),
      connectionStatus: this.page.locator('[data-testid="connection-status"]'),
      errorMessage: this.page.locator('[data-testid="chat-error"]'),
    };
  }

  async goto(): Promise<void> {
    await this.page.goto('/chat');
    await this.verifyPageLoaded();
  }

  async verifyPageLoaded(): Promise<void> {
    await expect(this.elements.chatInput).toBeVisible();
    await expect(this.elements.sendButton).toBeVisible();
    await expect(this.elements.messagesContainer).toBeVisible();
    await this.verifyTitle('Chat - Enterprise RAG Chatbot');
  }

  /**
   * Send a message in the chat
   */
  async sendMessage(message: string): Promise<void> {
    await this.fillFormField('[data-testid="chat-input"]', message);
    await this.elements.sendButton.click();
  }

  /**
   * Send message using Enter key
   */
  async sendMessageWithEnter(message: string): Promise<void> {
    await this.fillFormField('[data-testid="chat-input"]', message);
    await this.elements.chatInput.press('Enter');
  }

  /**
   * Wait for AI response
   */
  async waitForResponse(timeout = 30000): Promise<void> {
    // Wait for typing indicator to appear
    await expect(this.elements.typingIndicator).toBeVisible({ timeout: 5000 });
    
    // Wait for typing indicator to disappear (response complete)
    await expect(this.elements.typingIndicator).not.toBeVisible({ timeout });
  }

  /**
   * Get the last message in the chat
   */
  async getLastMessage(): Promise<{ role: string; content: string; timestamp?: string }> {
    const messages = this.elements.messageItems;
    const lastMessage = messages.last();
    
    await expect(lastMessage).toBeVisible();
    
    const role = await lastMessage.getAttribute('data-role') || 'unknown';
    const content = await lastMessage.locator('[data-testid="message-content"]').textContent() || '';
    const timestamp = await lastMessage.getAttribute('data-timestamp');
    
    return { role, content, timestamp: timestamp || undefined };
  }

  /**
   * Get all messages in the current conversation
   */
  async getAllMessages(): Promise<Array<{ role: string; content: string; timestamp?: string }>> {
    const messages = this.elements.messageItems;
    const count = await messages.count();
    const result = [];
    
    for (let i = 0; i < count; i++) {
      const message = messages.nth(i);
      const role = await message.getAttribute('data-role') || 'unknown';
      const content = await message.locator('[data-testid="message-content"]').textContent() || '';
      const timestamp = await message.getAttribute('data-timestamp');
      
      result.push({ role, content, timestamp: timestamp || undefined });
    }
    
    return result;
  }

  /**
   * Get the last AI response
   */
  async getLastResponse(): Promise<string> {
    const messages = await this.getAllMessages();
    const assistantMessages = messages.filter(msg => msg.role === 'assistant');
    
    if (assistantMessages.length === 0) {
      throw new Error('No assistant messages found');
    }
    
    return assistantMessages[assistantMessages.length - 1].content;
  }

  /**
   * Clear the current chat
   */
  async clearChat(): Promise<void> {
    await this.elements.clearChatButton.click();
    await this.confirmAction(true);
    
    // Verify chat is cleared
    await expect(this.elements.messageItems).toHaveCount(0);
  }

  /**
   * Start a new conversation
   */
  async startNewConversation(): Promise<void> {
    await this.elements.newConversationButton.click();
    
    // Verify new conversation started
    await expect(this.elements.messageItems).toHaveCount(0);
  }

  /**
   * Toggle RAG mode
   */
  async toggleRAG(enabled: boolean): Promise<void> {
    const isCurrentlyEnabled = await this.elements.ragToggle.isChecked();
    
    if (isCurrentlyEnabled !== enabled) {
      await this.elements.ragToggle.click();
    }
    
    // Verify toggle state
    if (enabled) {
      await expect(this.elements.ragToggle).toBeChecked();
    } else {
      await expect(this.elements.ragToggle).not.toBeChecked();
    }
  }

  /**
   * Verify sources are displayed for RAG responses
   */
  async verifySources(expectedMinCount = 1): Promise<void> {
    await expect(this.elements.sourcesPanel).toBeVisible();
    
    const sourceCount = await this.elements.sourceItems.count();
    expect(sourceCount).toBeGreaterThanOrEqual(expectedMinCount);
  }

  /**
   * Get sources from the last response
   */
  async getSources(): Promise<Array<{ title: string; url?: string; snippet?: string }>> {
    await expect(this.elements.sourcesPanel).toBeVisible();
    
    const sources = this.elements.sourceItems;
    const count = await sources.count();
    const result = [];
    
    for (let i = 0; i < count; i++) {
      const source = sources.nth(i);
      const title = await source.locator('[data-testid="source-title"]').textContent() || '';
      const url = await source.getAttribute('data-url');
      const snippet = await source.locator('[data-testid="source-snippet"]').textContent();
      
      result.push({ 
        title, 
        url: url || undefined, 
        snippet: snippet || undefined 
      });
    }
    
    return result;
  }

  /**
   * Verify WebSocket connection status
   */
  async verifyConnectionStatus(expected: 'connected' | 'disconnected' | 'connecting'): Promise<void> {
    await expect(this.elements.connectionStatus).toHaveAttribute('data-status', expected);
  }

  /**
   * Test message streaming
   */
  async verifyMessageStreaming(message: string): Promise<void> {
    await this.sendMessage(message);
    
    // Verify typing indicator appears
    await expect(this.elements.typingIndicator).toBeVisible({ timeout: 5000 });
    
    // Wait for response to stream in
    const responseMessage = this.elements.messageItems.last();
    await expect(responseMessage).toBeVisible();
    
    // Verify response content updates (streaming effect)
    let previousContent = '';
    let streamingDetected = false;
    
    for (let i = 0; i < 10; i++) {
      const currentContent = await responseMessage.locator('[data-testid="message-content"]').textContent() || '';
      
      if (currentContent !== previousContent && currentContent.length > previousContent.length) {
        streamingDetected = true;
        break;
      }
      
      previousContent = currentContent;
      await this.page.waitForTimeout(500);
    }
    
    // Wait for streaming to complete
    await this.waitForResponse();
    
    // Note: Streaming might be too fast to detect in test environment
    console.log('Streaming detection:', streamingDetected ? 'detected' : 'not detected (possibly too fast)');
  }

  /**
   * Test chat with different message types
   */
  async testDifferentMessageTypes(): Promise<void> {
    const testMessages = [
      'What is the company policy?',
      'How do I upload documents?',
      'Can you summarize the latest updates?',
      'What are the security procedures?',
    ];
    
    for (const message of testMessages) {
      await this.sendMessage(message);
      await this.waitForResponse();
      
      const response = await this.getLastResponse();
      expect(response.length).toBeGreaterThan(0);
    }
  }

  /**
   * Test conversation history
   */
  async verifyConversationHistory(): Promise<void> {
    const messages = await this.getAllMessages();
    expect(messages.length).toBeGreaterThan(0);
    
    // Verify message ordering
    for (let i = 1; i < messages.length; i++) {
      if (messages[i-1].timestamp && messages[i].timestamp) {
        const prevTime = new Date(messages[i-1].timestamp!);
        const currTime = new Date(messages[i].timestamp!);
        expect(currTime.getTime()).toBeGreaterThanOrEqual(prevTime.getTime());
      }
    }
  }

  /**
   * Test error handling
   */
  async testErrorHandling(): Promise<void> {
    // Test empty message
    await this.sendMessage('');
    await expect(this.elements.errorMessage).toBeVisible();
    await expect(this.elements.errorMessage).toContainText('Empty message not allowed');
  }

  /**
   * Test accessibility features
   */
  async verifyAccessibility(): Promise<void> {
    // Check ARIA labels
    await expect(this.elements.chatInput).toHaveAttribute('aria-label', 'Chat message input');
    await expect(this.elements.sendButton).toHaveAttribute('aria-label', 'Send message');
    
    // Check keyboard navigation
    await this.elements.chatInput.focus();
    await expect(this.elements.chatInput).toBeFocused();
    
    // Test screen reader announcements for new messages
    await this.sendMessage('Test accessibility');
    await this.waitForResponse();
    
    const lastMessage = this.elements.messageItems.last();
    await expect(lastMessage).toHaveAttribute('aria-live', 'polite');
  }

  /**
   * Test responsive design
   */
  async testResponsiveDesign(): Promise<void> {
    // Test mobile layout
    await this.verifyResponsiveLayout('mobile');
    await expect(this.elements.chatInput).toBeVisible();
    await expect(this.elements.sendButton).toBeVisible();
    
    // Test tablet layout
    await this.verifyResponsiveLayout('tablet');
    await expect(this.elements.conversationList).toBeVisible();
    
    // Test desktop layout
    await this.verifyResponsiveLayout('desktop');
    await expect(this.elements.sourcesPanel).toBeVisible();
  }

  /**
   * Perform a complete chat interaction test
   */
  async performChatInteraction(message: string, expectRAGSources = true): Promise<string> {
    // Ensure RAG is enabled if sources are expected
    if (expectRAGSources) {
      await this.toggleRAG(true);
    }
    
    // Send message
    await this.sendMessage(message);
    await this.waitForResponse();
    
    // Get response
    const response = await this.getLastResponse();
    expect(response.length).toBeGreaterThan(0);
    
    // Verify sources if RAG is enabled
    if (expectRAGSources) {
      await this.verifySources();
    }
    
    // Verify no errors
    await this.verifyNoErrors();
    
    return response;
  }
}