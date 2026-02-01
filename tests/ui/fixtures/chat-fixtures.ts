import { test as base, expect } from '@playwright/test';
import { WebSocket } from 'ws';

/**
 * Chat fixtures for UI tests
 * Provides WebSocket utilities and chat interface helpers
 */

type ChatFixtures = {
  chatInterface: ChatInterface;
  webSocketClient: WebSocketClient;
  sendMessage: (message: string, options?: MessageOptions) => Promise<void>;
  waitForResponse: (timeout?: number) => Promise<string>;
  verifyStreamingResponse: () => Promise<void>;
  uploadDocument: (filePath: string) => Promise<void>;
};

interface MessageOptions {
  useRag?: boolean;
  conversationId?: string;
  attachments?: string[];
}

export class ChatInterface {
  constructor(private page: any) {}
  
  async navigateToChat() {
    await this.page.goto('/chat');
    await expect(this.page.locator('[data-testid="chat-interface"]')).toBeVisible();
  }
  
  async sendMessage(message: string, options: MessageOptions = {}) {
    console.log(`📤 Sending message: ${message}`);
    
    // Fill message input
    await this.page.fill('[data-testid="message-input"]', message);
    
    // Configure options if provided
    if (options.useRag !== undefined) {
      await this.page.locator('[data-testid="use-rag-toggle"]').setChecked(options.useRag);
    }
    
    // Handle attachments
    if (options.attachments?.length) {
      for (const attachment of options.attachments) {
        await this.page.setInputFiles('[data-testid="file-upload"]', attachment);
      }
    }
    
    // Send message
    await this.page.click('[data-testid="send-button"]');
    
    // Wait for message to appear in chat
    await expect(this.page.locator('[data-testid="user-message"]').last()).toContainText(message);
  }
  
  async waitForResponse(timeout: number = 30000): Promise<string> {
    console.log('⏳ Waiting for AI response...');
    
    // Wait for typing indicator
    await expect(this.page.locator('[data-testid="typing-indicator"]')).toBeVisible();
    
    // Wait for response to appear
    await this.page.waitForSelector('[data-testid="assistant-message"]', { timeout });
    
    // Get the latest response
    const responseElement = this.page.locator('[data-testid="assistant-message"]').last();
    const response = await responseElement.textContent();
    
    console.log(`📥 Received response: ${response?.substring(0, 100)}...`);
    return response || '';
  }
  
  async verifyStreamingResponse() {
    console.log('🌊 Verifying streaming response...');
    
    // Check for streaming chunks
    const chunks = this.page.locator('[data-testid="response-chunk"]');
    await expect(chunks.first()).toBeVisible();
    
    // Verify progressive text appearance
    let previousLength = 0;
    for (let i = 0; i < 5; i++) {
      await this.page.waitForTimeout(500);
      const currentText = await this.page.locator('[data-testid="assistant-message"]').last().textContent();
      const currentLength = currentText?.length || 0;
      
      if (currentLength > previousLength) {
        console.log(`✅ Streaming detected: ${currentLength} characters`);
        previousLength = currentLength;
      }
    }
  }
  
  async clearChat() {
    await this.page.click('[data-testid="clear-chat-button"]');
    await expect(this.page.locator('[data-testid="chat-messages"]')).toBeEmpty();
  }
  
  async startNewConversation() {
    await this.page.click('[data-testid="new-conversation-button"]');
    await expect(this.page.locator('[data-testid="conversation-title"]')).toContainText('New Conversation');
  }
  
  async selectConversation(conversationId: string) {
    await this.page.click(`[data-testid="conversation-${conversationId}"]`);
    await this.page.waitForSelector('[data-testid="chat-messages"]');
  }
  
  async verifyMessageHistory(expectedMessages: string[]) {
    const messages = this.page.locator('[data-testid="chat-message"]');
    const count = await messages.count();
    
    expect(count).toBe(expectedMessages.length);
    
    for (let i = 0; i < count; i++) {
      const messageText = await messages.nth(i).textContent();
      expect(messageText).toContain(expectedMessages[i]);
    }
  }
  
  async uploadFile(filePath: string) {
    console.log(`📎 Uploading file: ${filePath}`);
    
    await this.page.setInputFiles('[data-testid="file-upload"]', filePath);
    
    // Wait for upload progress
    await expect(this.page.locator('[data-testid="upload-progress"]')).toBeVisible();
    
    // Wait for upload completion
    await expect(this.page.locator('[data-testid="upload-success"]')).toBeVisible();
    
    console.log('✅ File uploaded successfully');
  }
  
  async verifySourceCitations(expectedSources: string[]) {
    const citations = this.page.locator('[data-testid="source-citation"]');
    const citationCount = await citations.count();
    
    expect(citationCount).toBeGreaterThan(0);
    
    for (const expectedSource of expectedSources) {
      await expect(this.page.locator(`[data-testid="source-citation"]`))
        .toContainText(expectedSource);
    }
  }
}

export class WebSocketClient {
  private ws: WebSocket | null = null;
  private messages: any[] = [];
  
  constructor(private baseURL: string) {}
  
  async connect(clientId: string, token?: string): Promise<void> {
    const wsUrl = this.baseURL.replace('http', 'ws') + `/ws/${clientId}`;
    const fullUrl = token ? `${wsUrl}?token=${token}` : wsUrl;
    
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(fullUrl);
      
      this.ws.onopen = () => {
        console.log('🔌 WebSocket connected');
        resolve();
      };
      
      this.ws.onmessage = (event) => {
        const message = JSON.parse(event.data);
        this.messages.push(message);
        console.log('📨 WebSocket message received:', message.type);
      };
      
      this.ws.onerror = (error) => {
        console.error('❌ WebSocket error:', error);
        reject(error);
      };
      
      setTimeout(() => reject(new Error('WebSocket connection timeout')), 10000);
    });
  }
  
  async sendMessage(message: any): Promise<void> {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket not connected');
    }
    
    this.ws.send(JSON.stringify(message));
  }
  
  async waitForMessage(type: string, timeout: number = 10000): Promise<any> {
    const startTime = Date.now();
    
    while (Date.now() - startTime < timeout) {
      const message = this.messages.find(msg => msg.type === type);
      if (message) {
        return message;
      }
      
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    
    throw new Error(`Timeout waiting for message type: ${type}`);
  }
  
  getMessages(): any[] {
    return [...this.messages];
  }
  
  clearMessages(): void {
    this.messages = [];
  }
  
  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}

export const test = base.extend<ChatFixtures>({
  chatInterface: async ({ page }, use) => {
    const chatInterface = new ChatInterface(page);
    await use(chatInterface);
  },
  
  webSocketClient: async ({ baseURL }, use) => {
    const client = new WebSocketClient(baseURL || 'http://localhost:8000');
    await use(client);
    client.disconnect();
  },
  
  sendMessage: async ({ page }, use) => {
    const sendFunction = async (message: string, options: MessageOptions = {}) => {
      const chatInterface = new ChatInterface(page);
      await chatInterface.sendMessage(message, options);
    };
    
    await use(sendFunction);
  },
  
  waitForResponse: async ({ page }, use) => {
    const waitFunction = async (timeout: number = 30000) => {
      const chatInterface = new ChatInterface(page);
      return await chatInterface.waitForResponse(timeout);
    };
    
    await use(waitFunction);
  },
  
  verifyStreamingResponse: async ({ page }, use) => {
    const verifyFunction = async () => {
      const chatInterface = new ChatInterface(page);
      await chatInterface.verifyStreamingResponse();
    };
    
    await use(verifyFunction);
  },
  
  uploadDocument: async ({ page }, use) => {
    const uploadFunction = async (filePath: string) => {
      const chatInterface = new ChatInterface(page);
      await chatInterface.uploadFile(filePath);
    };
    
    await use(uploadFunction);
  }
});

export { expect };