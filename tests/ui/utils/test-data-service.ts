import { APIRequestContext, request } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';

export interface TestDocument {
  id: string;
  filename: string;
  content: string;
  type: 'pdf' | 'txt' | 'docx';
}

export interface TestConversation {
  id: string;
  userId: string;
  messages: Array<{
    role: 'user' | 'assistant';
    content: string;
    timestamp: string;
  }>;
}

export class TestDataService {
  private baseUrl: string;
  private apiContext: APIRequestContext | null = null;
  private testDocuments: TestDocument[] = [];
  private testConversations: TestConversation[] = [];

  constructor() {
    this.baseUrl = process.env.BACKEND_URL || 'http://localhost:8000';
  }

  async getApiContext(): Promise<APIRequestContext> {
    if (!this.apiContext) {
      this.apiContext = await request.newContext({
        baseURL: this.baseUrl,
        extraHTTPHeaders: {
          'Content-Type': 'application/json',
        },
      });
    }
    return this.apiContext;
  }

  async createTestDocuments(): Promise<void> {
    console.log('Creating test documents...');
    
    const documents = [
      {
        filename: 'test-policy.txt',
        content: 'This is a test company policy document. It contains information about employee guidelines, procedures, and best practices.',
        type: 'txt' as const,
      },
      {
        filename: 'test-manual.txt',
        content: 'User manual for the RAG system. This document explains how to use the chat interface, upload documents, and manage conversations.',
        type: 'txt' as const,
      },
      {
        filename: 'test-faq.txt',
        content: 'Frequently asked questions about the system. Q: How do I upload documents? A: Use the upload button in the documents section.',
        type: 'txt' as const,
      },
    ];

    const api = await this.getApiContext();
    
    // Get admin token for document upload
    const adminToken = process.env.ADMIN_TOKEN || await this.getAdminToken();

    for (const doc of documents) {
      try {
        // Create temporary file
        const tempDir = path.join(__dirname, '../data/temp');
        if (!fs.existsSync(tempDir)) {
          fs.mkdirSync(tempDir, { recursive: true });
        }
        
        const filePath = path.join(tempDir, doc.filename);
        fs.writeFileSync(filePath, doc.content);

        // Upload document
        const formData = new FormData();
        const fileBuffer = fs.readFileSync(filePath);
        const blob = new Blob([fileBuffer], { type: 'text/plain' });
        formData.append('file', blob, doc.filename);
        formData.append('title', doc.filename);
        formData.append('description', `Test document: ${doc.filename}`);

        const response = await api.post('/api/v1/documents/upload', {
          headers: {
            'Authorization': `Bearer ${adminToken}`,
          },
          multipart: {
            file: {
              name: doc.filename,
              mimeType: 'text/plain',
              buffer: fileBuffer,
            },
            title: doc.filename,
            description: `Test document: ${doc.filename}`,
          },
        });

        if (response.ok()) {
          const result = await response.json();
          this.testDocuments.push({
            id: result.document_id,
            filename: doc.filename,
            content: doc.content,
            type: doc.type,
          });
          console.log(`Created test document: ${doc.filename}`);
        } else {
          console.warn(`Failed to create document ${doc.filename}: ${response.status()}`);
        }

        // Clean up temp file
        fs.unlinkSync(filePath);
      } catch (error) {
        console.warn(`Error creating document ${doc.filename}:`, error);
      }
    }
  }

  async createTestConversations(): Promise<void> {
    console.log('Creating test conversations...');
    
    const conversations = [
      {
        userId: 'admin_test',
        messages: [
          { role: 'user' as const, content: 'What is the company policy?' },
          { role: 'assistant' as const, content: 'Based on the company policy document...' },
        ],
      },
      {
        userId: 'user_test',
        messages: [
          { role: 'user' as const, content: 'How do I use the system?' },
          { role: 'assistant' as const, content: 'According to the user manual...' },
        ],
      },
    ];

    const api = await this.getApiContext();
    const adminToken = process.env.ADMIN_TOKEN || await this.getAdminToken();

    for (const conv of conversations) {
      try {
        // Create conversation through chat endpoint
        const response = await api.post('/api/v1/chat', {
          headers: {
            'Authorization': `Bearer ${adminToken}`,
          },
          data: {
            message: conv.messages[0].content,
            conversation_id: null,
            use_rag: true,
          },
        });

        if (response.ok()) {
          const result = await response.json();
          this.testConversations.push({
            id: result.conversation_id,
            userId: conv.userId,
            messages: conv.messages.map(msg => ({
              ...msg,
              timestamp: new Date().toISOString(),
            })),
          });
          console.log(`Created test conversation for ${conv.userId}`);
        }
      } catch (error) {
        console.warn(`Error creating conversation for ${conv.userId}:`, error);
      }
    }
  }

  async getAdminToken(): Promise<string> {
    const api = await this.getApiContext();
    
    const response = await api.post('/api/v1/auth/login', {
      data: {
        username: 'admin',
        password: 'admin123',
      },
    });
    
    if (!response.ok()) {
      throw new Error('Failed to get admin token');
    }
    
    const result = await response.json();
    return result.access_token;
  }

  async cleanupTestDocuments(): Promise<void> {
    console.log('Cleaning up test documents...');
    
    const api = await this.getApiContext();
    const adminToken = process.env.ADMIN_TOKEN || await this.getAdminToken();

    for (const doc of this.testDocuments) {
      try {
        await api.delete(`/api/v1/documents/${doc.id}`, {
          headers: {
            'Authorization': `Bearer ${adminToken}`,
          },
        });
        console.log(`Cleaned up document: ${doc.filename}`);
      } catch (error) {
        console.warn(`Error cleaning up document ${doc.filename}:`, error);
      }
    }
    
    this.testDocuments = [];
  }

  async cleanupTestConversations(): Promise<void> {
    console.log('Cleaning up test conversations...');
    
    const api = await this.getApiContext();
    const adminToken = process.env.ADMIN_TOKEN || await this.getAdminToken();

    for (const conv of this.testConversations) {
      try {
        await api.delete(`/api/v1/conversations/${conv.id}`, {
          headers: {
            'Authorization': `Bearer ${adminToken}`,
          },
        });
        console.log(`Cleaned up conversation: ${conv.id}`);
      } catch (error) {
        console.warn(`Error cleaning up conversation ${conv.id}:`, error);
      }
    }
    
    this.testConversations = [];
  }

  async clearTestCaches(): Promise<void> {
    console.log('Clearing test caches...');
    
    try {
      // Clear temporary files
      const tempDir = path.join(__dirname, '../data/temp');
      if (fs.existsSync(tempDir)) {
        const files = fs.readdirSync(tempDir);
        for (const file of files) {
          fs.unlinkSync(path.join(tempDir, file));
        }
        fs.rmdirSync(tempDir);
      }
    } catch (error) {
      console.warn('Error clearing test caches:', error);
    }
  }

  getTestDocuments(): TestDocument[] {
    return [...this.testDocuments];
  }

  getTestConversations(): TestConversation[] {
    return [...this.testConversations];
  }

  async dispose(): Promise<void> {
    if (this.apiContext) {
      await this.apiContext.dispose();
      this.apiContext = null;
    }
  }
}