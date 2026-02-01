import { promises as fs } from 'fs';
import path from 'path';

/**
 * Test Data Manager
 * Manages test documents, conversations, and user data for UI tests
 */
export class TestDataManager {
  private testDataDir: string;
  private documentsDir: string;
  private conversationsDir: string;
  
  constructor() {
    this.testDataDir = path.join(process.cwd(), 'test-data');
    this.documentsDir = path.join(this.testDataDir, 'documents');
    this.conversationsDir = path.join(this.testDataDir, 'conversations');
  }
  
  async initialize() {
    console.log('📁 Initializing test data manager...');
    
    // Create test data directories
    await this.ensureDirectoryExists(this.testDataDir);
    await this.ensureDirectoryExists(this.documentsDir);
    await this.ensureDirectoryExists(this.conversationsDir);
    
    // Create test documents
    await this.createTestDocuments();
    
    // Create test conversations
    await this.createTestConversations();
    
    console.log('✅ Test data manager initialized');
  }
  
  async createTestDocuments() {
    const testDocuments = [
      {
        filename: 'sample_policy.pdf',
        content: 'This is a sample company policy document. It contains important information about employee guidelines, procedures, and company regulations.',
        metadata: {
          title: 'Company Policy Document',
          author: 'HR Department',
          created_at: '2024-01-15',
          category: 'policy'
        }
      },
      {
        filename: 'technical_manual.txt',
        content: 'Technical Manual for Enterprise RAG System. This document explains how to configure and maintain the RAG pipeline, including vector database setup, embedding models, and query optimization.',
        metadata: {
          title: 'Technical Manual',
          author: 'Engineering Team',
          created_at: '2024-02-01',
          category: 'technical'
        }
      },
      {
        filename: 'meeting_notes.md',
        content: '# Q1 Strategy Meeting\n\n## Attendees\n- CEO\n- CTO\n- VP Engineering\n\n## Key Decisions\n1. Implement RAG system\n2. Focus on enterprise features\n3. Prioritize security and compliance',
        metadata: {
          title: 'Q1 Strategy Meeting Notes',
          author: 'Executive Team',
          created_at: '2024-03-01',
          category: 'meeting'
        }
      },
      {
        filename: 'large_document.txt',
        content: 'Large Document Content. '.repeat(1000) + 'This is a large document used for testing document chunking, embedding generation, and retrieval performance.',
        metadata: {
          title: 'Large Test Document',
          author: 'Test Suite',
          created_at: '2024-01-01',
          category: 'test'
        }
      }
    ];
    
    for (const doc of testDocuments) {
      const filePath = path.join(this.documentsDir, doc.filename);
      await fs.writeFile(filePath, doc.content);
      
      // Create metadata file
      const metadataPath = path.join(this.documentsDir, `${doc.filename}.meta.json`);
      await fs.writeFile(metadataPath, JSON.stringify(doc.metadata, null, 2));
    }
    
    console.log(`📄 Created ${testDocuments.length} test documents`);
  }
  
  async createTestConversations() {
    const testConversations = [
      {
        id: 'conv_001',
        title: 'Company Policy Questions',
        messages: [
          {
            role: 'user',
            content: 'What is the company policy on remote work?',
            timestamp: '2024-01-15T10:00:00Z'
          },
          {
            role: 'assistant',
            content: 'Based on the company policy document, remote work is allowed up to 3 days per week with manager approval.',
            timestamp: '2024-01-15T10:00:05Z',
            sources: ['sample_policy.pdf']
          }
        ],
        user_id: 'user_test',
        created_at: '2024-01-15T10:00:00Z',
        updated_at: '2024-01-15T10:00:05Z'
      },
      {
        id: 'conv_002',
        title: 'Technical Support',
        messages: [
          {
            role: 'user',
            content: 'How do I configure the embedding model?',
            timestamp: '2024-02-01T14:30:00Z'
          },
          {
            role: 'assistant',
            content: 'To configure the embedding model, you need to update the model settings in the configuration file as described in the technical manual.',
            timestamp: '2024-02-01T14:30:08Z',
            sources: ['technical_manual.txt']
          }
        ],
        user_id: 'admin_test',
        created_at: '2024-02-01T14:30:00Z',
        updated_at: '2024-02-01T14:30:08Z'
      }
    ];
    
    for (const conv of testConversations) {
      const filePath = path.join(this.conversationsDir, `${conv.id}.json`);
      await fs.writeFile(filePath, JSON.stringify(conv, null, 2));
    }
    
    console.log(`💬 Created ${testConversations.length} test conversations`);
  }
  
  async getTestDocument(filename: string): Promise<string> {
    const filePath = path.join(this.documentsDir, filename);
    return await fs.readFile(filePath, 'utf-8');
  }
  
  async getTestConversation(conversationId: string): Promise<any> {
    const filePath = path.join(this.conversationsDir, `${conversationId}.json`);
    const content = await fs.readFile(filePath, 'utf-8');
    return JSON.parse(content);
  }
  
  async listTestDocuments(): Promise<string[]> {
    const files = await fs.readdir(this.documentsDir);
    return files.filter(file => !file.endsWith('.meta.json'));
  }
  
  async cleanup() {
    console.log('🧹 Cleaning up test data...');
    
    try {
      await fs.rmdir(this.testDataDir, { recursive: true });
      console.log('✅ Test data cleaned up');
    } catch (error) {
      console.warn('⚠️ Error cleaning up test data:', error);
    }
  }
  
  private async ensureDirectoryExists(dirPath: string) {
    try {
      await fs.access(dirPath);
    } catch {
      await fs.mkdir(dirPath, { recursive: true });
    }
  }
}