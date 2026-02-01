/**
 * Database Seeder for UI Tests
 * Seeds test database with users, documents, and conversations
 */
export class DatabaseSeeder {
  private baseURL: string;
  
  constructor() {
    this.baseURL = process.env.BASE_URL || 'http://localhost:8000';
  }
  
  async seedTestData() {
    console.log('🌱 Seeding test database...');
    
    try {
      // Seed users
      await this.seedUsers();
      
      // Seed documents
      await this.seedDocuments();
      
      // Seed conversations
      await this.seedConversations();
      
      console.log('✅ Test database seeded successfully');
      
    } catch (error) {
      console.error('❌ Failed to seed test database:', error);
      throw error;
    }
  }
  
  private async seedUsers() {
    const users = [
      {
        user_id: 'admin_test',
        username: 'admin_test',
        email: 'admin@test.com',
        password: 'AdminTest123!',
        role: 'admin',
        is_active: true
      },
      {
        user_id: 'user_test',
        username: 'user_test',
        email: 'user@test.com',
        password: 'UserTest123!',
        role: 'user',
        is_active: true
      },
      {
        user_id: 'viewer_test',
        username: 'viewer_test',
        email: 'viewer@test.com',
        password: 'ViewerTest123!',
        role: 'viewer',
        is_active: true
      }
    ];
    
    for (const user of users) {
      try {
        await this.apiRequest('POST', '/api/v1/admin/users', user);
        console.log(`👤 Seeded user: ${user.username}`);
      } catch (error) {
        console.log(`ℹ️ User ${user.username} may already exist`);
      }
    }
  }
  
  private async seedDocuments() {
    const documents = [
      {
        id: 'doc_001',
        title: 'Company Policy Document',
        content: 'This is a sample company policy document with important guidelines.',
        metadata: {
          author: 'HR Department',
          category: 'policy',
          file_type: 'pdf',
          file_size: 1024
        },
        created_by: 'admin_test'
      },
      {
        id: 'doc_002',
        title: 'Technical Manual',
        content: 'Technical documentation for the Enterprise RAG system.',
        metadata: {
          author: 'Engineering Team',
          category: 'technical',
          file_type: 'txt',
          file_size: 2048
        },
        created_by: 'admin_test'
      },
      {
        id: 'doc_003',
        title: 'Meeting Notes',
        content: 'Q1 strategy meeting notes with key decisions and action items.',
        metadata: {
          author: 'Executive Team',
          category: 'meeting',
          file_type: 'md',
          file_size: 512
        },
        created_by: 'user_test'
      }
    ];
    
    for (const doc of documents) {
      try {
        await this.apiRequest('POST', '/api/v1/documents', doc);
        console.log(`📄 Seeded document: ${doc.title}`);
      } catch (error) {
        console.log(`ℹ️ Document ${doc.title} may already exist`);
      }
    }
  }
  
  private async seedConversations() {
    const conversations = [
      {
        conversation_id: 'conv_001',
        user_id: 'user_test',
        title: 'Company Policy Questions',
        messages: [
          {
            role: 'user',
            content: 'What is the company policy on remote work?',
            timestamp: new Date('2024-01-15T10:00:00Z').toISOString()
          },
          {
            role: 'assistant',
            content: 'Based on the company policy document, remote work is allowed up to 3 days per week.',
            timestamp: new Date('2024-01-15T10:00:05Z').toISOString(),
            sources: ['doc_001']
          }
        ]
      },
      {
        conversation_id: 'conv_002',
        user_id: 'admin_test',
        title: 'Technical Questions',
        messages: [
          {
            role: 'user',
            content: 'How do I configure the embedding model?',
            timestamp: new Date('2024-02-01T14:30:00Z').toISOString()
          },
          {
            role: 'assistant',
            content: 'You can configure the embedding model in the settings panel.',
            timestamp: new Date('2024-02-01T14:30:08Z').toISOString(),
            sources: ['doc_002']
          }
        ]
      }
    ];
    
    for (const conv of conversations) {
      try {
        await this.apiRequest('POST', '/api/v1/conversations', conv);
        console.log(`💬 Seeded conversation: ${conv.title}`);
      } catch (error) {
        console.log(`ℹ️ Conversation ${conv.title} may already exist`);
      }
    }
  }
  
  async cleanupTestData() {
    console.log('🧹 Cleaning up test database...');
    
    try {
      // Delete test conversations
      await this.apiRequest('DELETE', '/api/v1/test/conversations');
      
      // Delete test documents
      await this.apiRequest('DELETE', '/api/v1/test/documents');
      
      // Delete test users
      await this.apiRequest('DELETE', '/api/v1/test/users');
      
      console.log('✅ Test database cleaned up');
      
    } catch (error) {
      console.warn('⚠️ Error cleaning up test database:', error);
    }
  }
  
  private async apiRequest(method: string, endpoint: string, data?: any) {
    const fetch = (await import('node-fetch')).default;
    
    const options: any = {
      method,
      headers: {
        'Content-Type': 'application/json',
        'X-Test-Mode': 'true'
      }
    };
    
    if (data) {
      options.body = JSON.stringify(data);
    }
    
    const response = await fetch(`${this.baseURL}${endpoint}`, options);
    
    if (!response.ok) {
      throw new Error(`API request failed: ${response.status} ${response.statusText}`);
    }
    
    return response.json();
  }
}