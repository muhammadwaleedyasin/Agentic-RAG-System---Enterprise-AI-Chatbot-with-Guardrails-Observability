import { APIRequestContext, request } from '@playwright/test';

export interface TestUser {
  username: string;
  password: string;
  email: string;
  role: 'admin' | 'user' | 'viewer';
}

export class AuthService {
  private baseUrl: string;
  private apiContext: APIRequestContext | null = null;

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

  async createTestUser(user: TestUser): Promise<void> {
    const api = await this.getApiContext();
    
    try {
      // First, get admin token to create users
      const adminToken = await this.getDefaultAdminToken();
      
      const response = await api.post('/api/v1/admin/users', {
        headers: {
          'Authorization': `Bearer ${adminToken}`,
        },
        data: {
          username: user.username,
          password: user.password,
          email: user.email,
          role: user.role,
        },
      });
      
      if (!response.ok()) {
        const error = await response.text();
        console.warn(`Failed to create user ${user.username}: ${error}`);
      }
    } catch (error) {
      console.warn(`Error creating user ${user.username}:`, error);
    }
  }

  async getAuthToken(username: string, password: string): Promise<string> {
    const api = await this.getApiContext();
    
    const response = await api.post('/api/v1/auth/login', {
      data: {
        username,
        password,
      },
    });
    
    if (!response.ok()) {
      throw new Error(`Authentication failed for ${username}: ${response.status()}`);
    }
    
    const result = await response.json();
    return result.access_token;
  }

  async getDefaultAdminToken(): Promise<string> {
    // Use the default admin user created at startup
    return this.getAuthToken('admin', 'admin123');
  }

  async removeTestUser(username: string): Promise<void> {
    const api = await this.getApiContext();
    
    try {
      const adminToken = await this.getDefaultAdminToken();
      
      const response = await api.delete(`/api/v1/admin/users/${username}`, {
        headers: {
          'Authorization': `Bearer ${adminToken}`,
        },
      });
      
      if (!response.ok()) {
        console.warn(`Failed to remove user ${username}: ${response.status()}`);
      }
    } catch (error) {
      console.warn(`Error removing user ${username}:`, error);
    }
  }

  async validateToken(token: string): Promise<boolean> {
    const api = await this.getApiContext();
    
    try {
      const response = await api.get('/api/v1/auth/me', {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });
      
      return response.ok();
    } catch (error) {
      return false;
    }
  }

  async logout(token: string): Promise<void> {
    const api = await this.getApiContext();
    
    try {
      await api.post('/api/v1/auth/logout', {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });
    } catch (error) {
      console.warn('Error during logout:', error);
    }
  }

  async dispose(): Promise<void> {
    if (this.apiContext) {
      await this.apiContext.dispose();
      this.apiContext = null;
    }
  }
}