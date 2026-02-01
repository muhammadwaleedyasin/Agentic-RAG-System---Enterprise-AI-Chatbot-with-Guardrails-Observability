import { http, HttpResponse } from 'msw';

const API_BASE_URL = 'http://localhost:3001';

export const handlers = [
  // Auth endpoints
  http.post(`${API_BASE_URL}/api/v1/auth/login`, async ({ request }) => {
    const body = await request.json() as { username: string; password: string };
    
    if (body.username === 'testuser' && body.password === 'testpass') {
      return HttpResponse.json({
        access_token: 'mock-jwt-token',
        token_type: 'bearer',
        user: {
          id: '1',
          username: 'testuser',
          email: 'test@example.com',
          roles: [
            {
              name: 'user',
              permissions: [
                { resource: 'documents', action: 'read' },
                { resource: 'conversations', action: 'read' },
              ],
            },
          ],
        },
      });
    }
    
    return HttpResponse.json(
      { message: 'Invalid credentials' },
      { status: 401 }
    );
  }),

  http.get(`${API_BASE_URL}/api/v1/auth/me`, ({ request }) => {
    const authHeader = request.headers.get('authorization');
    
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return HttpResponse.json(
        { message: 'Unauthorized' },
        { status: 401 }
      );
    }
    
    return HttpResponse.json({
      data: {
        id: '1',
        username: 'testuser',
        email: 'test@example.com',
        roles: [
          {
            name: 'user',
            permissions: [
              { resource: 'documents', action: 'read' },
              { resource: 'conversations', action: 'read' },
            ],
          },
        ],
      },
    });
  }),

  http.post(`${API_BASE_URL}/api/v1/auth/logout`, () => {
    return HttpResponse.json({ message: 'Logged out successfully' });
  }),

  // Conversation endpoints
  http.get(`${API_BASE_URL}/api/v1/memory/conversations`, () => {
    return HttpResponse.json({
      data: [
        {
          id: 'conv-1',
          title: 'Test Conversation',
          created_at: '2024-01-01T00:00:00Z',
          updated_at: '2024-01-01T00:00:00Z',
          metadata: {},
        },
      ],
    });
  }),

  http.post(`${API_BASE_URL}/api/v1/memory/conversations`, async ({ request }) => {
    const body = await request.json() as { title?: string; metadata?: Record<string, any> };
    
    return HttpResponse.json({
      data: {
        id: 'conv-new',
        title: body.title || 'New Conversation',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        metadata: body.metadata || {},
      },
    });
  }),

  http.get(`${API_BASE_URL}/api/v1/memory/conversations/:id`, ({ params }) => {
    return HttpResponse.json({
      data: {
        id: params.id,
        title: 'Test Conversation',
        created_at: '2024-01-01T00:00:00Z',
        updated_at: '2024-01-01T00:00:00Z',
        metadata: {},
      },
    });
  }),

  http.delete(`${API_BASE_URL}/api/v1/memory/conversations/:id`, () => {
    return HttpResponse.json({ message: 'Conversation deleted' });
  }),

  // Message endpoints
  http.get(`${API_BASE_URL}/api/v1/memory/conversations/:id/messages`, () => {
    return HttpResponse.json({
      data: [
        {
          id: 'msg-1',
          content: 'Hello, this is a test message',
          role: 'user',
          created_at: '2024-01-01T00:00:00Z',
          metadata: {},
        },
        {
          id: 'msg-2',
          content: 'This is a response',
          role: 'assistant',
          created_at: '2024-01-01T00:01:00Z',
          metadata: {},
        },
      ],
    });
  }),

  http.post(`${API_BASE_URL}/api/v1/memory/conversations/:id/messages`, async ({ request }) => {
    const body = await request.json() as { content: string; metadata?: Record<string, any> };
    
    return HttpResponse.json({
      data: {
        id: 'msg-new',
        content: body.content,
        role: 'user',
        created_at: new Date().toISOString(),
        metadata: body.metadata || {},
      },
    });
  }),

  // Document endpoints
  http.get(`${API_BASE_URL}/api/v1/documents`, () => {
    return HttpResponse.json({
      data: [
        {
          id: 'doc-1',
          filename: 'test-document.pdf',
          content_type: 'application/pdf',
          size: 1024,
          created_at: '2024-01-01T00:00:00Z',
          metadata: {},
          tags: ['test'],
        },
      ],
    });
  }),

  http.post(`${API_BASE_URL}/api/v1/documents/upload`, async ({ request }) => {
    const formData = await request.formData();
    const file = formData.get('file') as File;
    
    if (!file) {
      return HttpResponse.json(
        { message: 'No file provided' },
        { status: 400 }
      );
    }
    
    return HttpResponse.json({
      data: {
        id: 'doc-new',
        filename: file.name,
        content_type: file.type,
        size: file.size,
        created_at: new Date().toISOString(),
        metadata: {},
        tags: [],
      },
    });
  }),

  http.delete(`${API_BASE_URL}/api/v1/documents/:id`, () => {
    return HttpResponse.json({ message: 'Document deleted' });
  }),

  // Health endpoint
  http.get(`${API_BASE_URL}/api/v1/health`, () => {
    return HttpResponse.json({
      status: 'healthy',
      timestamp: new Date().toISOString(),
    });
  }),

  // Admin endpoints
  http.get(`${API_BASE_URL}/api/v1/admin/users`, () => {
    return HttpResponse.json({
      data: [
        {
          id: '1',
          username: 'testuser',
          email: 'test@example.com',
          created_at: '2024-01-01T00:00:00Z',
          roles: ['user'],
        },
      ],
    });
  }),

  http.get(`${API_BASE_URL}/api/v1/admin/settings`, () => {
    return HttpResponse.json({
      data: {
        site_name: 'Test Site',
        max_upload_size: 10485760,
        allowed_file_types: ['pdf', 'txt', 'docx'],
      },
    });
  }),

  http.put(`${API_BASE_URL}/api/v1/admin/settings`, async ({ request }) => {
    const body = await request.json();
    return HttpResponse.json({
      data: body,
    });
  }),

  http.get(`${API_BASE_URL}/api/v1/admin/audit-logs`, () => {
    return HttpResponse.json({
      data: [
        {
          id: 'log-1',
          action: 'login',
          user_id: '1',
          timestamp: '2024-01-01T00:00:00Z',
          details: {},
        },
      ],
    });
  }),

  // Analytics endpoint
  http.get(`${API_BASE_URL}/api/v1/analytics`, () => {
    return HttpResponse.json({
      data: {
        total_users: 10,
        total_documents: 25,
        total_conversations: 50,
        active_users_today: 5,
      },
    });
  }),
];