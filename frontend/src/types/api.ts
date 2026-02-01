// Base API response envelope
export interface ApiResponse<T = any> {
  data: T;
  message?: string;
  success: boolean;
  errors?: Record<string, string[]>;
}

export interface PaginatedResponse<T> {
  data: T[];
  total: number;
  page: number;
  per_page: number;
  pages: number;
}

// Authentication & User types
export interface LoginRequest {
  username: string;
  password: string;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
}

export interface User {
  id: string;
  username: string;
  email: string;
  full_name?: string;
  is_active: boolean;
  is_superuser: boolean;
  created_at: string;
  updated_at: string;
  roles: Role[];
}

export interface Role {
  id: string;
  name: string;
  description?: string;
  permissions: Permission[];
}

export interface Permission {
  id: string;
  name: string;
  resource: string;
  action: string;
}

// Chat & Conversation types
export interface Message {
  id: string;
  conversation_id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  metadata?: Record<string, any>;
  created_at: string;
  updated_at: string;
}

export interface Conversation {
  id: string;
  user_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  last_message_at?: string;
  message_count: number;
  metadata?: Record<string, any>;
}

export interface ChatMessage {
  id?: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp?: string;
  metadata?: {
    sources?: DocumentReference[];
    thinking?: string;
    confidence?: number;
  };
}

export interface StreamingResponse {
  type: 'token' | 'message_complete' | 'error';
  data: string;
  message_id?: string;
  metadata?: Record<string, any>;
}

// Document types
export interface Document {
  id: string;
  filename: string;
  original_filename: string;
  file_size: number;
  mime_type: string;
  upload_status: 'pending' | 'processing' | 'completed' | 'failed';
  processing_status?: 'pending' | 'chunking' | 'embedding' | 'completed' | 'failed';
  chunk_count?: number;
  metadata?: Record<string, any>;
  uploaded_by: string;
  uploaded_at: string;
  processed_at?: string;
  tags: string[];
}

export interface DocumentUploadRequest {
  file: File;
  tags?: string[];
  metadata?: Record<string, any>;
}

export interface DocumentUploadResponse {
  document_id: string;
  filename: string;
  upload_url?: string;
  status: string;
}

export interface DocumentReference {
  document_id: string;
  filename: string;
  chunk_id?: string;
  relevance_score?: number;
  snippet?: string;
}

// Analytics & Dashboard types
export interface AnalyticsData {
  conversations: {
    total: number;
    today: number;
    week_growth: number;
  };
  messages: {
    total: number;
    today: number;
    avg_per_conversation: number;
  };
  documents: {
    total: number;
    total_size: number;
    processing_queue: number;
  };
  users: {
    total: number;
    active_today: number;
    active_week: number;
  };
}

export interface SystemHealth {
  status: 'healthy' | 'degraded' | 'unhealthy';
  version: string;
  uptime: number;
  services: {
    database: ServiceStatus;
    vector_db: ServiceStatus;
    llm_provider: ServiceStatus;
    cache: ServiceStatus;
  };
  metrics: {
    memory_usage: number;
    cpu_usage: number;
    disk_usage: number;
    response_time: number;
  };
}

export interface ServiceStatus {
  status: 'healthy' | 'degraded' | 'unhealthy';
  last_check: string;
  message?: string;
}

// Admin types
export interface AdminSettings {
  site_name: string;
  max_upload_size: number;
  allowed_file_types: string[];
  rate_limits: {
    messages_per_minute: number;
    uploads_per_hour: number;
  };
  llm_settings: {
    default_model: string;
    max_tokens: number;
    temperature: number;
  };
  security: {
    session_timeout: number;
    max_login_attempts: number;
    require_2fa: boolean;
  };
}

export interface AuditLog {
  id: string;
  user_id: string;
  user_email: string;
  action: string;
  resource: string;
  resource_id?: string;
  details?: Record<string, any>;
  ip_address: string;
  user_agent: string;
  created_at: string;
}

// WebSocket types
export interface WebSocketMessage {
  type: string;
  data: any;
  message_id?: string;
  conversation_id?: string;
  timestamp: string;
}

export interface ChatSocketEvents {
  message: ChatMessage;
  typing_start: { user_id: string; conversation_id: string };
  typing_stop: { user_id: string; conversation_id: string };
  conversation_updated: Conversation;
  error: { message: string; code?: string };
}

// API Error types
export interface ApiError {
  message: string;
  code?: string;
  field?: string;
  details?: Record<string, any>;
}

// Query & Filter types
export interface QueryParams {
  page?: number;
  per_page?: number;
  sort_by?: string;
  sort_order?: 'asc' | 'desc';
  search?: string;
  filters?: Record<string, any>;
}

export interface ConversationFilters {
  user_id?: string;
  created_after?: string;
  created_before?: string;
  has_messages?: boolean;
}

export interface DocumentFilters {
  status?: Document['upload_status'];
  mime_type?: string;
  uploaded_after?: string;
  uploaded_before?: string;
  tags?: string[];
  uploaded_by?: string;
}