import axios from 'axios';

// Create axios instance with base configuration
const api = axios.create({
  baseURL: 'http://localhost:8001/api/v1',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    console.log('Making API request:', config.method?.toUpperCase(), config.url);
    return config;
  },
  (error) => {
    console.error('Request error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => {
    console.log('API response:', response.status, response.config.url);
    return response;
  },
  (error) => {
    console.error('Response error:', error.response?.status, error.response?.data);
    return Promise.reject(error);
  }
);

// Types for backend API
export interface BackendSearchRequest {
  query: string;
  max_results?: number;
  include_citations?: boolean;
  search_type?: 'hybrid' | 'vector' | 'keyword';
}

export interface BackendSearchResponse {
  query: string;
  answer: string;
  sources: Array<{
    url: string;
    title: string;
    content: string;
    relevance_score: number;
    domain: string;
    publish_date: string;
    author?: string;
  }>;
  citations: Array<{
    text: string;
    source_url: string;
    source_title: string;
    relevance_score: number;
    position: number;
  }>;
  search_time: number;
  model_used: string;
  search_type: string;
  total_sources_found: number;
}

// Legacy types (keeping for compatibility)
export interface SearchRequest {
  query: string;
  filters?: {
    content_type?: string[];
    date_range?: string;
    sort_by?: string;
  };
  limit?: number;
  offset?: number;
}

export interface SearchResult {
  id: string;
  title: string;
  content: string;
  score: number;
  content_type: 'text' | 'code' | 'image' | 'video' | 'audio' | 'structured_data';
  url?: string;
  metadata?: {
    author?: string;
    date?: string;
    tags?: string[];
  };
}

export interface SearchResponse {
  results: SearchResult[];
  total: number;
  query_info: {
    processed_query: string;
    intent: string;
    entities: Array<{
      text: string;
      type: string;
      confidence: number;
    }>;
  };
  search_stats: {
    total_time_ms: number;
    index_time_ms: number;
    ranking_time_ms: number;
  };
}

export interface ChatMessage {
  content: string;
  role: 'user' | 'assistant';
}

export interface ChatRequest {
  message: string;
  conversation_id?: string;
  context?: {
    search_results?: SearchResult[];
    previous_messages?: ChatMessage[];
  };
}

export interface ChatResponse {
  response: string;
  conversation_id: string;
  sources?: Array<{
    title: string;
    content: string;
    score: number;
  }>;
  metadata: {
    response_time_ms: number;
    model_used: string;
    confidence: number;
  };
}

export interface DocumentUploadRequest {
  file: File;
  metadata?: {
    title?: string;
    author?: string;
    tags?: string[];
    category?: string;
  };
}

export interface DocumentUploadResponse {
  document_id: string;
  filename: string;
  size: number;
  content_type: string;
  processing_status: 'pending' | 'processing' | 'completed' | 'failed';
  metadata: {
    upload_time: string;
    processing_time_ms?: number;
  };
}

export interface HealthResponse {
  status: 'healthy' | 'unhealthy';
  version: string;
  uptime: number;
  components: {
    database: 'healthy' | 'unhealthy';
    vector_index: 'healthy' | 'unhealthy';
    search_engine: 'healthy' | 'unhealthy';
  };
}

// API Service Class
class ApiService {
  // Health Check
  async health(): Promise<HealthResponse> {
    try {
      const response = await api.get('/health');
      return response.data;
    } catch (error) {
      console.error('Health check failed:', error);
      throw error;
    }
  }

  // Search API (Updated for backend compatibility)
  async search(query: string, options?: {
    max_results?: number;
    include_citations?: boolean;
    search_type?: 'hybrid' | 'vector' | 'keyword';
  }): Promise<BackendSearchResponse> {
    try {
      const request: BackendSearchRequest = {
        query,
        max_results: options?.max_results || 10,
        include_citations: options?.include_citations ?? true,
        search_type: options?.search_type || 'hybrid'
      };
      
      const response = await api.post('/search', request);
      return response.data;
    } catch (error) {
      console.error('Search request failed:', error);
      throw error;
    }
  }

  // Legacy search method (keeping for compatibility)
  async legacySearch(request: SearchRequest): Promise<SearchResponse> {
    try {
      const response = await api.post('/search', request);
      return response.data;
    } catch (error) {
      console.error('Search request failed:', error);
      throw error;
    }
  }

  // Advanced search with query understanding
  async advancedSearch(query: string, filters?: any): Promise<SearchResponse> {
    try {
      const request: SearchRequest = {
        query,
        filters,
        limit: 20,
        offset: 0,
      };
      
      const response = await api.post('/search/advanced', request);
      return response.data;
    } catch (error) {
      console.error('Advanced search failed:', error);
      throw error;
    }
  }

  // Chat API
  async chat(request: ChatRequest): Promise<ChatResponse> {
    try {
      const response = await api.post('/chat', request);
      return response.data;
    } catch (error) {
      console.error('Chat request failed:', error);
      throw error;
    }
  }

  // Streaming chat (using EventSource)
  async streamChat(
    request: ChatRequest,
    onMessage: (chunk: string) => void,
    onComplete: (response: ChatResponse) => void,
    onError: (error: Error) => void
  ): Promise<void> {
    try {
      const response = await api.post('/chat/stream', request, {
        responseType: 'stream',
      });

      // Note: In a real implementation, you'd use EventSource or WebSocket
      // This is a simplified version
      onComplete({
        response: 'This is a mock streaming response. In production, this would use Server-Sent Events.',
        conversation_id: request.conversation_id || 'new-conversation',
        metadata: {
          response_time_ms: 1500,
          model_used: 'gpt-4',
          confidence: 0.95,
        },
      });
    } catch (error) {
      console.error('Streaming chat failed:', error);
      onError(error as Error);
    }
  }

  // Document Upload API
  async uploadDocument(request: DocumentUploadRequest): Promise<DocumentUploadResponse> {
    try {
      const formData = new FormData();
      formData.append('file', request.file);
      
      if (request.metadata) {
        formData.append('metadata', JSON.stringify(request.metadata));
      }

      const response = await api.post('/documents/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round(
            (progressEvent.loaded * 100) / (progressEvent.total || 1)
          );
          console.log('Upload progress:', percentCompleted + '%');
        },
      });

      return response.data;
    } catch (error) {
      console.error('Document upload failed:', error);
      throw error;
    }
  }

  // Get document list
  async getDocuments(limit: number = 50, offset: number = 0): Promise<any> {
    try {
      const response = await api.get('/documents', {
        params: { limit, offset },
      });
      return response.data;
    } catch (error) {
      console.error('Get documents failed:', error);
      throw error;
    }
  }

  // Delete document
  async deleteDocument(documentId: string): Promise<void> {
    try {
      await api.delete(`/documents/${documentId}`);
    } catch (error) {
      console.error('Delete document failed:', error);
      throw error;
    }
  }

  // Get search suggestions
  async getSearchSuggestions(query: string): Promise<string[]> {
    try {
      const response = await api.get('/search/suggestions', {
        params: { q: query },
      });
      return response.data.suggestions || [];
    } catch (error) {
      console.error('Get suggestions failed:', error);
      // Return empty array on error
      return [];
    }
  }

  // Analytics and stats
  async getSearchStats(): Promise<any> {
    try {
      const response = await api.get('/search/stats');
      return response.data;
    } catch (error) {
      console.error('Get search stats failed:', error);
      throw error;
    }
  }

  // Real-time indexing status
  async getIndexingStatus(): Promise<any> {
    try {
      const response = await api.get('/indexing/status');
      return response.data;
    } catch (error) {
      console.error('Get indexing status failed:', error);
      throw error;
    }
  }
}

// Export singleton instance
export const apiService = new ApiService();
export default apiService;