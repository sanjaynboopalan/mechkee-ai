/**
 * Enhanced API Service with RAG Integration
 * Provides both standard chat and enhanced RAG-powered conversations
 */

export interface StandardChatRequest {
  message: string;
  session_id?: string;
}

export interface StandardChatResponse {
  response: string;
  session_id: string;
  conversation_length: number;
  processing_time: number;
}

export interface EnhancedChatRequest {
  message: string;
  session_id?: string;
  use_rag?: boolean;
  search_web?: boolean;
}

export interface SourceInfo {
  title: string;
  source: string;
  relevance_score: number;
  snippet?: string;
}

export interface EnhancedChatResponse {
  response: string;
  session_id: string;
  conversation_length: number;
  processing_time: number;
  context_used: boolean;
  sources: SourceInfo[];
  rag_enabled: boolean;
  response_type: string;
}

export interface DocumentUploadResponse {
  document_id: string;
  filename: string;
  chunks_created: number;
  status: string;
  message: string;
}

export interface KnowledgeBaseStats {
  total_documents: number;
  total_chunks: number;
  embedding_dimension: number;
  last_updated: string;
}

class EnhancedAPIService {
  private baseURL: string;

  constructor() {
    this.baseURL = process.env.REACT_APP_API_URL || 'http://localhost:8001/api/v1';
  }

  // Standard Chat Methods
  async sendMessage(request: StandardChatRequest): Promise<StandardChatResponse> {
    try {
      const response = await fetch(`${this.baseURL}/chat/enhanced`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Standard chat error:', error);
      throw error;
    }
  }

  async clearConversation(sessionId: string): Promise<void> {
    try {
      const response = await fetch(`${this.baseURL}/chat/${sessionId}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
    } catch (error) {
      console.error('Clear conversation error:', error);
      throw error;
    }
  }

  // Enhanced Chat Methods (RAG-powered)
  async sendEnhancedMessage(request: EnhancedChatRequest): Promise<EnhancedChatResponse> {
    try {
      const response = await fetch(`${this.baseURL}/enhanced-chat/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Enhanced chat error:', error);
      throw error;
    }
  }

  async clearEnhancedConversation(sessionId: string): Promise<void> {
    try {
      const response = await fetch(`${this.baseURL}/enhanced-chat/${sessionId}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
    } catch (error) {
      console.error('Clear enhanced conversation error:', error);
      throw error;
    }
  }

  // Document Management Methods
  async uploadDocument(file: File, title?: string, description?: string): Promise<DocumentUploadResponse> {
    try {
      const formData = new FormData();
      formData.append('file', file);
      if (title) formData.append('title', title);
      if (description) formData.append('description', description);

      const response = await fetch(`${this.baseURL}/enhanced-chat/upload-document`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Document upload error:', error);
      throw error;
    }
  }

  async addURL(url: string, title?: string): Promise<any> {
    try {
      const formData = new FormData();
      formData.append('url', url);
      if (title) formData.append('title', title);

      const response = await fetch(`${this.baseURL}/enhanced-chat/add-url`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Add URL error:', error);
      throw error;
    }
  }

  // Knowledge Base Methods
  async getKnowledgeBaseStats(): Promise<KnowledgeBaseStats> {
    try {
      const response = await fetch(`${this.baseURL}/enhanced-chat/knowledge-base/stats`);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Knowledge base stats error:', error);
      throw error;
    }
  }

  // Conversation History Methods
  async getConversationHistory(sessionId: string, enhanced: boolean = false): Promise<any> {
    try {
      const endpoint = enhanced ? '/chat/enhanced' : '/chat/';
      const response = await fetch(`${this.baseURL}${endpoint}${sessionId}/history`);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Get conversation history error:', error);
      throw error;
    }
  }

  async listSessions(enhanced: boolean = false): Promise<any> {
    try {
      const endpoint = enhanced ? '/chat/enhanced' : '/chat/sessions';
      const response = await fetch(`${this.baseURL}${endpoint}`);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('List sessions error:', error);
      throw error;
    }
  }

  // Health Check Methods
  async checkHealth(): Promise<any> {
    try {
      const response = await fetch(`${this.baseURL}/health/`);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Health check error:', error);
      throw error;
    }
  }

  async checkEnhancedHealth(): Promise<any> {
    try {
      const response = await fetch(`${this.baseURL}/enhanced-chat/health`);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Enhanced health check error:', error);
      throw error;
    }
  }
}

// Create and export a singleton instance
export const enhancedAPI = new EnhancedAPIService();

// Export the class for custom instances if needed
export default EnhancedAPIService;