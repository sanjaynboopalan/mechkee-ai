import axios from 'axios';

// Demo mode for GitHub Pages - simulate API responses
const DEMO_MODE = process.env.NODE_ENV === 'production';

// Create axios instance with base configuration
const api = axios.create({
  baseURL: DEMO_MODE ? '' : 'http://localhost:8001/api/v1',
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
    console.error('API error:', error.response?.status, error.response?.data);
    return Promise.reject(error);
  }
);

// Types
export interface Source {
  id: string;
  title: string;
  url: string;
  snippet: string;
  credibility: number;
  cite_number: number;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  sources?: Source[];
  key_points?: string[];
  confidence_score?: number;
  word_count?: number;
  search_metadata?: any;
}

export interface ChatRequest {
  message: string;
  session_id?: string;
  search_web?: boolean;
  detail_level?: string;
  response_style?: string;
  include_sources?: boolean;
  max_sources?: number;
}

export interface ChatResponse {
  response: string;
  session_id: string;
  conversation_length: number;
  processing_time: number;
  word_count: number;
  sources: Source[];
  key_points: string[];
  confidence_score: number;
  search_metadata: any;
}

export interface HealthResponse {
  status: string;
  service: string;
  timestamp: string;
}

// API functions
export const chatAPI = {
  // Send a chat message and get AI response
  async sendMessage(request: ChatRequest): Promise<ChatResponse> {
    try {
      // Demo mode for GitHub Pages
      if (DEMO_MODE) {
        // Simulate API delay
        await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 2000));
        
        // Generate demo response
        const demoResponses = [
          "Hello! I'm MechKee.ai, your intelligent assistant. I'm currently running in demo mode since this is hosted on GitHub Pages. In a full deployment, I would be connected to a backend server with AI capabilities, web search, and document processing features.",
          "I'm designed to help with various tasks including answering questions, searching the web for information, and processing documents. This mobile-responsive interface works great on all devices!",
          "While in demo mode, I can show you the interface and user experience. For full functionality, you would need to deploy the backend server with your API keys configured.",
          "Thanks for trying MechKee.ai! This demo showcases the responsive design and chat interface. The full version includes real-time search, document analysis, and advanced AI reasoning capabilities."
        ];
        
        const randomResponse = demoResponses[Math.floor(Math.random() * demoResponses.length)];
        
        return {
          response: randomResponse,
          session_id: request.session_id || 'demo-session',
          conversation_length: 1,
          processing_time: 1500,
          word_count: randomResponse.length,
          sources: [],
          key_points: ['Demo Mode Active', 'GitHub Pages Deployment', 'Mobile Responsive Design'],
          confidence_score: 1.0,
          search_metadata: { demo_mode: true }
        };
      }
      
      // Use basic chat endpoint for now since enhanced endpoint has configuration issues
      const basicRequest = {
        message: request.message,
        session_id: request.session_id
      };
      
      const response = await api.post('/chat', basicRequest);
      
      // Transform basic response to match expected ChatResponse interface
      return {
        response: response.data.response,
        session_id: response.data.session_id,
        conversation_length: response.data.conversation_length,
        processing_time: response.data.processing_time,
        word_count: response.data.response?.length || 0,
        sources: [], // Basic chat doesn't return sources
        key_points: [], // Basic chat doesn't return key points
        confidence_score: 1.0, // Default confidence
        search_metadata: {} // No search metadata
      };
    } catch (error) {
      console.error('Chat API error:', error);
      throw error;
    }
  },

  // Get conversation history
  async getHistory(sessionId: string): Promise<{session_id: string, conversation: ChatMessage[], message_count: number}> {
    try {
      const response = await api.get(`/chat/${sessionId}/history`);
      return response.data;
    } catch (error) {
      console.error('Get history error:', error);
      throw error;
    }
  },

  // Clear conversation
  async clearConversation(sessionId: string): Promise<{message: string}> {
    try {
      const response = await api.delete(`/chat/${sessionId}`);
      return response.data;
    } catch (error) {
      console.error('Clear conversation error:', error);
      throw error;
    }
  },

  // Health check
  async healthCheck(): Promise<HealthResponse> {
    try {
      const response = await api.get<HealthResponse>('/health/');
      return response.data;
    } catch (error) {
      console.error('Health check error:', error);
      throw error;
    }
  }
};

export default api;