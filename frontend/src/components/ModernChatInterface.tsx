import React, { useState, useEffect, useRef, ChangeEvent } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Send, 
  User, 
  Bot, 
  Copy, 
  Upload, 
  Sparkles,
  MoreVertical,
  Share,
  RefreshCw,
  ExternalLink,
  Zap,
  Brain,
  Plus,
  Menu
} from 'lucide-react';
import Button from './ui/Button';
import Input from './ui/Input';
import DocumentUploader from './DocumentUploader';
import { chatAPI } from '../services/chatapi';
import { enhancedAPI, type SourceInfo } from '../services/enhancedapi';
import toast from 'react-hot-toast';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  sources?: SourceInfo[];
  response_type?: string;
  suggested_questions?: string[];
}

interface SuggestedQuestion {
  text: string;
  category?: string;
}

const ModernChatInterface: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'assistant',
      content: 'Hello! I\'m MechKee.ai, your intelligent assistant. I can help you with research, analysis, and provide detailed answers.',
      timestamp: new Date().toISOString(),
      suggested_questions: [
        'What are the latest trends in artificial intelligence?',
        'How does machine learning work?',
        'Explain quantum computing in simple terms',
        'What are the benefits of renewable energy?'
      ]
    }
  ]);
  
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string | undefined>(undefined);
  const [chatMode, setChatMode] = useState<'standard' | 'enhanced'>('standard');
  const [showUploader, setShowUploader] = useState(false);
  const [showSidebar, setShowSidebar] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Clean response by removing markdown formatting like ** and *
  const cleanResponse = (text: string): string => {
    return text
      .replace(/\*\*(.*?)\*\*/g, '$1') // Remove bold markdown **text**
      .replace(/\*(.*?)\*/g, '$1')     // Remove italic markdown *text*
      .replace(/•\s*/g, '• ')          // Normalize bullet points
      .replace(/\n\s*\n\s*\n/g, '\n\n') // Remove excessive line breaks
      .trim();
  };

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      role: 'user',
      content: input,
      timestamp: new Date().toISOString(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      let assistantMessage: Message;

      if (chatMode === 'enhanced') {
        const response = await enhancedAPI.sendEnhancedMessage({
          message: input,
          session_id: sessionId,
          use_rag: true,
        });

        if (!sessionId) {
          setSessionId(response.session_id);
        }

        assistantMessage = {
          role: 'assistant',
          content: cleanResponse(response.response),
          timestamp: new Date().toISOString(),
          sources: response.sources,
          response_type: response.response_type,
          suggested_questions: generateSuggestedQuestions(response.response)
        };
      } else {
        const response = await chatAPI.sendMessage({
          message: input,
          session_id: sessionId,
        });

        if (!sessionId) {
          setSessionId(response.session_id);
        }

        assistantMessage = {
          role: 'assistant',
          content: cleanResponse(response.response),
          timestamp: new Date().toISOString(),
          suggested_questions: generateSuggestedQuestions(response.response)
        };
      }

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Chat error:', error);
      toast.error('Failed to send message. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const clearConversation = () => {
    setMessages([{
      role: 'assistant',
      content: 'Hello! I\'m MechKee.ai, your intelligent assistant. How can I assist you today?',
      timestamp: new Date().toISOString(),
    }]);
    setSessionId(undefined);
  };

  const copyMessage = (content: string) => {
    navigator.clipboard.writeText(content);
    toast.success('Message copied to clipboard');
  };

  const handleSuggestedQuestion = (question: string) => {
    setInput(question);
  };

  const generateSuggestedQuestions = (messageContent: string): string[] => {
    // Generate contextual follow-up questions based on the response
    const suggestions = [
      'Can you provide more details about this topic?',
      'What are the practical applications?',
      'Are there any recent developments?',
      'What are the potential challenges?',
      'How does this compare to alternatives?'
    ];
    
    // Simple logic to generate more relevant questions based on content
    if (messageContent.toLowerCase().includes('technology') || messageContent.toLowerCase().includes('ai')) {
      return [
        'What are the latest advancements in this technology?',
        'What are the ethical implications?',
        'How will this impact the future?',
        'What are the current limitations?'
      ];
    } else if (messageContent.toLowerCase().includes('health') || messageContent.toLowerCase().includes('medical')) {
      return [
        'Are there any side effects to consider?',
        'What does recent research show?',
        'How can I learn more about this?',
        'What are the prevention strategies?'
      ];
    } else if (messageContent.toLowerCase().includes('business') || messageContent.toLowerCase().includes('market')) {
      return [
        'What are the market trends?',
        'How does this affect consumers?',
        'What are the investment opportunities?',
        'What are the risks involved?'
      ];
    }
    
    return suggestions.slice(0, 4);
  };

  return (
    <div className="modern-chat-container">
      {/* Mobile overlay */}
      {showSidebar && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-50 z-40 lg:hidden"
          onClick={() => setShowSidebar(false)}
        />
      )}
      
      {/* Sidebar */}
      <div className={`modern-sidebar ${showSidebar ? 'show' : ''}`}>
        <div className="sidebar-header">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
              <Brain className="w-5 h-5 text-white" />
            </div>
            <h1 className="text-lg font-bold text-gray-800">MechKee.ai</h1>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowSidebar(false)}
            className="lg:hidden"
          >
            ×
          </Button>
        </div>

        <div className="sidebar-content">
          <Button
            className="new-chat-btn"
            onClick={clearConversation}
          >
            <Plus className="w-4 h-4 mr-2" />
            New Chat
          </Button>

          <div className="chat-modes">
            <h3 className="text-sm font-semibold text-gray-600 mb-3">Chat Mode</h3>
            <div className="space-y-2">
              <button
                onClick={() => setChatMode('standard')}
                className={`mode-btn ${chatMode === 'standard' ? 'active' : ''}`}
              >
                <Zap className="w-4 h-4" />
                <span>Standard</span>
                <div className="mode-indicator" />
              </button>
              <button
                onClick={() => setChatMode('enhanced')}
                className={`mode-btn ${chatMode === 'enhanced' ? 'active' : ''}`}
              >
                <Sparkles className="w-4 h-4" />
                <span>Enhanced</span>
                <div className="mode-indicator" />
              </button>
            </div>
          </div>

          {chatMode === 'enhanced' && (
            <div className="mt-6">
              <Button
                onClick={() => setShowUploader(true)}
                className="upload-btn"
                variant="outline"
              >
                <Upload className="w-4 h-4 mr-2" />
                Upload Documents
              </Button>
            </div>
          )}
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="modern-chat-main">
        {/* Header */}
        <div className="chat-header">
          <div className="flex items-center space-x-3">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowSidebar(!showSidebar)}
              className="lg:hidden"
            >
              <Menu className="w-5 h-5" />
            </Button>
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-gradient-to-r from-emerald-500 to-blue-500 rounded-full flex items-center justify-center">
                <Bot className="w-5 h-5 text-white" />
              </div>
              <div>
                <h2 className="font-semibold text-gray-800">MechKee.ai</h2>
                <p className="text-sm text-gray-500">
                  {chatMode === 'enhanced' ? 'Enhanced Mode' : 'Standard Mode'}
                </p>
              </div>
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            <Button variant="ghost" size="sm">
              <Share className="w-4 h-4" />
            </Button>
            <Button variant="ghost" size="sm">
              <MoreVertical className="w-4 h-4" />
            </Button>
          </div>
        </div>

        {/* Messages */}
        <div className="messages-container">
          <div className="messages-inner">
            <AnimatePresence>
              {messages.map((message, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.3 }}
                  className={`message-wrapper ${message.role}`}
                >
                  <div className="message-content">
                    <div className="message-avatar">
                      {message.role === 'user' ? (
                        <User className="w-4 h-4" />
                      ) : (
                        <Bot className="w-4 h-4" />
                      )}
                    </div>
                    
                    <div className="message-body">
                      <div className="message-text">
                        {message.content.split('\n').map((line, i) => (
                          <p key={i} className={line.trim() === '' ? 'mb-2' : ''}>
                            {line}
                          </p>
                        ))}
                      </div>
                      
                      {/* Sources */}
                      {message.sources && message.sources.length > 0 && (
                        <div className="message-sources">
                          <div className="sources-header">
                            <ExternalLink className="w-4 h-4" />
                            <span>Sources ({message.sources.length})</span>
                          </div>
                          <div className="sources-list">
                            {message.sources.map((source, idx) => (
                              <div key={idx} className="source-item">
                                <div className="source-title">
                                  <a 
                                    href={source.source}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="source-link"
                                  >
                                    {source.title}
                                    <ExternalLink className="w-3 h-3 ml-1 inline" />
                                  </a>
                                </div>
                                <div className="source-meta">{new URL(source.source).hostname}</div>
                                {source.snippet && (
                                  <div className="source-snippet">"{source.snippet}"</div>
                                )}
                                <div className="source-relevance">
                                  {Math.round(source.relevance_score * 100)}% relevant
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Suggested Questions */}
                      {message.suggested_questions && message.suggested_questions.length > 0 && (
                        <div className="suggested-questions">
                          <div className="suggested-header">
                            <span>Related questions</span>
                          </div>
                          <div className="questions-grid">
                            {message.suggested_questions.map((question, idx) => (
                              <button
                                key={idx}
                                onClick={() => handleSuggestedQuestion(question)}
                                className="suggested-question-btn"
                              >
                                {question}
                              </button>
                            ))}
                          </div>
                        </div>
                      )}
                      
                      <div className="message-actions">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => copyMessage(message.content)}
                          className="action-btn"
                        >
                          <Copy className="w-3 h-3" />
                        </Button>
                      </div>
                    </div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>

            {/* Typing Indicator */}
            {isLoading && (
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                className="message-wrapper assistant"
              >
                <div className="message-content">
                  <div className="message-avatar">
                    <Bot className="w-4 h-4" />
                  </div>
                  <div className="typing-indicator">
                    <div className="typing-dots">
                      <div></div>
                      <div></div>
                      <div></div>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}

            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input Area */}
        <div className="input-container">
          <div className="input-wrapper">
            <div className="input-field">
              <Input
                value={input}
                onChange={(e: ChangeEvent<HTMLInputElement>) => setInput(e.target.value)}
                onKeyDown={handleKeyPress}
                placeholder="Message MechKee.ai..."
                className="modern-input"
                disabled={isLoading}
              />
              <Button
                onClick={handleSendMessage}
                disabled={!input.trim() || isLoading}
                className="send-btn"
                size="sm"
              >
                {isLoading ? (
                  <RefreshCw className="w-4 h-4 animate-spin" />
                ) : (
                  <Send className="w-4 h-4" />
                )}
              </Button>
            </div>
            <div className="input-footer">
              <p className="text-xs text-gray-500">
                MechKee.ai can make mistakes. Check important info.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Document Uploader Modal */}
      <DocumentUploader
        isOpen={showUploader}
        onClose={() => setShowUploader(false)}
        onUploadSuccess={() => {}}
      />
    </div>
  );
};

export default ModernChatInterface;