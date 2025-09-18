import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Send, 
  User, 
  Bot, 
  Copy, 
  Trash2, 
  Upload, 
  Database,
  Settings,
  MessageSquare,
  Sparkles,
  FileText,
  MoreVertical,
  Download,
  Share,
  RefreshCw,
  ExternalLink
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
}

interface ChatMode {
  id: 'standard' | 'enhanced';
  name: string;
  description: string;
  icon: React.ReactNode;
}

const ChatInterface: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'assistant',
      content: '� **Welcome to your Professional AI Assistant!**\n\nI\'m here to provide intelligent, context-aware assistance with:\n\n✨ **Expert Analysis** - Deep insights and comprehensive responses\n📚 **Document Intelligence** - Smart search through your uploaded files\n🎯 **Precise Citations** - Verified sources and references\n💡 **Strategic Thinking** - Complex problem-solving and planning\n\n*Ready to transform your productivity? Let\'s get started!*',
      timestamp: new Date().toISOString(),
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string | undefined>(undefined);
  const [chatMode, setChatMode] = useState<'standard' | 'enhanced'>('standard');
  const [showUploader, setShowUploader] = useState(false);
  const [knowledgeBaseStats, setKnowledgeBaseStats] = useState<any>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const chatModes: ChatMode[] = [
    {
      id: 'standard',
      name: 'Standard Chat',
      description: 'Fast responses using AI training data',
      icon: <Bot className="w-4 h-4" />
    },
    {
      id: 'enhanced',
      name: 'Enhanced RAG',
      description: 'Context-aware responses with document retrieval',
      icon: <FileText className="w-4 h-4" />
    }
  ];

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Load knowledge base stats when enhanced mode is selected
  useEffect(() => {
    if (chatMode === 'enhanced') {
      loadKnowledgeBaseStats();
    }
  }, [chatMode]);

  const loadKnowledgeBaseStats = async () => {
    try {
      const stats = await enhancedAPI.getKnowledgeBaseStats();
      setKnowledgeBaseStats(stats);
    } catch (error) {
      console.error('Failed to load knowledge base stats:', error);
    }
  };

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
      
      console.log('Sending message:', input, 'in mode:', chatMode);
      
      if (chatMode === 'enhanced') {
        // Use enhanced RAG chat
        console.log('Using enhanced API...');
        const response = await enhancedAPI.sendEnhancedMessage({
          message: input,
          session_id: sessionId,
          use_rag: true,
        });

        console.log('Enhanced API response:', response);

        // Update session ID if this is the first message
        if (!sessionId) {
          setSessionId(response.session_id);
        }

        assistantMessage = {
          role: 'assistant',
          content: response.response,
          timestamp: new Date().toISOString(),
          sources: response.sources,
          response_type: response.response_type
        };
      } else {
        // Use standard chat
        console.log('Using standard API...');
        const response = await chatAPI.sendMessage({
          message: input,
          session_id: sessionId,
        });

        console.log('Standard API response:', response);

        // Update session ID if this is the first message
        if (!sessionId) {
          setSessionId(response.session_id);
        }

        assistantMessage = {
          role: 'assistant',
          content: response.response,
          timestamp: new Date().toISOString(),
        };
      }

      console.log('Adding assistant message:', assistantMessage);
      setMessages(prev => [...prev, assistantMessage]);

    } catch (error) {
      console.error('Chat error:', error);
      toast.error('Failed to send message. Please try again.');
      
      // Remove the user message on error
      setMessages(prev => prev.slice(0, -1));
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

  const clearConversation = async () => {
    if (sessionId) {
      try {
        if (chatMode === 'enhanced') {
          await enhancedAPI.clearEnhancedConversation(sessionId);
        } else {
          await chatAPI.clearConversation(sessionId);
        }
        toast.success('Conversation cleared');
      } catch (error) {
        console.error('Clear conversation error:', error);
      }
    }
    
    setMessages([
      {
        role: 'assistant',
        content: 'Hello! I\'m your enhanced AI assistant. How can I assist you today?',
        timestamp: new Date().toISOString(),
      }
    ]);
    setSessionId(undefined);
  };

  const copyMessage = (content: string) => {
    navigator.clipboard.writeText(content);
    toast.success('Message copied to clipboard');
  };

  return (
    <div className="flex flex-col h-screen max-w-7xl mx-auto p-6">
      {/* Enhanced Header */}
      <div className="corporate-card mb-6 p-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-6">
            <h2 className="text-2xl font-bold text-corporate-gradient">🤖 Professional AI Assistant</h2>
            <div className="hidden sm:flex items-center space-x-3">
              {chatModes.map((mode) => (
                <button
                  key={mode.id}
                  onClick={() => setChatMode(mode.id)}
                  className={`flex items-center space-x-3 px-6 py-3 rounded-xl text-sm font-semibold transition-all ${
                    chatMode === mode.id
                      ? 'corporate-button text-white'
                      : 'bg-slate-100 text-slate-600 hover:bg-slate-200 border border-slate-300'
                  }`}
                >
                  {mode.icon}
                  <span className="hidden md:inline">{mode.name}</span>
                </button>
              ))}
            </div>
          </div>
          <div className="flex items-center space-x-4">
            <span className="text-base text-slate-600 font-medium px-4 py-2 bg-slate-50 rounded-full">
              {messages.length > 1 ? `${Math.floor((messages.length - 1) / 2)} conversations` : 'New session'}
            </span>
            {chatMode === 'enhanced' && (
              <>
                <Button
                  onClick={() => setShowUploader(true)}
                  className="corporate-button-outline text-sm px-4 py-2"
                  variant="ghost"
                  size="sm"
                >
                  <Upload className="w-5 h-5 mr-2" />
                  Upload
                </Button>
                {knowledgeBaseStats && (
                  <div className="hidden sm:flex items-center space-x-2 text-sm text-slate-600 bg-slate-100 px-4 py-2 rounded-full border">
                    <Database className="w-5 h-5 text-blue-500" />
                    <span className="font-medium">{knowledgeBaseStats.total_documents} documents</span>
                  </div>
                )}
              </>
            )}
            <Button
              variant="ghost"
              size="sm"
              onClick={clearConversation}
              className="text-red-500 hover:text-red-600 hover:bg-red-50 px-4 py-2 rounded-xl transition-all"
            >
              <Trash2 className="w-5 h-5" />
            </Button>
          </div>
        </div>
        
        {/* Mode description for mobile */}
        <div className="sm:hidden mt-4 p-4 bg-slate-50 rounded-xl border border-slate-200">
          <p className="text-sm text-slate-600 font-medium">
            {chatModes.find(m => m.id === chatMode)?.description}
          </p>
        </div>
      </div>

      {/* Messages Container */}
      <div className="corporate-card flex-1 overflow-hidden p-0 mb-6">
        <div className="h-full overflow-y-auto px-8 py-6 space-y-6 bg-gradient-to-b from-slate-50/30 to-transparent">
          <AnimatePresence>
            {messages.map((message, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.4, ease: "easeOut" }}
                className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div className={`flex items-start space-x-4 max-w-4xl ${message.role === 'user' ? 'flex-row-reverse space-x-reverse' : ''}`}>
                  {/* Avatar */}
                  <div className={message.role === 'user' ? 'avatar-user' : 'avatar-assistant'}>
                    {message.role === 'user' ? (
                      <User className="w-5 h-5" />
                    ) : (
                      <Bot className="w-6 h-6" />
                    )}
                  </div>

                  {/* Message Content */}
                  <div className="flex-1">
                    <div className={message.role === 'user' ? 'message-bubble-user' : 'message-bubble-assistant'}>
                      <div className="whitespace-pre-wrap break-words leading-relaxed text-base">
                        {message.content}
                      </div>
                      
                      {/* Sources */}
                      {message.sources && message.sources.length > 0 && (
                        <div className="mt-6 pt-6 border-t border-slate-200">
                          <div className="text-base text-slate-600 mb-4 flex items-center space-x-3 font-semibold">
                            <ExternalLink className="w-5 h-5 text-blue-500" />
                            <span>Sources ({message.sources.length})</span>
                          </div>
                          <div className="space-y-4">
                            {message.sources.map((source, idx) => (
                              <div key={idx} className="bg-slate-50 rounded-2xl p-6 border border-slate-200 hover:border-blue-300 transition-all">
                                <div className="font-bold text-lg text-slate-800 mb-3">{source.title}</div>
                                <div className="text-slate-600 text-base mb-3">{source.source}</div>
                                {source.snippet && (
                                  <div className="text-slate-700 text-base italic bg-white p-4 rounded-xl border border-slate-100">
                                    "{source.snippet}"
                                  </div>
                                )}
                                <div className="flex justify-between items-center mt-4">
                                  <span className="text-base text-slate-500 bg-slate-100 px-4 py-2 rounded-full font-semibold">
                                    {Math.round(source.relevance_score * 100)}% relevant
                                  </span>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                      
                      {/* Action Button */}
                      <div className="flex items-center justify-end mt-4">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => copyMessage(message.content)}
                          className="opacity-60 hover:opacity-100 p-3 hover:bg-slate-100 rounded-xl transition-all"
                        >
                          <Copy className="w-5 h-5" />
                        </Button>
                      </div>
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
              className="flex justify-start"
            >
              <div className="flex items-start space-x-4 max-w-4xl">
                <div className="avatar-assistant">
                  <Bot className="w-6 h-6" />
                </div>
                <div className="typing-indicator bg-white border border-slate-200 shadow-sm px-6 py-4">
                  <div className="typing-dot" />
                  <div className="typing-dot" />
                  <div className="typing-dot" />
                </div>
              </div>
            </motion.div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <div className="corporate-card p-6">
        <div className="flex items-end space-x-6">
          <div className="flex-1">
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyPress}
              placeholder={`💭 ${chatMode === 'enhanced' ? 'Ask me anything with document context...' : 'What can I help you with today?'}`}
              className="w-full input-corporate text-lg py-4 px-6"
              disabled={isLoading}
            />
          </div>
          <Button
            onClick={handleSendMessage}
            disabled={!input.trim() || isLoading}
            className="corporate-button shrink-0 px-8 py-4 text-lg"
          >
            <Send className="w-6 h-6" />
          </Button>
        </div>
        
        {/* Status Bar */}
        <div className="flex items-center justify-between mt-6 pt-4 border-t border-slate-200">
          <div className="flex items-center space-x-4">
            <div className={`w-4 h-4 rounded-full ${chatMode === 'enhanced' ? 'bg-green-400' : 'bg-blue-400'} animate-pulse`} />
            <span className="text-base text-slate-600 font-semibold">
              {chatModes.find(m => m.id === chatMode)?.description}
            </span>
          </div>
          <div className="text-base text-slate-500 font-medium">Press Enter to send</div>
        </div>
      </div>

      {/* Document Uploader Modal */}
      <DocumentUploader
        isOpen={showUploader}
        onClose={() => setShowUploader(false)}
        onUploadSuccess={loadKnowledgeBaseStats}
      />
    </div>
  );
};

export default ChatInterface;