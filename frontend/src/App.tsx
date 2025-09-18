import React, { useState } from 'react';
import { Toaster } from 'react-hot-toast';
import ModernChatInterface from './components/ModernChatInterface';
import AdvancedAIInterface from './components/AdvancedAIInterface';
import { Brain, MessageCircle, Settings, Search } from 'lucide-react';
import './styles/ModernChat.css';

function App() {
  const [activeTab, setActiveTab] = useState<'perplexity' | 'chat' | 'advanced' | 'settings'>('perplexity');

  return (
    <div className="app-container min-h-screen bg-gray-50">
      {/* Navigation */}
      <nav className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center">
              <Brain className="h-8 w-8 text-blue-600 mr-3" />
              <h1 className="text-xl font-bold text-gray-900">BlueMech AI</h1>
            </div>
            
            <div className="flex space-x-1">
              {[
                { id: 'perplexity', label: 'AI Search', icon: Search },
                { id: 'chat', label: 'Chat', icon: MessageCircle },
                { id: 'advanced', label: 'Advanced AI', icon: Brain },
                { id: 'settings', label: 'Settings', icon: Settings }
              ].map(({ id, label, icon: Icon }) => (
                <button
                  key={id}
                  onClick={() => setActiveTab(id as any)}
                  className={`flex items-center space-x-2 px-4 py-2 rounded-md transition-all ${
                    activeTab === id
                      ? 'bg-blue-100 text-blue-700 font-medium'
                      : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{label}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="flex-1">
        {activeTab === 'perplexity' && <ModernChatInterface />}
        {activeTab === 'chat' && <ModernChatInterface />}
        {activeTab === 'advanced' && (
          <div className="container mx-auto py-6">
            <AdvancedAIInterface />
          </div>
        )}
        {activeTab === 'settings' && (
          <div className="container mx-auto py-6">
            <div className="max-w-4xl mx-auto p-6 bg-white rounded-lg shadow-lg">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">Settings</h2>
              <p className="text-gray-600">Settings panel will be implemented here.</p>
            </div>
          </div>
        )}
      </main>
      
      {/* Toast Notifications */}
      <Toaster
        position="top-right"
        toastOptions={{
          style: {
            background: '#374151',
            color: '#f9fafb',
            borderRadius: '8px',
            padding: '12px 16px',
          },
          success: {
            iconTheme: {
              primary: '#10a37f',
              secondary: '#ffffff'
            }
          },
          error: {
            iconTheme: {
              primary: '#ef4444',
              secondary: '#ffffff'
            }
          }
        }}
      />
    </div>
  );
}

export default App;