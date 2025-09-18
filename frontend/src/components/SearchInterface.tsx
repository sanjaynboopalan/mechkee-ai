import React, { useState, useCallback, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, Mic, Camera, FileText, Code, Image, Video, Filter, X, Loader2 } from 'lucide-react';
import Button from './ui/Button';
import Input from './ui/Input';
import Card from './ui/Card';
import { apiService, BackendSearchResponse, SearchResult } from '../services/api';
import toast from 'react-hot-toast';

interface SearchInterfaceProps {
  onSearch?: (query: string, filters?: SearchFilters) => Promise<any[]>;
}

interface SearchFilters {
  contentType?: string[];
  dateRange?: 'day' | 'week' | 'month' | 'year' | 'all';
  sortBy?: 'relevance' | 'date' | 'title';
}

// Local interface for display results
interface DisplayResult {
  id: string;
  title: string;
  content: string;
  score: number;
  content_type: 'text' | 'code' | 'image' | 'video' | 'audio' | 'structured_data';
  url?: string;
  metadata?: {
    author?: string;
    date?: string;
    domain?: string;
    tags?: string[];
  };
}

const SearchInterface: React.FC<SearchInterfaceProps> = ({ onSearch }) => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<DisplayResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [showFilters, setShowFilters] = useState(false);
  const [filters, setFilters] = useState<SearchFilters>({
    contentType: [],
    dateRange: 'all',
    sortBy: 'relevance',
  });
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  
  const searchInputRef = useRef<HTMLInputElement>(null);

  // Mock suggestions for demo
  const mockSuggestions = [
    'machine learning algorithms',
    'neural networks',
    'data visualization',
    'web development',
    'artificial intelligence',
    'deep learning models',
    'computer vision',
    'natural language processing',
  ];

  const contentTypes = [
    { id: 'text', label: 'Text', icon: FileText },
    { id: 'code', label: 'Code', icon: Code },
    { id: 'image', label: 'Images', icon: Image },
    { id: 'video', label: 'Videos', icon: Video },
  ];

  const handleSearch = useCallback(async (searchQuery: string = query) => {
    if (!searchQuery.trim()) return;

    setLoading(true);
    setShowSuggestions(false);
    
    try {
      // Use the new backend-compatible search API
      const response: BackendSearchResponse = await apiService.search(searchQuery, {
        max_results: 20,
        include_citations: true,
        search_type: 'hybrid'
      });
      
      // Transform backend response to display format
      const transformedResults = response.sources.map((source: { url: string; title: string; content: string; relevance_score: number; domain: string; publish_date: string; author?: string }, index: number) => ({
        id: `${index}`,
        title: source.title,
        content: source.content,
        score: source.relevance_score,
        content_type: 'text' as const,
        url: source.url,
        metadata: {
          author: source.author || '',
          date: source.publish_date,
          domain: source.domain
        }
      }));
      
      setResults(transformedResults);
      
      if (transformedResults.length === 0) {
        toast.error('No results found. Try different search terms.');
      } else {
        toast.success(`Found ${transformedResults.length} results`);
      }
      
      // Also show the AI-generated answer if available
      if (response.answer && !response.answer.includes('⚠️') && !response.answer.includes('unable to generate')) {
        toast.success('AI Answer: ' + response.answer.substring(0, 100) + '...', {
          duration: 5000
        });
      } else if (response.answer.includes('⚠️')) {
        toast.error(response.answer, { duration: 8000 });
      }
      
    } catch (error) {
      console.error('Search error:', error);
      toast.error('Search failed. Please try again.');
      
      // Fallback to mock data for demo
      const mockResults: DisplayResult[] = [
        {
          id: '1',
          title: 'Advanced Machine Learning Techniques',
          content: 'Comprehensive guide to advanced machine learning algorithms including neural networks, deep learning, and ensemble methods...',
          score: 0.95,
          content_type: 'text' as const,
          metadata: {
            author: 'Dr. Smith',
            date: '2024-01-15',
            tags: ['ML', 'AI', 'Neural Networks'],
          },
        },
        {
          id: '2',
          title: 'Python Implementation of HNSW Algorithm',
          content: 'def hnsw_search(query_vector, graph, ef=50):\\n    # Hierarchical Navigable Small World search\\n    candidates = []\\n    visited = set()',
          score: 0.89,
          content_type: 'code' as const,
          metadata: {
            author: 'CodeMaster',
            date: '2024-01-10',
            tags: ['Python', 'HNSW', 'Vector Search'],
          },
        },
        {
          id: '3',
          title: 'Deep Learning Visualization',
          content: 'Interactive visualization of neural network architectures and training processes...',
          score: 0.82,
          content_type: 'image' as const,
          metadata: {
            author: 'VisualAI',
            date: '2024-01-08',
            tags: ['Visualization', 'Deep Learning'],
          },
        },
      ].filter(result => {
        if (filters.contentType && filters.contentType.length > 0) {
          return filters.contentType.includes(result.content_type);
        }
        return true;
      });

      setResults(mockResults);
    } finally {
      setLoading(false);
    }
  }, [query, filters, onSearch]);

  const handleQueryChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setQuery(value);
    
    // Show suggestions when typing
    if (value.length > 2) {
      const filtered = mockSuggestions.filter(suggestion =>
        suggestion.toLowerCase().includes(value.toLowerCase())
      );
      setSuggestions(filtered.slice(0, 5));
      setShowSuggestions(true);
    } else {
      setShowSuggestions(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
    if (e.key === 'Escape') {
      setShowSuggestions(false);
    }
  };

  const toggleContentTypeFilter = (contentType: string) => {
    setFilters(prev => ({
      ...prev,
      contentType: prev.contentType?.includes(contentType)
        ? prev.contentType.filter(type => type !== contentType)
        : [...(prev.contentType || []), contentType],
    }));
  };

  const getContentTypeIcon = (type: string) => {
    const contentType = contentTypes.find(ct => ct.id === type);
    return contentType?.icon || FileText;
  };

  const renderSearchResult = (result: DisplayResult) => {
    const Icon = getContentTypeIcon(result.content_type);
    
    return (
      <Card
        key={result.id}
        variant="cosmic"
        hover
        className="search-result"
      >
        <div className="flex items-start space-x-4">
          <div className="flex-shrink-0">
            <div className="w-8 h-8 bg-gradient-to-r from-space-blue-500 to-cosmic-purple-500 rounded-lg flex items-center justify-center">
              <Icon className="w-4 h-4 text-white" />
            </div>
          </div>
          
          <div className="flex-1 min-w-0">
            <h3 className="text-lg font-semibold text-white mb-2">
              {result.title}
            </h3>
            
            <p className="text-gray-300 text-sm mb-3 line-clamp-3">
              {result.content}
            </p>
            
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4 text-xs text-gray-400">
                {result.metadata?.author && (
                  <span>By {result.metadata.author}</span>
                )}
                {result.metadata?.date && (
                  <span>{new Date(result.metadata.date).toLocaleDateString()}</span>
                )}
                <span className="px-2 py-1 bg-space-blue-500/20 rounded text-space-blue-400">
                  {Math.round(result.score * 100)}% match
                </span>
              </div>
              
              {result.metadata?.tags && (
                <div className="flex space-x-1">
                  {result.metadata.tags.slice(0, 3).map(tag => (
                    <span
                      key={tag}
                      className="px-2 py-1 bg-cosmic-purple-500/20 text-cosmic-purple-400 text-xs rounded"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </Card>
    );
  };

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-6">
      {/* Search Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center space-y-4"
      >
        <h1 className="text-4xl font-bold gradient-text">
          Cosmic Search Engine
        </h1>
        <p className="text-gray-400 text-lg">
          Discover knowledge across the universe with advanced AI-powered search
        </p>
      </motion.div>

      {/* Search Bar */}
      <Card variant="cosmic" size="lg" className="relative">
        <div className="flex items-center space-x-4">
          <div className="flex-1 relative">
            <Input
              type="search"
              placeholder="Search across documents, code, images, and more..."
              value={query}
              onChange={handleQueryChange}
              onKeyDown={handleKeyDown}
              icon={Search}
              size="lg"
              variant="transparent"
              className="text-lg"
              autoFocus
            />
            
            {/* Search Suggestions */}
            <AnimatePresence>
              {showSuggestions && suggestions.length > 0 && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 10 }}
                  className="absolute top-full left-0 right-0 mt-2 glass-morphism border border-white/20 rounded-lg overflow-hidden z-50"
                >
                  {suggestions.map((suggestion, index) => (
                    <motion.div
                      key={suggestion}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.05 }}
                      className="px-4 py-3 text-gray-300 hover:bg-white/10 cursor-pointer transition-colors"
                      onClick={() => {
                        setQuery(suggestion);
                        setShowSuggestions(false);
                        handleSearch(suggestion);
                      }}
                    >
                      <Search className="w-4 h-4 inline mr-2 text-space-blue-400" />
                      {suggestion}
                    </motion.div>
                  ))}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
          
          <div className="flex items-center space-x-2">
            <Button
              variant="ghost"
              size="lg"
              icon={Mic}
              onClick={() => console.log('Voice search')}
            >
              Voice
            </Button>
            
            <Button
              variant="ghost"
              size="lg"
              icon={Camera}
              onClick={() => console.log('Visual search')}
            >
              Visual
            </Button>
            
            <Button
              variant={showFilters ? 'cosmic' : 'ghost'}
              size="lg"
              icon={Filter}
              onClick={() => setShowFilters(!showFilters)}
            >
              Filters
            </Button>
            
            <Button
              variant="primary"
              size="lg"
              onClick={() => handleSearch()}
              loading={loading}
              glow
            >
              Search
            </Button>
          </div>
        </div>

        {/* Search Filters */}
        <AnimatePresence>
          {showFilters && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="mt-6 pt-6 border-t border-white/10"
            >
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Content Type
                  </label>
                  <div className="flex flex-wrap gap-2">
                    {contentTypes.map(({ id, label, icon: Icon }) => (
                      <Button
                        key={id}
                        variant={filters.contentType?.includes(id) ? 'cosmic' : 'outline'}
                        size="sm"
                        icon={Icon}
                        onClick={() => toggleContentTypeFilter(id)}
                      >
                        {label}
                      </Button>
                    ))}
                  </div>
                </div>

                <div className="flex items-center space-x-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Date Range
                    </label>
                    <select
                      value={filters.dateRange}
                      onChange={(e) => setFilters(prev => ({ ...prev, dateRange: e.target.value as any }))}
                      className="glass-morphism bg-white/5 border border-white/20 text-white rounded-lg px-3 py-2"
                    >
                      <option value="all">All time</option>
                      <option value="day">Past day</option>
                      <option value="week">Past week</option>
                      <option value="month">Past month</option>
                      <option value="year">Past year</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Sort by
                    </label>
                    <select
                      value={filters.sortBy}
                      onChange={(e) => setFilters(prev => ({ ...prev, sortBy: e.target.value as any }))}
                      className="glass-morphism bg-white/5 border border-white/20 text-white rounded-lg px-3 py-2"
                    >
                      <option value="relevance">Relevance</option>
                      <option value="date">Date</option>
                      <option value="title">Title</option>
                    </select>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </Card>

      {/* Search Results */}
      <AnimatePresence>
        {loading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="flex items-center justify-center py-12"
          >
            <div className="flex items-center space-x-3 text-space-blue-400">
              <Loader2 className="w-6 h-6 animate-spin" />
              <span className="text-lg">Searching the cosmos...</span>
            </div>
          </motion.div>
        )}

        {!loading && results.length > 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="space-y-4"
          >
            <div className="flex items-center justify-between">
              <p className="text-gray-400">
                Found {results.length} results for "{query}"
              </p>
              
              {(filters.contentType && filters.contentType.length > 0) && (
                <div className="flex items-center space-x-2">
                  {filters.contentType.map(type => (
                    <span
                      key={type}
                      className="px-2 py-1 bg-space-blue-500/20 text-space-blue-400 text-sm rounded flex items-center space-x-1"
                    >
                      <span>{type}</span>
                      <X
                        className="w-3 h-3 cursor-pointer hover:text-white"
                        onClick={() => toggleContentTypeFilter(type)}
                      />
                    </span>
                  ))}
                </div>
              )}
            </div>

            <div className="space-y-4">
              {results.map((result, index) => (
                <motion.div
                  key={result.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                >
                  {renderSearchResult(result)}
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}

        {!loading && query && results.length === 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-center py-12"
          >
            <div className="text-gray-400 space-y-2">
              <p className="text-lg">No results found for "{query}"</p>
              <p className="text-sm">Try adjusting your search terms or filters</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default SearchInterface;