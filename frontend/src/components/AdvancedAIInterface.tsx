import React, { useState, useEffect, ChangeEvent } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Brain, 
  Target, 
  Shield, 
  User, 
  TrendingUp, 
  CheckCircle, 
  AlertTriangle,
  Info,
  Settings,
  Lock,
  Eye,
  BarChart3,
  Lightbulb,
  RefreshCw
} from 'lucide-react';
import Button from './ui/Button';
import Input from './ui/Input';

// Types for advanced AI features
interface Option {
  name: string;
  description: string;
  features: Record<string, any>;
  cost?: number;
  risk_level?: number;
  expected_value?: number;
}

interface DecisionRequest {
  user_id: string;
  decision_title: string;
  options: Option[];
  criteria: Record<string, number>;
  run_simulation: boolean;
  simulation_runs: number;
  context?: string;
}

interface TruthfulnessRequest {
  user_id: string;
  claim: string;
  sources: Array<Record<string, string>>;
  context?: string;
  confidence_threshold: number;
}

interface PersonalizationRequest {
  user_id: string;
  interaction_type: string;
  content: string;
  feedback_score?: number;
  preferences?: Record<string, any>;
  update_profile: boolean;
}

interface AdvancedAIResponse {
  decision?: {
    decision_id: string;
    recommended_option: string;
    confidence_score: number;
    reasoning_path: string[];
    ranked_options: Array<{
      name: string;
      score: number;
      rank: number;
      strengths: string[];
      weaknesses: string[];
    }>;
    simulation_results?: any;
    risk_analysis: Record<string, any>;
  };
  truthfulness?: {
    analysis_id: string;
    truthfulness_score: number;
    confidence_level: number;
    bias_indicators: string[];
    cross_check_results: Array<{
      source: string;
      agreement: number;
      evidence: string[];
      conflicts: string[];
    }>;
    reasoning_path: string[];
    recommendations: string[];
  };
  personalization?: {
    user_id: string;
    personalized_content: string;
    adaptation_score: number;
    learned_preferences: Record<string, any>;
    recommendations: string[];
    privacy_status: string;
  };
}

const AdvancedAIInterface: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'decision' | 'truthfulness' | 'personalization'>('decision');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AdvancedAIResponse | null>(null);
  const [userId] = useState(() => `user_${Date.now()}`);
  
  // Decision Support State
  const [decisionTitle, setDecisionTitle] = useState('');
  const [options, setOptions] = useState<Option[]>([]);
  const [criteria, setCriteria] = useState({
    cost: 0.3,
    value: 0.4,
    risk: 0.2,
    alignment: 0.1
  });
  
  // Truthfulness Check State
  const [claim, setClaim] = useState('');
  const [sources, setSources] = useState<Array<Record<string, string>>>([]);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.7);
  
  // Personalization State
  const [interactionType, setInteractionType] = useState('general');
  const [content, setContent] = useState('');
  const [feedbackScore, setFeedbackScore] = useState<number | undefined>();

  const addOption = () => {
    setOptions([...options, {
      name: '',
      description: '',
      features: {},
      cost: 0,
      risk_level: 0,
      expected_value: 0
    }]);
  };

  const updateOption = (index: number, field: keyof Option, value: any) => {
    const newOptions = [...options];
    newOptions[index] = { ...newOptions[index], [field]: value };
    setOptions(newOptions);
  };

  const removeOption = (index: number) => {
    setOptions(options.filter((_, i) => i !== index));
  };

  const addSource = () => {
    setSources([...sources, { content: '', type: 'web', url: '', credibility: '0.5' }]);
  };

  const updateSource = (index: number, field: string, value: string) => {
    const newSources = [...sources];
    newSources[index] = { ...newSources[index], [field]: value };
    setSources(newSources);
  };

  const removeSource = (index: number) => {
    setSources(sources.filter((_, i) => i !== index));
  };

  const handleDecisionSupport = async () => {
    if (!decisionTitle || options.length === 0) {
      alert('Please provide a decision title and at least one option');
      return;
    }

    setLoading(true);
    try {
      const request: DecisionRequest = {
        user_id: userId,
        decision_title: decisionTitle,
        options,
        criteria,
        run_simulation: true,
        simulation_runs: 1000,
        context: `Decision about: ${decisionTitle}`
      };

      const response = await fetch('/api/v1/advanced/decision-support', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request)
      });

      if (!response.ok) throw new Error('Decision support failed');
      
      const data = await response.json();
      setResult({ decision: data });
    } catch (error) {
      console.error('Decision support error:', error);
      alert('Failed to get decision support');
    } finally {
      setLoading(false);
    }
  };

  const handleTruthfulnessCheck = async () => {
    if (!claim) {
      alert('Please provide a claim to check');
      return;
    }

    setLoading(true);
    try {
      const request: TruthfulnessRequest = {
        user_id: userId,
        claim,
        sources,
        confidence_threshold: confidenceThreshold
      };

      const response = await fetch('/api/v1/advanced/truthfulness-check', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request)
      });

      if (!response.ok) throw new Error('Truthfulness check failed');
      
      const data = await response.json();
      setResult({ truthfulness: data });
    } catch (error) {
      console.error('Truthfulness check error:', error);
      alert('Failed to check truthfulness');
    } finally {
      setLoading(false);
    }
  };

  const handlePersonalization = async () => {
    if (!content) {
      alert('Please provide content for personalization');
      return;
    }

    setLoading(true);
    try {
      const request: PersonalizationRequest = {
        user_id: userId,
        interaction_type: interactionType,
        content,
        feedback_score: feedbackScore,
        update_profile: true
      };

      const response = await fetch('/api/v1/advanced/personalized-response', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request)
      });

      if (!response.ok) throw new Error('Personalization failed');
      
      const data = await response.json();
      setResult({ personalization: data });
    } catch (error) {
      console.error('Personalization error:', error);
      alert('Failed to get personalized response');
    } finally {
      setLoading(false);
    }
  };

  const ConfidenceIndicator: React.FC<{ score: number; label: string }> = ({ score, label }) => (
    <div className="flex items-center space-x-2">
      <span className="text-sm font-medium text-gray-700">{label}:</span>
      <div className="flex-1 bg-gray-200 rounded-full h-2 max-w-32">
        <div 
          className={`h-2 rounded-full transition-all duration-300 ${
            score >= 0.8 ? 'bg-green-500' : 
            score >= 0.6 ? 'bg-yellow-500' : 
            'bg-red-500'
          }`}
          style={{ width: `${score * 100}%` }}
        />
      </div>
      <span className="text-sm font-bold">{(score * 100).toFixed(0)}%</span>
    </div>
  );

  return (
    <div className="max-w-6xl mx-auto p-6 bg-white rounded-lg shadow-lg">
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900 mb-2 flex items-center">
          <Brain className="mr-3 text-blue-600" />
          Advanced AI Features
        </h1>
        <p className="text-gray-600">
          Intelligent decision-making, truthfulness verification, and personalized responses
        </p>
      </div>

      {/* Tab Navigation */}
      <div className="flex space-x-1 bg-gray-100 p-1 rounded-lg mb-6">
        {[
          { id: 'decision', label: 'Decision Support', icon: Target },
          { id: 'truthfulness', label: 'Truthfulness Check', icon: Shield },
          { id: 'personalization', label: 'Personalization', icon: User }
        ].map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setActiveTab(id as any)}
            className={`flex items-center space-x-2 px-4 py-2 rounded-md flex-1 transition-all ${
              activeTab === id
                ? 'bg-white shadow-sm text-blue-600 font-medium'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            <Icon className="w-4 h-4" />
            <span>{label}</span>
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Input Panel */}
        <div className="space-y-6">
          {activeTab === 'decision' && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-4"
            >
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Decision Title
                </label>
                <Input
                  value={decisionTitle}
                  onChange={(e: ChangeEvent<HTMLInputElement>) => setDecisionTitle(e.target.value)}
                  placeholder="What decision are you trying to make?"
                  className="w-full"
                />
              </div>

              <div>
                <div className="flex justify-between items-center mb-2">
                  <label className="block text-sm font-medium text-gray-700">
                    Options
                  </label>
                  <Button onClick={addOption} size="sm" variant="outline">
                    Add Option
                  </Button>
                </div>
                
                {options.map((option, index) => (
                  <div key={index} className="border rounded-lg p-4 mb-4 bg-gray-50">
                    <div className="flex justify-between items-start mb-2">
                      <h4 className="font-medium">Option {index + 1}</h4>
                      <button 
                        onClick={() => removeOption(index)}
                        className="text-red-500 hover:text-red-700"
                      >
                        ×
                      </button>
                    </div>
                    
                    <div className="grid grid-cols-1 gap-2">
                      <Input
                        value={option.name}
                        onChange={(e: ChangeEvent<HTMLInputElement>) => updateOption(index, 'name', e.target.value)}
                        placeholder="Option name"
                        size="sm"
                      />
                      <Input
                        value={option.description}
                        onChange={(e: ChangeEvent<HTMLInputElement>) => updateOption(index, 'description', e.target.value)}
                        placeholder="Description"
                        size="sm"
                      />
                      <div className="grid grid-cols-3 gap-2">
                        <Input
                          type="number"
                          value={option.cost?.toString() || ''}
                          onChange={(e: ChangeEvent<HTMLInputElement>) => updateOption(index, 'cost', parseFloat(e.target.value) || 0)}
                          placeholder="Cost"
                          size="sm"
                        />
                        <Input
                          type="number"
                          value={option.risk_level?.toString() || ''}
                          onChange={(e: ChangeEvent<HTMLInputElement>) => updateOption(index, 'risk_level', parseFloat(e.target.value) || 0)}
                          placeholder="Risk (0-1)"
                          size="sm"
                        />
                        <Input
                          type="number"
                          value={option.expected_value?.toString() || ''}
                          onChange={(e: ChangeEvent<HTMLInputElement>) => updateOption(index, 'expected_value', parseFloat(e.target.value) || 0)}
                          placeholder="Expected Value"
                          size="sm"
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Decision Criteria Weights
                </label>
                <div className="grid grid-cols-2 gap-2">
                  {Object.entries(criteria).map(([key, value]) => (
                    <div key={key} className="flex items-center space-x-2">
                      <label className="text-sm capitalize">{key}:</label>
                      <Input
                        type="number"
                        value={value.toString()}
                        onChange={(e: ChangeEvent<HTMLInputElement>) => setCriteria({
                          ...criteria,
                          [key]: parseFloat(e.target.value) || 0
                        })}
                        size="sm"
                        className="w-20"
                      />
                    </div>
                  ))}
                </div>
              </div>

              <Button 
                onClick={handleDecisionSupport}
                disabled={loading}
                className="w-full"
              >
                {loading ? <RefreshCw className="w-4 h-4 mr-2 animate-spin" /> : <Target className="w-4 h-4 mr-2" />}
                Analyze Decision
              </Button>
            </motion.div>
          )}

          {activeTab === 'truthfulness' && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-4"
            >
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Claim to Verify
                </label>
                <textarea
                  value={claim}
                  onChange={(e) => setClaim(e.target.value)}
                  placeholder="Enter the claim or statement you want to verify..."
                  className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  rows={4}
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Confidence Threshold
                </label>
                <Input
                  type="number"
                  value={confidenceThreshold.toString()}
                  onChange={(e: ChangeEvent<HTMLInputElement>) => setConfidenceThreshold(parseFloat(e.target.value) || 0.7)}
                  placeholder="0.7"
                />
              </div>

              <div>
                <div className="flex justify-between items-center mb-2">
                  <label className="block text-sm font-medium text-gray-700">
                    Sources (Optional)
                  </label>
                  <Button onClick={addSource} size="sm" variant="outline">
                    Add Source
                  </Button>
                </div>
                
                {sources.map((source, index) => (
                  <div key={index} className="border rounded-lg p-4 mb-4 bg-gray-50">
                    <div className="flex justify-between items-start mb-2">
                      <h4 className="font-medium">Source {index + 1}</h4>
                      <button 
                        onClick={() => removeSource(index)}
                        className="text-red-500 hover:text-red-700"
                      >
                        ×
                      </button>
                    </div>
                    
                    <div className="grid grid-cols-1 gap-2">
                      <Input
                        value={source.url || ''}
                        onChange={(e: ChangeEvent<HTMLInputElement>) => updateSource(index, 'url', e.target.value)}
                        placeholder="Source URL"
                        size="sm"
                      />
                      <Input
                        value={source.content || ''}
                        onChange={(e: ChangeEvent<HTMLInputElement>) => updateSource(index, 'content', e.target.value)}
                        placeholder="Source content/summary"
                        size="sm"
                      />
                      <Input
                        type="number"
                        value={source.credibility || ''}
                        onChange={(e: ChangeEvent<HTMLInputElement>) => updateSource(index, 'credibility', e.target.value)}
                        placeholder="Credibility (0-1)"
                        size="sm"
                      />
                    </div>
                  </div>
                ))}
              </div>

              <Button 
                onClick={handleTruthfulnessCheck}
                disabled={loading}
                className="w-full"
              >
                {loading ? <RefreshCw className="w-4 h-4 mr-2 animate-spin" /> : <Shield className="w-4 h-4 mr-2" />}
                Verify Truthfulness
              </Button>
            </motion.div>
          )}

          {activeTab === 'personalization' && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-4"
            >
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Interaction Type
                </label>
                <select
                  value={interactionType}
                  onChange={(e) => setInteractionType(e.target.value)}
                  className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
                >
                  <option value="general">General</option>
                  <option value="question">Question</option>
                  <option value="decision">Decision</option>
                  <option value="research">Research</option>
                  <option value="analysis">Analysis</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Content
                </label>
                <textarea
                  value={content}
                  onChange={(e) => setContent(e.target.value)}
                  placeholder="Enter your query or content for personalized response..."
                  className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  rows={4}
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Feedback Score (Optional)
                </label>
                <Input
                  type="number"
                  value={feedbackScore?.toString() || ''}
                  onChange={(e: ChangeEvent<HTMLInputElement>) => setFeedbackScore(e.target.value ? parseFloat(e.target.value) : undefined)}
                  placeholder="Rate previous response (-1 to 1)"
                />
              </div>

              <Button 
                onClick={handlePersonalization}
                disabled={loading}
                className="w-full"
              >
                {loading ? <RefreshCw className="w-4 h-4 mr-2 animate-spin" /> : <User className="w-4 h-4 mr-2" />}
                Get Personalized Response
              </Button>
            </motion.div>
          )}
        </div>

        {/* Results Panel */}
        <div className="space-y-6">
          {result && (
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="bg-gray-50 rounded-lg p-6"
            >
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <BarChart3 className="w-5 h-5 mr-2 text-blue-600" />
                Results
              </h3>

              {result.decision && (
                <div className="space-y-4">
                  <div className="bg-white rounded-lg p-4 border-l-4 border-blue-500">
                    <h4 className="font-semibold text-lg text-gray-900 mb-2">
                      Recommended: {result.decision.recommended_option}
                    </h4>
                    <ConfidenceIndicator 
                      score={result.decision.confidence_score} 
                      label="Confidence" 
                    />
                  </div>

                  <div className="bg-white rounded-lg p-4">
                    <h5 className="font-medium text-gray-900 mb-2">Ranking Results</h5>
                    <div className="space-y-2">
                      {result.decision.ranked_options.map((option, index) => (
                        <div key={index} className="flex justify-between items-center p-2 bg-gray-50 rounded">
                          <span className="font-medium">#{option.rank} {option.name}</span>
                          <span className="text-sm text-gray-600">{(option.score * 100).toFixed(1)}%</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="bg-white rounded-lg p-4">
                    <h5 className="font-medium text-gray-900 mb-2">Reasoning Path</h5>
                    <ul className="space-y-1 text-sm text-gray-700">
                      {result.decision.reasoning_path.map((step, index) => (
                        <li key={index} className="flex items-start">
                          <span className="w-4 h-4 bg-blue-100 text-blue-600 rounded-full text-xs flex items-center justify-center mr-2 mt-0.5">
                            {index + 1}
                          </span>
                          {step}
                        </li>
                      ))}
                    </ul>
                  </div>

                  {result.decision.risk_analysis && (
                    <div className="bg-white rounded-lg p-4">
                      <h5 className="font-medium text-gray-900 mb-2 flex items-center">
                        <AlertTriangle className="w-4 h-4 mr-1 text-orange-500" />
                        Risk Analysis
                      </h5>
                      <div className="text-sm text-gray-700">
                        <p><strong>Overall Risk:</strong> {result.decision.risk_analysis.overall_risk}</p>
                        {result.decision.risk_analysis.risk_factors && (
                          <div className="mt-2">
                            <strong>Risk Factors:</strong>
                            <ul className="list-disc list-inside ml-2">
                              {result.decision.risk_analysis.risk_factors.map((factor: string, index: number) => (
                                <li key={index}>{factor}</li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {result.truthfulness && (
                <div className="space-y-4">
                  <div className="bg-white rounded-lg p-4 border-l-4 border-green-500">
                    <h4 className="font-semibold text-lg text-gray-900 mb-2">
                      Truthfulness Analysis
                    </h4>
                    <div className="space-y-2">
                      <ConfidenceIndicator 
                        score={result.truthfulness.truthfulness_score} 
                        label="Truthfulness" 
                      />
                      <ConfidenceIndicator 
                        score={result.truthfulness.confidence_level} 
                        label="Confidence" 
                      />
                    </div>
                  </div>

                  {result.truthfulness.bias_indicators.length > 0 && (
                    <div className="bg-white rounded-lg p-4">
                      <h5 className="font-medium text-gray-900 mb-2 flex items-center">
                        <Eye className="w-4 h-4 mr-1 text-yellow-500" />
                        Bias Indicators
                      </h5>
                      <ul className="space-y-1 text-sm text-gray-700">
                        {result.truthfulness.bias_indicators.map((indicator, index) => (
                          <li key={index} className="flex items-center">
                            <AlertTriangle className="w-3 h-3 text-yellow-500 mr-2" />
                            {indicator}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  <div className="bg-white rounded-lg p-4">
                    <h5 className="font-medium text-gray-900 mb-2">Cross-Check Results</h5>
                    <div className="space-y-2">
                      {result.truthfulness.cross_check_results.map((check, index) => (
                        <div key={index} className="p-2 bg-gray-50 rounded">
                          <div className="flex justify-between items-center mb-1">
                            <span className="text-sm font-medium">{check.source}</span>
                            <span className={`text-xs px-2 py-1 rounded ${
                              check.agreement > 0.7 ? 'bg-green-100 text-green-800' :
                              check.agreement > 0.4 ? 'bg-yellow-100 text-yellow-800' :
                              'bg-red-100 text-red-800'
                            }`}>
                              {(check.agreement * 100).toFixed(0)}% agreement
                            </span>
                          </div>
                          {check.evidence.length > 0 && (
                            <div className="text-xs text-gray-600">
                              <strong>Evidence:</strong> {check.evidence.slice(0, 2).join(', ')}
                              {check.evidence.length > 2 && '...'}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="bg-white rounded-lg p-4">
                    <h5 className="font-medium text-gray-900 mb-2">Recommendations</h5>
                    <ul className="space-y-1 text-sm text-gray-700">
                      {result.truthfulness.recommendations.map((rec, index) => (
                        <li key={index} className="flex items-start">
                          <Lightbulb className="w-3 h-3 text-blue-500 mr-2 mt-1" />
                          {rec}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}

              {result.personalization && (
                <div className="space-y-4">
                  <div className="bg-white rounded-lg p-4 border-l-4 border-purple-500">
                    <h4 className="font-semibold text-lg text-gray-900 mb-2">
                      Personalized Response
                    </h4>
                    <p className="text-gray-700">{result.personalization.personalized_content}</p>
                  </div>

                  <div className="bg-white rounded-lg p-4">
                    <h5 className="font-medium text-gray-900 mb-2">Adaptation Score</h5>
                    <ConfidenceIndicator 
                      score={result.personalization.adaptation_score} 
                      label="Learning Progress" 
                    />
                  </div>

                  <div className="bg-white rounded-lg p-4">
                    <h5 className="font-medium text-gray-900 mb-2">Learned Preferences</h5>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      {Object.entries(result.personalization.learned_preferences).map(([key, value]) => (
                        <div key={key} className="flex justify-between p-2 bg-gray-50 rounded">
                          <span className="capitalize">{key.replace('_', ' ')}</span>
                          <span className="font-medium">
                            {typeof value === 'number' ? (value * 100).toFixed(0) + '%' : String(value)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="bg-white rounded-lg p-4">
                    <h5 className="font-medium text-gray-900 mb-2 flex items-center">
                      <Lock className="w-4 h-4 mr-1 text-green-500" />
                      Privacy Status: {result.personalization.privacy_status}
                    </h5>
                    <ul className="space-y-1 text-sm text-gray-700">
                      {result.personalization.recommendations.map((rec, index) => (
                        <li key={index} className="flex items-start">
                          <Lightbulb className="w-3 h-3 text-purple-500 mr-2 mt-1" />
                          {rec}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}
            </motion.div>
          )}

          {!result && !loading && (
            <div className="bg-gray-50 rounded-lg p-8 text-center">
              <Brain className="w-12 h-12 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">Ready for Advanced AI</h3>
              <p className="text-gray-600">
                Use the controls on the left to access advanced reasoning, truthfulness checking, and personalization features.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default AdvancedAIInterface;