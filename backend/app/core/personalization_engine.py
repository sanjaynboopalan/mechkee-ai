"""
Personalization Engine with Reinforcement Learning
Learns user preferences, style, and goals over time with privacy protection
"""

import asyncio
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import hashlib
import pickle
import base64
from cryptography.fernet import Fernet
from collections import defaultdict, deque
import uuid

logger = logging.getLogger(__name__)

class InteractionType(Enum):
    QUERY = "query"
    FEEDBACK = "feedback"
    PREFERENCE = "preference"
    GOAL_SETTING = "goal_setting"
    DECISION_MAKING = "decision_making"

class FeedbackType(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    VERY_POSITIVE = "very_positive"
    VERY_NEGATIVE = "very_negative"

class LearningMode(Enum):
    EXPLORATION = "exploration"
    EXPLOITATION = "exploitation"
    BALANCED = "balanced"

@dataclass
class UserInteraction:
    """Represents a user interaction for learning"""
    interaction_id: str
    user_id: str
    timestamp: datetime
    interaction_type: InteractionType
    content: str
    context: Dict[str, Any]
    feedback: Optional[FeedbackType] = None
    outcome_quality: Optional[float] = None  # 0-1 score
    
@dataclass
class UserPreference:
    """Represents a learned user preference"""
    preference_id: str
    category: str
    preference_type: str
    value: Any
    confidence: float
    learned_from: List[str]  # interaction IDs
    last_updated: datetime
    stability_score: float  # How stable this preference is
    
@dataclass
class UserGoal:
    """Represents a user goal"""
    goal_id: str
    description: str
    category: str
    priority: float
    target_date: Optional[datetime]
    progress: float  # 0-1
    sub_goals: List[str]
    success_metrics: Dict[str, Any]
    
@dataclass
class UserContext:
    """Current context about the user"""
    user_id: str
    current_goals: List[UserGoal]
    preferences: Dict[str, UserPreference]
    interaction_history: deque
    learning_style: str
    expertise_areas: Dict[str, float]
    communication_style: Dict[str, float]
    decision_patterns: Dict[str, Any]
    last_active: datetime
    
@dataclass
class PersonalizedRecommendation:
    """Personalized recommendation based on learned preferences"""
    recommendation_id: str
    user_id: str
    content: str
    reasoning: str
    confidence: float
    personalization_factors: Dict[str, float]
    alternatives: List[str]
    expected_satisfaction: float
    
class ReinforcementLearningAgent:
    """
    Reinforcement learning agent for personalization
    Uses Q-learning with experience replay
    """
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = 0.95
        self.epsilon = 0.1  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Q-table for simple implementation (could be neural network)
        self.q_table = defaultdict(lambda: np.zeros(action_size))
        self.experience_replay = deque(maxlen=10000)
        
    def get_state_representation(self, interaction: UserInteraction, context: UserContext) -> str:
        """Convert interaction and context to state representation"""
        
        # Create a simplified state representation
        features = [
            interaction.interaction_type.value,
            len(interaction.content) // 10,  # Content length bucket
            interaction.timestamp.hour,  # Time of day
            len(context.current_goals),
            context.learning_style,
            len(context.interaction_history)
        ]
        
        # Convert to string for Q-table indexing
        return "_".join(str(f) for f in features)
    
    def choose_action(self, state: str, mode: LearningMode = LearningMode.BALANCED) -> int:
        """Choose action using epsilon-greedy strategy"""
        
        if mode == LearningMode.EXPLORATION or np.random.random() <= self.epsilon:
            return np.random.choice(self.action_size)
        else:
            return np.argmax(self.q_table[state])
    
    def learn(self, state: str, action: int, reward: float, next_state: str, done: bool):
        """Update Q-values using Q-learning"""
        
        current_q = self.q_table[state][action]
        
        if done:
            target_q = reward
        else:
            target_q = reward + self.discount_factor * np.max(self.q_table[next_state])
        
        # Q-learning update
        self.q_table[state][action] = current_q + self.learning_rate * (target_q - current_q)
        
        # Store experience for replay
        self.experience_replay.append((state, action, reward, next_state, done))
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def experience_replay_learning(self, batch_size: int = 32):
        """Learn from stored experiences"""
        
        if len(self.experience_replay) < batch_size:
            return
        
        # Sample random batch
        batch_indices = np.random.choice(len(self.experience_replay), batch_size, replace=False)
        
        for idx in batch_indices:
            state, action, reward, next_state, done = self.experience_replay[idx]
            self.learn(state, action, reward, next_state, done)

class PersonalizationEngine:
    """
    Main personalization engine that learns user preferences and adapts responses
    """
    
    def __init__(self, encryption_key: bytes = None):
        self.user_contexts: Dict[str, UserContext] = {}
        self.rl_agents: Dict[str, ReinforcementLearningAgent] = {}
        self.preference_patterns = {}
        self.goal_templates = self._load_goal_templates()
        
        # Privacy and encryption
        if encryption_key is None:
            encryption_key = Fernet.generate_key()
        self.encryption_key = encryption_key
        self.cipher_suite = Fernet(encryption_key)
        
        # Learning parameters
        self.min_interactions_for_learning = 5
        self.preference_confidence_threshold = 0.6
        self.goal_progress_update_frequency = timedelta(days=1)
        
    async def initialize_user(self, user_id: str) -> UserContext:
        """Initialize a new user context"""
        
        if user_id in self.user_contexts:
            return self.user_contexts[user_id]
        
        context = UserContext(
            user_id=user_id,
            current_goals=[],
            preferences={},
            interaction_history=deque(maxlen=1000),
            learning_style="adaptive",
            expertise_areas={},
            communication_style={
                "formality": 0.5,
                "detail_level": 0.5,
                "technical_depth": 0.5,
                "examples_preference": 0.5
            },
            decision_patterns={},
            last_active=datetime.now()
        )
        
        self.user_contexts[user_id] = context
        
        # Initialize RL agent for this user
        self.rl_agents[user_id] = ReinforcementLearningAgent(
            state_size=10,  # Simplified state representation
            action_size=5   # Number of response styles
        )
        
        return context
    
    async def learn_from_interaction(
        self,
        user_id: str,
        interaction: UserInteraction
    ) -> None:
        """Learn from a user interaction"""
        
        context = await self.initialize_user(user_id)
        context.interaction_history.append(interaction)
        context.last_active = datetime.now()
        
        # Extract preferences from interaction
        await self._extract_preferences(interaction, context)
        
        # Update expertise areas
        await self._update_expertise(interaction, context)
        
        # Update communication style
        await self._update_communication_style(interaction, context)
        
        # Learn from RL perspective
        await self._reinforcement_learning_update(interaction, context)
        
        # Update goals if relevant
        if interaction.interaction_type == InteractionType.GOAL_SETTING:
            await self._update_goals(interaction, context)
    
    async def get_personalized_response_style(
        self,
        user_id: str,
        query_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get personalized response style for a user"""
        
        context = await self.initialize_user(user_id)
        
        # Use RL agent to choose response style
        if user_id in self.rl_agents:
            state = self._get_current_state(context, query_context)
            action = self.rl_agents[user_id].choose_action(state, LearningMode.EXPLOITATION)
            
            response_style = self._action_to_response_style(action, context)
        else:
            response_style = self._default_response_style()
        
        return response_style
    
    async def personalize_recommendations(
        self,
        user_id: str,
        base_recommendations: List[str],
        context: Dict[str, Any]
    ) -> List[PersonalizedRecommendation]:
        """Personalize recommendations based on user preferences"""
        
        user_context = await self.initialize_user(user_id)
        personalized_recs = []
        
        for i, base_rec in enumerate(base_recommendations):
            # Calculate personalization factors
            personalization_factors = await self._calculate_personalization_factors(
                base_rec, user_context, context
            )
            
            # Adjust recommendation based on preferences
            personalized_content = await self._adjust_recommendation_content(
                base_rec, user_context, personalization_factors
            )
            
            # Calculate expected satisfaction
            expected_satisfaction = await self._predict_satisfaction(
                personalized_content, user_context, personalization_factors
            )
            
            # Generate reasoning
            reasoning = self._generate_personalization_reasoning(
                user_context, personalization_factors
            )
            
            rec = PersonalizedRecommendation(
                recommendation_id=f"rec_{user_id}_{i}_{int(datetime.now().timestamp())}",
                user_id=user_id,
                content=personalized_content,
                reasoning=reasoning,
                confidence=personalization_factors.get('confidence', 0.5),
                personalization_factors=personalization_factors,
                alternatives=base_recommendations[:3],  # Alternative options
                expected_satisfaction=expected_satisfaction
            )
            
            personalized_recs.append(rec)
        
        # Sort by expected satisfaction
        personalized_recs.sort(key=lambda x: x.expected_satisfaction, reverse=True)
        
        return personalized_recs
    
    async def provide_feedback(
        self,
        user_id: str,
        interaction_id: str,
        feedback: FeedbackType,
        outcome_quality: float = None
    ) -> None:
        """Receive feedback on a previous interaction"""
        
        context = await self.initialize_user(user_id)
        
        # Find the interaction in history
        target_interaction = None
        for interaction in context.interaction_history:
            if interaction.interaction_id == interaction_id:
                target_interaction = interaction
                break
        
        if not target_interaction:
            logger.warning(f"Interaction {interaction_id} not found for user {user_id}")
            return
        
        # Update interaction with feedback
        target_interaction.feedback = feedback
        target_interaction.outcome_quality = outcome_quality
        
        # Convert feedback to reward signal
        reward = self._feedback_to_reward(feedback, outcome_quality)
        
        # Update RL agent
        if user_id in self.rl_agents:
            state = self._get_interaction_state(target_interaction, context)
            # For this example, we'll use the same state as next_state
            # In practice, you'd want the state after the action was taken
            next_state = state
            action = 0  # You'd need to store which action was taken
            
            self.rl_agents[user_id].learn(state, action, reward, next_state, True)
        
        # Update preferences based on feedback
        await self._update_preferences_from_feedback(target_interaction, feedback, context)
    
    async def get_user_insights(self, user_id: str) -> Dict[str, Any]:
        """Get insights about user preferences and patterns"""
        
        context = await self.initialize_user(user_id)
        
        insights = {
            'total_interactions': len(context.interaction_history),
            'learning_progress': self._calculate_learning_progress(context),
            'top_preferences': self._get_top_preferences(context),
            'communication_style': context.communication_style,
            'expertise_areas': context.expertise_areas,
            'active_goals': [goal.description for goal in context.current_goals],
            'personalization_confidence': self._calculate_personalization_confidence(context),
            'recent_activity': self._analyze_recent_activity(context)
        }
        
        return insights
    
    async def _extract_preferences(self, interaction: UserInteraction, context: UserContext) -> None:
        """Extract preferences from user interaction"""
        
        content = interaction.content.lower()
        
        # Extract communication preferences
        if any(word in content for word in ['detailed', 'thorough', 'comprehensive']):
            await self._update_preference(
                context, 'communication', 'detail_level', 0.8, interaction.interaction_id
            )
        elif any(word in content for word in ['brief', 'quick', 'summary']):
            await self._update_preference(
                context, 'communication', 'detail_level', 0.2, interaction.interaction_id
            )
        
        # Extract format preferences
        if any(word in content for word in ['example', 'examples', 'demonstrate']):
            await self._update_preference(
                context, 'format', 'examples_preference', 0.9, interaction.interaction_id
            )
        
        # Extract topic preferences
        if 'investment' in content or 'finance' in content:
            await self._update_preference(
                context, 'topics', 'finance_interest', 0.8, interaction.interaction_id
            )
        
        if 'technology' in content or 'ai' in content or 'machine learning' in content:
            await self._update_preference(
                context, 'topics', 'technology_interest', 0.8, interaction.interaction_id
            )
    
    async def _update_preference(
        self,
        context: UserContext,
        category: str,
        preference_type: str,
        value: float,
        interaction_id: str
    ) -> None:
        """Update a specific preference"""
        
        pref_key = f"{category}_{preference_type}"
        
        if pref_key in context.preferences:
            # Update existing preference
            existing = context.preferences[pref_key]
            
            # Weighted average with recency bias
            weight = 0.3  # Weight for new observation
            new_value = (1 - weight) * existing.value + weight * value
            
            existing.value = new_value
            existing.confidence = min(0.95, existing.confidence + 0.1)
            existing.learned_from.append(interaction_id)
            existing.last_updated = datetime.now()
            
            # Update stability score
            existing.stability_score = self._calculate_stability_score(existing)
        else:
            # Create new preference
            preference = UserPreference(
                preference_id=str(uuid.uuid4()),
                category=category,
                preference_type=preference_type,
                value=value,
                confidence=0.3,  # Start with low confidence
                learned_from=[interaction_id],
                last_updated=datetime.now(),
                stability_score=0.1
            )
            
            context.preferences[pref_key] = preference
    
    def _calculate_stability_score(self, preference: UserPreference) -> float:
        """Calculate how stable a preference is over time"""
        
        # Simple stability calculation based on consistency
        days_since_update = (datetime.now() - preference.last_updated).days
        
        # More interactions and recent updates = higher stability
        interaction_factor = min(len(preference.learned_from) / 10, 1.0)
        recency_factor = max(0.1, 1.0 - days_since_update / 30)
        
        return min(0.95, preference.confidence * interaction_factor * recency_factor)
    
    async def _update_expertise(self, interaction: UserInteraction, context: UserContext) -> None:
        """Update user expertise areas based on interaction"""
        
        content = interaction.content.lower()
        
        # Simple keyword-based expertise detection
        expertise_keywords = {
            'finance': ['investment', 'stock', 'portfolio', 'dividend', 'roi', 'finance'],
            'technology': ['ai', 'machine learning', 'algorithm', 'programming', 'software'],
            'health': ['health', 'medicine', 'doctor', 'symptoms', 'treatment'],
            'business': ['business', 'strategy', 'marketing', 'management', 'company'],
            'science': ['research', 'study', 'experiment', 'data', 'analysis']
        }
        
        for domain, keywords in expertise_keywords.items():
            keyword_count = sum(1 for keyword in keywords if keyword in content)
            
            if keyword_count > 0:
                current_level = context.expertise_areas.get(domain, 0.0)
                # Increase expertise level gradually
                new_level = min(1.0, current_level + keyword_count * 0.05)
                context.expertise_areas[domain] = new_level
    
    async def _update_communication_style(self, interaction: UserInteraction, context: UserContext) -> None:
        """Update communication style preferences"""
        
        content = interaction.content
        
        # Analyze formality
        formal_indicators = ['please', 'thank you', 'could you', 'would you']
        informal_indicators = ['hey', 'hi', 'thanks', 'cool', 'awesome']
        
        formal_count = sum(1 for indicator in formal_indicators if indicator in content.lower())
        informal_count = sum(1 for indicator in informal_indicators if indicator in content.lower())
        
        if formal_count > informal_count:
            context.communication_style['formality'] = min(1.0, context.communication_style['formality'] + 0.1)
        elif informal_count > formal_count:
            context.communication_style['formality'] = max(0.0, context.communication_style['formality'] - 0.1)
        
        # Analyze technical depth preference
        technical_words = ['algorithm', 'implementation', 'methodology', 'analysis', 'technical']
        tech_count = sum(1 for word in technical_words if word in content.lower())
        
        if tech_count > 2:
            context.communication_style['technical_depth'] = min(1.0, context.communication_style['technical_depth'] + 0.1)
        elif tech_count == 0 and len(content.split()) > 10:
            context.communication_style['technical_depth'] = max(0.0, context.communication_style['technical_depth'] - 0.05)
    
    async def _reinforcement_learning_update(self, interaction: UserInteraction, context: UserContext) -> None:
        """Update RL agent based on interaction"""
        
        if context.user_id not in self.rl_agents:
            return
        
        rl_agent = self.rl_agents[context.user_id]
        
        # Get state representation
        state = self._get_interaction_state(interaction, context)
        
        # For new interactions without feedback, we can't learn yet
        # This would be updated when feedback is received
        if interaction.feedback is not None:
            reward = self._feedback_to_reward(interaction.feedback, interaction.outcome_quality)
            action = 0  # You'd need to store which action was actually taken
            next_state = state  # Simplified
            
            rl_agent.learn(state, action, reward, next_state, True)
    
    def _get_interaction_state(self, interaction: UserInteraction, context: UserContext) -> str:
        """Get state representation for RL agent"""
        
        return self.rl_agents[context.user_id].get_state_representation(interaction, context)
    
    def _get_current_state(self, context: UserContext, query_context: Dict[str, Any]) -> str:
        """Get current state for choosing action"""
        
        # Create a synthetic interaction for state representation
        synthetic_interaction = UserInteraction(
            interaction_id="current",
            user_id=context.user_id,
            timestamp=datetime.now(),
            interaction_type=InteractionType.QUERY,
            content=query_context.get('query', ''),
            context=query_context
        )
        
        return self._get_interaction_state(synthetic_interaction, context)
    
    def _action_to_response_style(self, action: int, context: UserContext) -> Dict[str, Any]:
        """Convert RL action to response style parameters"""
        
        # Define different response styles
        styles = [
            {  # Action 0: Detailed and formal
                'detail_level': 0.9,
                'formality': 0.8,
                'examples': True,
                'technical_depth': 0.7
            },
            {  # Action 1: Brief and casual
                'detail_level': 0.3,
                'formality': 0.2,
                'examples': False,
                'technical_depth': 0.3
            },
            {  # Action 2: Balanced
                'detail_level': 0.6,
                'formality': 0.5,
                'examples': True,
                'technical_depth': 0.5
            },
            {  # Action 3: Technical focus
                'detail_level': 0.8,
                'formality': 0.6,
                'examples': True,
                'technical_depth': 0.9
            },
            {  # Action 4: Simple explanation
                'detail_level': 0.5,
                'formality': 0.4,
                'examples': True,
                'technical_depth': 0.2
            }
        ]
        
        if action < len(styles):
            base_style = styles[action]
        else:
            base_style = styles[2]  # Default to balanced
        
        # Adjust based on learned communication style
        adjusted_style = base_style.copy()
        adjusted_style['formality'] = (base_style['formality'] + context.communication_style['formality']) / 2
        adjusted_style['technical_depth'] = (base_style['technical_depth'] + context.communication_style['technical_depth']) / 2
        
        return adjusted_style
    
    def _default_response_style(self) -> Dict[str, Any]:
        """Default response style for new users"""
        return {
            'detail_level': 0.6,
            'formality': 0.5,
            'examples': True,
            'technical_depth': 0.5
        }
    
    def _feedback_to_reward(self, feedback: FeedbackType, outcome_quality: float = None) -> float:
        """Convert user feedback to reward signal for RL"""
        
        base_rewards = {
            FeedbackType.VERY_POSITIVE: 1.0,
            FeedbackType.POSITIVE: 0.5,
            FeedbackType.NEUTRAL: 0.0,
            FeedbackType.NEGATIVE: -0.5,
            FeedbackType.VERY_NEGATIVE: -1.0
        }
        
        reward = base_rewards.get(feedback, 0.0)
        
        # Adjust based on outcome quality if provided
        if outcome_quality is not None:
            reward = (reward + outcome_quality) / 2
        
        return reward
    
    async def _calculate_personalization_factors(
        self,
        base_recommendation: str,
        context: UserContext,
        query_context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate personalization factors for a recommendation"""
        
        factors = {
            'preference_match': 0.5,
            'expertise_alignment': 0.5,
            'goal_relevance': 0.5,
            'style_match': 0.5,
            'confidence': 0.5
        }
        
        # Calculate preference match
        relevant_preferences = [
            pref for pref in context.preferences.values()
            if any(keyword in base_recommendation.lower() 
                  for keyword in [pref.preference_type, pref.category])
        ]
        
        if relevant_preferences:
            avg_confidence = sum(pref.confidence for pref in relevant_preferences) / len(relevant_preferences)
            factors['preference_match'] = avg_confidence
        
        # Calculate expertise alignment
        query_content = query_context.get('query', '').lower()
        relevant_expertise = [
            level for domain, level in context.expertise_areas.items()
            if domain in query_content or domain in base_recommendation.lower()
        ]
        
        if relevant_expertise:
            factors['expertise_alignment'] = max(relevant_expertise)
        
        # Calculate goal relevance
        goal_relevance_scores = []
        for goal in context.current_goals:
            if any(keyword in base_recommendation.lower() 
                  for keyword in goal.description.lower().split()):
                goal_relevance_scores.append(goal.priority)
        
        if goal_relevance_scores:
            factors['goal_relevance'] = max(goal_relevance_scores)
        
        # Calculate overall confidence
        factors['confidence'] = sum(factors.values()) / len(factors)
        
        return factors
    
    async def _adjust_recommendation_content(
        self,
        base_recommendation: str,
        context: UserContext,
        personalization_factors: Dict[str, float]
    ) -> str:
        """Adjust recommendation content based on personalization"""
        
        # This is a simplified version - in practice, you'd use more sophisticated NLP
        adjusted_content = base_recommendation
        
        # Adjust based on communication style
        if context.communication_style['formality'] > 0.7:
            adjusted_content = f"I would respectfully recommend: {adjusted_content}"
        elif context.communication_style['formality'] < 0.3:
            adjusted_content = f"Here's what I think: {adjusted_content}"
        
        # Add technical details if preferred
        if context.communication_style['technical_depth'] > 0.7:
            adjusted_content += " (This recommendation is based on algorithmic analysis of multiple factors)"
        
        # Add examples if preferred
        if context.communication_style.get('examples_preference', 0.5) > 0.7:
            adjusted_content += " For example, this approach has been successful in similar scenarios."
        
        return adjusted_content
    
    async def _predict_satisfaction(
        self,
        recommendation: str,
        context: UserContext,
        personalization_factors: Dict[str, float]
    ) -> float:
        """Predict user satisfaction with the recommendation"""
        
        # Simple satisfaction prediction based on personalization factors
        base_satisfaction = 0.5
        
        # Weight different factors
        weights = {
            'preference_match': 0.3,
            'expertise_alignment': 0.2,
            'goal_relevance': 0.3,
            'style_match': 0.2
        }
        
        weighted_satisfaction = sum(
            personalization_factors.get(factor, 0.5) * weight
            for factor, weight in weights.items()
        )
        
        # Add bonus for high confidence
        confidence_bonus = personalization_factors.get('confidence', 0.5) * 0.1
        
        final_satisfaction = min(0.95, weighted_satisfaction + confidence_bonus)
        
        return max(0.1, final_satisfaction)
    
    def _generate_personalization_reasoning(
        self,
        context: UserContext,
        personalization_factors: Dict[str, float]
    ) -> str:
        """Generate explanation for why recommendation was personalized this way"""
        
        reasoning_parts = []
        
        if personalization_factors['preference_match'] > 0.6:
            reasoning_parts.append("matches your previously expressed preferences")
        
        if personalization_factors['expertise_alignment'] > 0.6:
            reasoning_parts.append("aligns with your expertise level")
        
        if personalization_factors['goal_relevance'] > 0.6:
            reasoning_parts.append("supports your stated goals")
        
        if context.communication_style['formality'] > 0.7:
            reasoning_parts.append("presented in your preferred formal style")
        elif context.communication_style['formality'] < 0.3:
            reasoning_parts.append("presented in your preferred casual style")
        
        if not reasoning_parts:
            return "Based on general best practices"
        
        return "Personalized because it " + " and ".join(reasoning_parts)
    
    async def _update_goals(self, interaction: UserInteraction, context: UserContext) -> None:
        """Update user goals based on goal-setting interaction"""
        
        content = interaction.content.lower()
        
        # Simple goal extraction (in practice, use more sophisticated NLP)
        if 'goal' in content or 'want to' in content or 'planning to' in content:
            goal_description = interaction.content
            
            goal = UserGoal(
                goal_id=str(uuid.uuid4()),
                description=goal_description,
                category='general',
                priority=0.7,
                target_date=None,
                progress=0.0,
                sub_goals=[],
                success_metrics={}
            )
            
            context.current_goals.append(goal)
    
    async def _update_preferences_from_feedback(
        self,
        interaction: UserInteraction,
        feedback: FeedbackType,
        context: UserContext
    ) -> None:
        """Update preferences based on feedback received"""
        
        # Analyze what worked or didn't work
        if feedback in [FeedbackType.POSITIVE, FeedbackType.VERY_POSITIVE]:
            # Reinforce preferences that led to positive feedback
            content_length = len(interaction.content)
            
            if content_length > 200:  # Long response got positive feedback
                await self._update_preference(
                    context, 'communication', 'detail_level', 0.8, interaction.interaction_id
                )
        
        elif feedback in [FeedbackType.NEGATIVE, FeedbackType.VERY_NEGATIVE]:
            # Adjust preferences based on negative feedback
            content_length = len(interaction.content)
            
            if content_length > 200:  # Long response got negative feedback
                await self._update_preference(
                    context, 'communication', 'detail_level', 0.3, interaction.interaction_id
                )
    
    def _calculate_learning_progress(self, context: UserContext) -> float:
        """Calculate how much we've learned about the user"""
        
        total_preferences = len(context.preferences)
        confident_preferences = sum(1 for pref in context.preferences.values() if pref.confidence > 0.6)
        
        if total_preferences == 0:
            return 0.0
        
        preference_progress = confident_preferences / total_preferences
        interaction_progress = min(len(context.interaction_history) / 50, 1.0)
        
        return (preference_progress + interaction_progress) / 2
    
    def _get_top_preferences(self, context: UserContext) -> List[Dict[str, Any]]:
        """Get top learned preferences"""
        
        sorted_prefs = sorted(
            context.preferences.values(),
            key=lambda p: p.confidence * p.stability_score,
            reverse=True
        )
        
        return [
            {
                'category': pref.category,
                'type': pref.preference_type,
                'value': pref.value,
                'confidence': pref.confidence
            }
            for pref in sorted_prefs[:5]
        ]
    
    def _calculate_personalization_confidence(self, context: UserContext) -> float:
        """Calculate overall confidence in personalization"""
        
        if not context.preferences:
            return 0.1
        
        avg_confidence = sum(pref.confidence for pref in context.preferences.values()) / len(context.preferences)
        interaction_factor = min(len(context.interaction_history) / 20, 1.0)
        
        return avg_confidence * interaction_factor
    
    def _analyze_recent_activity(self, context: UserContext) -> Dict[str, Any]:
        """Analyze recent user activity patterns"""
        
        recent_cutoff = datetime.now() - timedelta(days=7)
        recent_interactions = [
            interaction for interaction in context.interaction_history
            if interaction.timestamp > recent_cutoff
        ]
        
        return {
            'interactions_last_7_days': len(recent_interactions),
            'most_common_interaction_type': self._most_common_interaction_type(recent_interactions),
            'avg_session_length': self._calculate_avg_session_length(recent_interactions),
            'topics_discussed': self._extract_recent_topics(recent_interactions)
        }
    
    def _most_common_interaction_type(self, interactions: List[UserInteraction]) -> str:
        """Find most common interaction type"""
        
        if not interactions:
            return "none"
        
        type_counts = {}
        for interaction in interactions:
            type_counts[interaction.interaction_type.value] = type_counts.get(interaction.interaction_type.value, 0) + 1
        
        return max(type_counts, key=type_counts.get)
    
    def _calculate_avg_session_length(self, interactions: List[UserInteraction]) -> float:
        """Calculate average session length in minutes"""
        
        if len(interactions) < 2:
            return 0.0
        
        # Simple approximation: time between first and last interaction
        time_span = (interactions[-1].timestamp - interactions[0].timestamp).total_seconds() / 60
        return time_span / len(interactions) if interactions else 0.0
    
    def _extract_recent_topics(self, interactions: List[UserInteraction]) -> List[str]:
        """Extract topics from recent interactions"""
        
        all_content = " ".join(interaction.content for interaction in interactions)
        content_lower = all_content.lower()
        
        topic_keywords = {
            'finance': ['investment', 'money', 'stock', 'portfolio'],
            'technology': ['ai', 'algorithm', 'software', 'tech'],
            'health': ['health', 'medical', 'doctor', 'symptoms'],
            'business': ['business', 'company', 'strategy', 'market'],
            'education': ['learn', 'study', 'course', 'education']
        }
        
        detected_topics = []
        for topic, keywords in topic_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                detected_topics.append(topic)
        
        return detected_topics
    
    def _load_goal_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load templates for common goals"""
        
        return {
            'financial': {
                'investment': {
                    'metrics': ['roi', 'risk_level', 'time_horizon'],
                    'milestones': ['research', 'allocate_funds', 'monitor', 'rebalance']
                },
                'savings': {
                    'metrics': ['target_amount', 'monthly_contribution', 'deadline'],
                    'milestones': ['set_budget', 'automate_savings', 'track_progress']
                }
            },
            'learning': {
                'skill_development': {
                    'metrics': ['competency_level', 'practice_hours', 'projects_completed'],
                    'milestones': ['foundation', 'intermediate', 'advanced', 'expert']
                }
            },
            'health': {
                'fitness': {
                    'metrics': ['weight', 'exercise_frequency', 'strength_measures'],
                    'milestones': ['baseline', 'initial_improvement', 'habit_formation', 'target_achievement']
                }
            }
        }