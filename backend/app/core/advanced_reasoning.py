"""
Advanced Reasoning Engine with Ranking, Simulation, and Recommendation Capabilities
"""

import asyncio
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

class ConfidenceLevel(Enum):
    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.95

class DecisionType(Enum):
    INVESTMENT = "investment"
    CAREER = "career"
    PURCHASE = "purchase"
    STRATEGY = "strategy"
    COMPARISON = "comparison"
    PREDICTION = "prediction"

@dataclass
class Option:
    """Represents an option to be evaluated"""
    id: str
    name: str
    description: str
    parameters: Dict[str, Any]
    category: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class SimulationResult:
    """Results from running a simulation"""
    scenario: str
    probability: float
    expected_value: float
    risk_score: float
    outcomes: List[Dict[str, Any]]
    confidence: float
    
@dataclass
class RankingCriterion:
    """Criteria for ranking options"""
    name: str
    weight: float
    metric_type: str  # 'benefit', 'cost', 'risk'
    scale: Tuple[float, float]  # min, max values
    
@dataclass
class ReasoningPath:
    """Documents the reasoning process"""
    steps: List[Dict[str, Any]]
    evidence: List[Dict[str, Any]]
    assumptions: List[str]
    counterarguments: List[str]
    confidence_factors: Dict[str, float]
    
@dataclass
class Decision:
    """Final decision with recommendation"""
    recommended_option: Option
    confidence_score: float
    reasoning_path: ReasoningPath
    alternatives: List[Tuple[Option, float]]
    risk_assessment: Dict[str, Any]
    simulation_results: List[SimulationResult]
    timestamp: datetime
    decision_id: str
    
    def __post_init__(self):
        if self.decision_id is None:
            self.decision_id = str(uuid.uuid4())

class AdvancedReasoningEngine:
    """
    Advanced reasoning engine that can:
    1. Rank options using multiple criteria
    2. Run simulations for different scenarios
    3. Make recommendations with confidence scores
    4. Track reasoning paths
    """
    
    def __init__(self):
        self.criteria_weights = {}
        self.simulation_cache = {}
        self.decision_history = []
        
    async def rank_options(
        self,
        options: List[Option],
        criteria: List[RankingCriterion],
        user_preferences: Dict[str, Any] = None
    ) -> List[Tuple[Option, float]]:
        """
        Rank options using weighted criteria and user preferences
        """
        if not options or not criteria:
            return []
            
        scores = {}
        
        for option in options:
            total_score = 0.0
            total_weight = 0.0
            
            for criterion in criteria:
                # Get metric value for this option
                metric_value = await self._evaluate_metric(option, criterion)
                
                # Normalize the value to 0-1 scale
                normalized_value = self._normalize_value(
                    metric_value, 
                    criterion.scale, 
                    criterion.metric_type
                )
                
                # Apply user preference adjustment
                adjusted_weight = self._adjust_weight_for_user(
                    criterion, user_preferences
                )
                
                total_score += normalized_value * adjusted_weight
                total_weight += adjusted_weight
            
            # Calculate final score
            final_score = total_score / total_weight if total_weight > 0 else 0.0
            scores[option.id] = (option, final_score)
        
        # Sort by score descending
        ranked_options = sorted(
            scores.values(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return ranked_options
    
    async def run_simulation(
        self,
        option: Option,
        scenario: str,
        parameters: Dict[str, Any],
        iterations: int = 1000
    ) -> SimulationResult:
        """
        Run Monte Carlo simulation for an option under given scenario
        """
        cache_key = f"{option.id}_{scenario}_{hash(str(parameters))}"
        
        if cache_key in self.simulation_cache:
            return self.simulation_cache[cache_key]
        
        outcomes = []
        
        for i in range(iterations):
            # Generate random variables based on parameters
            random_factors = self._generate_random_factors(parameters)
            
            # Calculate outcome for this iteration
            outcome = await self._calculate_outcome(
                option, scenario, random_factors
            )
            outcomes.append(outcome)
        
        # Analyze results
        values = [outcome.get('value', 0) for outcome in outcomes]
        expected_value = np.mean(values)
        risk_score = np.std(values) / expected_value if expected_value != 0 else float('inf')
        
        # Calculate probability of success based on scenario
        success_threshold = parameters.get('success_threshold', expected_value)
        probability = sum(1 for v in values if v >= success_threshold) / len(values)
        
        # Calculate confidence based on variance and sample size
        confidence = min(0.95, 1.0 - (np.std(values) / (np.sqrt(iterations) * abs(expected_value))))
        confidence = max(0.1, confidence)
        
        result = SimulationResult(
            scenario=scenario,
            probability=probability,
            expected_value=expected_value,
            risk_score=risk_score,
            outcomes=outcomes[:10],  # Store first 10 for inspection
            confidence=confidence
        )
        
        self.simulation_cache[cache_key] = result
        return result
    
    async def make_recommendation(
        self,
        options: List[Option],
        criteria: List[RankingCriterion],
        scenarios: List[str] = None,
        user_preferences: Dict[str, Any] = None,
        decision_type: DecisionType = DecisionType.COMPARISON
    ) -> Decision:
        """
        Make a comprehensive recommendation with confidence scoring
        """
        # Step 1: Rank options
        ranked_options = await self.rank_options(options, criteria, user_preferences)
        
        if not ranked_options:
            raise ValueError("No options to evaluate")
        
        # Step 2: Run simulations for top options
        simulation_results = []
        top_options = ranked_options[:min(3, len(ranked_options))]
        
        if scenarios:
            for option, score in top_options:
                for scenario in scenarios:
                    sim_params = self._get_simulation_parameters(
                        option, scenario, decision_type
                    )
                    sim_result = await self.run_simulation(
                        option, scenario, sim_params
                    )
                    simulation_results.append(sim_result)
        
        # Step 3: Build reasoning path
        reasoning_path = await self._build_reasoning_path(
            ranked_options, simulation_results, criteria
        )
        
        # Step 4: Calculate final confidence
        final_confidence = self._calculate_final_confidence(
            ranked_options[0][1], simulation_results, reasoning_path
        )
        
        # Step 5: Assess risks
        risk_assessment = await self._assess_risks(
            ranked_options[0][0], simulation_results
        )
        
        # Step 6: Create decision
        decision = Decision(
            recommended_option=ranked_options[0][0],
            confidence_score=final_confidence,
            reasoning_path=reasoning_path,
            alternatives=ranked_options[1:4],  # Top 3 alternatives
            risk_assessment=risk_assessment,
            simulation_results=simulation_results,
            timestamp=datetime.now(),
            decision_id=None  # Will be auto-generated
        )
        
        self.decision_history.append(decision)
        return decision
    
    async def _evaluate_metric(self, option: Option, criterion: RankingCriterion) -> float:
        """Evaluate a specific metric for an option"""
        # This would contain logic to extract or calculate the metric value
        # For now, using synthetic values based on option parameters
        
        if criterion.name in option.parameters:
            return float(option.parameters[criterion.name])
        
        # Synthetic evaluation based on criterion type
        if criterion.name == "cost":
            return option.parameters.get("price", 100) * np.random.uniform(0.8, 1.2)
        elif criterion.name == "quality":
            return np.random.uniform(3, 5)  # Quality score 1-5
        elif criterion.name == "risk":
            return np.random.uniform(0.1, 0.9)  # Risk score 0-1
        elif criterion.name == "roi":
            return np.random.uniform(0.05, 0.25)  # ROI 5-25%
        else:
            return np.random.uniform(0, 1)
    
    def _normalize_value(self, value: float, scale: Tuple[float, float], metric_type: str) -> float:
        """Normalize value to 0-1 scale"""
        min_val, max_val = scale
        
        if max_val == min_val:
            return 0.5
        
        normalized = (value - min_val) / (max_val - min_val)
        normalized = max(0, min(1, normalized))
        
        # For cost and risk metrics, invert the scale (lower is better)
        if metric_type in ['cost', 'risk']:
            normalized = 1 - normalized
        
        return normalized
    
    def _adjust_weight_for_user(
        self, 
        criterion: RankingCriterion, 
        user_preferences: Dict[str, Any]
    ) -> float:
        """Adjust criterion weight based on user preferences"""
        if not user_preferences:
            return criterion.weight
        
        # Apply user preference multipliers
        preference_key = f"{criterion.name}_importance"
        multiplier = user_preferences.get(preference_key, 1.0)
        
        return criterion.weight * multiplier
    
    def _generate_random_factors(self, parameters: Dict[str, Any]) -> Dict[str, float]:
        """Generate random factors for simulation"""
        factors = {}
        
        for key, value in parameters.items():
            if isinstance(value, dict) and 'distribution' in value:
                dist_type = value['distribution']
                if dist_type == 'normal':
                    factors[key] = np.random.normal(
                        value['mean'], value['std']
                    )
                elif dist_type == 'uniform':
                    factors[key] = np.random.uniform(
                        value['min'], value['max']
                    )
                else:
                    factors[key] = float(value.get('default', 1.0))
            else:
                # Add some randomness to fixed values
                factors[key] = float(value) * np.random.uniform(0.9, 1.1)
        
        return factors
    
    async def _calculate_outcome(
        self, 
        option: Option, 
        scenario: str, 
        random_factors: Dict[str, float]
    ) -> Dict[str, Any]:
        """Calculate outcome for a single simulation iteration"""
        # This would contain scenario-specific calculation logic
        # For investment scenario example:
        
        if scenario == "investment":
            initial_value = random_factors.get('initial_amount', 1000)
            return_rate = random_factors.get('return_rate', 0.07)
            market_factor = random_factors.get('market_factor', 1.0)
            time_horizon = random_factors.get('time_horizon', 5)
            
            # Simple compound interest with market volatility
            final_value = initial_value * ((1 + return_rate * market_factor) ** time_horizon)
            
            return {
                'value': final_value,
                'return': final_value - initial_value,
                'factors': random_factors
            }
        
        elif scenario == "career":
            base_salary = random_factors.get('base_salary', 50000)
            growth_rate = random_factors.get('growth_rate', 0.05)
            market_demand = random_factors.get('market_demand', 1.0)
            years = random_factors.get('years', 10)
            
            final_salary = base_salary * ((1 + growth_rate * market_demand) ** years)
            
            return {
                'value': final_salary,
                'total_earnings': sum(
                    base_salary * ((1 + growth_rate * market_demand) ** year)
                    for year in range(years)
                ),
                'factors': random_factors
            }
        
        else:
            # Generic outcome calculation
            base_value = random_factors.get('base_value', 100)
            multiplier = random_factors.get('multiplier', 1.0)
            
            return {
                'value': base_value * multiplier,
                'factors': random_factors
            }
    
    def _get_simulation_parameters(
        self, 
        option: Option, 
        scenario: str, 
        decision_type: DecisionType
    ) -> Dict[str, Any]:
        """Get simulation parameters based on scenario and decision type"""
        
        base_params = option.parameters.copy()
        
        if scenario == "investment" and decision_type == DecisionType.INVESTMENT:
            return {
                'initial_amount': {'distribution': 'uniform', 'min': 500, 'max': 2000},
                'return_rate': {'distribution': 'normal', 'mean': 0.07, 'std': 0.02},
                'market_factor': {'distribution': 'normal', 'mean': 1.0, 'std': 0.3},
                'time_horizon': base_params.get('time_horizon', 5),
                'success_threshold': base_params.get('target_return', 1200)
            }
        
        elif scenario == "career" and decision_type == DecisionType.CAREER:
            return {
                'base_salary': base_params.get('salary', 50000),
                'growth_rate': {'distribution': 'normal', 'mean': 0.05, 'std': 0.02},
                'market_demand': {'distribution': 'uniform', 'min': 0.8, 'max': 1.2},
                'years': base_params.get('years', 10),
                'success_threshold': base_params.get('target_salary', 80000)
            }
        
        else:
            # Generic parameters
            return {
                'base_value': base_params.get('value', 100),
                'multiplier': {'distribution': 'normal', 'mean': 1.0, 'std': 0.2},
                'success_threshold': base_params.get('threshold', 110)
            }
    
    async def _build_reasoning_path(
        self,
        ranked_options: List[Tuple[Option, float]],
        simulation_results: List[SimulationResult],
        criteria: List[RankingCriterion]
    ) -> ReasoningPath:
        """Build a detailed reasoning path"""
        
        steps = [
            {
                'step': 'option_evaluation',
                'description': f'Evaluated {len(ranked_options)} options using {len(criteria)} criteria',
                'details': {
                    'top_option': ranked_options[0][0].name,
                    'score': ranked_options[0][1],
                    'criteria_used': [c.name for c in criteria]
                }
            }
        ]
        
        if simulation_results:
            steps.append({
                'step': 'simulation_analysis',
                'description': f'Ran simulations for {len(simulation_results)} scenarios',
                'details': {
                    'avg_confidence': np.mean([sr.confidence for sr in simulation_results]),
                    'scenarios': [sr.scenario for sr in simulation_results]
                }
            })
        
        evidence = [
            {
                'type': 'ranking_scores',
                'data': [(opt.name, score) for opt, score in ranked_options[:3]]
            }
        ]
        
        if simulation_results:
            evidence.append({
                'type': 'simulation_results',
                'data': [
                    {
                        'scenario': sr.scenario,
                        'probability': sr.probability,
                        'expected_value': sr.expected_value,
                        'confidence': sr.confidence
                    }
                    for sr in simulation_results
                ]
            })
        
        assumptions = [
            "User preferences are accurately represented by provided weights",
            "Historical patterns will continue into the future",
            "Market conditions remain relatively stable"
        ]
        
        counterarguments = [
            "Alternative ranking criteria might yield different results",
            "Unexpected market changes could affect outcomes",
            "Personal circumstances might change priorities"
        ]
        
        confidence_factors = {
            'data_quality': 0.8,
            'model_accuracy': 0.75,
            'prediction_horizon': 0.7,
            'market_stability': 0.6
        }
        
        return ReasoningPath(
            steps=steps,
            evidence=evidence,
            assumptions=assumptions,
            counterarguments=counterarguments,
            confidence_factors=confidence_factors
        )
    
    def _calculate_final_confidence(
        self,
        ranking_score: float,
        simulation_results: List[SimulationResult],
        reasoning_path: ReasoningPath
    ) -> float:
        """Calculate final confidence score"""
        
        # Base confidence from ranking
        base_confidence = ranking_score
        
        # Simulation confidence
        sim_confidence = 0.5  # Default
        if simulation_results:
            sim_confidence = np.mean([sr.confidence for sr in simulation_results])
        
        # Reasoning path confidence
        reasoning_confidence = np.mean(list(reasoning_path.confidence_factors.values()))
        
        # Weighted combination
        final_confidence = (
            0.4 * base_confidence +
            0.35 * sim_confidence +
            0.25 * reasoning_confidence
        )
        
        return min(0.95, max(0.1, final_confidence))
    
    async def _assess_risks(
        self,
        option: Option,
        simulation_results: List[SimulationResult]
    ) -> Dict[str, Any]:
        """Assess risks for the recommended option"""
        
        risks = {
            'financial_risk': 'medium',
            'market_risk': 'medium',
            'execution_risk': 'low',
            'opportunity_cost': 'low'
        }
        
        if simulation_results:
            avg_risk = np.mean([sr.risk_score for sr in simulation_results])
            
            if avg_risk > 1.0:
                risks['financial_risk'] = 'high'
            elif avg_risk < 0.3:
                risks['financial_risk'] = 'low'
        
        # Add specific risk factors based on option type
        risks['risk_factors'] = [
            'Market volatility',
            'Regulatory changes',
            'Competition',
            'Technology disruption'
        ]
        
        risks['mitigation_strategies'] = [
            'Diversification',
            'Regular monitoring',
            'Exit strategy planning',
            'Risk assessment updates'
        ]
        
        return risks

# Helper functions for common use cases
async def analyze_investment_options(
    investments: List[Dict[str, Any]],
    amount: float,
    time_horizon: int,
    risk_tolerance: str = "medium"
) -> Decision:
    """Analyze investment options"""
    
    engine = AdvancedReasoningEngine()
    
    # Convert to Option objects
    options = [
        Option(
            id=str(i),
            name=inv['name'],
            description=inv.get('description', ''),
            parameters={
                'expected_return': inv['expected_return'],
                'risk_level': inv['risk_level'],
                'liquidity': inv.get('liquidity', 0.5),
                'fees': inv.get('fees', 0.01),
                'time_horizon': time_horizon
            },
            category='investment'
        )
        for i, inv in enumerate(investments)
    ]
    
    # Define criteria
    criteria = [
        RankingCriterion('expected_return', 0.4, 'benefit', (0.0, 0.3)),
        RankingCriterion('risk_level', 0.3, 'risk', (0.0, 1.0)),
        RankingCriterion('liquidity', 0.2, 'benefit', (0.0, 1.0)),
        RankingCriterion('fees', 0.1, 'cost', (0.0, 0.05))
    ]
    
    # Adjust weights based on risk tolerance
    risk_weights = {
        'low': {'expected_return': 0.2, 'risk_level': 0.5},
        'medium': {'expected_return': 0.4, 'risk_level': 0.3},
        'high': {'expected_return': 0.6, 'risk_level': 0.1}
    }
    
    if risk_tolerance in risk_weights:
        for criterion in criteria:
            if criterion.name in risk_weights[risk_tolerance]:
                criterion.weight = risk_weights[risk_tolerance][criterion.name]
    
    decision = await engine.make_recommendation(
        options=options,
        criteria=criteria,
        scenarios=['investment'],
        decision_type=DecisionType.INVESTMENT
    )
    
    return decision