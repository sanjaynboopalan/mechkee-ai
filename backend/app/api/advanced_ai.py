"""
Advanced AI API Router
Integrates advanced reasoning, truthfulness, personalization, and privacy features
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime
import asyncio
import uuid

from ..core.advanced_reasoning import (
    AdvancedReasoningEngine, 
    Option, 
    RankingCriterion, 
    DecisionType,
    analyze_investment_options
)
from ..core.truthfulness_engine import TruthfulnessEngine, SourceType
from ..core.personalization_engine import (
    PersonalizationEngine, 
    UserInteraction, 
    InteractionType, 
    FeedbackType
)
from ..core.privacy_engine import PrivacyEngine, DataCategory, PrivacyLevel

logger = logging.getLogger(__name__)

# Initialize engines
reasoning_engine = AdvancedReasoningEngine()
truthfulness_engine = TruthfulnessEngine()
personalization_engine = PersonalizationEngine()
privacy_engine = PrivacyEngine()

router = APIRouter(prefix="/advanced", tags=["Advanced AI"])

# Request/Response Models
class AdvancedQuery(BaseModel):
    query: str = Field(..., description="User's query or question")
    user_id: str = Field(..., description="Unique user identifier")
    query_type: str = Field(default="general", description="Type of query: general, investment, decision, comparison")
    options: Optional[List[Dict[str, Any]]] = Field(None, description="Options to evaluate (for decision queries)")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    require_sources: bool = Field(default=True, description="Whether to verify information with sources")
    personalize: bool = Field(default=True, description="Whether to personalize response")

class FeedbackRequest(BaseModel):
    user_id: str
    interaction_id: str
    feedback_type: str  # "positive", "negative", "neutral", "very_positive", "very_negative"
    outcome_quality: Optional[float] = Field(None, ge=0.0, le=1.0)
    comments: Optional[str] = None

class PreferenceUpdate(BaseModel):
    user_id: str
    preferences: Dict[str, Any]
    privacy_preferences: Optional[Dict[str, Any]] = None

class InvestmentAnalysisRequest(BaseModel):
    user_id: str
    investments: List[Dict[str, Any]]
    amount: float
    time_horizon: int
    risk_tolerance: str = "medium"

class AdvancedResponse(BaseModel):
    response_id: str
    user_id: str
    content: str
    confidence_score: float
    reasoning_path: List[str]
    sources: Optional[List[Dict[str, Any]]] = None
    suggested_questions: Optional[List[str]] = None
    personalization_applied: bool
    privacy_preserved: bool
    bias_assessment: Optional[Dict[str, Any]] = None
    recommendations: Optional[List[Dict[str, Any]]] = None
    timestamp: datetime

@router.post("/query", response_model=AdvancedResponse)
async def advanced_query(
    request: AdvancedQuery,
    background_tasks: BackgroundTasks
) -> AdvancedResponse:
    """
    Process an advanced query with reasoning, truthfulness verification, and personalization
    """
    
    try:
        response_id = str(uuid.uuid4())
        
        # Step 1: Initialize user privacy and personalization
        await privacy_engine.initialize_user_privacy(request.user_id)
        user_context = await personalization_engine.initialize_user(request.user_id)
        
        # Step 2: Record interaction for learning
        interaction = UserInteraction(
            interaction_id=response_id,
            user_id=request.user_id,
            timestamp=datetime.now(),
            interaction_type=InteractionType.QUERY,
            content=request.query,
            context=request.context or {}
        )
        
        # Learn from interaction (background task)
        background_tasks.add_task(
            personalization_engine.learn_from_interaction,
            request.user_id,
            interaction
        )
        
        # Step 3: Get personalized response style
        response_style = await personalization_engine.get_personalized_response_style(
            request.user_id,
            {"query": request.query, "type": request.query_type}
        )
        
        # Step 4: Verify information truthfulness if required
        truthfulness_report = None
        bias_assessment = None
        sources = None
        
        if request.require_sources:
            truthfulness_report = await truthfulness_engine.verify_information(
                query=request.query,
                min_sources=3,
                max_sources=10
            )
            
            bias_assessment = truthfulness_report.bias_assessment
            sources = [
                {
                    "title": claim.statement[:100] + "..." if len(claim.statement) > 100 else claim.statement,
                    "confidence": claim.confidence_score,
                    "verification_status": claim.verification_status,
                    "supporting_sources": len(claim.supporting_sources)
                }
                for claim in truthfulness_report.verified_claims[:5]  # Top 5 claims
            ]
        
        # Step 5: Generate reasoning and recommendations
        recommendations = None
        reasoning_path = ["Query processed with advanced AI reasoning"]
        
        if request.query_type == "decision" and request.options:
            # Decision-making query
            options = [
                Option(
                    id=str(i),
                    name=opt.get("name", f"Option {i+1}"),
                    description=opt.get("description", ""),
                    parameters=opt.get("parameters", {}),
                    category=opt.get("category", "general")
                )
                for i, opt in enumerate(request.options)
            ]
            
            criteria = [
                RankingCriterion("quality", 0.4, "benefit", (0.0, 1.0)),
                RankingCriterion("cost", 0.3, "cost", (0.0, 1000.0)),
                RankingCriterion("risk", 0.3, "risk", (0.0, 1.0))
            ]
            
            decision = await reasoning_engine.make_recommendation(
                options=options,
                criteria=criteria,
                scenarios=["optimistic", "pessimistic"],
                user_preferences={"risk_tolerance": response_style.get("risk_preference", 0.5)},
                decision_type=DecisionType.COMPARISON
            )
            
            recommendations = [
                {
                    "option": decision.recommended_option.name,
                    "confidence": decision.confidence_score,
                    "reasoning": decision.reasoning_path.steps[0]["description"] if decision.reasoning_path.steps else "AI analysis"
                }
            ]
            
            reasoning_path.extend([step["description"] for step in decision.reasoning_path.steps])
        
        # Step 6: Apply personalization to content
        base_content = f"Based on your query about '{request.query}', here's my analysis:"
        
        if truthfulness_report:
            confidence_score = truthfulness_report.confidence_score
            base_content += f"\\n\\nI've verified this information with {len(truthfulness_report.verified_claims)} verified claims "
            base_content += f"(confidence: {confidence_score:.2f})."
        else:
            confidence_score = 0.8  # Default confidence
        
        if recommendations:
            base_content += f"\\n\\nBased on my analysis, I recommend: {recommendations[0]['option']} "
            base_content += f"(confidence: {recommendations[0]['confidence']:.2f})"
        
        # Adjust content based on user's communication style
        if response_style.get("formality", 0.5) > 0.7:
            personalized_content = f"I would respectfully provide the following analysis: {base_content}"
        elif response_style.get("formality", 0.5) < 0.3:
            personalized_content = f"Here's what I found: {base_content}"
        else:
            personalized_content = base_content
        
        if response_style.get("technical_depth", 0.5) > 0.7:
            personalized_content += "\\n\\n(This analysis uses advanced reasoning algorithms with bias detection and source verification.)"
        
        # Step 7: Generate suggested questions
        suggested_questions = _generate_contextual_questions(request.query, request.query_type)
        
        # Step 8: Encrypt and store interaction data
        encrypted_interaction = await privacy_engine.encrypt_user_data(
            user_id=request.user_id,
            data=interaction.__dict__,
            data_category=DataCategory.INTERACTIONS,
            privacy_level=PrivacyLevel.CONFIDENTIAL
        )
        
        response = AdvancedResponse(
            response_id=response_id,
            user_id=request.user_id,
            content=personalized_content,
            confidence_score=confidence_score,
            reasoning_path=reasoning_path,
            sources=sources,
            suggested_questions=suggested_questions,
            personalization_applied=True,
            privacy_preserved=True,
            bias_assessment=bias_assessment,
            recommendations=recommendations,
            timestamp=datetime.now()
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing advanced query: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/investment-analysis")
async def investment_analysis(request: InvestmentAnalysisRequest) -> Dict[str, Any]:
    """
    Analyze investment options with advanced reasoning and personalization
    """
    
    try:
        # Analyze investments
        decision = await analyze_investment_options(
            investments=request.investments,
            amount=request.amount,
            time_horizon=request.time_horizon,
            risk_tolerance=request.risk_tolerance
        )
        
        # Personalize recommendations
        personalized_recs = await personalization_engine.personalize_recommendations(
            user_id=request.user_id,
            base_recommendations=[decision.recommended_option.name],
            context={
                "amount": request.amount,
                "time_horizon": request.time_horizon,
                "risk_tolerance": request.risk_tolerance
            }
        )
        
        return {
            "recommended_investment": decision.recommended_option.name,
            "confidence_score": decision.confidence_score,
            "expected_return": decision.simulation_results[0].expected_value if decision.simulation_results else None,
            "risk_assessment": decision.risk_assessment,
            "personalized_reasoning": personalized_recs[0].reasoning if personalized_recs else "Standard analysis",
            "alternatives": [alt.name for alt, _ in decision.alternatives],
            "simulation_results": [
                {
                    "scenario": sr.scenario,
                    "probability": sr.probability,
                    "expected_value": sr.expected_value,
                    "risk_score": sr.risk_score
                }
                for sr in decision.simulation_results
            ]
        }
        
    except Exception as e:
        logger.error(f"Error in investment analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/feedback")
async def provide_feedback(request: FeedbackRequest) -> Dict[str, str]:
    """
    Provide feedback on a previous interaction for learning
    """
    
    try:
        feedback_type = FeedbackType(request.feedback_type.upper())
        
        await personalization_engine.provide_feedback(
            user_id=request.user_id,
            interaction_id=request.interaction_id,
            feedback=feedback_type,
            outcome_quality=request.outcome_quality
        )
        
        return {"status": "feedback_received", "message": "Thank you for your feedback!"}
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid feedback type: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/preferences")
async def update_preferences(request: PreferenceUpdate) -> Dict[str, str]:
    """
    Update user preferences and privacy settings
    """
    
    try:
        # Update privacy preferences if provided
        if request.privacy_preferences:
            await privacy_engine.update_privacy_preferences(
                user_id=request.user_id,
                new_preferences=request.privacy_preferences
            )
        
        # Store general preferences (encrypted)
        if request.preferences:
            await privacy_engine.encrypt_user_data(
                user_id=request.user_id,
                data=request.preferences,
                data_category=DataCategory.PREFERENCES,
                privacy_level=PrivacyLevel.CONFIDENTIAL
            )
        
        return {"status": "preferences_updated", "message": "Your preferences have been updated successfully"}
        
    except Exception as e:
        logger.error(f"Error updating preferences: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/user-insights/{user_id}")
async def get_user_insights(user_id: str) -> Dict[str, Any]:
    """
    Get insights about user's learned preferences and patterns
    """
    
    try:
        # Get personalization insights
        insights = await personalization_engine.get_user_insights(user_id)
        
        # Get privacy report
        privacy_report = await privacy_engine.generate_privacy_report(user_id)
        
        combined_insights = {
            "personalization": insights,
            "privacy": {
                "privacy_measures_active": privacy_report["privacy_measures"],
                "compliance_status": privacy_report["compliance_status"],
                "data_access_summary": privacy_report["data_access_summary"]
            },
            "recommendations": [
                "Continue engaging to improve personalization",
                "Consider updating privacy preferences periodically",
                "Provide feedback to enhance AI responses"
            ]
        }
        
        return combined_insights
        
    except Exception as e:
        logger.error(f"Error getting user insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/verify-information")
async def verify_information(
    query: str,
    sources: Optional[List[str]] = None,
    min_sources: int = 3
) -> Dict[str, Any]:
    """
    Verify information using truthfulness engine with bias detection
    """
    
    try:
        report = await truthfulness_engine.verify_information(
            query=query,
            initial_sources=sources,
            min_sources=min_sources,
            max_sources=15
        )
        
        return {
            "query": report.query,
            "overall_confidence": report.confidence_score,
            "verified_claims": len(report.verified_claims),
            "disputed_claims": len(report.disputed_claims),
            "bias_risk": report.bias_assessment["overall_bias_risk"],
            "source_diversity": report.source_diversity,
            "reasoning_path": report.reasoning_path,
            "recommendations": report.bias_assessment.get("recommendations", []),
            "falsification_attempts": len(report.falsification_attempts),
            "detailed_claims": [
                {
                    "statement": claim.statement[:200] + "..." if len(claim.statement) > 200 else claim.statement,
                    "confidence": claim.confidence_score,
                    "status": claim.verification_status,
                    "sources": len(claim.supporting_sources)
                }
                for claim in (report.verified_claims + report.disputed_claims)[:10]
            ]
        }
        
    except Exception as e:
        logger.error(f"Error verifying information: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/user-data/{user_id}")
async def delete_user_data(user_id: str, reason: str = "user_request") -> Dict[str, str]:
    """
    Delete all user data (GDPR right to be forgotten)
    """
    
    try:
        success = await privacy_engine.delete_user_data(user_id, reason)
        
        if success:
            return {"status": "deleted", "message": f"All data for user {user_id} has been deleted"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete user data")
            
    except Exception as e:
        logger.error(f"Error deleting user data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/privacy-report/{user_id}")
async def get_privacy_report(user_id: str) -> Dict[str, Any]:
    """
    Get detailed privacy report for a user
    """
    
    try:
        report = await privacy_engine.generate_privacy_report(user_id)
        return report
        
    except Exception as e:
        logger.error(f"Error generating privacy report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/simulate-scenario")
async def simulate_scenario(
    user_id: str,
    scenario_name: str,
    parameters: Dict[str, Any],
    iterations: int = 1000
) -> Dict[str, Any]:
    """
    Run simulation for a specific scenario
    """
    
    try:
        # Create option from parameters
        option = Option(
            id="simulation_option",
            name=scenario_name,
            description=f"Simulation for {scenario_name}",
            parameters=parameters,
            category="simulation"
        )
        
        # Run simulation
        result = await reasoning_engine.run_simulation(
            option=option,
            scenario=scenario_name,
            parameters=parameters,
            iterations=iterations
        )
        
        return {
            "scenario": result.scenario,
            "probability_of_success": result.probability,
            "expected_value": result.expected_value,
            "risk_score": result.risk_score,
            "confidence": result.confidence,
            "sample_outcomes": result.outcomes[:5],  # First 5 outcomes
            "interpretation": _interpret_simulation_results(result)
        }
        
    except Exception as e:
        logger.error(f"Error running simulation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper functions
def _generate_contextual_questions(query: str, query_type: str) -> List[str]:
    """Generate contextual follow-up questions"""
    
    base_questions = [
        f"What are the potential risks of {query.lower()}?",
        f"How does this compare to alternative approaches?",
        f"What factors should I consider when deciding about {query.lower()}?"
    ]
    
    if query_type == "investment":
        return [
            "What's the expected return on this investment?",
            "How does market volatility affect this strategy?",
            "What's the optimal time horizon for this investment?"
        ]
    elif query_type == "decision":
        return [
            "What are the long-term implications of this decision?",
            "How confident are you in this recommendation?",
            "What would change your recommendation?"
        ]
    else:
        return base_questions

def _interpret_simulation_results(result) -> str:
    """Interpret simulation results for users"""
    
    if result.probability > 0.8:
        probability_desc = "very high"
    elif result.probability > 0.6:
        probability_desc = "high"
    elif result.probability > 0.4:
        probability_desc = "moderate"
    else:
        probability_desc = "low"
    
    if result.risk_score < 0.2:
        risk_desc = "low risk"
    elif result.risk_score < 0.5:
        risk_desc = "moderate risk"
    else:
        risk_desc = "high risk"
    
    return (
        f"This scenario has a {probability_desc} probability of success "
        f"with {risk_desc}. Expected outcome: {result.expected_value:.2f}"
    )