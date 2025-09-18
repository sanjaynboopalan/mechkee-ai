"""
Truthfulness System with Cross-checking and Falsification Loop
Implements bias-resistant reasoning with confidence scoring
"""

import asyncio
import json
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from datetime import datetime
import re
import aiohttp
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class SourceType(Enum):
    ACADEMIC = "academic"
    NEWS = "news"
    GOVERNMENT = "government"
    COMMERCIAL = "commercial"
    SOCIAL = "social"
    ENCYCLOPEDIA = "encyclopedia"
    EXPERT = "expert"

class BiasType(Enum):
    CONFIRMATION = "confirmation_bias"
    SELECTION = "selection_bias"
    ANCHORING = "anchoring_bias"
    AVAILABILITY = "availability_bias"
    RECENCY = "recency_bias"
    AUTHORITY = "authority_bias"

@dataclass
class Source:
    """Represents a source of information"""
    url: str
    title: str
    content: str
    source_type: SourceType
    credibility_score: float
    bias_indicators: List[BiasType]
    publication_date: Optional[datetime]
    author: Optional[str]
    domain_authority: float
    
@dataclass
class Claim:
    """Represents a factual claim to be verified"""
    statement: str
    claim_id: str
    supporting_sources: List[Source]
    contradicting_sources: List[Source]
    confidence_score: float
    verification_status: str  # "verified", "disputed", "unverified"
    
@dataclass
class FalsificationAttempt:
    """Represents an attempt to falsify a claim"""
    claim_id: str
    counter_hypothesis: str
    counter_evidence: List[Source]
    falsification_strength: float
    reasoning: str

@dataclass
class TruthfulnessReport:
    """Comprehensive report on information truthfulness"""
    query: str
    verified_claims: List[Claim]
    disputed_claims: List[Claim]
    unverified_claims: List[Claim]
    bias_assessment: Dict[str, Any]
    confidence_score: float
    reasoning_path: List[str]
    falsification_attempts: List[FalsificationAttempt]
    source_diversity: Dict[str, int]
    timestamp: datetime

class TruthfulnessEngine:
    """
    Engine for truthful, bias-resistant reasoning that:
    1. Cross-checks multiple sources
    2. Uses falsification loop to test claims
    3. Outputs confidence scores with reasoning paths
    4. Detects and mitigates various types of bias
    """
    
    def __init__(self):
        self.source_credibility_db = {}
        self.bias_patterns = self._load_bias_patterns()
        self.domain_authorities = {}
        self.falsification_strategies = self._load_falsification_strategies()
        
    async def verify_information(
        self,
        query: str,
        initial_sources: List[str] = None,
        min_sources: int = 5,
        max_sources: int = 20
    ) -> TruthfulnessReport:
        """
        Main method to verify information with cross-checking and falsification
        """
        logger.info(f"Starting verification for query: {query}")
        
        # Step 1: Gather diverse sources
        sources = await self._gather_diverse_sources(query, initial_sources, min_sources, max_sources)
        
        # Step 2: Extract and categorize claims
        claims = await self._extract_claims(sources, query)
        
        # Step 3: Cross-check claims across sources
        cross_checked_claims = await self._cross_check_claims(claims, sources)
        
        # Step 4: Apply falsification loop
        falsification_attempts = await self._apply_falsification_loop(cross_checked_claims)
        
        # Step 5: Assess bias
        bias_assessment = await self._assess_bias(sources, cross_checked_claims)
        
        # Step 6: Calculate final confidence and categorize claims
        verified_claims, disputed_claims, unverified_claims = await self._categorize_claims(
            cross_checked_claims, falsification_attempts
        )
        
        # Step 7: Generate reasoning path
        reasoning_path = self._generate_reasoning_path(
            sources, cross_checked_claims, falsification_attempts, bias_assessment
        )
        
        # Step 8: Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(
            verified_claims, disputed_claims, bias_assessment
        )
        
        # Step 9: Analyze source diversity
        source_diversity = self._analyze_source_diversity(sources)
        
        return TruthfulnessReport(
            query=query,
            verified_claims=verified_claims,
            disputed_claims=disputed_claims,
            unverified_claims=unverified_claims,
            bias_assessment=bias_assessment,
            confidence_score=overall_confidence,
            reasoning_path=reasoning_path,
            falsification_attempts=falsification_attempts,
            source_diversity=source_diversity,
            timestamp=datetime.now()
        )
    
    async def _gather_diverse_sources(
        self,
        query: str,
        initial_sources: List[str],
        min_sources: int,
        max_sources: int
    ) -> List[Source]:
        """Gather diverse sources from different types and perspectives"""
        
        sources = []
        
        # Process initial sources if provided
        if initial_sources:
            for url in initial_sources:
                source = await self._process_source(url, query)
                if source:
                    sources.append(source)
        
        # Search for additional sources from different categories
        search_strategies = [
            ("academic", f"site:edu {query}"),
            ("government", f"site:gov {query}"),
            ("news", f"{query} news"),
            ("encyclopedia", f"{query} wikipedia OR britannica"),
            ("expert", f"{query} expert opinion research")
        ]
        
        for source_type, search_query in search_strategies:
            if len(sources) >= max_sources:
                break
                
            new_sources = await self._search_sources(search_query, source_type, 3)
            sources.extend(new_sources)
        
        # Ensure minimum diversity
        if len(sources) < min_sources:
            logger.warning(f"Could not gather minimum sources ({len(sources)} < {min_sources})")
        
        return sources[:max_sources]
    
    async def _process_source(self, url: str, query: str) -> Optional[Source]:
        """Process a single source and extract relevant information"""
        
        try:
            # Get cached credibility score
            credibility_score = self.source_credibility_db.get(url, 0.5)
            
            # Determine source type from URL
            source_type = self._classify_source_type(url)
            
            # Get domain authority
            domain = urlparse(url).netloc
            domain_authority = self.domain_authorities.get(domain, 0.5)
            
            # For demo purposes, create synthetic content
            # In real implementation, this would scrape/fetch actual content
            content = f"Sample content from {url} related to {query}"
            title = f"Article about {query}"
            
            # Detect bias indicators
            bias_indicators = self._detect_bias_indicators(content, url)
            
            return Source(
                url=url,
                title=title,
                content=content,
                source_type=source_type,
                credibility_score=credibility_score,
                bias_indicators=bias_indicators,
                publication_date=datetime.now(),
                author="Unknown",
                domain_authority=domain_authority
            )
            
        except Exception as e:
            logger.error(f"Error processing source {url}: {e}")
            return None
    
    async def _search_sources(self, query: str, source_type: str, limit: int) -> List[Source]:
        """Search for sources using various strategies"""
        
        # In real implementation, this would use search APIs
        # For demo, return synthetic sources
        sources = []
        
        for i in range(limit):
            url = f"https://example-{source_type}-{i}.com/article"
            source = await self._process_source(url, query)
            if source:
                sources.append(source)
        
        return sources
    
    def _classify_source_type(self, url: str) -> SourceType:
        """Classify source type based on URL"""
        
        domain = urlparse(url).netloc.lower()
        
        if '.edu' in domain:
            return SourceType.ACADEMIC
        elif '.gov' in domain:
            return SourceType.GOVERNMENT
        elif any(news in domain for news in ['news', 'times', 'post', 'guardian', 'bbc']):
            return SourceType.NEWS
        elif 'wikipedia' in domain or 'britannica' in domain:
            return SourceType.ENCYCLOPEDIA
        elif any(social in domain for social in ['twitter', 'facebook', 'linkedin', 'reddit']):
            return SourceType.SOCIAL
        else:
            return SourceType.COMMERCIAL
    
    def _detect_bias_indicators(self, content: str, url: str) -> List[BiasType]:
        """Detect potential bias indicators in content"""
        
        bias_indicators = []
        content_lower = content.lower()
        
        # Check for confirmation bias patterns
        confirmation_patterns = [
            r'\\b(obviously|clearly|undoubtedly|without question)\\b',
            r'\\b(everyone knows|common sense|it\'s clear that)\\b'
        ]
        
        for pattern in confirmation_patterns:
            if re.search(pattern, content_lower):
                bias_indicators.append(BiasType.CONFIRMATION)
                break
        
        # Check for selection bias
        if any(word in content_lower for word in ['cherry-pick', 'selective', 'biased sample']):
            bias_indicators.append(BiasType.SELECTION)
        
        # Check for authority bias
        authority_patterns = [
            r'\\b(expert says|according to authorities|official statement)\\b',
            r'\\b(prestigious|renowned|famous) .* (says|claims|argues)\\b'
        ]
        
        for pattern in authority_patterns:
            if re.search(pattern, content_lower):
                bias_indicators.append(BiasType.AUTHORITY)
                break
        
        # Check for recency bias
        if any(word in content_lower for word in ['latest', 'breaking', 'just announced', 'trending']):
            bias_indicators.append(BiasType.RECENCY)
        
        return bias_indicators
    
    async def _extract_claims(self, sources: List[Source], query: str) -> List[Claim]:
        """Extract factual claims from sources"""
        
        claims = []
        claim_map = {}
        
        for source in sources:
            # Extract claims from content
            # In real implementation, this would use NLP to extract factual statements
            
            # For demo, create synthetic claims based on source content
            source_claims = [
                f"Claim from {source.source_type.value}: Statement about {query}",
                f"Secondary claim from {source.source_type.value}: Additional information about {query}"
            ]
            
            for claim_text in source_claims:
                claim_id = hashlib.md5(claim_text.encode()).hexdigest()[:8]
                
                if claim_id not in claim_map:
                    claim_map[claim_id] = Claim(
                        statement=claim_text,
                        claim_id=claim_id,
                        supporting_sources=[],
                        contradicting_sources=[],
                        confidence_score=0.0,
                        verification_status="unverified"
                    )
                
                # Add this source as supporting (simplified logic)
                claim_map[claim_id].supporting_sources.append(source)
        
        return list(claim_map.values())
    
    async def _cross_check_claims(self, claims: List[Claim], sources: List[Source]) -> List[Claim]:
        """Cross-check claims against multiple sources"""
        
        for claim in claims:
            # Calculate support strength
            total_credibility = sum(s.credibility_score for s in claim.supporting_sources)
            source_diversity = len(set(s.source_type for s in claim.supporting_sources))
            
            # Look for contradicting sources
            for source in sources:
                if source not in claim.supporting_sources:
                    # Simplified logic: some sources contradict based on type
                    if (source.source_type == SourceType.SOCIAL and 
                        any(s.source_type == SourceType.ACADEMIC for s in claim.supporting_sources)):
                        claim.contradicting_sources.append(source)
            
            # Calculate confidence based on support strength and contradictions
            contradiction_penalty = len(claim.contradicting_sources) * 0.1
            diversity_bonus = min(source_diversity * 0.2, 0.6)
            
            confidence = min(0.95, (total_credibility / len(claim.supporting_sources)) + diversity_bonus - contradiction_penalty)
            claim.confidence_score = max(0.1, confidence)
            
            # Set verification status
            if claim.confidence_score > 0.8 and not claim.contradicting_sources:
                claim.verification_status = "verified"
            elif claim.contradicting_sources:
                claim.verification_status = "disputed"
            else:
                claim.verification_status = "unverified"
        
        return claims
    
    async def _apply_falsification_loop(self, claims: List[Claim]) -> List[FalsificationAttempt]:
        """Apply falsification loop to test claims"""
        
        falsification_attempts = []
        
        for claim in claims:
            if claim.verification_status in ["verified", "disputed"]:
                # Generate counter-hypotheses
                counter_hypotheses = self._generate_counter_hypotheses(claim)
                
                for counter_hypothesis in counter_hypotheses:
                    # Look for evidence supporting the counter-hypothesis
                    counter_evidence = await self._find_counter_evidence(claim, counter_hypothesis)
                    
                    if counter_evidence:
                        falsification_strength = self._calculate_falsification_strength(
                            claim, counter_evidence
                        )
                        
                        reasoning = self._generate_falsification_reasoning(
                            claim, counter_hypothesis, counter_evidence
                        )
                        
                        attempt = FalsificationAttempt(
                            claim_id=claim.claim_id,
                            counter_hypothesis=counter_hypothesis,
                            counter_evidence=counter_evidence,
                            falsification_strength=falsification_strength,
                            reasoning=reasoning
                        )
                        
                        falsification_attempts.append(attempt)
                        
                        # Adjust claim confidence based on falsification strength
                        if falsification_strength > 0.5:
                            claim.confidence_score *= (1 - falsification_strength * 0.3)
                            if claim.verification_status == "verified" and falsification_strength > 0.7:
                                claim.verification_status = "disputed"
        
        return falsification_attempts
    
    def _generate_counter_hypotheses(self, claim: Claim) -> List[str]:
        """Generate counter-hypotheses for a claim"""
        
        statement = claim.statement.lower()
        counter_hypotheses = []
        
        # Simple patterns for generating counter-hypotheses
        if "increases" in statement:
            counter_hypotheses.append(statement.replace("increases", "decreases"))
        elif "decreases" in statement:
            counter_hypotheses.append(statement.replace("decreases", "increases"))
        elif "causes" in statement:
            counter_hypotheses.append(statement.replace("causes", "does not cause"))
        elif "effective" in statement:
            counter_hypotheses.append(statement.replace("effective", "ineffective"))
        
        # Add general negation
        if not any(neg in statement for neg in ["not", "no", "never"]):
            counter_hypotheses.append(f"It is not true that {statement}")
        
        return counter_hypotheses[:3]  # Limit to top 3
    
    async def _find_counter_evidence(self, claim: Claim, counter_hypothesis: str) -> List[Source]:
        """Find evidence that supports the counter-hypothesis"""
        
        # In real implementation, this would search for contradicting evidence
        # For demo, simulate finding counter-evidence for some claims
        
        counter_evidence = []
        
        # Simulate finding counter-evidence based on claim confidence
        if claim.confidence_score < 0.7:  # Lower confidence claims more likely to have counter-evidence
            # Create synthetic counter-evidence source
            counter_source = Source(
                url="https://counter-evidence.com/article",
                title=f"Counter-evidence for: {claim.statement[:50]}...",
                content=f"Evidence supporting: {counter_hypothesis}",
                source_type=SourceType.ACADEMIC,
                credibility_score=0.8,
                bias_indicators=[],
                publication_date=datetime.now(),
                author="Counter Expert",
                domain_authority=0.7
            )
            counter_evidence.append(counter_source)
        
        return counter_evidence
    
    def _calculate_falsification_strength(self, claim: Claim, counter_evidence: List[Source]) -> float:
        """Calculate how strong the falsification attempt is"""
        
        if not counter_evidence:
            return 0.0
        
        # Calculate strength based on counter-evidence quality
        total_credibility = sum(source.credibility_score for source in counter_evidence)
        avg_credibility = total_credibility / len(counter_evidence)
        
        # Factor in source diversity
        source_types = set(source.source_type for source in counter_evidence)
        diversity_factor = min(len(source_types) * 0.2, 0.8)
        
        # Compare against original claim strength
        claim_strength = claim.confidence_score
        
        falsification_strength = min(0.9, avg_credibility + diversity_factor - claim_strength * 0.3)
        
        return max(0.0, falsification_strength)
    
    def _generate_falsification_reasoning(
        self, 
        claim: Claim, 
        counter_hypothesis: str, 
        counter_evidence: List[Source]
    ) -> str:
        """Generate reasoning for falsification attempt"""
        
        reasoning_parts = [
            f"Testing falsification of claim: '{claim.statement[:100]}...'",
            f"Counter-hypothesis: {counter_hypothesis}",
            f"Found {len(counter_evidence)} sources supporting counter-hypothesis:",
        ]
        
        for i, source in enumerate(counter_evidence, 1):
            reasoning_parts.append(
                f"  {i}. {source.source_type.value} source "
                f"(credibility: {source.credibility_score:.2f}): {source.title}"
            )
        
        reasoning_parts.append(
            f"Falsification strength suggests "
            f"{'strong' if self._calculate_falsification_strength(claim, counter_evidence) > 0.6 else 'moderate'} "
            f"challenge to original claim."
        )
        
        return "\\n".join(reasoning_parts)
    
    async def _assess_bias(self, sources: List[Source], claims: List[Claim]) -> Dict[str, Any]:
        """Assess potential bias in sources and claims"""
        
        bias_assessment = {
            'overall_bias_risk': 'low',
            'detected_biases': [],
            'source_balance': {},
            'recommendations': []
        }
        
        # Analyze bias indicators across sources
        all_bias_indicators = []
        for source in sources:
            all_bias_indicators.extend(source.bias_indicators)
        
        # Count bias types
        bias_counts = {}
        for bias in all_bias_indicators:
            bias_counts[bias.value] = bias_counts.get(bias.value, 0) + 1
        
        bias_assessment['detected_biases'] = bias_counts
        
        # Assess source type balance
        source_type_counts = {}
        for source in sources:
            source_type_counts[source.source_type.value] = source_type_counts.get(source.source_type.value, 0) + 1
        
        bias_assessment['source_balance'] = source_type_counts
        
        # Calculate overall bias risk
        total_bias_indicators = len(all_bias_indicators)
        bias_risk_score = min(total_bias_indicators / len(sources), 1.0)
        
        if bias_risk_score > 0.7:
            bias_assessment['overall_bias_risk'] = 'high'
        elif bias_risk_score > 0.4:
            bias_assessment['overall_bias_risk'] = 'medium'
        
        # Generate recommendations
        recommendations = []
        
        if 'social' in source_type_counts and source_type_counts['social'] > len(sources) * 0.3:
            recommendations.append("Consider reducing reliance on social media sources")
        
        if len(source_type_counts) < 3:
            recommendations.append("Seek more diverse source types for better balance")
        
        if bias_counts.get('confirmation_bias', 0) > 2:
            recommendations.append("Be aware of potential confirmation bias in sources")
        
        bias_assessment['recommendations'] = recommendations
        
        return bias_assessment
    
    async def _categorize_claims(
        self, 
        claims: List[Claim], 
        falsification_attempts: List[FalsificationAttempt]
    ) -> Tuple[List[Claim], List[Claim], List[Claim]]:
        """Categorize claims based on verification status and falsification results"""
        
        # Adjust claims based on falsification attempts
        falsification_map = {attempt.claim_id: attempt for attempt in falsification_attempts}
        
        for claim in claims:
            if claim.claim_id in falsification_map:
                attempt = falsification_map[claim.claim_id]
                if attempt.falsification_strength > 0.7:
                    claim.verification_status = "disputed"
                elif attempt.falsification_strength > 0.4 and claim.verification_status == "verified":
                    claim.verification_status = "unverified"
        
        verified_claims = [claim for claim in claims if claim.verification_status == "verified"]
        disputed_claims = [claim for claim in claims if claim.verification_status == "disputed"]
        unverified_claims = [claim for claim in claims if claim.verification_status == "unverified"]
        
        return verified_claims, disputed_claims, unverified_claims
    
    def _generate_reasoning_path(
        self,
        sources: List[Source],
        claims: List[Claim],
        falsification_attempts: List[FalsificationAttempt],
        bias_assessment: Dict[str, Any]
    ) -> List[str]:
        """Generate detailed reasoning path"""
        
        reasoning_steps = []
        
        reasoning_steps.append(f"1. Gathered {len(sources)} sources from {len(set(s.source_type for s in sources))} different types")
        
        reasoning_steps.append(f"2. Extracted and analyzed {len(claims)} factual claims")
        
        verified_count = sum(1 for claim in claims if claim.verification_status == "verified")
        disputed_count = sum(1 for claim in claims if claim.verification_status == "disputed")
        
        reasoning_steps.append(
            f"3. Cross-verification resulted in {verified_count} verified claims, "
            f"{disputed_count} disputed claims"
        )
        
        reasoning_steps.append(f"4. Applied {len(falsification_attempts)} falsification attempts to test claims")
        
        if bias_assessment['detected_biases']:
            bias_types = list(bias_assessment['detected_biases'].keys())
            reasoning_steps.append(f"5. Detected potential biases: {', '.join(bias_types)}")
        else:
            reasoning_steps.append("5. No significant bias patterns detected")
        
        reasoning_steps.append(
            f"6. Overall bias risk assessed as {bias_assessment['overall_bias_risk']}"
        )
        
        return reasoning_steps
    
    def _calculate_overall_confidence(
        self,
        verified_claims: List[Claim],
        disputed_claims: List[Claim],
        bias_assessment: Dict[str, Any]
    ) -> float:
        """Calculate overall confidence in the information"""
        
        if not verified_claims and not disputed_claims:
            return 0.1
        
        # Base confidence from verified claims
        if verified_claims:
            avg_verified_confidence = sum(claim.confidence_score for claim in verified_claims) / len(verified_claims)
        else:
            avg_verified_confidence = 0.0
        
        # Penalty for disputed claims
        dispute_penalty = len(disputed_claims) * 0.1
        
        # Penalty for bias risk
        bias_risk_penalty = {
            'low': 0.0,
            'medium': 0.1,
            'high': 0.2
        }.get(bias_assessment['overall_bias_risk'], 0.1)
        
        overall_confidence = avg_verified_confidence - dispute_penalty - bias_risk_penalty
        
        return max(0.1, min(0.95, overall_confidence))
    
    def _analyze_source_diversity(self, sources: List[Source]) -> Dict[str, int]:
        """Analyze diversity of sources"""
        
        diversity = {
            'total_sources': len(sources),
            'unique_domains': len(set(urlparse(s.url).netloc for s in sources)),
            'source_types': len(set(s.source_type for s in sources)),
            'high_credibility_sources': sum(1 for s in sources if s.credibility_score > 0.7)
        }
        
        # Add breakdown by source type
        for source in sources:
            key = f"{source.source_type.value}_count"
            diversity[key] = diversity.get(key, 0) + 1
        
        return diversity
    
    def _load_bias_patterns(self) -> Dict[str, List[str]]:
        """Load patterns for detecting bias"""
        return {
            'confirmation': [
                r'\\bobviously\\b', r'\\bclearly\\b', r'\\bundoubtedly\\b',
                r'\\beveryone knows\\b', r'\\bcommon sense\\b'
            ],
            'authority': [
                r'\\bexpert says\\b', r'\\baccording to authorities\\b',
                r'\\bofficial statement\\b', r'\\bprestigious.*says\\b'
            ],
            'emotional': [
                r'\\bshocking\\b', r'\\balarming\\b', r'\\bdevastating\\b',
                r'\\bincredible\\b', r'\\bamazing\\b'
            ]
        }
    
    def _load_falsification_strategies(self) -> List[str]:
        """Load strategies for falsification attempts"""
        return [
            "Look for contradictory evidence",
            "Test alternative explanations",
            "Check for sample bias",
            "Verify methodology",
            "Examine conflicting studies",
            "Question underlying assumptions"
        ]