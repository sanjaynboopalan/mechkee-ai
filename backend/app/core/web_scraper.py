"""
Advanced Web Scraper for real-time content retrieval like Perplexity
"""

import asyncio
import aiohttp
import time
import json
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse, quote
from bs4 import BeautifulSoup
import logging
from datetime import datetime, timedelta
import re

from app.utils.config import get_settings
from app.models.search import Source

logger = logging.getLogger(__name__)

class WebScraper:
    """
    Intelligent web scraper for retrieving fresh content like Perplexity AI
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.session = None
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        # Rate limiting
        self.last_request_time = {}
        self.min_delay = 1.0  # Minimum delay between requests to same domain
        
        # Content filters
        self.quality_indicators = [
            'article', 'main', 'content', 'post', 'entry', '.content-body',
            '.article-content', '.post-content', '.entry-content', 'p'
        ]
        
        self.noise_selectors = [
            'nav', 'header', 'footer', 'sidebar', 'aside',
            '.nav', '.header', '.footer', '.sidebar', '.aside',
            '.advertisement', '.ads', '.popup', '.modal', 'script', 'style'
        ]
        
        # Trusted sources for real-time content
        self.trusted_sources = {
            'wikipedia': 'https://en.wikipedia.org/wiki/',
            'news': [
                'https://www.reuters.com/search/?query=',
                'https://www.bbc.com/search?q=',
                'https://www.cnn.com/search?q='
            ],
            'academic': [
                'https://scholar.google.com/scholar?q=',
                'https://arxiv.org/search/?query='
            ],
            'tech': [
                'https://stackoverflow.com/search?q=',
                'https://github.com/search?q='
            ]
        }
    
    async def search_and_scrape(self, query: str, max_sources: int = 10) -> List[Source]:
        """
        Search for and scrape real-time content like Perplexity AI
        """
        sources = []
        
        try:
            if not self.session:
                await self._create_session()
            
            # Get search results from multiple sources
            search_tasks = [
                self._search_wikipedia(query),
                self._search_news(query),
                self._search_tech_sources(query)
            ]
            
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # Combine all URLs found
            all_urls = []
            for result in search_results:
                if isinstance(result, list):
                    all_urls.extend(result)
            
            # Scrape content from found URLs
            scrape_tasks = [self._scrape_url(url) for url in all_urls[:max_sources]]
            scraped_content = await asyncio.gather(*scrape_tasks, return_exceptions=True)
            
            # Process scraped content
            for content in scraped_content:
                if isinstance(content, Source) and content.content.strip():
                    sources.append(content)
            
            return sources[:max_sources]
            
        except Exception as e:
            logger.error(f"Search and scrape failed: {str(e)}")
            return []
    
    async def _search_wikipedia(self, query: str) -> List[str]:
        """Search Wikipedia for relevant articles"""
        try:
            # Use Wikipedia API to search for articles
            wiki_search_url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={quote(query)}&limit=3&format=json"
            
            async with self.session.get(wiki_search_url) as response:
                if response.status == 200:
                    data = await response.json()
                    if len(data) > 3 and data[3]:
                        logger.info(f"✅ Found {len(data[3])} Wikipedia articles for '{query}'")
                        return data[3]  # URLs
            
        except Exception as e:
            logger.error(f"Wikipedia search failed: {str(e)}")
        
        return []
    
    async def _search_news(self, query: str) -> List[str]:
        """Search news sources for relevant articles"""
        urls = []
        try:
            # For now, let's use direct construction of news URLs
            # This is more reliable than depending on external APIs
            
            # Construct search URLs for major news sources
            news_searches = [
                f"https://www.bbc.com/search?q={quote(query)}",
                f"https://www.reuters.com/search/news?blob={quote(query)}",
            ]
            
            # Add the search URLs directly (they will be scraped for actual content)
            urls.extend(news_searches)
            
            logger.info(f"✅ Prepared {len(urls)} news search URLs for '{query}'")
            
        except Exception as e:
            logger.error(f"News search failed: {str(e)}")
        
        return urls
    
    async def _search_tech_sources(self, query: str) -> List[str]:
        """Search technical sources"""
        urls = []
        
        # For tech queries, construct direct URLs
        if any(term in query.lower() for term in ['programming', 'code', 'python', 'javascript', 'ai', 'machine learning']):
            # Add some tech-focused URLs
            tech_urls = [
                f"https://stackoverflow.com/search?q={quote(query)}",
                f"https://docs.python.org/3/search.html?q={quote(query)}"
            ]
            urls.extend(tech_urls)
        
        return urls
    
    async def _scrape_url(self, url: str) -> Optional[Source]:
        """Scrape content from a specific URL"""
        try:
            domain = urlparse(url).netloc
            
            # Rate limiting
            if domain in self.last_request_time:
                time_since_last = time.time() - self.last_request_time[domain]
                if time_since_last < self.min_delay:
                    await asyncio.sleep(self.min_delay - time_since_last)
            
            self.last_request_time[domain] = time.time()
            
            async with self.session.get(url, timeout=10) as response:
                if response.status == 200:
                    html = await response.text()
                    return self._extract_content(html, url)
        
        except Exception as e:
            logger.error(f"Failed to scrape {url}: {str(e)}")
        
        return None
    
    def _extract_content(self, html: str, url: str) -> Optional[Source]:
        """Extract clean content from HTML"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove noise elements
            for selector in self.noise_selectors:
                for element in soup.select(selector):
                    element.decompose()
            
            # Extract title
            title_elem = soup.find('title')
            title = title_elem.get_text().strip() if title_elem else ""
            
            # Extract main content
            content_text = ""
            
            # Try to find main content area
            main_content = None
            for indicator in self.quality_indicators:
                main_content = soup.select_one(indicator)
                if main_content:
                    break
            
            if main_content:
                # Extract paragraphs from main content
                paragraphs = main_content.find_all('p')
                content_text = " ".join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
            else:
                # Fallback: get all paragraphs
                paragraphs = soup.find_all('p')
                content_text = " ".join([p.get_text().strip() for p in paragraphs[:5] if p.get_text().strip()])
            
            # Clean and limit content
            content_text = re.sub(r'\s+', ' ', content_text).strip()
            content_text = content_text[:2000]  # Limit content length
            
            if len(content_text) > 100:  # Only return if we have substantial content
                return Source(
                    url=url,
                    title=title,
                    content=content_text,
                    relevance_score=0.8,
                    domain=urlparse(url).netloc,
                    publish_date=datetime.now()
                )
        
        except Exception as e:
            logger.error(f"Content extraction failed for {url}: {str(e)}")
        
        return None
    
    async def _create_session(self):
        """Create aiohttp session"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=10)
            self.session = aiohttp.ClientSession(
                headers=self.headers,
                timeout=timeout
            )
    
    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()
    
    async def __aenter__(self):
        """Async context manager entry"""
        connector = aiohttp.TCPConnector(limit=self.settings.max_concurrent_requests)
        timeout = aiohttp.ClientTimeout(total=self.settings.request_timeout)
        self.session = aiohttp.ClientSession(
            headers=self.headers,
            connector=connector,
            timeout=timeout
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def scrape_urls(self, urls: List[str]) -> List[Source]:
        """
        Scrape multiple URLs concurrently
        """
        tasks = [self.scrape_url(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        sources = []
        for result in results:
            if isinstance(result, Source):
                sources.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Scraping failed: {str(result)}")
        
        return sources
    
    async def scrape_url(self, url: str) -> Optional[Source]:
        """
        Scrape content from a single URL
        """
        try:
            # Rate limiting
            await self._rate_limit(url)
            
            # Fetch content
            async with self.session.get(url) as response:
                if response.status != 200:
                    logger.warning(f"HTTP {response.status} for {url}")
                    return None
                
                html_content = await response.text()
            
            # Parse content
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract metadata
            title = self._extract_title(soup)
            content = self._extract_main_content(soup)
            publish_date = self._extract_publish_date(soup)
            author = self._extract_author(soup)
            
            if not content or len(content.strip()) < 100:
                logger.warning(f"Insufficient content from {url}")
                return None
            
            # Calculate relevance score (basic quality assessment)
            relevance_score = self._calculate_quality_score(soup, content)
            
            return Source(
                url=url,
                title=title,
                content=content,
                relevance_score=relevance_score,
                domain=urlparse(url).netloc,
                publish_date=publish_date,
                author=author
            )
            
        except Exception as e:
            logger.error(f"Failed to scrape {url}: {str(e)}")
            return None
    
    async def _rate_limit(self, url: str):
        """Apply rate limiting per domain"""
        domain = urlparse(url).netloc
        
        if domain in self.last_request_time:
            elapsed = time.time() - self.last_request_time[domain]
            if elapsed < self.min_delay:
                await asyncio.sleep(self.min_delay - elapsed)
        
        self.last_request_time[domain] = time.time()
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title"""
        # Try multiple sources for title
        title_sources = [
            soup.find('title'),
            soup.find('h1'),
            soup.find('meta', {'property': 'og:title'}),
            soup.find('meta', {'name': 'twitter:title'})
        ]
        
        for source in title_sources:
            if source:
                if source.name == 'meta':
                    title = source.get('content', '')
                else:
                    title = source.get_text(strip=True)
                
                if title:
                    return title[:200]  # Limit title length
        
        return "Untitled"
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from page"""
        # Remove noise elements
        for selector in self.noise_selectors:
            for element in soup.select(selector):
                element.decompose()
        
        # Try to find main content area
        content_candidates = []
        
        # Look for semantic content areas
        for selector in self.quality_indicators:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text(strip=True)
                if len(text) > 100:  # Minimum content length
                    content_candidates.append((text, len(text)))
        
        # If no semantic areas found, use body
        if not content_candidates:
            body = soup.find('body')
            if body:
                text = body.get_text(strip=True)
                content_candidates.append((text, len(text)))
        
        # Return the longest content
        if content_candidates:
            content = max(content_candidates, key=lambda x: x[1])[0]
            return self._clean_text(content)
        
        return ""
    
    def _extract_publish_date(self, soup: BeautifulSoup) -> Optional[datetime]:
        """Extract publish date from page"""
        date_selectors = [
            'meta[property="article:published_time"]',
            'meta[name="publish_date"]',
            'meta[name="date"]',
            'time[datetime]',
            '.publish-date',
            '.date'
        ]
        
        for selector in date_selectors:
            element = soup.select_one(selector)
            if element:
                date_str = element.get('content') or element.get('datetime') or element.get_text(strip=True)
                try:
                    # Basic date parsing (extend as needed)
                    return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                except:
                    continue
        
        return None
    
    def _extract_author(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract author from page"""
        author_selectors = [
            'meta[name="author"]',
            'meta[property="article:author"]',
            '.author',
            '.byline',
            '.writer'
        ]
        
        for selector in author_selectors:
            element = soup.select_one(selector)
            if element:
                author = element.get('content') or element.get_text(strip=True)
                if author:
                    return author[:100]  # Limit author length
        
        return None
    
    def _calculate_quality_score(self, soup: BeautifulSoup, content: str) -> float:
        """Calculate content quality score"""
        score = 0.5  # Base score
        
        # Content length bonus
        if len(content) > 1000:
            score += 0.2
        elif len(content) > 500:
            score += 0.1
        
        # Semantic structure bonus
        if soup.find('article') or soup.find('main'):
            score += 0.1
        
        # Title quality
        title = soup.find('title')
        if title and len(title.get_text(strip=True)) > 10:
            score += 0.1
        
        # Meta description
        if soup.find('meta', {'name': 'description'}):
            score += 0.1
        
        return min(score, 1.0)
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        lines = [line.strip() for line in text.split('\n')]
        lines = [line for line in lines if line]
        
        # Join with single spaces
        cleaned = ' '.join(lines)
        
        # Limit content length
        return cleaned[:5000] if len(cleaned) > 5000 else cleaned