"""
Real-time content updater for keeping search index fresh
"""

import asyncio
import aiohttp
import feedparser
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
import json
import os

from app.core.web_scraper import WebScraper
from app.core.search_engine import SearchEngine
from app.models.search import Source
from app.utils.config import get_settings

logger = logging.getLogger(__name__)

class ContentUpdater:
    """
    Manages real-time content updates and index refreshing
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.search_engine = SearchEngine()
        self.update_sources = self._load_update_sources()
        self.last_update = datetime.utcnow()
        
    def _load_update_sources(self) -> Dict[str, List[str]]:
        """Load configured content sources"""
        # In production, load from configuration or database
        return {
            'rss_feeds': [
                'https://feeds.ycombinator.com/news.rss',
                'https://rss.cnn.com/rss/edition.rss',
                'https://feeds.techcrunch.com/TechCrunch/',
                'https://www.reddit.com/r/technology/.rss'
            ],
            'news_sites': [
                'https://news.ycombinator.com',
                'https://techcrunch.com',
                'https://arstechnica.com'
            ],
            'search_engines': [
                'https://www.google.com/search',
                'https://duckduckgo.com'
            ]
        }
    
    async def start_real_time_updates(self, interval_minutes: int = 30):
        """
        Start background task for real-time content updates
        """
        logger.info(f"Starting real-time updates with {interval_minutes} minute intervals")
        
        while True:
            try:
                await self.update_content()
                await asyncio.sleep(interval_minutes * 60)
            except Exception as e:
                logger.error(f"Real-time update failed: {str(e)}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry
    
    async def update_content(self) -> Dict[str, Any]:
        """
        Perform content update from all sources
        """
        logger.info("Starting content update cycle")
        start_time = datetime.utcnow()
        
        results = {
            'rss_updates': 0,
            'web_updates': 0,
            'errors': [],
            'start_time': start_time,
            'duration': 0
        }
        
        try:
            # Update from RSS feeds
            rss_results = await self._update_from_rss_feeds()
            results['rss_updates'] = rss_results['count']
            results['errors'].extend(rss_results['errors'])
            
            # Update from web scraping
            web_results = await self._update_from_web_scraping()
            results['web_updates'] = web_results['count']
            results['errors'].extend(web_results['errors'])
            
            # Update trending topics
            await self._update_trending_topics()
            
            self.last_update = datetime.utcnow()
            results['duration'] = (self.last_update - start_time).total_seconds()
            
            logger.info(f"Content update completed: {results['rss_updates']} RSS + {results['web_updates']} web updates")
            
        except Exception as e:
            logger.error(f"Content update failed: {str(e)}")
            results['errors'].append(str(e))
        
        return results
    
    async def _update_from_rss_feeds(self) -> Dict[str, Any]:
        """Update content from RSS feeds"""
        results = {'count': 0, 'errors': []}
        
        for feed_url in self.update_sources['rss_feeds']:
            try:
                feed_data = await self._fetch_rss_feed(feed_url)
                
                for entry in feed_data.entries[:10]:  # Process latest 10 items
                    if await self._is_new_content(entry.link):
                        source = await self._create_source_from_rss_entry(entry)
                        if source:
                            await self.search_engine.index_document(
                                url=source.url,
                                title=source.title,
                                content=source.content
                            )
                            results['count'] += 1
                
            except Exception as e:
                error_msg = f"RSS feed error ({feed_url}): {str(e)}"
                logger.error(error_msg)
                results['errors'].append(error_msg)
        
        return results
    
    async def _update_from_web_scraping(self) -> Dict[str, Any]:
        """Update content through web scraping"""
        results = {'count': 0, 'errors': []}
        
        # Get trending URLs to scrape
        trending_urls = await self._get_trending_urls()
        
        if trending_urls:
            async with WebScraper() as scraper:
                sources = await scraper.scrape_urls(trending_urls[:5])  # Limit to 5 URLs
                
                for source in sources:
                    if source and await self._is_new_content(source.url):
                        await self.search_engine.index_document(
                            url=source.url,
                            title=source.title,
                            content=source.content
                        )
                        results['count'] += 1
        
        return results
    
    async def _fetch_rss_feed(self, feed_url: str) -> Any:
        """Fetch and parse RSS feed"""
        async with aiohttp.ClientSession() as session:
            async with session.get(feed_url) as response:
                if response.status == 200:
                    content = await response.text()
                    return feedparser.parse(content)
                else:
                    raise Exception(f"HTTP {response.status}")
    
    async def _create_source_from_rss_entry(self, entry: Any) -> Optional[Source]:
        """Create Source object from RSS entry"""
        try:
            # Parse publish date
            publish_date = None
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                publish_date = datetime(*entry.published_parsed[:6])
            
            # Extract content
            content = ""
            if hasattr(entry, 'summary'):
                content = entry.summary
            elif hasattr(entry, 'description'):
                content = entry.description
            
            # If content is too short, try to scrape the full article
            if len(content) < 200:
                async with WebScraper() as scraper:
                    scraped_source = await scraper.scrape_url(entry.link)
                    if scraped_source:
                        content = scraped_source.content
            
            if not content or len(content) < 50:
                return None
            
            return Source(
                url=entry.link,
                title=entry.title,
                content=content,
                relevance_score=0.8,  # RSS content is generally high quality
                domain=entry.link.split('/')[2] if '/' in entry.link else '',
                publish_date=publish_date
            )
            
        except Exception as e:
            logger.error(f"Error creating source from RSS entry: {str(e)}")
            return None
    
    async def _get_trending_urls(self) -> List[str]:
        """Get trending URLs to scrape"""
        # Mock implementation - in production, integrate with trending APIs
        trending_urls = [
            "https://techcrunch.com/latest/",
            "https://arstechnica.com/",
            "https://www.theverge.com/"
        ]
        
        return trending_urls
    
    async def _is_new_content(self, url: str) -> bool:
        """Check if content is new and should be indexed"""
        # Simple implementation - check if we've seen this URL recently
        # In production, use database to track indexed URLs
        cache_file = "data/indexed_urls.json"
        
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    indexed_urls = json.load(f)
            else:
                indexed_urls = {}
            
            # Check if URL was indexed in the last 24 hours
            if url in indexed_urls:
                last_indexed = datetime.fromisoformat(indexed_urls[url])
                if datetime.utcnow() - last_indexed < timedelta(hours=24):
                    return False
            
            # Mark URL as indexed
            indexed_urls[url] = datetime.utcnow().isoformat()
            
            # Save updated cache (keep only last 1000 URLs)
            if len(indexed_urls) > 1000:
                # Keep only recent entries
                recent_entries = {
                    k: v for k, v in sorted(indexed_urls.items(), 
                    key=lambda x: x[1], reverse=True)[:1000]
                }
                indexed_urls = recent_entries
            
            with open(cache_file, 'w') as f:
                json.dump(indexed_urls, f)
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking content freshness: {str(e)}")
            return True  # Default to indexing if check fails
    
    async def _update_trending_topics(self):
        """Update trending topics for better search relevance"""
        # Mock implementation - in production, analyze search patterns
        trending_topics = [
            "artificial intelligence",
            "machine learning",
            "climate change",
            "cryptocurrency",
            "space exploration"
        ]
        
        # Store trending topics for query enhancement
        cache_file = "data/trending_topics.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    'topics': trending_topics,
                    'updated_at': datetime.utcnow().isoformat()
                }, f)
        except Exception as e:
            logger.error(f"Error updating trending topics: {str(e)}")
    
    async def get_update_status(self) -> Dict[str, Any]:
        """Get current update status"""
        return {
            'last_update': self.last_update.isoformat(),
            'sources_configured': len(self.update_sources['rss_feeds']) + len(self.update_sources['news_sites']),
            'update_interval': '30 minutes',
            'status': 'active'
        }