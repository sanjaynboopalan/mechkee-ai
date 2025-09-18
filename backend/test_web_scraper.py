#!/usr/bin/env python3
"""
Test script for the web scraper functionality
"""

import asyncio
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.core.web_scraper import WebScraper

async def test_web_scraper():
    """Test the web scraper functionality"""
    print("🔍 Testing Web Scraper...")
    
    scraper = WebScraper()
    
    # Test queries
    test_queries = [
        "SpaceX latest news",
        "Python programming",
        "artificial intelligence"
    ]
    
    for query in test_queries:
        print(f"\n📰 Testing query: '{query}'")
        try:
            sources = await scraper.search_and_scrape(query, max_sources=3)
            print(f"✅ Found {len(sources)} sources")
            
            for i, source in enumerate(sources, 1):
                print(f"  {i}. {source.title}")
                print(f"     URL: {source.url}")
                print(f"     Content: {source.content[:100]}...")
                print()
        
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Close the session
    if scraper.session:
        await scraper.session.close()

if __name__ == "__main__":
    asyncio.run(test_web_scraper())