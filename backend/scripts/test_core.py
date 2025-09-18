#!/usr/bin/env python3
"""
Simple test script to verify the Perplexity AI clone is working
"""

import sys
import os
import asyncio
import json

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

async def test_core_functionality():
    """Test the core RAG functionality"""
    print("🧪 Testing Perplexity AI Clone Core Functionality...")
    print("=" * 50)
    
    try:
        # Test imports
        print("1. Testing imports...")
        from app.core.rag_pipeline import RAGPipeline
        from app.core.search_engine import SearchEngine
        from app.core.vector_store import VectorStore
        print("   ✅ All core imports successful")
        
        # Test search engine
        print("2. Testing search engine...")
        search_engine = SearchEngine()
        results = await search_engine.search("artificial intelligence", max_results=3)
        print(f"   ✅ Search returned {len(results)} results")
        
        # Test vector store
        print("3. Testing vector store...")
        vector_store = VectorStore()
        stats = await vector_store.get_stats()
        print(f"   ✅ Vector store initialized with {stats['total_documents']} documents")
        
        # Test RAG pipeline
        print("4. Testing RAG pipeline...")
        rag = RAGPipeline()
        result = await rag.search_and_generate(
            query="What is artificial intelligence?",
            max_results=3,
            include_citations=True
        )
        print(f"   ✅ RAG pipeline generated {len(result['answer'])} character response")
        print(f"   ✅ Found {len(result['sources'])} sources")
        print(f"   ✅ Generated {len(result['citations'])} citations")
        
        # Show sample result
        print("\n📋 Sample Query Result:")
        print("-" * 30)
        print(f"Query: What is artificial intelligence?")
        print(f"Answer: {result['answer'][:200]}...")
        print(f"Sources: {len(result['sources'])}")
        print(f"Citations: {len(result['citations'])}")
        print(f"Search Time: {result['search_time']:.2f}s")
        
        print("\n🎉 All tests passed! The Perplexity AI clone is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_core_functionality())
    sys.exit(0 if success else 1)