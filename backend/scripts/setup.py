"""
Setup script for Perplexity AI Clone
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the backend directory to Python path
backend_path = Path(__file__).parent.parent
sys.path.append(str(backend_path))

from app.utils.config import get_settings, validate_config
from app.core.vector_store import VectorStore

async def setup_database():
    """Setup database tables and indexes"""
    print("Setting up database...")
    
    # Create necessary directories
    settings = get_settings()
    
    # Create data directories
    data_dirs = [
        "data/embeddings",
        "data/uploads", 
        "data/chroma",
        "data/scraped",
        "data/indexes"
    ]
    
    for dir_path in data_dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    print("Database setup completed!")

async def setup_vector_store():
    """Initialize vector store"""
    print("Setting up vector store...")
    
    vector_store = VectorStore()
    stats = await vector_store.get_stats()
    print(f"Vector store initialized with {stats['total_documents']} documents")
    
    print("Vector store setup completed!")

async def seed_initial_data():
    """Seed initial data for testing"""
    print("Seeding initial data...")
    
    vector_store = VectorStore()
    
    # Add some sample documents
    sample_docs = [
        {
            "id": "sample_1",
            "content": "Artificial Intelligence is transforming how we process and understand information. Modern AI systems can analyze vast amounts of data and provide insights that were previously impossible to obtain.",
            "metadata": {
                "title": "AI and Information Processing",
                "source": "AI Research Blog",
                "type": "article"
            }
        },
        {
            "id": "sample_2", 
            "content": "Retrieval Augmented Generation combines the power of large language models with real-time information retrieval. This approach allows AI systems to provide accurate, up-to-date responses while maintaining conversational abilities.",
            "metadata": {
                "title": "Understanding RAG Systems",
                "source": "Tech Innovation Weekly",
                "type": "article"
            }
        },
        {
            "id": "sample_3",
            "content": "Vector databases enable semantic search by storing high-dimensional representations of text. Unlike traditional keyword search, vector search can understand context and meaning.",
            "metadata": {
                "title": "Vector Search Technology",
                "source": "Database Technologies",
                "type": "article"
            }
        }
    ]
    
    for doc in sample_docs:
        await vector_store.add_document(
            document_id=doc["id"],
            content=doc["content"],
            metadata=doc["metadata"]
        )
        print(f"Added sample document: {doc['metadata']['title']}")
    
    print("Initial data seeding completed!")

async def main():
    """Main setup function"""
    print("🚀 Setting up Perplexity AI Clone...")
    print("=" * 50)
    
    try:
        # Validate configuration
        print("Validating configuration...")
        validate_config()
        print("✅ Configuration valid")
        
        # Setup components
        await setup_database()
        await setup_vector_store()
        await seed_initial_data()
        
        print("=" * 50)
        print("🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Copy .env.example to .env and fill in your API keys")
        print("2. Start the backend server: uvicorn app.main:app --reload")
        print("3. Access the API at: http://localhost:8000")
        print("4. View API docs at: http://localhost:8000/docs")
        
    except Exception as e:
        print(f"❌ Setup failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())