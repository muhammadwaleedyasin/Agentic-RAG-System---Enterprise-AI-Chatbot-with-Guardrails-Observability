#!/usr/bin/env python3
"""Database initialization script for Enterprise RAG Chatbot."""

import asyncio
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sqlalchemy.ext.asyncio import create_async_engine
from src.utils.config_loader import ConfigLoader
from src.utils.logging_config import setup_logging


async def create_tables():
    """Create database tables."""
    config = ConfigLoader()
    logger = setup_logging()
    
    # Create async engine
    engine = create_async_engine(
        config.database.url,
        echo=config.database.echo
    )
    
    try:
        # Import all models to ensure they're registered
        from src.models import Base
        
        # Create all tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("Database tables created successfully")
        
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        raise
    finally:
        await engine.dispose()


async def create_default_data():
    """Create default data for the application."""
    config = ConfigLoader()
    logger = setup_logging()
    
    # Add any default data creation logic here
    logger.info("Default data creation completed")


async def main():
    """Main initialization function."""
    print("Initializing Enterprise RAG Chatbot database...")
    
    try:
        await create_tables()
        await create_default_data()
        print("✅ Database initialization completed successfully!")
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())