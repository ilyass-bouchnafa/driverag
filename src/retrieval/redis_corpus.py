import json
import logging
import redis
from typing import List, Dict, Optional
from src.config import REDIS_URL

logger = logging.getLogger(__name__)

# Try connecting to Redis (optional)
redis_client = None
REDIS_AVAILABLE = False

try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=2)
    # Test connection
    redis_client.ping()
    REDIS_AVAILABLE = True
    logger.info(f"✅ Redis connected: {REDIS_URL}")
except Exception as e:
    logger.warning(f"⚠️ Redis not available: {e}. Falling back to ChromaDB.")
    redis_client = None
    REDIS_AVAILABLE = False

# Main hash key used to store all chunks in Redis
CHUNKS_HASH_KEY = "driverag:chunks"


def save_chunks_to_redis(chunks: List[Dict]):
    """Save all chunks into Redis using a hash for efficient storage."""
    if not REDIS_AVAILABLE or not redis_client or not chunks:
        return

    try:
        pipe = redis_client.pipeline()

        for chunk in chunks:
            chunk_id = (
                f"{chunk['metadata']['source']}_p{chunk['metadata']['page']}"
                f"_c{chunk['metadata']['chunk_index']}"
            )
            
            # Save the full chunk JSON
            pipe.hset(CHUNKS_HASH_KEY, chunk_id, json.dumps(chunk))
        
        pipe.execute()
        logger.info(f"✅ {len(chunks)} chunks saved to Redis")
    except Exception as e:
        logger.warning(f"⚠️ Redis save_chunks error: {e}. Continuing without Redis...")


def get_all_chunks_from_redis() -> List[Dict]:
    """Retrieve all chunks from Redis (much faster than querying ChromaDB)."""
    if not REDIS_AVAILABLE or not redis_client:
        return []

    try:
        chunks_data = redis_client.hgetall(CHUNKS_HASH_KEY)
        
        chunks = []
        for chunk_json in chunks_data.values():
            chunks.append(json.loads(chunk_json))
        
        logger.debug(f"📦 {len(chunks)} chunks loaded from Redis")
        return chunks

    except Exception as e:
        logger.warning(f"⚠️ Redis get_all_chunks error: {e}")
        return []


def delete_chunks_by_source_redis(source_name: str):
    """Delete all chunks in Redis associated with a given source file."""
    if not REDIS_AVAILABLE or not redis_client:
        return

    try:
        all_entries = redis_client.hgetall(CHUNKS_HASH_KEY)
        keys_to_delete = [
            key for key in all_entries.keys() 
            if key.startswith(source_name + "_p")
        ]
        
        if keys_to_delete:
            redis_client.hdel(CHUNKS_HASH_KEY, *keys_to_delete)
            logger.info(f"🗑️ {len(keys_to_delete)} chunks deleted from Redis for {source_name}")
    except Exception as e:
        logger.warning(f"⚠️ Redis deletion error: {e}")


def clear_redis_corpus():
    """Completely clear the Redis corpus (useful for reset)."""
    if not REDIS_AVAILABLE or not redis_client:
        return

    try:
        redis_client.delete(CHUNKS_HASH_KEY)
        logger.info("🧹 Redis corpus fully cleared")
    except Exception as e:
        logger.warning(f"⚠️ Redis clear error: {e}")


def get_redis_stats() -> Dict:
    """Return statistics about the Redis corpus."""
    if not REDIS_AVAILABLE or not redis_client:
        return {"status": "⚠️ Redis not available", "available": False}

    try:
        chunks_count = redis_client.hlen(CHUNKS_HASH_KEY)
        return {
            "chunks_count": chunks_count,
            "redis_url": REDIS_URL,
            "status": "✅ Connected",
            "available": True
        }
    except Exception as e:
        logger.warning(f"⚠️ Redis stats error: {e}")
        return {"error": str(e), "status": "❌ Connection failed", "available": False}