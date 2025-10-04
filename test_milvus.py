#!/usr/bin/env python3
"""Test Milvus vector database setup and operations"""

import time
import numpy as np
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

def test_milvus_connection():
    """Test connection to Milvus"""
    try:
        # Connect to Milvus (default localhost:19530)
        connections.connect("default", host="localhost", port="19530")
        print("[OK] Connected to Milvus")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to connect to Milvus: {e}")
        print("[INFO] Make sure Milvus is running via Docker:")
        print("       docker run -d --name milvus-standalone -p 19530:19530 -p 9091:9091 -v milvus:/var/lib/milvus milvusdb/milvus:latest standalone")
        return False

def test_milvus_operations():
    """Test basic Milvus operations: create collection, insert, search"""
    collection_name = "test_collection"
    
    try:
        # Drop collection if it exists
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            print(f"[OK] Dropped existing collection: {collection_name}")
        
        # Define collection schema (384-dim for bge-small-en-v1.5)
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
        ]
        schema = CollectionSchema(fields, "Test collection for RAG")
        
        # Create collection
        collection = Collection(collection_name, schema)
        print(f"[OK] Created collection: {collection_name}")
        
        # Create index for vector field
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index("embedding", index_params)
        print("[OK] Created vector index")
        
        # Insert test data
        test_texts = [
            "This is a test document about machine learning.",
            "Python is a popular programming language.",
            "Vector databases are useful for similarity search."
        ]
        
        # Generate random embeddings (384-dim)
        test_embeddings = np.random.random((3, 384)).tolist()
        
        data = [test_texts, test_embeddings]
        collection.insert(data)
        print(f"[OK] Inserted {len(test_texts)} test documents")
        
        # Load collection for search
        collection.load()
        print("[OK] Loaded collection for search")
        
        # Wait a moment for indexing
        time.sleep(1)
        
        # Search test
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        query_embedding = np.random.random((1, 384)).tolist()
        
        results = collection.search(
            data=query_embedding,
            anns_field="embedding",
            param=search_params,
            limit=2,
            output_fields=["text"]
        )
        
        print(f"[OK] Search returned {len(results[0])} results")
        for i, hit in enumerate(results[0]):
            print(f"       Result {i+1}: {hit.entity.get('text')[:50]}... (score: {hit.score:.3f})")
        
        # Cleanup
        utility.drop_collection(collection_name)
        print(f"[OK] Cleaned up test collection")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Milvus operations failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Milvus vector database...")
    
    if test_milvus_connection():
        print("\n=== Testing Milvus Operations ===")
        if test_milvus_operations():
            print("\n[SUCCESS] Milvus is working correctly!")
        else:
            print("\n[FAILED] Milvus operations failed!")
    else:
        print("\n[FAILED] Could not connect to Milvus!")
