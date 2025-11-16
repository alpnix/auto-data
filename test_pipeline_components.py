#!/usr/bin/env python3
"""
Test individual pipeline components before running the full pipeline
"""

import os
from dotenv import load_dotenv
from education_pipeline import EducationDataPipeline

load_dotenv()

def test_api_keys():
    """Test that required API keys are configured"""
    print("\n" + "="*80)
    print("Testing API Keys Configuration")
    print("="*80)

    keys = {
        "BRAVE_KEY": os.getenv('BRAVE_KEY'),
        "OPENAI_API_KEY": os.getenv('OPENAI_API_KEY'),
        "GOOGLE_API_KEY": os.getenv('GOOGLE_API_KEY'),
        "GOOGLE_CSE_ID": os.getenv('GOOGLE_CSE_ID'),
    }

    for key_name, key_value in keys.items():
        if key_value:
            masked = key_value[:8] + "..." + key_value[-4:] if len(key_value) > 12 else "***"
            print(f"[OK] {key_name}: {masked}")
        else:
            print(f"[WARN] {key_name}: Not configured")

    return bool(keys["BRAVE_KEY"])


def test_search_api():
    """Test search API functionality"""
    print("\n" + "="*80)
    print("Testing Search APIs")
    print("="*80)

    pipeline = EducationDataPipeline()

    print("\n[TEST] Testing Brave Search...")
    results = pipeline.search_brave("Stanford University food bank", count=3)

    if results:
        print(f"[OK] Brave Search returned {len(results)} results")
        print(f"[OK] First result: {results[0]['title'][:60]}...")
        return True
    else:
        print("[WARN] Brave Search returned no results")
        print("\n[TEST] Testing Google Custom Search...")
        results = pipeline.search_google("Stanford University food bank", count=3)

        if results:
            print(f"[OK] Google CSE returned {len(results)} results")
            return True
        else:
            print("[ERROR] Both search APIs failed")
            return False


def test_embedding_model():
    """Test embedding model"""
    print("\n" + "="*80)
    print("Testing Embedding Model")
    print("="*80)

    pipeline = EducationDataPipeline()

    print("[TEST] Loading sentence transformer model...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('BAAI/bge-small-en-v1.5')

    test_text = "Stanford University has a food pantry for students in need."
    embedding = model.encode(test_text, normalize_embeddings=True)

    print(f"[OK] Model loaded: {model}")
    print(f"[OK] Embedding dimension: {len(embedding)}")
    print(f"[OK] Sample embedding values: {embedding[:5]}")

    return True


def test_milvus_connection():
    """Test Milvus connection (requires Docker)"""
    print("\n" + "="*80)
    print("Testing Milvus Connection")
    print("="*80)

    try:
        from pymilvus import connections
        connections.connect("default", host="localhost", port="19530")
        print("[OK] Connected to Milvus successfully")
        connections.disconnect("default")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to connect to Milvus: {e}")
        print("\n[INFO] Milvus is not running. To start Milvus:")
        print("  Option 1 - Docker Compose (recommended):")
        print("    wget https://github.com/milvus-io/milvus/releases/download/v2.3.4/milvus-standalone-docker-compose.yml -O docker-compose.yml")
        print("    docker-compose up -d")
        print("\n  Option 2 - Simple Docker command:")
        print("    docker run -d --name milvus-standalone -p 19530:19530 milvusdb/milvus:latest standalone")
        print("\n  Option 3 - Milvus Lite (embedded, no Docker needed):")
        print("    pip install milvus")
        print("    Then use: connections.connect(uri='./milvus_demo.db')")
        return False


def test_openai_connection():
    """Test OpenAI API connection"""
    print("\n" + "="*80)
    print("Testing OpenAI API")
    print("="*80)

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("[WARN] OPENAI_API_KEY not configured")
        return False

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'test successful' if you can read this."}],
            max_tokens=10
        )

        answer = response.choices[0].message.content
        print(f"[OK] OpenAI API connected successfully")
        print(f"[OK] Test response: {answer}")
        return True

    except Exception as e:
        print(f"[ERROR] OpenAI API test failed: {e}")
        return False


def main():
    """Run all component tests"""
    print("\n" + "="*80)
    print("TESTING PIPELINE COMPONENTS")
    print("="*80)

    results = {
        "API Keys": test_api_keys(),
        "Search API": test_search_api(),
        "Embedding Model": test_embedding_model(),
        "Milvus Connection": test_milvus_connection(),
        "OpenAI API": test_openai_connection()
    }

    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for component, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{status} - {component}")

    passed_count = sum(results.values())
    total_count = len(results)

    print(f"\nPassed: {passed_count}/{total_count}")

    if results["Search API"] and results["Embedding Model"]:
        print("\n[INFO] Core components working! You can run the pipeline without Milvus for testing.")
        print("[INFO] For full functionality, start Milvus and re-run tests.")
    elif results["Milvus Connection"] and results["Search API"] and results["Embedding Model"]:
        print("\n[SUCCESS] All components ready! You can run the full pipeline.")
    else:
        print("\n[WARN] Some components need configuration. See errors above.")


if __name__ == "__main__":
    main()
