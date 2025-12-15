#!/usr/bin/env python3
"""
Education Establishment Data Collection Pipeline
Searches for information about educational institutions (e.g., food banks, services)
and stores results in a vector database for LLM-based decision making.
"""

import os
import requests
from typing import List, Dict, Optional
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility
)
from openai import OpenAI
from bs4 import BeautifulSoup

load_dotenv()


class EducationDataPipeline:
    """Main pipeline for collecting and processing education establishment data"""

    def __init__(self, collection_name: str = "education_data"):
        self.collection_name = collection_name
        self.embedding_model = None
        self.milvus_collection = None
        self.openai_client = None
        self.brave_key = os.getenv('BRAVE_KEY')
        self.google_api_key = os.getenv('GOOGLE_SEARCH_API')
        self.google_cse_id = os.getenv('GOOGLE_CSE_ID')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')

    def initialize(self):
        """Initialize all components"""
        print("[INIT] Initializing pipeline components...")
        print("[INIT] Loading sentence transformer model...")
        self.embedding_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
        print(f"[OK] Embedding model loaded (dim={self.embedding_model.get_sentence_embedding_dimension()})")

        if self.openai_api_key:
            self.openai_client = OpenAI(api_key=self.openai_api_key)
            print("[OK] OpenAI client initialized")
        else:
            print("[WARN] OpenAI API key not found, LLM decision-making will be skipped")

        self._init_milvus()

    def _init_milvus(self):
        """Initialize Milvus vector database connection and collection"""
        try:
            try:
                connections.connect("default", uri="./milvus_education.db")
                print("[OK] Connected to Milvus Lite (embedded)")
            except:
                connections.connect("default", host="localhost", port="19530")
                print("[OK] Connected to Milvus Server")

            if utility.has_collection(self.collection_name):
                print(f"[INFO] Collection '{self.collection_name}' already exists, using existing")
                self.milvus_collection = Collection(self.collection_name)
            else:
                embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
                fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=5000),
                    FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=500),
                    FieldSchema(name="query", dtype=DataType.VARCHAR, max_length=500),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim)
                ]
                schema = CollectionSchema(fields, "Education establishment data collection")
                self.milvus_collection = Collection(self.collection_name, schema)
                print(f"[OK] Created collection: {self.collection_name}")

                index_params = {
                    "metric_type": "COSINE",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 128}
                }
                self.milvus_collection.create_index("embedding", index_params)
                print("[OK] Created vector index")

            self.milvus_collection.load()
            print("[OK] Collection loaded and ready")

        except Exception as e:
            print(f"[ERROR] Failed to initialize Milvus: {e}")
            print("[INFO] Make sure Milvus is running. Start it with:")
            print("       docker run -d --name milvus-standalone -p 19530:19530 milvusdb/milvus:latest standalone")
            raise

    def search_brave(self, query: str, count: int = 5) -> List[Dict]:
        """Search using Brave Search API"""
        if not self.brave_key:
            print("[SKIP] BRAVE_KEY not found")
            return []

        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.brave_key
        }
        params = {"q": query, "count": count}

        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            results = data.get('web', {}).get('results', [])
            print(f"[OK] Brave Search: found {len(results)} results for '{query}'")

            return [
                {
                    "title": r.get('title', 'N/A'),
                    "url": r.get('url', 'N/A'),
                    "description": r.get('description', 'N/A')
                }
                for r in results
            ]
        except Exception as e:
            print(f"[ERROR] Brave Search failed: {e}")
            return []

    def search_google(self, query: str, count: int = 5) -> List[Dict]:
        """Search using Google Custom Search API as fallback"""
        if not self.google_api_key or not self.google_cse_id:
            print("[SKIP] Google CSE credentials not found")
            return []

        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.google_api_key,
            "cx": self.google_cse_id,
            "q": query,
            "num": min(count, 10)
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            items = data.get('items', [])
            print(f"[OK] Google CSE: found {len(items)} results for '{query}'")

            return [
                {
                    "title": item.get('title', 'N/A'),
                    "url": item.get('link', 'N/A'),
                    "description": item.get('snippet', 'N/A')
                }
                for item in items
            ]
        except Exception as e:
            print(f"[ERROR] Google CSE failed: {e}")
            return []

    def search(self, query: str, count: int = 5) -> List[Dict]:
        """Search using available APIs with fallback"""
        # Use Google first, fallback to Brave
        results = self.search_google(query, count)
        if not results:
            results = self.search_brave(query, count)
        return results

    def scrape_content(self, url: str) -> Optional[str]:
        """Scrape content from URL using requests + BeautifulSoup"""
        try:
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            })
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'lxml')
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()

            text = soup.get_text(separator=' ', strip=True)
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)

            content = text[:5000]
            print(f"[OK] Scraped {len(content)} chars from {url[:50]}...")
            return content

        except Exception as e:
            print(f"[WARN] Failed to scrape {url}: {type(e).__name__}")
            return None

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text"""
        embedding = self.embedding_model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def store_in_vector_db(self, texts: List[str], urls: List[str], queries: List[str]):
        """Store text and embeddings in Milvus"""
        if not texts:
            print("[WARN] No texts to store")
            return

        print(f"[INFO] Generating embeddings for {len(texts)} documents...")
        embeddings = [self.embed_text(text) for text in texts]

        data = [texts, urls, queries, embeddings]

        insert_result = self.milvus_collection.insert(data)
        print(f"[OK] Stored {len(texts)} documents in Milvus (IDs: {insert_result.primary_keys[:3]}...)")

    def semantic_search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search vector database for semantically similar content"""
        query_embedding = self.embed_text(query)

        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}

        results = self.milvus_collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["text", "url", "query"]
        )

        print(f"[OK] Found {len(results[0])} similar documents")

        return [
            {
                "text": hit.entity.get('text'),
                "url": hit.entity.get('url'),
                "original_query": hit.entity.get('query'),
                "score": hit.score
            }
            for hit in results[0]
        ]

    def llm_decision(self, query: str, context: List[Dict]) -> str:
        """Use LLM to make decision based on retrieved context"""
        if not self.openai_client:
            return "LLM decision skipped (no API key)"

        context_str = "\n\n".join([
            f"Source: {c['url']}\n{c['text'][:500]}..."
            for c in context
        ])

        prompt = f"""Based on the following information gathered from web searches, answer this question:

Question: {query}

Retrieved Information:
{context_str}

Please provide a clear yes/no answer followed by supporting evidence."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions about educational institutions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300
            )

            answer = response.choices[0].message.content
            print(f"[OK] LLM decision generated")
            return answer

        except Exception as e:
            print(f"[ERROR] LLM decision failed: {e}")
            return f"Error: {e}"

    def process_query(self, query: str, variations: Optional[List[str]] = None) -> Dict:
        """
        Main pipeline: process a query about an educational establishment

        Args:
            query: Main question (e.g., "Does Stanford University have a food bank?")
            variations: Alternative phrasings of the same question
        """
        print(f"\n{'='*80}")
        print(f"Processing: {query}")
        print(f"{'='*80}\n")

        queries_to_search = [query] + (variations or [])
        all_results = []
        texts_to_store = []
        urls_to_store = []
        queries_to_store = []

        for q in queries_to_search:
            print(f"\n[SEARCH] Searching for: '{q}'")
            search_results = self.search(q, count=3)

            for result in search_results:
                text = f"{result['title']}. {result['description']}"
                url = result['url']
                texts_to_store.append(text)
                urls_to_store.append(url)
                queries_to_store.append(q)
                all_results.append({
                    "query": q,
                    "url": url,
                    "text": text
                })

        if texts_to_store:
            print(f"\n[STORE] Storing {len(texts_to_store)} results in vector database...")
            self.store_in_vector_db(texts_to_store, urls_to_store, queries_to_store)

        print(f"\n[RETRIEVE] Finding most relevant documents...")
        relevant_docs = self.semantic_search(query, top_k=5)

        print(f"\n[LLM] Making decision with LLM...")
        decision = self.llm_decision(query, relevant_docs)

        return {
            "query": query,
            "search_results_count": len(all_results),
            "relevant_docs": relevant_docs,
            "llm_decision": decision
        }

    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.milvus_collection:
                self.milvus_collection.release()
            connections.disconnect("default")
            print("\n[CLEANUP] Resources released")
        except:
            pass


def main():
    """Example usage"""
    pipeline = EducationDataPipeline()

    try:
        pipeline.initialize()
        result = pipeline.process_query(
            query="Does Stanford University have a food bank?",
            variations=[
                "Stanford food pantry",
                "Stanford student food assistance",
                "food bank at Stanford"
            ]
        )

        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)
        print(f"\nQuery: {result['query']}")
        print(f"Search results collected: {result['search_results_count']}")
        print(f"\nTop relevant documents:")
        for i, doc in enumerate(result['relevant_docs'][:3], 1):
            print(f"\n{i}. Score: {doc['score']:.3f}")
            print(f"URL: {doc['url']}")
            print(f"Text: {doc['text'][:200]}...")

        print(f"\nLLM Decision:\n{result['llm_decision']}")

    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    main()
