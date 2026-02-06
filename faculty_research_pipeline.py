#!/usr/bin/env python3
"""
Research Data Collection Pipeline
Collects detailed entity information from web sources using strategic search queries.
Dataset-agnostic: reads field names and search templates from pipeline_config.py.
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
import json
import time
from results_manager import ResultsManager
import importlib
from config_helpers import (
    load_config, get_entity_schema, get_primary_field,
    get_entity_primary_value, get_entity_display, get_entity_fields
)

load_dotenv()


class ResearchPipeline:
    """Pipeline for collecting research data about any entity type"""

    def __init__(self, collection_name: str = "research_data"):
        self.collection_name = collection_name
        self.embedding_model = None
        self.milvus_collection = None
        self.llm_client = None
        self.serper_api_key = os.getenv('SERPER_API_KEY') or os.getenv('SERPER')
        self.google_api_key = os.getenv('GOOGLE_CUSTOM_SEARCH_KEY')
        self.google_cse_id = os.getenv('GOOGLE_CSE_ID')
        self.deepseek_api_key = os.getenv('DEEPSEEK_API_KEY') or os.getenv('DEEPSEEK_API')
        self.search_count = 0

    def initialize(self):
        """Initialize all components"""
        print("[INIT] Loading sentence transformer model...")
        self.embedding_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
        print(f"[OK] Embedding model loaded (dim={self.embedding_model.get_sentence_embedding_dimension()})")

        if self.deepseek_api_key:
            self.llm_client = OpenAI(
                api_key=self.deepseek_api_key,
                base_url="https://api.deepseek.com"
            )
            print("[OK] DeepSeek client initialized")
        else:
            print("[WARN] DEEPSEEK_API_KEY not found, LLM extraction will be skipped")

        self._init_milvus()

    def _init_milvus(self):
        """Initialize Milvus vector database"""
        try:
            connections.connect("default", uri="./milvus_research.db")
            print("[OK] Connected to Milvus Lite")

            if utility.has_collection(self.collection_name):
                print(f"[INFO] Collection '{self.collection_name}' exists, using existing")
                self.milvus_collection = Collection(self.collection_name)
            else:
                embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
                fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="entity_id", dtype=DataType.VARCHAR, max_length=200),
                    FieldSchema(name="entity_meta", dtype=DataType.VARCHAR, max_length=500),
                    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=5000),
                    FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=500),
                    FieldSchema(name="query_type", dtype=DataType.VARCHAR, max_length=100),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim)
                ]
                schema = CollectionSchema(fields, "Research data collection")
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
            raise

    def generate_search_queries(self, entity: Dict) -> List[Dict]:
        """
        Generate search queries using SEARCH_QUERY_TEMPLATES from pipeline_config.
        Falls back to generic queries built from entity field values.
        """
        config = load_config()
        templates = getattr(config, 'SEARCH_QUERY_TEMPLATES', []) if config else []

        if templates:
            queries = []
            for i, template in enumerate(templates):
                # Substitute all entity fields into the template
                query = template
                for field_key, field_value in entity.items():
                    query = query.replace(f'{{{field_key}}}', str(field_value))

                query_type = self._infer_query_type(query, i)
                queries.append({
                    'query': query,
                    'type': query_type,
                    'description': f'Search: {template[:60]}'
                })
            return queries

        # Fallback: build generic queries from entity field values
        primary = get_entity_primary_value(entity, config)
        parts = [f'"{v}"' for v in entity.values() if v]
        base_query = " ".join(parts)

        return [
            {'query': f'{base_query} profile', 'type': 'profile', 'description': 'Profile search'},
            {'query': f'{base_query} information', 'type': 'info', 'description': 'General information'},
            {'query': f'{base_query} contact', 'type': 'contact', 'description': 'Contact information'},
            {'query': f'"{primary}"', 'type': 'direct', 'description': 'Direct name search'},
        ]

    def _infer_query_type(self, query: str, index: int) -> str:
        """Infer query type from query content for scraping decisions."""
        query_lower = query.lower()
        if 'profile' in query_lower or 'cv' in query_lower or 'curriculum' in query_lower:
            return 'profile'
        elif 'email' in query_lower or 'contact' in query_lower:
            return 'contact'
        elif 'site:' in query_lower:
            return 'site_specific'
        elif index == 0:
            return 'profile'
        else:
            return f'search_{index}'

    def search_serper(self, query: str) -> List[Dict]:
        """Search using Serper API (primary, cheaper)"""
        if not self.serper_api_key:
            return []

        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": self.serper_api_key,
            "Content-Type": "application/json"
        }
        payload = {"q": query, "num": 3}

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            self.search_count += 1

            results = data.get('organic', [])
            print(f"[OK] Serper: found {len(results)} results (total searches: {self.search_count})")

            return [
                {
                    "title": r.get('title', 'N/A'),
                    "url": r.get('link', 'N/A'),
                    "description": r.get('snippet', 'N/A')
                }
                for r in results
            ]
        except Exception as e:
            print(f"[ERROR] Serper failed: {e}")
            return []

    def search_google(self, query: str) -> List[Dict]:
        """Search using Google Custom Search API (backup)"""
        if not self.google_api_key or not self.google_cse_id:
            return []

        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.google_api_key,
            "cx": self.google_cse_id,
            "q": query,
            "num": 3
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            self.search_count += 1

            items = data.get('items', [])
            print(f"[OK] Google CSE: found {len(items)} results (total searches: {self.search_count})")

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

    def search(self, query: str) -> List[Dict]:
        """Execute search using Serper (primary) with Google backup"""
        if self.serper_api_key:
            results = self.search_serper(query)
            if results:
                return results
        
        if self.google_api_key and self.google_cse_id:
            return self.search_google(query)
        
        return []

    def scrape_content(self, url: str) -> Optional[str]:
        """Scrape content from URL"""
        try:
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
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

    def store_in_vector_db(self, entity_id: str, entity_meta: str,
                          texts: List[str], urls: List[str], query_types: List[str]):
        """Store entity data in Milvus"""
        if not texts:
            print("[WARN] No texts to store")
            return

        print(f"[INFO] Generating embeddings for {len(texts)} documents...")
        embeddings = [self.embed_text(text) for text in texts]

        entity_ids = [entity_id] * len(texts)
        entity_metas = [entity_meta] * len(texts)

        data = [entity_ids, entity_metas, texts, urls, query_types, embeddings]

        insert_result = self.milvus_collection.insert(data)
        print(f"[OK] Stored {len(texts)} documents in Milvus")

    def extract_entity_info(self, entity: Dict, documents: List[Dict]) -> Dict:
        """Use LLM to extract structured entity information from documents"""
        if not self.llm_client:
            return {"error": "LLM not configured"}

        config = load_config()
        entity_display = get_entity_display(entity, config)

        context_str = "\n\n".join([
            f"Source {i+1} ({doc['query_type']}): {doc['url']}\n{doc['text'][:1500]}"
            for i, doc in enumerate(documents)
        ])

        # Load config dynamically to get the latest extraction prompt
        try:
            import pipeline_config
            importlib.reload(pipeline_config)

            # Try new signature first: get_extraction_prompt(entity, context)
            try:
                prompt = pipeline_config.get_extraction_prompt(entity, context_str)
            except TypeError:
                # Backward compat: old configs use (entity_name, institution, context)
                entity_name = get_entity_primary_value(entity, config)
                fields = get_entity_fields(config)
                secondary = entity.get(fields[1], '') if len(fields) > 1 else ''
                prompt = pipeline_config.get_extraction_prompt(entity_name, secondary, context_str)

        except Exception as e:
            print(f"[WARN] Could not load config prompt, using fallback: {e}")
            # Fallback prompt: build from EXTRACTION_FIELDS if available
            extraction_fields = getattr(config, 'EXTRACTION_FIELDS', []) if config else []

            if extraction_fields:
                fields_json = json.dumps(
                    {f['key']: f.get('description', f['label']) for f in extraction_fields},
                    indent=4
                )
            else:
                fields_json = '{"info": "Any relevant information"}'

            prompt = f"""Extract information about {entity_display} from the following sources.

Sources:
{context_str}

Please extract and return ONLY a JSON object with these fields (use "Not available" if not found):
{fields_json}

Return ONLY valid JSON, no additional text."""

        try:
            response = self.llm_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a data extraction assistant. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500
            )

            answer = response.choices[0].message.content
            print(f"[OK] LLM extraction completed")

            # Try to parse JSON
            try:
                # Remove markdown code blocks if present
                if "```json" in answer:
                    answer = answer.split("```json")[1].split("```")[0].strip()
                elif "```" in answer:
                    answer = answer.split("```")[1].split("```")[0].strip()

                return json.loads(answer)
            except json.JSONDecodeError:
                print(f"[WARN] Failed to parse JSON, returning raw response")
                return {"raw_response": answer}

        except Exception as e:
            print(f"[ERROR] LLM extraction failed: {e}")
            return {"error": str(e)}

    def process_entity(self, entity: Dict) -> Dict:
        """
        Process a single entity.

        Args:
            entity: Dict with entity field keys (e.g., {'name': '...', 'institution': '...'})
        """
        config = load_config()
        entity_display = get_entity_display(entity, config)
        entity_id = get_entity_primary_value(entity, config)
        entity_meta = json.dumps(entity)

        print(f"\n{'='*80}")
        print(f"Processing: {entity_display}")
        print(f"{'='*80}\n")

        # Generate search queries
        search_queries = self.generate_search_queries(entity)

        all_documents = []
        texts_to_store = []
        urls_to_store = []
        query_types_to_store = []

        # Execute searches
        for query_info in search_queries:
            query = query_info['query']
            query_type = query_info['type']

            print(f"\n[SEARCH] {query_info['description']}")
            print(f"  Query: '{query}'")

            search_results = self.search(query)
            time.sleep(1)  # Rate limiting

            for result in search_results:
                text = f"{result['title']}. {result['description']}"
                url = result['url']

                # Store snippet
                texts_to_store.append(text)
                urls_to_store.append(url)
                query_types_to_store.append(query_type)

                all_documents.append({
                    "query_type": query_type,
                    "url": url,
                    "text": text
                })

                # Try to scrape full content for profile/CV URLs
                if query_type in ['profile', 'cv', 'site_specific'] and len(all_documents) < 5:
                    scraped = self.scrape_content(url)
                    if scraped:
                        texts_to_store.append(scraped)
                        urls_to_store.append(url)
                        query_types_to_store.append(f"{query_type}_full")
                        all_documents.append({
                            "query_type": f"{query_type}_full",
                            "url": url,
                            "text": scraped
                        })

        # Store in vector database
        if texts_to_store:
            print(f"\n[STORE] Storing {len(texts_to_store)} documents in vector database...")
            self.store_in_vector_db(
                entity_id,
                entity_meta,
                texts_to_store,
                urls_to_store,
                query_types_to_store
            )

        # Extract structured information using LLM
        print(f"\n[EXTRACT] Using LLM to extract structured information...")
        extracted_info = self.extract_entity_info(entity, all_documents[:8])

        return {
            "entity": entity,
            "documents_collected": len(all_documents),
            "sources": all_documents,
            "extracted_info": extracted_info,
            "total_searches": self.search_count
        }

    # Backward compatibility aliases
    def process_faculty(self, faculty: Dict) -> Dict:
        return self.process_entity(faculty)

    def extract_faculty_info(self, faculty_name: str, institution: str,
                            documents: List[Dict]) -> Dict:
        entity = {"name": faculty_name, "institution": institution}
        return self.extract_entity_info(entity, documents)

    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.milvus_collection:
                self.milvus_collection.release()
            connections.disconnect("default")
            print("\n[CLEANUP] Resources released")
        except:
            pass


# Backward compatibility alias
FacultyResearchPipeline = ResearchPipeline


def main():
    """Test with entities from pipeline_config"""
    from datetime import datetime

    config = load_config()
    if config and hasattr(config, 'TARGET_ENTITIES') and config.TARGET_ENTITIES:
        entity_list = config.TARGET_ENTITIES[:3]  # Test with first 3
    else:
        # Fallback test data
        entity_list = [
            {"name": "Francis Couvares", "institution": "Amherst College"},
            {"name": "Solsiree Del Moral", "institution": "Amherst College"},
            {"name": "Pawan Dhingra", "institution": "Amherst College"}
        ]

    pipeline = ResearchPipeline()
    results_manager = ResultsManager()

    try:
        # Create session folder
        session_folder = results_manager.create_session_folder()
        print(f"\n[SESSION] Results will be saved to: {session_folder}\n")

        start_time = time.time()
        pipeline.initialize()

        results = []
        log_entries = []
        all_sources = set()

        for entity in entity_list:
            result = pipeline.process_entity(entity)
            results.append(result)

            # Collect unique sources
            for source in result.get('sources', []):
                all_sources.add(source.get('url', ''))

            entity_display = get_entity_display(entity, config)

            # Add to log
            log_entries.append(f"\n{'='*80}")
            log_entries.append(f"PROCESSED: {entity_display}")
            log_entries.append(f"{'='*80}")
            log_entries.append(f"Documents collected: {result['documents_collected']}")
            log_entries.append(f"Extracted Info: {json.dumps(result['extracted_info'], indent=2)}")

            print("\n" + "="*80)
            print(f"RESULT FOR {entity_display}")
            print("="*80)
            print(f"Documents collected: {result['documents_collected']}")
            print(f"\nExtracted Information:")
            print(json.dumps(result['extracted_info'], indent=2))
            print("\n")

        end_time = time.time()
        processing_time = end_time - start_time

        # Save results
        print("\n[SAVING] Saving results...")

        # Compile stats
        stats = {
            'total_entities': len(results),
            'total_documents': sum(r['documents_collected'] for r in results),
            'unique_sources': len(all_sources),
            'total_searches': pipeline.search_count,
            'processing_time': f"{processing_time:.1f} seconds",
            'processing_time_seconds': processing_time,
            'avg_time_per_entity': f"{processing_time / len(results):.1f} seconds"
        }

        # Save all formats
        saved_files = results_manager.save_results(results, session_folder)
        stats_file = results_manager.save_stats(stats, session_folder)
        log_file = results_manager.save_session_log(log_entries, session_folder)
        summary_file = results_manager.create_summary_report(results, stats, session_folder)

        print(f"\n[COMPLETE] Processed {len(results)} entities")
        print(f"Total Google searches used: {pipeline.search_count}")
        print(f"Processing time: {processing_time:.1f} seconds\n")

        print("="*80)
        print("FILES SAVED")
        print("="*80)
        for file_type, file_path in saved_files.items():
            print(f"  {file_type}: {file_path}")
        print(f"  stats: {stats_file}")
        print(f"  log: {log_file}")
        print(f"  summary: {summary_file}")
        print("\n" + "="*80)
        print(f"Session folder: {session_folder}")
        print("="*80 + "\n")

    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    main()
