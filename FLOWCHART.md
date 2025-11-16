## Function Flowchart

Complete function flowchart for all modules (excluding fallbacks).

### education_pipeline.py

```
main()
  └─> EducationDataPipeline.__init__(collection_name="education_data")
      └─> [Sets: collection_name, embedding_model=None, milvus_collection=None, 
           openai_client=None, API keys from env]
  └─> pipeline.initialize()
      ├─> SentenceTransformer('BAAI/bge-small-en-v1.5')
      │   └─> [Sets: self.embedding_model]
      ├─> OpenAI(api_key=self.openai_api_key) [if key exists]
      │   └─> [Sets: self.openai_client]
      └─> _init_milvus()
          ├─> connections.connect("default", uri="./milvus_education.db")
          ├─> utility.has_collection(self.collection_name)
          │   ├─> [IF EXISTS] Collection(self.collection_name)
          │   │   └─> [Sets: self.milvus_collection]
          │   └─> [IF NOT EXISTS]
          │       ├─> CollectionSchema(fields, description)
          │       ├─> Collection(self.collection_name, schema)
          │       │   └─> [Sets: self.milvus_collection]
          │       └─> create_index("embedding", index_params)
          └─> self.milvus_collection.load()
  └─> pipeline.process_query(query, variations)
      ├─> queries_to_search = [query] + (variations or [])
      ├─> FOR EACH q IN queries_to_search:
      │   └─> search(q, count=3)
      │       └─> search_brave(q, count=3)
      │           ├─> requests.get("https://api.search.brave.com/res/v1/web/search", ...)
      │           └─> [Returns: List[Dict{title, url, description}]]
      │   └─> FOR EACH result IN search_results:
      │       ├─> text = f"{result['title']}. {result['description']}"
      │       ├─> texts_to_store.append(text)
      │       ├─> urls_to_store.append(url)
      │       └─> queries_to_store.append(q)
      ├─> store_in_vector_db(texts_to_store, urls_to_store, queries_to_store)
      │   ├─> FOR EACH text IN texts:
      │   │   └─> embed_text(text)
      │   │       └─> self.embedding_model.encode(text, normalize_embeddings=True)
      │   │           └─> [Returns: List[float]]
      │   └─> self.milvus_collection.insert([texts, urls, queries, embeddings])
      ├─> semantic_search(query, top_k=5)
      │   ├─> embed_text(query)
      │   │   └─> self.embedding_model.encode(query, normalize_embeddings=True)
      │   └─> self.milvus_collection.search(
      │       data=[query_embedding],
      │       anns_field="embedding",
      │       param=search_params,
      │       limit=top_k,
      │       output_fields=["text", "url", "query"]
      │   )
      │   └─> [Returns: List[Dict{text, url, original_query, score}]]
      └─> llm_decision(query, relevant_docs)
          ├─> [Build context_str from relevant_docs]
          ├─> self.openai_client.chat.completions.create(
          │   model="gpt-4o-mini",
          │   messages=[system, user],
          │   max_tokens=300
          │   )
          └─> [Returns: str (LLM response)]
  └─> pipeline.cleanup()
      ├─> self.milvus_collection.release()
      └─> connections.disconnect("default")
```

## web_dashboard.py

```
check_system_status()
  ├─> os.getenv('BRAVE_KEY')
  ├─> os.getenv('OPENAI_API_KEY')
  ├─> SentenceTransformer [import check]
  └─> connections.connect("health_check", uri="./milvus_education.db")
      └─> connections.disconnect("health_check")
  └─> [Returns: Dict{timestamp, components}]

index()
  └─> render_template('dashboard.html')

api_status()
  └─> check_system_status()
      └─> jsonify(status)

api_errors()
  └─> jsonify(list(error_log))

api_query()
  ├─> request.json
  ├─> EducationDataPipeline.__init__()
  ├─> pipeline.initialize()
  │   └─> [Same flow as above]
  ├─> pipeline.process_query(query=query)
  │   └─> [Same flow as above]
  └─> pipeline.cleanup()
      └─> [Same flow as above]
  └─> jsonify({success, query, results_count, answer, sources})
```

## test_pipeline_components.py

```
test_api_keys()
  ├─> os.getenv('BRAVE_KEY')
  ├─> os.getenv('OPENAI_API_KEY')
  ├─> os.getenv('GOOGLE_API_KEY')
  └─> os.getenv('GOOGLE_CSE_ID')
  └─> [Returns: bool]

test_search_api()
  ├─> EducationDataPipeline.__init__()
  └─> pipeline.search_brave("Stanford University food bank", count=3)
      └─> [Same flow as search_brave above]
      └─> [Returns: bool]

test_embedding_model()
  └─> SentenceTransformer('BAAI/bge-small-en-v1.5')
      └─> model.encode(test_text, normalize_embeddings=True)
      └─> [Returns: bool]

test_milvus_connection()
  └─> connections.connect("default", host="localhost", port="19530")
      └─> connections.disconnect("default")
      └─> [Returns: bool]

test_openai_connection()
  ├─> os.getenv('OPENAI_API_KEY')
  └─> OpenAI(api_key=api_key)
      └─> client.chat.completions.create(
          model="gpt-4o-mini",
          messages=[...],
          max_tokens=10
      )
      └─> [Returns: bool]

main()
  ├─> test_api_keys()
  ├─> test_search_api()
  ├─> test_embedding_model()
  ├─> test_milvus_connection()
  └─> test_openai_connection()
```

## example_queries.py

```
run_example_queries()
  ├─> EducationDataPipeline.__init__()
  ├─> pipeline.initialize()
  │   └─> [Same flow as above]
  └─> FOR EACH query_set IN queries:
      └─> pipeline.process_query(query=query_set["main"], variations=...)
          └─> [Same flow as process_query above]
  └─> pipeline.cleanup()
      └─> [Same flow as above]
```

## test_search_apis.py

```
test_brave_search(query="test query")
  ├─> os.getenv('BRAVE_KEY')
  └─> requests.get("https://api.search.brave.com/res/v1/web/search", ...)
      └─> [Returns: bool]

test_google_custom_search(query="test query")
  ├─> os.getenv('GOOGLE_API_KEY')
  ├─> os.getenv('GOOGLE_CSE_ID')
  └─> requests.get("https://www.googleapis.com/customsearch/v1", ...)
      └─> [Returns: bool]
```

## test_milvus.py

```
test_milvus_connection()
  └─> connections.connect("default", host="localhost", port="19530")
      └─> [Returns: bool]

test_milvus_operations()
  ├─> utility.has_collection(collection_name)
  ├─> utility.drop_collection(collection_name) [if exists]
  ├─> CollectionSchema(fields, description)
  ├─> Collection(collection_name, schema)
  ├─> collection.create_index("embedding", index_params)
  ├─> collection.insert(data)
  ├─> collection.load()
  └─> collection.search(data, anns_field, param, limit, output_fields)
  └─> utility.drop_collection(collection_name)
  └─> [Returns: bool]
```

## test_playwright.py

```
test_playwright_headless()
  ├─> async_playwright()
  ├─> p.chromium.launch(headless=True, channel="chromium", args=[...])
  ├─> browser.new_page()
  ├─> page.goto("https://httpbin.org/html", timeout=30000)
  ├─> page.title()
  ├─> page.content()
  └─> browser.close()
  └─> [Returns: bool]

test_playwright_headed()
  ├─> async_playwright()
  ├─> p.chromium.launch(headless=False, channel="chromium")
  ├─> browser.new_page()
  ├─> page.goto("https://httpbin.org/html", timeout=30000)
  ├─> page.title()
  ├─> page.content()
  └─> browser.close()
  └─> [Returns: bool]
```

## Key Flow Paths

**PRIMARY PIPELINE FLOW (process_query):**
```
process_query()
  → search() → search_brave()
  → store_in_vector_db() → embed_text() [for each text]
  → semantic_search() → embed_text() [for query]
  → llm_decision() → OpenAI API
```

**INITIALIZATION FLOW:**
```
initialize()
  → SentenceTransformer.load()
  → OpenAI() [if key exists]
  → _init_milvus()
      → connections.connect()
      → Collection() or Collection.create()
      → create_index()
      → collection.load()
```

**WEB DASHBOARD FLOW:**
```
api_query()
  → EducationDataPipeline.__init__()
  → initialize()
  → process_query()
  → cleanup()
```

