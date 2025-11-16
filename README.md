## Education Establishment Data Collection Pipeline

A RAG (Retrieval-Augmented Generation) pipeline for collecting and analyzing information about educational establishments. The system searches for answers to questions about schools, universities, and other educational institutions.

### Installation

```bash
pip install -r requirements.txt
pip install "pymilvus[milvus_lite]"
```

### Configuration

Create a `.env` file with your API keys:

```bash
BRAVE_KEY=your_brave_search_api_key
OPENAI_API_KEY=your_openai_api_key
```

**Getting API Keys:**
- Brave Search: https://brave.com/search/api/
- OpenAI: https://platform.openai.com/api-keys

### Usage

#### Test Components

```bash
python test_pipeline_components.py
```

#### Run Pipeline

```bash
python education_pipeline.py
```

#### Web Dashboard

```bash
python web_dashboard.py
```

Then open: http://localhost:8080

#### Custom Queries

```python
from education_pipeline import EducationDataPipeline

pipeline = EducationDataPipeline()
pipeline.initialize()

try:
    result = pipeline.process_query(
        query="Does Harvard have mental health services?",
        variations=[
            "Harvard mental health support",
            "Harvard counseling services"
        ]
    )
    print(result['llm_decision'])
finally:
    pipeline.cleanup()
```

### How It Works

1. **Search**: Uses Brave Search API to find relevant web pages
2. **Embed**: Converts text to vector embeddings using BAAI/bge-small-en-v1.5
3. **Store**: Saves embeddings in Milvus vector database
4. **Retrieve**: Finds semantically similar content
5. **Decide**: Uses OpenAI GPT-4o-mini to make informed decisions

### Architecture

- **Search**: Brave Search API (primary), Google Custom Search (fallback)
- **Embeddings**: BAAI/bge-small-en-v1.5 (384-dim)
- **Vector DB**: Milvus Lite (embedded) or Milvus Server
- **LLM**: OpenAI GPT-4o-mini

### Project Structure

```
mini-rag-c2i/
├── education_pipeline.py      # Main pipeline
├── test_pipeline_components.py # Component tests
├── web_dashboard.py           # Web interface
├── example_queries.py         # Example usage
└── requirements.txt          # Dependencies
```

## Troubleshooting

**Milvus Connection**: Uses Milvus Lite (embedded) by default. No Docker required.

For detailed function flowcharts, see `FLOWCHART.md`.
