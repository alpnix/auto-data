# Research Data Collection Pipeline

**Dataset-agnostic RAG pipeline: CSV + instructions → config → scraper → vector DB → LLM extraction → QA & display**

---

# Faculty Research Data Collection System

This system collects detailed faculty information from web sources using strategic Google searches, web scraping, vector database storage, and LLM-based data extraction.

## Overview

The faculty research pipeline is designed to automatically gather information about faculty members including:
- PhD and Bachelor's institutions
- Email addresses
- Academic rank (Assistant/Associate/Full Professor)
- Department affiliations
- Gender (inferred)
- Profile URLs

## Components

### 1. Faculty Research Pipeline (`faculty_research_pipeline.py`)
Core pipeline that executes strategic searches and data collection.

**Search Strategy:**
For each entity, the pipeline generates strategic searches based on config templates.
Example for faculty: profile, CV, education, contact, department, site-specific searches.
Fully customizable via `SEARCH_QUERY_TEMPLATES` in `pipeline_config.py`.

**Process Flow:**
1. Generate strategic search queries
2. Execute Google Custom Search API calls (3 results per query)
3. Scrape full content from profile/CV pages
4. Store all content in Milvus vector database
5. Use LLM (DeepSeek) to extract structured information

### 2. Setup & Research Server (`setup_server.py`)
Combined server for the complete workflow:
- **Setup phase**: Upload CSV + instructions → generate config → apply
- **Research phase**: Run pipeline → view results with evaluation

### 3. Faculty Dashboard (`faculty_dashboard.py`)
Flask web server with real-time process visualization using Server-Sent Events (SSE).

**Features:**
- Live progress tracking for each faculty member
- Real-time search logs
- Document collection statistics
- Extracted information display

### 3. Process Viewer UI (`templates/faculty_process.html`)
Dark-themed web interface matching the existing dashboard aesthetic.

**UI Elements:**
- Faculty cards with live progress bars
- Real-time search logs
- Extracted information display
- Global statistics (documents, sources, completion)

## Setup

### Prerequisites
```bash
pip install flask sentence-transformers pymilvus openai python-dotenv beautifulsoup4 lxml requests
```

### Environment Variables
Create a `.env` file with:
```env
GOOGLE_CUSTOM_SEARCH_KEY=your_google_api_key
GOOGLE_CSE_ID=your_cse_id
DEEPSEEK_API_KEY=your_deepseek_key
```

## Usage

### Option 1: Web Dashboard (Recommended)
Start the web server:
```bash
python faculty_dashboard.py
```

Then open your browser to:
```
http://localhost:8081
```

Click "Start Research Process" to begin collecting data for all 3 faculty members.

### Option 2: Command Line
Run the pipeline directly:
```bash
python faculty_research_pipeline.py
```

This will process all 3 faculty members sequentially and print results to console.

## Current Test Data

The system is configured to collect data for these 3 Amherst College faculty members:

1. **Francis Couvares**
   - Institution: Amherst College
   - Expected: American Studies & History departments

2. **Solsiree Del Moral**
   - Institution: Amherst College
   - Expected: American Studies & Black Studies departments

3. **Pawan Dhingra**
   - Institution: Amherst College
   - Expected: American Studies & Education Studies departments

## Search Quota Management

Google Custom Search API has a limit of 100 searches/day (free tier).

**Current configuration:**
- 6 query types per faculty member
- 3 results per query
- ~18 searches per faculty member
- 3 faculty = ~54 searches total
- Additional scraping attempts for full content

The pipeline tracks search usage and displays it in the dashboard.

## Data Storage

### Vector Database (Milvus Lite)
- Database file: `milvus_faculty.db`
- Collection: `faculty_data`
- Embeddings: BAAI/bge-small-en-v1.5 (384 dimensions)
- Stored fields:
  - faculty_name
  - institution
  - text (content)
  - url (source)
  - query_type (search category)
  - embedding (vector)

### Output Format

Extracted information is returned as JSON:
```json
{
  "name": "Faculty Name",
  "institution": "Institution Name",
  "email": "email@domain.edu",
  "phd_institution": "PhD University",
  "bachelors_institution": "Undergrad College",
  "rank": "Full Professor",
  "gender": "Man/Woman/Unknown",
  "department_1": "Primary Department",
  "department_2": "Secondary Department",
  "profile_url": "https://..."
}
```

## Customization

### Adding More Faculty
Edit the `faculty_list` in either file:

```python
faculty_list = [
    {"name": "Professor Name", "institution": "University Name"},
    # Add more...
]
```

### Modifying Search Queries
Edit `generate_search_queries()` in `faculty_research_pipeline.py`:

```python
queries = [
    {
        'query': 'your custom query',
        'type': 'custom_type',
        'description': 'What this query finds'
    }
]
```

### Adjusting LLM Extraction
Modify the prompt in `extract_faculty_info()` to extract different fields or change the output format.

## Rate Limiting

The pipeline includes automatic rate limiting:
- 1 second delay between Google searches
- Maximum 30 searches per session (configurable)
- Selective scraping (only profile/CV/site-specific pages)

## Troubleshooting

### "Missing credentials" error
- Check that `.env` file exists and contains valid API keys
- Verify GOOGLE_CUSTOM_SEARCH_KEY and GOOGLE_CSE_ID are set

### "Reached search limit" warning
- Google CSE free tier: 100 searches/day
- Wait until next day or upgrade to paid tier
- Reduce number of faculty members or queries per faculty

### LLM extraction returns "Unknown"
- Check DEEPSEEK_API_KEY is valid
- Verify scraped content contains relevant information
- Review search results quality (may need different queries)

### Milvus connection errors
- The pipeline uses Milvus Lite (embedded), no separate server needed
- Check write permissions in current directory
- Delete `milvus_faculty.db` to reset database

## Architecture

```
User Browser
    ↓
Flask Server (SSE)
    ↓
Faculty Research Pipeline
    ├→ Google Custom Search API (searches)
    ├→ Web Scraping (BeautifulSoup)
    ├→ Vector Database (Milvus)
    ├→ Embeddings (Sentence Transformers)
    └→ LLM Extraction (DeepSeek)
```

## Next Steps

To scale this system:

1. **CSV/Google Sheets Integration**: Read faculty lists from external sources
2. **Batch Processing**: Process large lists with pagination
3. **Export Results**: Save to CSV, JSON, or Google Sheets
4. **Enhanced Scraping**: Add Playwright for JavaScript-heavy sites
5. **Improved Extraction**: Fine-tune prompts or use structured extraction
6. **Caching**: Avoid re-processing same faculty members

## Performance

Expected timing (per faculty member):
- Search queries: ~30 seconds (6 queries × 5 seconds)
- Web scraping: ~15 seconds (3-5 pages)
- Vector storage: ~2 seconds
- LLM extraction: ~5 seconds

**Total per faculty**: ~50-60 seconds
**Total for 3 faculty**: ~3 minutes
