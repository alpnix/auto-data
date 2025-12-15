# Faculty Data Extraction - Quick Start Guide

## What You Have Now

I've created a complete system to automatically extract faculty demographic data using your existing RAG pipeline. Here's what's been added:

### 📄 New Files Created

1. **`FACULTY_DATA_COLLECTION_PROMPTS.md`** - Comprehensive documentation
   - Detailed explanation of how the RAG flow maps to faculty data collection
   - Specific search queries for each of the 11 features
   - LLM extraction prompts for each feature
   - Evaluation strategy
   - Implementation options

2. **`faculty_data_extractor.py`** - Working implementation
   - `FacultyDataExtractor` class that uses your RAG pipeline
   - Extracts all 11 features automatically
   - Includes evaluation against ground truth
   - Ready-to-run test suite for your 3 example faculty

---

## How It Works

### Input (What You Provide)
- **Institution Name**: e.g., "Amherst College"
- **Faculty Name**: e.g., "Francis Couvares"
- **Institution Domain** (optional): e.g., "amherst.edu"

### Output (What You Get)
```json
{
  "Institution": "Amherst College",
  "Name": "Francis Couvares",
  "profile_link": "https://www.amherst.edu/people/facstaff/fgcouvares",
  "email": "fgcouvares@amherst.edu",
  "department_1": "American Studies",
  "department_2": "History",
  "rank": "Full",
  "gender": "Man",
  "phd": "University of Michigan, Ann Arbor",
  "bachelors": "University of Pittsburgh",
  "alum_status": "Non-Alum",
  "google_scholar_link": "NOT_FOUND",
  "h_index": "NOT_FOUND",
  "i10_index": "NOT_FOUND"
}
```

---

## Quick Start

### Step 1: Test with Ground Truth Examples

Run the test suite on your 3 ground truth faculty members:

```bash
python faculty_data_extractor.py
```

This will:
1. Extract data for Francis Couvares, Solsiree Del Moral, and Pawan Dhingra
2. Compare results against ground truth
3. Generate accuracy metrics
4. Save results to `faculty_extraction_results.json`

### Step 2: Extract Data for a Single Faculty Member

```python
from education_pipeline import EducationDataPipeline
from faculty_data_extractor import FacultyDataExtractor

# Initialize pipeline
pipeline = EducationDataPipeline(collection_name="faculty_data")
pipeline.initialize()

# Initialize extractor
extractor = FacultyDataExtractor(pipeline)

try:
    # Extract all features
    faculty_data = extractor.extract_all_features(
        faculty_name="Francis Couvares",
        institution="Amherst College",
        institution_domain="amherst.edu"
    )

    print(faculty_data)

finally:
    pipeline.cleanup()
```

### Step 3: Extract a Single Feature

```python
# Extract just the email
email = extractor.extract_feature(
    feature='email',
    faculty_name="Francis Couvares",
    institution="Amherst College",
    institution_domain="amherst.edu"
)

print(f"Email: {email}")
```

---

## Features Being Extracted

| # | Feature | Description | Example |
|---|---------|-------------|---------|
| 1 | `profile_link` | Official faculty profile URL | https://www.amherst.edu/people/facstaff/fgcouvares |
| 2 | `email` | Faculty email address | fgcouvares@amherst.edu |
| 3 | `department_1` | Primary department | American Studies |
| 4 | `department_2` | Secondary department | History (or "NONE") |
| 5 | `rank` | Academic rank | Full, Associate, Assistant, etc. |
| 6 | `gender` | Gender (if explicitly stated) | Man, Woman, Non-binary |
| 7 | `phd` | PhD institution | University of Michigan, Ann Arbor |
| 8 | `bachelors` | Undergraduate institution | University of Pittsburgh |
| 9 | `alum_status` | Alumni of current institution? | Alum or Non-Alum |
| 10 | `google_scholar_link` | Google Scholar profile URL | https://scholar.google.com/citations?... |
| 11 | `h_index` | Google Scholar h-index | 12 |
| 12 | `i10_index` | Google Scholar i10-index | 14 |

---

## How Each Feature is Extracted

For each feature, the system:

1. **Generates targeted search queries**
   ```python
   # Example for email:
   queries = [
       "Francis Couvares Amherst College email",
       "Francis Couvares Amherst College contact information",
       "site:amherst.edu Francis Couvares email"
   ]
   ```

2. **Searches the web** using Brave/Google Search API
   - Retrieves 3 results per query
   - Collects titles and descriptions

3. **Embeds and stores** results in Milvus vector database
   - Uses BAAI/bge-small-en-v1.5 embeddings
   - 384-dimensional vectors

4. **Performs semantic search** to find most relevant content
   - Cosine similarity ranking
   - Returns top 5 most relevant results

5. **Extracts with LLM** using GPT-4o-mini
   - Feature-specific extraction prompts
   - Returns clean, structured value

---

## Understanding the RAG Flow for Faculty Data

### Original Flow (Yes/No Questions)
```
"Does Harvard have mental health services?"
  → Search web
  → Store results
  → Semantic search
  → LLM decides: "Yes, based on..."
```

### Adapted Flow (Faculty Data Extraction)
```
"What is Francis Couvares' email at Amherst?"
  → Search: "Francis Couvares Amherst email"
  → Store results
  → Semantic search for relevant docs
  → LLM extracts: "fgcouvares@amherst.edu"
```

**Key Differences:**
- **Multiple features**: Run 11 separate extractions (or batch them)
- **Structured output**: Extract specific values, not yes/no answers
- **Validation**: Compare against ground truth for accuracy

---

## Evaluation Metrics

After extraction, the system calculates:

- **Accuracy**: `(Correct / Total Features) × 100`
- **Precision**: `(Correct / (Total - Not Found)) × 100`
- **Per-feature results**: Which features succeeded/failed

Example output:
```
Correct: 8/11
Incorrect: 1/11
Not Found: 2/11
Accuracy: 72.7%
Precision: 88.9%
```

---

## Customization

### Add More Features

Edit `faculty_data_extractor.py` and add to the `generate_search_queries()` and `build_extraction_prompt()` methods:

```python
'new_feature': [
    f"{faculty_name} {institution} custom query",
    f"another variation"
]
```

### Modify Search Strategy

Change the number of results per query:
```python
result = self.pipeline.process_query(
    query=queries[0],
    variations=queries[1:],
    count=5  # Default is 3
)
```

### Adjust LLM Extraction

Modify prompts in `build_extraction_prompt()` to fine-tune extraction logic.

---

## Expected Performance

Based on the ground truth examples:

### High Accuracy (>80%) Expected:
- Profile Link (institution websites are well-structured)
- Email (usually on profile pages)
- Department 1 (prominently displayed)
- Rank (standard academic titles)

### Medium Accuracy (50-80%) Expected:
- Department 2 (not all faculty have joint appointments)
- PhD Institution (depends on CV availability)
- Bachelor's Institution (less commonly listed)

### Lower Accuracy (<50%) Expected:
- Google Scholar metrics (not all faculty have profiles)
- Gender (only if explicitly stated with pronouns)
- Alum Status (requires cross-referencing education history)

---

## Next Steps

1. **Run the test suite** to see baseline accuracy
2. **Review extraction results** and identify failure patterns
3. **Refine prompts** for features with low accuracy
4. **Add more test cases** from different institutions
5. **Consider profile scraping** for higher accuracy (see Option 3 in PROMPTS.md)

---

## Troubleshooting

### Issue: Many "NOT_FOUND" results
**Solution**: The faculty member may not have a strong web presence. Try:
- More query variations
- Direct profile page scraping
- Manual data entry for missing fields

### Issue: Incorrect extractions
**Solution**: Check the LLM prompts. You may need to:
- Add more specific instructions
- Provide better examples
- Increase context window (more retrieved documents)

### Issue: Rate limiting
**Solution**: Add delays between searches:
```python
import time
time.sleep(1)  # Wait 1 second between API calls
```

---

## File Structure

```
mini-rag-c2i/
├── education_pipeline.py                    # Existing RAG pipeline
├── faculty_data_extractor.py               # NEW: Faculty extractor
├── FACULTY_DATA_COLLECTION_PROMPTS.md      # NEW: Detailed documentation
├── FACULTY_EXTRACTION_QUICKSTART.md        # NEW: This guide
└── faculty_extraction_results.json         # Generated after test run
```

---

## Key Concepts

### Ground Truth
The 3 faculty examples you provided are the "ground truth" - known correct answers used to evaluate the system's accuracy.

### Feature
Each column in your dataset (email, rank, PhD, etc.) is a "feature" that needs to be extracted.

### Extraction Prompt
The specific instructions given to the LLM to extract a particular feature from search results.

### Semantic Search
Finding relevant documents based on meaning/context, not just keyword matching. Your pipeline uses vector embeddings for this.

---

## Questions for Each Feature

Here's a summary of what the system "asks" to find each feature:

1. **Profile Link**: "What is the official faculty page URL?"
2. **Email**: "What is their email address?"
3. **Department 1**: "What is their primary department?"
4. **Department 2**: "Do they have a secondary department?"
5. **Rank**: "What is their academic rank (Full/Associate/Assistant)?"
6. **Gender**: "What gender pronouns are used in their bio?"
7. **PhD**: "Where did they earn their PhD?"
8. **Bachelor's**: "Where did they earn their Bachelor's degree?"
9. **Alum Status**: "Did they graduate from this institution?"
10. **Google Scholar**: "What is their Google Scholar profile URL?"
11. **H-Index**: "What is their h-index?"
12. **i10-Index**: "What is their i10-index?"

Each question is translated into search queries and extraction prompts as shown in the detailed documentation.

---

## Summary

You now have:
✅ A complete understanding of how your RAG flowchart works
✅ Specific prompts/questions for all 11 features
✅ A working implementation you can test immediately
✅ Evaluation framework to measure accuracy
✅ Documentation for customization and extension

**To get started**: Run `python faculty_data_extractor.py` and review the results!
