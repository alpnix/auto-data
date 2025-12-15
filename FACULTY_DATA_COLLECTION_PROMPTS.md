# Faculty Demographics Data Collection: Query & Prompt Design

## Overview

This document maps the existing RAG pipeline flow to the faculty demographics collection use case. Given **Institution Name** and **Faculty Name** as inputs, we need to automatically collect all other features from the ground truth dataset.

## Current RAG Pipeline Flow → Faculty Data Extraction

### Existing Flow
```
User Query → Search API (Brave/Google) → Embed Results → Store in Milvus
→ Semantic Search → LLM Decision → Final Answer
```

### Adapted Flow for Faculty Data
```
Faculty Name + Institution → Multiple Targeted Searches (one per feature)
→ Embed Results → Store in Milvus → Semantic Search (per feature)
→ LLM Extraction (structured) → Structured Faculty Profile
```

---

## Ground Truth Features Analysis

**Example Data:**
- **Institution**: 164465 - Amherst College (MA)
- **Name**: Francis Couvares
- **PhD**: University of Michigan, Ann Arbor
- **Bachelor's**: University of Pittsburgh
- **Alum Status**: Non-Alum
- **Email**: fgcouvares@amherst.edu
- **Rank**: Full
- **Gender**: Man
- **Department 1**: American Studies
- **Department 2**: History
- **Profile Link**: https://www.amherst.edu/people/facstaff/fgcouvares
- **Google Scholar Link**: https://scholar.google.com/citations?hl=en&user=2Y0_sPsAAAAJ
- **Google Scholar H-Index**: 12
- **Google Scholar i10 Index**: 14

---

## Feature Extraction Strategy

### Given Inputs (No Extraction Needed)
1. **Institution** - Provided by user
2. **Name** - Provided by user

### Features to Extract (11 total)

---

## Detailed Prompt Design for Each Feature

### 1. PROFILE LINK
**Purpose**: Find the official faculty profile URL

**Search Queries:**
```python
queries = [
    f"{faculty_name} {institution_name} faculty profile",
    f"{faculty_name} {institution_name} biography",
    f"{faculty_name} {institution_name} contact",
    f"site:{institution_domain} {faculty_name}"
]
```

**Search Query Variations Example** (Francis Couvares, Amherst College):
- "Francis Couvares Amherst College faculty profile"
- "Francis Couvares Amherst College biography"
- "Francis Couvares Amherst College contact"
- "site:amherst.edu Francis Couvares"

**LLM Extraction Prompt:**
```
System: You are a data extraction assistant that identifies faculty profile URLs from search results.

User: Based on the following search results, extract the official faculty profile URL for {faculty_name} at {institution_name}.

Retrieved Information:
{context_str}

Instructions:
- Return ONLY the URL of the official faculty profile page
- The URL should be from the institution's official website (usually .edu domain)
- If multiple URLs are found, prioritize the main faculty directory profile
- If no URL is found, return "NOT_FOUND"
- Format: Return only the URL, nothing else

Example output: https://www.amherst.edu/people/facstaff/fgcouvares
```

**Expected Output**: `https://www.amherst.edu/people/facstaff/fgcouvares`

---

### 2. EMAIL
**Purpose**: Extract faculty email address

**Search Queries:**
```python
queries = [
    f"{faculty_name} {institution_name} email",
    f"{faculty_name} {institution_name} contact information",
    f"site:{institution_domain} {faculty_name} email"
]
```

**Search Query Variations Example:**
- "Francis Couvares Amherst College email"
- "Francis Couvares Amherst College contact information"
- "site:amherst.edu Francis Couvares email"

**LLM Extraction Prompt:**
```
System: You are a data extraction assistant that identifies faculty email addresses from search results.

User: Based on the following search results, extract the email address for {faculty_name} at {institution_name}.

Retrieved Information:
{context_str}

Instructions:
- Return ONLY the email address
- The email should belong to the faculty member at the institution
- Common formats: firstname.lastname@domain.edu, firstinitiallastname@domain.edu
- If no email is found, return "NOT_FOUND"
- Format: Return only the email address, nothing else

Example output: fgcouvares@amherst.edu
```

**Expected Output**: `fgcouvares@amherst.edu`

---

### 3. DEPARTMENT 1 (Primary Department)
**Purpose**: Identify the primary academic department

**Search Queries:**
```python
queries = [
    f"{faculty_name} {institution_name} department",
    f"{faculty_name} {institution_name} faculty",
    f"{faculty_name} {institution_name} professor",
    f"site:{institution_domain} {faculty_name} department"
]
```

**Search Query Variations Example:**
- "Francis Couvares Amherst College department"
- "Francis Couvares Amherst College faculty"
- "Francis Couvares Amherst professor"
- "site:amherst.edu Francis Couvares department"

**LLM Extraction Prompt:**
```
System: You are a data extraction assistant that identifies faculty departmental affiliations from search results.

User: Based on the following search results, extract the PRIMARY department for {faculty_name} at {institution_name}.

Retrieved Information:
{context_str}

Instructions:
- Return ONLY the primary department name
- Use the official department name as it appears on the institution's website
- If the faculty is affiliated with multiple departments, return the one listed FIRST or marked as PRIMARY
- Common departments: History, Biology, Computer Science, American Studies, etc.
- If no department is found, return "NOT_FOUND"
- Format: Return only the department name, nothing else

Example output: American Studies
```

**Expected Output**: `American Studies`

---

### 4. DEPARTMENT 2 (Secondary Department)
**Purpose**: Identify secondary/joint departmental affiliation

**Search Queries:**
```python
queries = [
    f"{faculty_name} {institution_name} joint appointment",
    f"{faculty_name} {institution_name} affiliated faculty",
    f"{faculty_name} {institution_name} departments",
    f"site:{institution_domain} {faculty_name} secondary appointment"
]
```

**Search Query Variations Example:**
- "Francis Couvares Amherst College joint appointment"
- "Francis Couvares Amherst College affiliated faculty"
- "Francis Couvares Amherst College departments"
- "site:amherst.edu Francis Couvares secondary appointment"

**LLM Extraction Prompt:**
```
System: You are a data extraction assistant that identifies secondary faculty departmental affiliations from search results.

User: Based on the following search results, extract the SECONDARY department (if any) for {faculty_name} at {institution_name}.

Retrieved Information:
{context_str}

Instructions:
- Return ONLY the secondary/joint department name (not the primary one)
- This is for faculty with joint appointments or affiliate positions
- If the faculty is affiliated with multiple departments, return the SECOND one listed
- If the faculty only has ONE department, return "NONE"
- If no secondary department is found, return "NONE"
- Format: Return only the department name or "NONE"

Example output: History
```

**Expected Output**: `History` or `NONE`

---

### 5. RANK (Academic Rank)
**Purpose**: Determine academic rank/title

**Search Queries:**
```python
queries = [
    f"{faculty_name} {institution_name} professor rank",
    f"{faculty_name} {institution_name} title",
    f"{faculty_name} {institution_name} academic rank",
    f"site:{institution_domain} {faculty_name} professor"
]
```

**Search Query Variations Example:**
- "Francis Couvares Amherst College professor rank"
- "Francis Couvares Amherst College title"
- "Francis Couvares Amherst academic rank"
- "site:amherst.edu Francis Couvares professor"

**LLM Extraction Prompt:**
```
System: You are a data extraction assistant that identifies faculty academic ranks from search results.

User: Based on the following search results, extract the academic rank for {faculty_name} at {institution_name}.

Retrieved Information:
{context_str}

Instructions:
- Return ONLY the standardized academic rank
- Standardize to one of: "Full", "Associate", "Assistant", "Emeritus", "Visiting", "Lecturer", "Adjunct"
- "Full Professor" → "Full"
- "Associate Professor" → "Associate"
- "Assistant Professor" → "Assistant"
- "Professor Emeritus" → "Emeritus"
- If no rank is found, return "NOT_FOUND"
- Format: Return only the standardized rank, nothing else

Example output: Full
```

**Expected Output**: `Full`

---

### 6. GENDER
**Purpose**: Identify gender (Note: This is sensitive and may require different approaches)

**Search Queries:**
```python
queries = [
    f"{faculty_name} {institution_name} biography",
    f"{faculty_name} {institution_name} profile",
    f"{faculty_name} {institution_name} about"
]
```

**Search Query Variations Example:**
- "Francis Couvares Amherst College biography"
- "Francis Couvares Amherst College profile"
- "Francis Couvares Amherst about"

**LLM Extraction Prompt:**
```
System: You are a data extraction assistant that identifies gender information from faculty profiles when explicitly stated.

User: Based on the following search results, extract the gender for {faculty_name} at {institution_name}.

Retrieved Information:
{context_str}

Instructions:
- Return ONLY the gender if EXPLICITLY stated in the text
- Look for pronouns used in biographies (he/him, she/her, they/them)
- Standardize to: "Man", "Woman", "Non-binary", or "NOT_FOUND"
- DO NOT infer or assume gender based on names
- Only extract if pronouns or gender identity are clearly stated in the text
- If no explicit gender information is found, return "NOT_FOUND"
- Format: Return only the gender category or "NOT_FOUND"

Example output: Man
```

**Expected Output**: `Man` or `Woman` or `Non-binary` or `NOT_FOUND`

**Note**: Gender inference is sensitive. Consider:
1. Using pronoun detection from biography text
2. Explicitly stating limitations to users
3. Providing "NOT_FOUND" when uncertain rather than guessing

---

### 7. PhD (Doctoral Institution)
**Purpose**: Identify where faculty earned PhD

**Search Queries:**
```python
queries = [
    f"{faculty_name} {institution_name} PhD",
    f"{faculty_name} {institution_name} doctorate",
    f"{faculty_name} {institution_name} education",
    f"{faculty_name} {institution_name} CV",
    f"{faculty_name} PhD dissertation"
]
```

**Search Query Variations Example:**
- "Francis Couvares Amherst College PhD"
- "Francis Couvares Amherst doctorate"
- "Francis Couvares Amherst education"
- "Francis Couvares Amherst CV"
- "Francis Couvares PhD dissertation"

**LLM Extraction Prompt:**
```
System: You are a data extraction assistant that identifies doctoral institutions from faculty profiles.

User: Based on the following search results, extract the institution where {faculty_name} earned their PhD.

Retrieved Information:
{context_str}

Instructions:
- Return ONLY the university/institution name where the PhD was earned
- Use the full official institution name (e.g., "University of Michigan, Ann Arbor" not "UMich")
- Include location if specified (e.g., "University of Wisconsin, Madison")
- If multiple PhDs, return the most recent or most relevant
- If no PhD institution is found, return "NOT_FOUND"
- Format: Return only the institution name, nothing else

Example output: University of Michigan, Ann Arbor
```

**Expected Output**: `University of Michigan, Ann Arbor`

---

### 8. BACHELOR'S (Undergraduate Institution)
**Purpose**: Identify where faculty earned Bachelor's degree

**Search Queries:**
```python
queries = [
    f"{faculty_name} {institution_name} bachelor's degree",
    f"{faculty_name} {institution_name} undergraduate",
    f"{faculty_name} {institution_name} BA BS",
    f"{faculty_name} {institution_name} education",
    f"{faculty_name} alma mater"
]
```

**Search Query Variations Example:**
- "Francis Couvares Amherst College bachelor's degree"
- "Francis Couvares Amherst undergraduate"
- "Francis Couvares Amherst BA BS"
- "Francis Couvares Amherst education"
- "Francis Couvares alma mater"

**LLM Extraction Prompt:**
```
System: You are a data extraction assistant that identifies undergraduate institutions from faculty profiles.

User: Based on the following search results, extract the institution where {faculty_name} earned their Bachelor's degree.

Retrieved Information:
{context_str}

Instructions:
- Return ONLY the university/institution name where the Bachelor's degree was earned
- Use the full official institution name
- Look for terms: BA, BS, AB, Bachelor of Arts, Bachelor of Science, undergraduate degree
- If multiple bachelor's degrees, return the institution of the first one
- If no undergraduate institution is found, return "NOT_FOUND"
- Format: Return only the institution name, nothing else

Example output: University of Pittsburgh
```

**Expected Output**: `University of Pittsburgh`

---

### 9. ALUM STATUS
**Purpose**: Determine if faculty is an alumnus of the institution

**Search Queries:**
```python
queries = [
    f"{faculty_name} {institution_name} alumni",
    f"{faculty_name} {institution_name} graduate",
    f"{faculty_name} {institution_name} alum",
    f"{faculty_name} {institution_name} education history"
]
```

**Search Query Variations Example:**
- "Francis Couvares Amherst College alumni"
- "Francis Couvares Amherst graduate"
- "Francis Couvares Amherst alum"
- "Francis Couvares Amherst education history"

**LLM Extraction Prompt:**
```
System: You are a data extraction assistant that determines if faculty members are alumni of their current institution.

User: Based on the following search results, determine if {faculty_name} is an alumnus of {institution_name}.

Retrieved Information:
{context_str}

Instructions:
- Return ONLY "Alum" or "Non-Alum"
- "Alum" = The faculty member earned ANY degree (Bachelor's, Master's, PhD) from {institution_name}
- "Non-Alum" = The faculty member did NOT earn any degree from {institution_name}
- Check if {institution_name} appears in their educational background
- If unclear or not enough information, return "NOT_FOUND"
- Format: Return only "Alum", "Non-Alum", or "NOT_FOUND"

Example output: Non-Alum
```

**Expected Output**: `Alum` or `Non-Alum`

---

### 10. GOOGLE SCHOLAR LINK
**Purpose**: Find Google Scholar profile URL

**Search Queries:**
```python
queries = [
    f"{faculty_name} {institution_name} Google Scholar",
    f"{faculty_name} scholar.google.com",
    f"site:scholar.google.com {faculty_name}",
    f"{faculty_name} citations"
]
```

**Search Query Variations Example:**
- "Francis Couvares Amherst College Google Scholar"
- "Francis Couvares scholar.google.com"
- "site:scholar.google.com Francis Couvares"
- "Francis Couvares citations"

**LLM Extraction Prompt:**
```
System: You are a data extraction assistant that identifies Google Scholar profile URLs.

User: Based on the following search results, extract the Google Scholar profile URL for {faculty_name} at {institution_name}.

Retrieved Information:
{context_str}

Instructions:
- Return ONLY the Google Scholar profile URL
- The URL should start with "https://scholar.google.com/citations"
- Verify the profile belongs to {faculty_name} (check name match and institution affiliation)
- If multiple profiles found, prioritize the one affiliated with {institution_name}
- If no Google Scholar profile is found, return "NOT_FOUND"
- Format: Return only the URL, nothing else

Example output: https://scholar.google.com/citations?hl=en&user=2Y0_sPsAAAAJ
```

**Expected Output**: `https://scholar.google.com/citations?hl=en&user=2Y0_sPsAAAAJ` or `NOT_FOUND`

---

### 11. GOOGLE SCHOLAR H-INDEX
**Purpose**: Extract h-index from Google Scholar

**Search Queries:**
```python
queries = [
    f"{faculty_name} {institution_name} h-index",
    f"{faculty_name} Google Scholar h-index",
    f"{faculty_name} citations metrics",
    f"site:scholar.google.com {faculty_name}"
]
```

**Search Query Variations Example:**
- "Francis Couvares Amherst College h-index"
- "Francis Couvares Google Scholar h-index"
- "Francis Couvares citations metrics"
- "site:scholar.google.com Francis Couvares"

**LLM Extraction Prompt:**
```
System: You are a data extraction assistant that extracts h-index values from Google Scholar profiles.

User: Based on the following search results, extract the h-index for {faculty_name} at {institution_name}.

Retrieved Information:
{context_str}

Instructions:
- Return ONLY the numerical h-index value
- The h-index is a citation metric shown on Google Scholar profiles
- Look for phrases like "h-index: 12" or "h-index 12"
- Return only the number (e.g., "12" not "h-index: 12")
- If no h-index is found, return "NOT_FOUND"
- Format: Return only the number or "NOT_FOUND"

Example output: 12
```

**Expected Output**: `12` or `NOT_FOUND`

**Note**: For most accurate results, consider scraping the actual Google Scholar page if the profile link is found.

---

### 12. GOOGLE SCHOLAR i10-INDEX
**Purpose**: Extract i10-index from Google Scholar

**Search Queries:**
```python
queries = [
    f"{faculty_name} {institution_name} i10-index",
    f"{faculty_name} Google Scholar i10",
    f"{faculty_name} citations metrics",
    f"site:scholar.google.com {faculty_name}"
]
```

**Search Query Variations Example:**
- "Francis Couvares Amherst College i10-index"
- "Francis Couvares Google Scholar i10"
- "Francis Couvares citations metrics"
- "site:scholar.google.com Francis Couvares"

**LLM Extraction Prompt:**
```
System: You are a data extraction assistant that extracts i10-index values from Google Scholar profiles.

User: Based on the following search results, extract the i10-index for {faculty_name} at {institution_name}.

Retrieved Information:
{context_str}

Instructions:
- Return ONLY the numerical i10-index value
- The i10-index is a citation metric shown on Google Scholar profiles (number of publications with at least 10 citations)
- Look for phrases like "i10-index: 14" or "i10-index 14"
- Return only the number (e.g., "14" not "i10-index: 14")
- If no i10-index is found, return "NOT_FOUND"
- Format: Return only the number or "NOT_FOUND"

Example output: 14
```

**Expected Output**: `14` or `NOT_FOUND`

**Note**: For most accurate results, consider scraping the actual Google Scholar page if the profile link is found.

---

## Implementation Strategy

### Option 1: Sequential Feature Extraction
Process each feature one at a time through the RAG pipeline:

```python
faculty_data = {
    "Institution": institution,
    "Name": faculty_name
}

features = [
    "profile_link", "email", "department_1", "department_2",
    "rank", "gender", "phd", "bachelors", "alum_status",
    "google_scholar_link", "h_index", "i10_index"
]

for feature in features:
    queries = generate_queries(feature, faculty_name, institution)
    result = pipeline.process_query(
        query=queries[0],
        variations=queries[1:]
    )
    faculty_data[feature] = extract_from_llm_response(result['llm_decision'])
```

### Option 2: Parallel Feature Extraction
Search for all features simultaneously, then extract from combined context:

```python
# Aggregate all search results
all_results = []
for feature in features:
    queries = generate_queries(feature, faculty_name, institution)
    for query in queries:
        results = pipeline.search(query, count=3)
        all_results.extend(results)

# Store all in vector DB
pipeline.store_in_vector_db(all_results)

# Extract each feature with targeted semantic search
for feature in features:
    feature_query = generate_feature_query(feature, faculty_name)
    relevant_docs = pipeline.semantic_search(feature_query, top_k=5)
    faculty_data[feature] = pipeline.llm_decision(feature_query, relevant_docs)
```

### Option 3: Profile-First Approach (RECOMMENDED)
1. First, find and scrape the faculty profile page
2. Use the profile content as primary context
3. Fall back to web search for missing fields

```python
# Step 1: Find profile link
profile_queries = generate_queries("profile_link", faculty_name, institution)
profile_result = pipeline.process_query(profile_queries[0], profile_queries[1:])
profile_url = extract_profile_url(profile_result)

# Step 2: Scrape full profile content
profile_content = pipeline.scrape_content(profile_url)

# Step 3: Extract all features from profile using single LLM call
faculty_data = extract_all_features_structured(profile_content, faculty_name, institution)

# Step 4: For missing fields, run targeted searches
for field, value in faculty_data.items():
    if value == "NOT_FOUND":
        targeted_result = search_for_feature(field, faculty_name, institution)
        faculty_data[field] = targeted_result
```

---

## Structured Extraction Prompt (All Features at Once)

If using a profile-first approach, you can extract all features in one LLM call:

```
System: You are a data extraction assistant that extracts structured faculty information from profile pages.

User: Extract the following information for {faculty_name} at {institution_name} from their profile page.

Profile Content:
{scraped_profile_content}

Extract the following fields in JSON format:
1. email - Faculty email address
2. department_1 - Primary department
3. department_2 - Secondary department (or "NONE")
4. rank - Academic rank (Full/Associate/Assistant/Emeritus/Visiting/Lecturer/Adjunct)
5. gender - Gender (Man/Woman/Non-binary, only if explicitly stated via pronouns)
6. phd - Institution where PhD was earned
7. bachelors - Institution where Bachelor's degree was earned
8. alum_status - "Alum" if they graduated from {institution_name}, else "Non-Alum"

For any field not found in the profile, use "NOT_FOUND".

Return ONLY a valid JSON object with these exact keys:
{
  "email": "...",
  "department_1": "...",
  "department_2": "...",
  "rank": "...",
  "gender": "...",
  "phd": "...",
  "bachelors": "...",
  "alum_status": "..."
}
```

---

## Evaluation Metrics

To evaluate accuracy against ground truth:

```python
def evaluate_extraction(extracted_data, ground_truth):
    """
    Compare extracted data against ground truth
    """
    metrics = {
        "total_features": 11,  # Excluding Institution and Name
        "correct": 0,
        "incorrect": 0,
        "not_found": 0,
        "accuracy_per_feature": {}
    }

    features = [
        "profile_link", "email", "department_1", "department_2",
        "rank", "gender", "phd", "bachelors", "alum_status",
        "google_scholar_link", "h_index", "i10_index"
    ]

    for feature in features:
        extracted = extracted_data.get(feature, "NOT_FOUND")
        truth = ground_truth.get(feature, "")

        if extracted == "NOT_FOUND":
            metrics["not_found"] += 1
            metrics["accuracy_per_feature"][feature] = "NOT_FOUND"
        elif extracted.lower().strip() == truth.lower().strip():
            metrics["correct"] += 1
            metrics["accuracy_per_feature"][feature] = "CORRECT"
        else:
            metrics["incorrect"] += 1
            metrics["accuracy_per_feature"][feature] = f"INCORRECT (got: {extracted}, expected: {truth})"

    metrics["accuracy_percentage"] = (metrics["correct"] / metrics["total_features"]) * 100
    metrics["recall_percentage"] = (metrics["correct"] / (metrics["total_features"] - metrics["not_found"])) * 100 if (metrics["total_features"] - metrics["not_found"]) > 0 else 0

    return metrics
```

---

## Next Steps

1. **Implement Query Generator**: Create function to generate search queries for each feature
2. **Implement Feature Extractors**: Create LLM prompts for each feature extraction
3. **Build Orchestrator**: Coordinate multi-feature extraction workflow
4. **Add Profile Scraper**: Enhance `scrape_content()` to handle faculty profile pages
5. **Create Evaluation Suite**: Test against the 3 ground truth examples
6. **Refine Prompts**: Iterate based on evaluation results
7. **Handle Edge Cases**: Missing data, ambiguous results, multiple matches

---

## Ground Truth Test Cases

### Test Case 1: Francis Couvares
- **Institution**: 164465 - Amherst College (MA)
- **Name**: Francis Couvares
- **Expected Profile**: https://www.amherst.edu/people/facstaff/fgcouvares
- **Has Google Scholar**: NO (based on data)

### Test Case 2: Solsiree Del Moral
- **Institution**: 164465 - Amherst College (MA)
- **Name**: Solsiree Del Moral
- **Expected Profile**: https://www.amherst.edu/people/facstaff/sdelmoral
- **Has Google Scholar**: NO (based on data)

### Test Case 3: Pawan Dhingra
- **Institution**: 164465 - Amherst College (MA)
- **Name**: Pawan Dhingra
- **Expected Profile**: https://www.amherst.edu/people/facstaff/pdhingra
- **Has Google Scholar**: YES
- **H-Index**: 12
- **i10-Index**: 14
- **Scholar URL**: https://scholar.google.com/citations?hl=en&user=2Y0_sPsAAAAJ

---

## Additional Considerations

### Error Handling
- **Disambiguation**: Handle faculty with common names (add institution context)
- **Name Variations**: Handle nicknames, middle initials, married names
- **Incomplete Profiles**: Gracefully handle missing information
- **Rate Limiting**: Respect API rate limits for search and scraping

### Optimization
- **Caching**: Cache profile pages to avoid repeated scraping
- **Batch Processing**: Process multiple faculty from same institution together
- **Incremental Updates**: Only re-fetch changed fields

### Data Quality
- **Validation**: Check extracted emails match expected format
- **Consistency**: Cross-validate PhD institution with CV/bio mentions
- **Confidence Scores**: Return confidence levels for each extracted field
