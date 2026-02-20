#!/usr/bin/env python3
"""
Pipeline Configuration - Dynamic field definitions and prompts
"""

DATASET_NAME = "faculty_research_dataset"
DATASET_DESCRIPTION = "Compiled faculty dataset with structured information about education, professional details, and demographics for faculty members at Amherst College"

ENTITY_SCHEMA = {"primary_field": "name", "fields": ["name", "institution"]}

TARGET_ENTITIES = [
    {
        "institution": "164465 - Amherst College (MA)",
        "name": "Francis Couvares"
    },
    {
        "institution": "164465 - Amherst College (MA)",
        "name": "Solsiree Del Moral"
    },
    {
        "institution": "164465 - Amherst College (MA)",
        "name": "Pawan Dhingra"
    },
    {
        "institution": "164465 - Amherst College (MA)",
        "name": "Judith Frank"
    },
    {
        "institution": "164465 - Amherst College (MA)",
        "name": "Robert Hayashi"
    },
    {
        "institution": "164465 - Amherst College (MA)",
        "name": "Jallicia Jolly"
    },
    {
        "institution": "164465 - Amherst College (MA)",
        "name": "Karen Sanchez-Eppler"
    },
    {
        "institution": "164465 - Amherst College (MA)",
        "name": "Kiara Vigil"
    },
    {
        "institution": "164465 - Amherst College (MA)",
        "name": "Vanessa Fong"
    },
    {
        "institution": "164465 - Amherst College (MA)",
        "name": "Deborah Gewertz"
    },
    {
        "institution": "164465 - Amherst College (MA)",
        "name": "Jerome Himmelstein"
    },
    {
        "institution": "164465 - Amherst College (MA)",
        "name": "Victoria Nguyen"
    },
    {
        "institution": "164465 - Amherst College (MA)",
        "name": "Li Zhang"
    },
    {
        "institution": "164465 - Amherst College (MA)",
        "name": "Gabriel Arboleda"
    },
    {
        "institution": "164465 - Amherst College (MA)",
        "name": "Dwight Carey"
    },
    {
        "institution": "164465 - Amherst College (MA)",
        "name": "Nicola Courtright"
    },
    {
        "institution": "164465 - Amherst College (MA)",
        "name": "Brian House"
    },
    {
        "institution": "164465 - Amherst College (MA)",
        "name": "Samuel Morse"
    },
    {
        "institution": "164465 - Amherst College (MA)",
        "name": "Yael Rice"
    },
    {
        "institution": "164465 - Amherst College (MA)",
        "name": "Natasha Staller"
    }
]

EXTRACTION_FIELDS = [
    {
        "key": "phd_institution",
        "label": "PhD Institution",
        "description": "Specific university where PhD was obtained",
        "example": "Harvard University",
        "required": False
    },
    {
        "key": "bachelors_institution",
        "label": "Bachelor's Institution",
        "description": "Specific university where bachelor's degree was obtained",
        "example": "Stanford University",
        "required": False
    },
    {
        "key": "alumni_status",
        "label": "Alumni Status",
        "description": "Alumni status where applicable",
        "example": "Alumni of Amherst College",
        "required": False
    },
    {
        "key": "institutional_email",
        "label": "Institutional Email",
        "description": "Institutional email address",
        "example": "faculty@amherst.edu",
        "required": False
    },
    {
        "key": "academic_rank",
        "label": "Academic Rank",
        "description": "Academic rank (e.g., assistant professor, associate professor, full professor)",
        "example": "Associate Professor",
        "required": False
    },
    {
        "key": "gender",
        "label": "Gender",
        "description": "Gender (man or woman), if publicly stated",
        "example": "woman",
        "required": False
    },
    {
        "key": "department_affiliations",
        "label": "Department Affiliations",
        "description": "One or multiple departments (e.g., American Studies and History)",
        "example": "Department of History and American Studies",
        "required": False
    },
    {
        "key": "profile_link",
        "label": "Profile Link",
        "description": "Link to official university profile page",
        "example": "https://www.amherst.edu/people/faculty/jsmith",
        "required": False
    }
]

SEARCH_QUERY_TEMPLATES = [
    "{{name}} {{institution}} faculty profile",
    "{{name}} Amherst College professor",
    "{{name}} CV Amherst College",
    "{{name}} Amherst College department",
    "{{name}} Amherst College email contact",
    "{{name}} Amherst College education background PhD"
]

def get_extraction_prompt(entity: dict, context: str) -> str:
    name = entity.get('name', 'Unknown')
    institution = entity.get('institution', 'Unknown')
    
    prompt = f"""Extract structured information about faculty member {name} from {institution}.

CONTEXT:
{context}

=== FIELD-SPECIFIC EXTRACTION INSTRUCTIONS ===

1. **phd_institution** - PhD/Doctoral Institution
   - Look for: "Ph.D.", "PhD", "Doctorate", "doctoral degree" followed by university name
   - Where to find: Education section, CV, biography, "About" sections
   - Pattern: "Ph.D. in [Field] from [University]" or "received her doctorate from [University]"
   - Return: Just the university name (e.g., "Harvard University", "University of Michigan")

2. **bachelors_institution** - Bachelor's/Undergraduate Institution
   - Look for: "B.A.", "B.S.", "Bachelor", "undergraduate", "A.B." followed by university name
   - Where to find: Education section, CV, biography
   - Pattern: "B.A. from [University]" or "graduated from [University]"
   - Return: Just the university name

3. **alumni_status** - Alumni of Current Institution
   - Check if: Person received ANY degree from {institution}
   - If they got BA/MA/PhD from {institution}: "Alumni"
   - If not: "Non-Alum"
   - If unclear: "Not available"

4. **institutional_email** - Email Address
   - Look for: Email addresses containing the institution domain
   - Pattern: [username]@amherst.edu, [firstname][lastname]@[domain]
   - Where to find: Contact section, profile header, CV
   - Return: Full email address (e.g., "jsmith@amherst.edu")

5. **academic_rank** - Professor Rank
   - Look for exact titles: "Assistant Professor", "Associate Professor", "Full Professor", "Professor"
   - Also look for: Named/endowed chairs (e.g., "John Smith Professor of History")
   - Normalize to: "Assistant", "Associate", "Full", or the full title if endowed
   - Don't confuse with: Visiting positions, adjunct, lecturer, emeritus

6. **gender** - Gender
   - Look for: Pronouns used (he/him = Man, she/her = Woman)
   - Also: Direct statements about gender
   - Return: "Man" or "Woman" only if clearly indicated
   - If unclear: "Not available"

7. **department_affiliations** - Department(s)
   - Look for: Department names in profile, title, or affiliation sections
   - Pattern: "Department of [X]", "Professor of [X]", "[X] Department"
   - Multiple departments: Combine with "and" (e.g., "History and American Studies")
   - Return: Department names without "Department of" prefix

8. **profile_link** - Official Profile URL
   - Look for: URLs from the institution's domain (amherst.edu)
   - Pattern: https://www.amherst.edu/people/facstaff/[username]
   - Prefer: Official faculty profile pages over news articles

=== OUTPUT FORMAT ===

Return ONLY a valid JSON object with these exact keys:

{{
  "phd_institution": "[University name or 'Not available']",
  "bachelors_institution": "[University name or 'Not available']",
  "alumni_status": "[Alumni/Non-Alum/Not available]",
  "institutional_email": "[email@domain.edu or 'Not available']",
  "academic_rank": "[Rank or 'Not available']",
  "gender": "[Man/Woman/Not available]",
  "department_affiliations": "[Department(s) or 'Not available']",
  "profile_link": "[URL or 'Not available']"
}}

Return ONLY the JSON, no explanation or additional text."""
    
    return prompt

# Ground truth for evaluation (keyed by primary field - faculty name)
# Add entries here to enable accuracy evaluation during research
GROUND_TRUTH = {
    # Example entry:
    # "Francis Couvares": {
    #     "phd_institution": "University of Michigan",
    #     "institutional_email": "fgcouvares@amherst.edu",
    #     "gender": "Man",
    #     ...
    # }
}