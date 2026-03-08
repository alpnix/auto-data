import json
from typing import Dict, List, Any

# 1. DATASET METADATA
DATASET_NAME = "faculty_education_demographics"
DATASET_DESCRIPTION = "Dataset of faculty members with education, professional, and demographic information extracted from profile links."

# 2. ENTITY SCHEMA
ENTITY_SCHEMA = {"primary_field": "name", "fields": ["name", "institution"]}

# 3. TARGET ENTITIES (to be populated automatically)
TARGET_ENTITIES = [
    {
        "name": "Francis Couvares",
        "institution": "164465 - Amherst College (MA)"
    },
    {
        "name": "Solsiree Del Moral",
        "institution": "164465 - Amherst College (MA)"
    },
    {
        "name": "Pawan Dhingra",
        "institution": "164465 - Amherst College (MA)"
    },
    {
        "name": "Judith Frank",
        "institution": "164465 - Amherst College (MA)"
    },
    {
        "name": "Robert Hayashi",
        "institution": "164465 - Amherst College (MA)"
    },
    {
        "name": "Jallicia Jolly",
        "institution": "164465 - Amherst College (MA)"
    },
    {
        "name": "Karen Sanchez-Eppler",
        "institution": "164465 - Amherst College (MA)"
    },
    {
        "name": "Kiara Vigil",
        "institution": "164465 - Amherst College (MA)"
    },
    {
        "name": "Vanessa Fong",
        "institution": "164465 - Amherst College (MA)"
    },
    {
        "name": "Deborah Gewertz",
        "institution": "164465 - Amherst College (MA)"
    }
]

# 5. SEARCH QUERY TEMPLATES
SEARCH_QUERY_TEMPLATES = [
    "{{name}} {{institution}} faculty profile",
    "{{name}} education PhD bachelor's",
    "{{institution}} {{name}} CV curriculum vitae",
    "{{name}} academic rank department",
    "{{name}} email contact {{institution}}",
    "{{name}} alumni {{institution}}"
]

# 6. EXTRACTION PROMPT FUNCTION
def get_extraction_prompt(entity: Dict[str, Any], context: str) -> str:
    name = entity.get('name', '')
    institution = entity.get('institution', '')

    prompt = f"""
You are extracting structured information about a faculty member from provided text.

ENTITY:
- Name: {name}
- Institution: {institution}

CONTEXT TEXT:
{context}

INSTRUCTIONS:
Extract the following fields from the context text. If information is not found, leave the value as an empty string.

FIELD-SPECIFIC GUIDANCE:
1. **name**: Faculty member's full name. Usually at the start of a record or prominently displayed.
2. **institutional_affiliation**: The institution they are affiliated with, often including an identifier.
3. **phd_institution**: Look for 'PhD', 'Doctorate', 'Ph.D.' followed by institution name. Check education sections, bios, CVs.
4. **bachelors_institution**: Look for 'BA', 'BS', 'Bachelor', 'B.A.', 'B.S.' followed by institution name.
5. **alumni_status**: Check if they are described as 'alum', 'alumni', 'graduate' of any institution, especially their current one.
6. **academic_rank**: Search for titles like 'Assistant Professor', 'Associate Professor', 'Professor', 'Full Professor', 'Professor Emeritus', 'Lecturer', 'Instructor'.
7. **email**: Look for email patterns (name@institution.edu). Often in contact sections or page footers.
8. **gender**: May be inferred from pronouns (he/him, she/her) or explicit mentions in bios.
9. **department_1**: Primary department. Look for 'Department of X', 'X Department', affiliations listed under name.
10. **department_2**: Secondary/joint department. Check for multiple listings, 'joint appointment', 'also affiliated with'.
11. **profile_link**: URL of the current page or explicitly linked profile. Use the most canonical link.

Return ONLY a JSON object with the extracted values. Use the exact keys below.

OUTPUT FORMAT:
{{
    "name": "",
    "institutional_affiliation": "",
    "phd_institution": "",
    "bachelors_institution": "",
    "alumni_status": "",
    "academic_rank": "",
    "email": "",
    "gender": "",
    "department_1": "",
    "department_2": "",
    "profile_link": ""
}}
"""
    return prompt.strip()

# 7. GROUND TRUTH (for validation)
GROUND_TRUTH = {
    "Francis Couvares": {
        "phd": "University of Michigan, Ann Arbor",
        "bachelors": "University of Pittsburgh",
        "alum_status": "Non-Alum",
        "email": "fgcouvares@amherst.edu",
        "rank": "Full",
        "gender": "Man",
        "department_1": "American Studies",
        "department_2": "History",
        "profile_link": "https://www.amherst.edu/people/facstaff/fgcouvares"
    },
    "Solsiree Del Moral": {
        "phd": "University of Wisconsin, Madison",
        "bachelors": "University of Wisconsin, Madison",
        "alum_status": "Non-Alum",
        "email": "sdelmoral@amherst.edu",
        "rank": "Full",
        "gender": "Woman",
        "department_1": "American Studies",
        "department_2": "Black Studies",
        "profile_link": "https://www.amherst.edu/people/facstaff/sdelmoral"
    },
    "Pawan Dhingra": {
        "phd": "Cornell University",
        "bachelors": "Carleton College",
        "alum_status": "Non-Alum",
        "email": "pdhingra@amherst.edu",
        "rank": "Full",
        "gender": "Man",
        "department_1": "American Studies",
        "department_2": "Education Studies",
        "profile_link": "https://www.amherst.edu/people/facstaff/pdhingra"
    },
    "Judith Frank": {
        "phd": "Cornell University",
        "bachelors": "Hebrew University of Jerusalem",
        "alum_status": "Non-Alum",
        "email": "jefrank@amherst.edu",
        "rank": "Full",
        "gender": "Woman",
        "department_1": "English",
        "profile_link": "https://www.amherst.edu/people/facstaff/jefrank"
    },
    "Robert Hayashi": {
        "phd": "University of Massachusetts, Amherst",
        "bachelors": "Denison University",
        "alum_status": "Non-Alum",
        "email": "rhayashi@amherst.edu",
        "rank": "Full",
        "gender": "Man",
        "department_1": "American Studies",
        "profile_link": "https://www.amherst.edu/people/facstaff/rhayashi"
    },
    "Jallicia Jolly": {
        "phd": "University of Michigan, Ann Arbor",
        "bachelors": "Williams College",
        "alum_status": "Non-Alum",
        "email": "jjolly@amherst.edu",
        "rank": "Assistant",
        "gender": "Woman",
        "department_1": "American Studies",
        "department_2": "Black Studies",
        "profile_link": "https://www.amherst.edu/people/facstaff/jjolly"
    },
    "Karen Sanchez-Eppler": {
        "phd": "Johns Hopkins University",
        "bachelors": "Williams College",
        "alum_status": "Non-Alum",
        "email": "kjsanchezepp@amherst.edu",
        "rank": "Full",
        "gender": "Woman",
        "department_1": "American Studies",
        "department_2": "English",
        "profile_link": "https://www.amherst.edu/people/facstaff/kjsanchezepp"
    },
    "Kiara Vigil": {
        "phd": "University of Michigan, Ann Arbor",
        "bachelors": "Tufts University",
        "alum_status": "Non-Alum",
        "email": "kvigil@amherst.edu",
        "rank": "Associate",
        "gender": "Woman",
        "department_1": "American Studies",
        "department_2": "Education Studies",
        "profile_link": "https://www.amherst.edu/people/facstaff/kvigil"
    },
    "Vanessa Fong": {
        "phd": "Harvard University",
        "bachelors": "Amherst College",
        "alum_status": "Alum",
        "email": "vfong@amherst.edu",
        "rank": "Full",
        "gender": "Woman",
        "department_1": "Asian Studies",
        "department_2": "Anthropology",
        "profile_link": "https://www.amherst.edu/people/facstaff/vfong"
    },
    "Deborah Gewertz": {
        "phd": "City University of New York",
        "bachelors": "City University of New York, Queens College",
        "alum_status": "Non-Alum",
        "email": "dbgewertz@amherst.edu",
        "rank": "Full",
        "gender": "Woman",
        "department_1": "Anthropology",
        "department_2": "Sociology",
        "profile_link": "https://www.amherst.edu/people/facstaff/dbgewertz"
    }
}