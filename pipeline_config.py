#!/usr/bin/env python3
"""
Pipeline Configuration - Dynamic field definitions and prompts
"""

DATASET_NAME = "faculty_research_dataset"
DATASET_DESCRIPTION = "Compile a faculty dataset with structured information about education, professional details, and demographic data for each faculty member."

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
    }
]

EXTRACTION_FIELDS = [
    {
        "key": "name",
        "label": "Faculty Name",
        "description": "Faculty member's name (begins each record)",
        "example": "Francis Couvares",
        "required": True
    },
    {
        "key": "institutional_affiliation",
        "label": "Institutional Affiliation",
        "description": "Institution where faculty member is affiliated",
        "example": "Amherst College",
        "required": True
    },
    {
        "key": "phd_institution",
        "label": "PhD Institution",
        "description": "Specific university where PhD was earned",
        "example": "University of Michigan",
        "required": True
    },
    {
        "key": "bachelors_institution",
        "label": "Bachelor's Institution",
        "description": "Specific university where bachelor's degree was earned",
        "example": "Harvard University",
        "required": True
    },
    {
        "key": "alumni_status",
        "label": "Alumni Status",
        "description": "Alumni status where applicable",
        "example": "Alumni of University of Michigan",
        "required": False
    },
    {
        "key": "academic_rank",
        "label": "Academic Rank",
        "description": "Academic rank (e.g., assistant professor, associate professor, full professor)",
        "example": "Professor",
        "required": True
    },
    {
        "key": "department_affiliations",
        "label": "Department Affiliations",
        "description": "One or multiple departments (e.g., American Studies and History)",
        "example": "American Studies and History",
        "required": True
    },
    {
        "key": "institutional_email",
        "label": "Institutional Email",
        "description": "Institutional email address",
        "example": "fcouvares@amherst.edu",
        "required": False
    },
    {
        "key": "gender",
        "label": "Gender",
        "description": "Gender (man or woman), only if publicly stated",
        "example": "man",
        "required": False
    },
    {
        "key": "profile_link",
        "label": "Profile Link",
        "description": "Provided in Excel sheet for each faculty member",
        "example": "https://www.amherst.edu/people/faculty/fcouvares",
        "required": True
    }
]

SEARCH_QUERY_TEMPLATES = [
    "{name} Amherst College faculty profile",
    "{name} {institution} professor",
    "{name} Amherst College CV",
    "{name} Amherst College department",
    "Amherst College {name} research",
    "{name} Amherst College email contact"
]

def get_extraction_prompt(entity_name: str, institution: str, context: str) -> str:
    return f"""
Extract structured information about the faculty member from the provided context.

FACULTY MEMBER: {entity_name}
INSTITUTION: {institution}

CONTEXT:
{context}

INSTRUCTIONS:
1. Extract ONLY the information explicitly stated in the context above.
2. If information is not found, leave the field empty.
3. For gender, only include if explicitly stated as "man" or "woman".
4. For department affiliations, list all departments mentioned.
5. For academic rank, use standard titles (e.g., Assistant Professor, Associate Professor, Professor).

Please provide the extracted information in the following format:

Name: [Full name]
Institutional Affiliation: [Institution name]
PhD Institution: [University where PhD was earned]
Bachelor's Institution: [University where bachelor's degree was earned]
Alumni Status: [Alumni status if mentioned]
Academic Rank: [Academic title]
Department Affiliations: [Department(s)]
Institutional Email: [Email address]
Gender: [man/woman only if explicitly stated]
Profile Link: [URL from context]

If any field cannot be found, write "Not found" for that field.
"""

GROUND_TRUTH = {
    "Francis Couvares": {
        "name": "Francis Couvares",
        "institutional_affiliation": "164465 - Amherst College (MA)"
    },
    "Solsiree Del Moral": {
        "name": "Solsiree Del Moral",
        "institutional_affiliation": "164465 - Amherst College (MA)"
    },
    "Pawan Dhingra": {
        "name": "Pawan Dhingra",
        "institutional_affiliation": "164465 - Amherst College (MA)"
    },
    "Judith Frank": {
        "name": "Judith Frank",
        "institutional_affiliation": "164465 - Amherst College (MA)"
    }
}

EVALUATION_CONFIG = {
    "required_fields": ["name", "institutional_affiliation", "phd_institution", "bachelors_institution", "academic_rank", "department_affiliations", "profile_link"],
    "optional_fields": ["alumni_status", "institutional_email", "gender"],
    "validation_rules": {
        "name": "Must match the target entity name",
        "institutional_affiliation": "Should contain 'Amherst College'",
        "phd_institution": "Should be a recognized university name",
        "bachelors_institution": "Should be a recognized university name",
        "academic_rank": "Should be a standard academic title",
        "department_affiliations": "Should contain department names",
        "profile_link": "Should be a valid URL",
        "gender": "Should be either 'man', 'woman', or empty"
    },
    "scoring": {
        "required_field_weight": 2.0,
        "optional_field_weight": 1.0,
        "accuracy_threshold": 0.8
    }
}