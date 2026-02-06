#!/usr/bin/env python3
"""
Agentic Setup System
Analyzes instruction documents and dataset schemas to auto-configure the pipeline.
This agent has RESTRICTED write permissions - only modifies pipeline_config.py
"""

import os
import json
import csv
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# =============================================================================
# SECURITY: Define allowed write targets
# =============================================================================
ALLOWED_WRITE_FILES = [
    "pipeline_config.py"
]


@dataclass
class DatasetSchema:
    """Parsed dataset schema from CSV/sheet"""
    entity_type: str  # e.g., "faculty", "university", "state"
    entity_fields: List[str]  # Fields that identify the entity (e.g., "name", "institution")
    target_fields: List[Dict]  # Fields to extract with metadata
    sample_entities: List[Dict]  # Sample rows from dataset
    ground_truth: Optional[Dict]  # Ground truth if provided


@dataclass
class Instructions:
    """Parsed instructions from document"""
    goal: str  # Overall research goal
    hints: List[str]  # Extraction hints/tips
    search_strategies: List[str]  # Suggested search approaches
    field_descriptions: Dict[str, str]  # Field-specific guidance
    raw_text: str  # Original instruction text


class AgenticSetup:
    """
    Agent that analyzes input files and configures the research pipeline.

    SECURITY: This agent can ONLY write to files in ALLOWED_WRITE_FILES.
    All other file operations will be rejected.
    """

    def __init__(self):
        self.api_key = os.getenv('DEEPSEEK_API_KEY') or os.getenv('DEEPSEEK_API')
        if self.api_key:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com"
            )
            print("[OK] LLM client initialized")
        else:
            self.client = None
            print("[WARN] No API key found - agent will not function")

        self.logs = []

    def log(self, message: str, level: str = "INFO"):
        """Add to log"""
        entry = f"[{level}] {message}"
        self.logs.append(entry)
        print(entry)

    def get_logs(self) -> List[str]:
        """Get all logs"""
        return self.logs

    def clear_logs(self):
        """Clear logs"""
        self.logs = []

    # =========================================================================
    # FILE PARSING
    # =========================================================================

    def parse_csv_dataset(self, csv_content: str, start_row: int = 0, end_row: int = None) -> DatasetSchema:
        """
        Parse CSV content to extract dataset schema.

        Args:
            csv_content: CSV string content
            start_row: Starting row index (0-based, after header)
            end_row: Ending row index (exclusive), None = all rows

        Expected format:
        - First row: headers (field names)
        - Subsequent rows: data (some may be ground truth)
        """
        self.log("Parsing CSV dataset...")

        lines = csv_content.strip().split('\n')
        reader = csv.DictReader(lines)

        headers = reader.fieldnames or []
        all_rows = list(reader)

        # Apply row range
        if end_row is None:
            end_row = len(all_rows)
        rows = all_rows[start_row:end_row]

        self.log(f"Found {len(headers)} columns, {len(all_rows)} total rows")
        self.log(f"Using rows {start_row} to {end_row} ({len(rows)} entities)")

        # Identify entity fields vs target fields
        # Common entity identifiers
        entity_identifiers = ['name', 'institution', 'university', 'organization',
                             'company', 'state', 'country', 'id', 'entity']

        entity_fields = []
        target_fields = []

        for header in headers:
            header_lower = header.lower().strip()
            is_entity = any(eid in header_lower for eid in entity_identifiers)

            if is_entity and len(entity_fields) < 5:
                entity_fields.append(header)
            else:
                # This is a target field to extract
                target_fields.append({
                    'key': self._to_snake_case(header),
                    'label': header,
                    'original_header': header
                })

        # Determine entity type from first entity field
        entity_type = "entity"
        if entity_fields:
            first_field = entity_fields[0].lower()
            if 'name' in first_field and len(entity_fields) > 1:
                entity_type = "person"
            elif 'university' in first_field or 'institution' in first_field:
                entity_type = "institution"
            elif 'state' in first_field:
                entity_type = "state"
            elif 'company' in first_field:
                entity_type = "company"

        # Extract sample entities and ground truth
        sample_entities = []
        ground_truth = {}

        for row in rows:  # Process all rows in range
            entity = {}
            for ef in entity_fields:
                entity[self._to_snake_case(ef)] = row.get(ef, '')
            sample_entities.append(entity)

            # Build ground truth
            entity_key = row.get(entity_fields[0], '') if entity_fields else ''
            if entity_key:
                truth = {}
                for tf in target_fields:
                    value = row.get(tf['original_header'], '')
                    if value:
                        truth[tf['key']] = value
                if truth:
                    ground_truth[entity_key] = truth

        self.log(f"Entity type: {entity_type}")
        self.log(f"Entity fields: {entity_fields}")
        self.log(f"Target fields: {[f['key'] for f in target_fields]}")

        return DatasetSchema(
            entity_type=entity_type,
            entity_fields=entity_fields,
            target_fields=target_fields,
            sample_entities=sample_entities,
            ground_truth=ground_truth if ground_truth else None
        )

    def parse_instructions(self, instruction_text: str) -> Instructions:
        """
        Parse instruction document to extract research guidance.
        Uses LLM to understand and structure the instructions.
        """
        self.log("Parsing instructions with LLM...")

        if not self.client:
            self.log("No LLM client - using basic parsing", "WARN")
            return Instructions(
                goal="Extract information from web sources",
                hints=[],
                search_strategies=[],
                field_descriptions={},
                raw_text=instruction_text
            )

        prompt = f"""Analyze these research instructions and extract structured guidance.

INSTRUCTIONS:
{instruction_text[:3000]}

Return a JSON object with:
{{
    "goal": "Brief description of the research goal",
    "hints": ["List of extraction hints/tips mentioned"],
    "search_strategies": ["Suggested search approaches"],
    "field_descriptions": {{"field_name": "guidance for extracting this field"}}
}}

Return ONLY valid JSON."""

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a research assistant that extracts structured information from instruction documents. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000
            )

            result_text = response.choices[0].message.content

            # Clean up JSON
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()

            result = json.loads(result_text)

            self.log(f"Extracted goal: {result.get('goal', 'N/A')[:50]}...")

            return Instructions(
                goal=result.get('goal', ''),
                hints=result.get('hints', []),
                search_strategies=result.get('search_strategies', []),
                field_descriptions=result.get('field_descriptions', {}),
                raw_text=instruction_text
            )

        except Exception as e:
            self.log(f"LLM parsing failed: {e}", "ERROR")
            return Instructions(
                goal="Extract information from web sources",
                hints=[],
                search_strategies=[],
                field_descriptions={},
                raw_text=instruction_text
            )

    # =========================================================================
    # CONFIG GENERATION
    # =========================================================================

    def generate_config(
        self,
        schema: DatasetSchema,
        instructions: Instructions,
        extra_instructions: str = ""
    ) -> str:
        """
        Use LLM to generate pipeline_config.py content based on schema and instructions.
        """
        self.log("Generating pipeline configuration with LLM...")

        if not self.client:
            self.log("No LLM client - cannot generate config", "ERROR")
            return None

        # Build context for LLM
        # Include ALL selected entities (not just samples)
        all_entities = schema.sample_entities
        entity_count = len(all_entities)

        schema_info = f"""
DATASET SCHEMA:
- Entity Type: {schema.entity_type}
- Entity Fields: {schema.entity_fields}
- Target Fields to Extract: {json.dumps(schema.target_fields, indent=2)}
- Total Entities to Process: {entity_count}
- ALL Entities (MUST include ALL {entity_count} in TARGET_ENTITIES):
{json.dumps(all_entities, indent=2)}
"""

        instructions_info = f"""
RESEARCH INSTRUCTIONS:
- Goal: {instructions.goal}
- Hints: {json.dumps(instructions.hints)}
- Search Strategies: {json.dumps(instructions.search_strategies)}
- Field Guidance: {json.dumps(instructions.field_descriptions)}
"""

        if extra_instructions:
            instructions_info += f"\nADDITIONAL INSTRUCTIONS:\n{extra_instructions}"

        # Build entity field placeholders for search template instructions
        entity_field_keys = [self._to_snake_case(f) for f in schema.entity_fields]
        entity_placeholder_examples = ", ".join([f'{{{{{k}}}}}' for k in entity_field_keys])

        prompt = f"""Generate a Python configuration file for a research data pipeline.

{schema_info}

{instructions_info}

Generate a complete pipeline_config.py file with:

1. DATASET_NAME and DATASET_DESCRIPTION
2. ENTITY_SCHEMA dict - with "primary_field" (the key used for display/lookup, e.g. the first entity field key) and "fields" (list of all entity dict key names)
3. TARGET_ENTITIES list - CRITICAL: You MUST include ALL {entity_count} entities provided above. Copy them exactly.
4. EXTRACTION_FIELDS list - each field should have:
   - key: snake_case identifier
   - label: Human readable label
   - description: What to look for (use field guidance if available)
   - example: An example value
   - required: True/False
5. SEARCH_QUERY_TEMPLATES list - generate 4-6 search templates using entity field placeholders like {entity_placeholder_examples}
6. get_extraction_prompt(entity: dict, context: str) function that generates the LLM prompt. Access entity fields via entity.get('field_name').
7. GROUND_TRUTH dict (if sample data has values, use them as ground truth)
8. EVALUATION_CONFIG dict

The file should be a complete, valid Python file that can be imported.
Use the exact structure shown below as a template:

```python
#!/usr/bin/env python3
\"\"\"
Pipeline Configuration - Dynamic field definitions and prompts
\"\"\"

DATASET_NAME = "..."
DATASET_DESCRIPTION = "..."

ENTITY_SCHEMA = {{
    "primary_field": "<first entity field key>",
    "fields": [<all entity field keys>]
}}

TARGET_ENTITIES = [...]

EXTRACTION_FIELDS = [...]

SEARCH_QUERY_TEMPLATES = [...]

def get_extraction_prompt(entity: dict, context: str) -> str:
    ...

GROUND_TRUTH = {{...}}

EVALUATION_CONFIG = {{...}}
```

Return ONLY the Python code, no explanation."""

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a Python developer generating configuration files. Return only valid Python code."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4000
            )

            config_code = response.choices[0].message.content

            # Clean up code blocks
            if "```python" in config_code:
                config_code = config_code.split("```python")[1].split("```")[0].strip()
            elif "```" in config_code:
                config_code = config_code.split("```")[1].split("```")[0].strip()

            # POST-PROCESSING: Ensure ALL entities are included
            # The LLM might truncate or modify entities, so we force-replace them
            config_code = self._inject_entities(config_code, all_entities, schema.ground_truth)

            self.log("Configuration generated successfully")
            return config_code

        except Exception as e:
            self.log(f"Config generation failed: {e}", "ERROR")
            return None

    # =========================================================================
    # SECURE FILE WRITING
    # =========================================================================

    def write_config(self, config_content: str, filename: str = "pipeline_config.py") -> Tuple[bool, str]:
        """
        Write configuration to file.

        SECURITY: Only allows writing to files in ALLOWED_WRITE_FILES.
        """
        # Security check
        if filename not in ALLOWED_WRITE_FILES:
            msg = f"SECURITY: Cannot write to '{filename}'. Allowed files: {ALLOWED_WRITE_FILES}"
            self.log(msg, "ERROR")
            return False, msg

        # Validate Python syntax before writing
        try:
            compile(config_content, filename, 'exec')
            self.log("Config syntax validated")
        except SyntaxError as e:
            msg = f"Invalid Python syntax: {e}"
            self.log(msg, "ERROR")
            return False, msg

        # Write file
        try:
            with open(filename, 'w') as f:
                f.write(config_content)
            self.log(f"Configuration written to {filename}")
            return True, f"Successfully wrote {filename}"
        except Exception as e:
            msg = f"Failed to write file: {e}"
            self.log(msg, "ERROR")
            return False, msg

    # =========================================================================
    # MAIN WORKFLOW
    # =========================================================================

    def process(
        self,
        dataset_csv: str,
        instructions_text: str,
        extra_instructions: str = "",
        auto_write: bool = False,
        start_row: int = 0,
        end_row: int = None
    ) -> Dict:
        """
        Main workflow: Parse inputs → Generate config → Optionally write.

        Args:
            dataset_csv: CSV content with dataset schema
            instructions_text: Research instructions
            extra_instructions: Additional user instructions
            auto_write: If True, automatically write config file
            start_row: Starting row index (0-based)
            end_row: Ending row index (exclusive), None = all

        Returns:
            Dict with results and generated config
        """
        self.clear_logs()
        self.log("Starting agentic setup process...")

        result = {
            'success': False,
            'schema': None,
            'instructions': None,
            'config': None,
            'config_written': False,
            'logs': [],
            'total_rows': 0
        }

        # Step 1: Parse dataset
        try:
            schema = self.parse_csv_dataset(dataset_csv, start_row, end_row)
            result['schema'] = {
                'entity_type': schema.entity_type,
                'entity_fields': schema.entity_fields,
                'target_fields': [f['key'] for f in schema.target_fields],
                'sample_count': len(schema.sample_entities),
                'has_ground_truth': schema.ground_truth is not None
            }
        except Exception as e:
            self.log(f"Failed to parse dataset: {e}", "ERROR")
            result['logs'] = self.get_logs()
            return result

        # Step 2: Parse instructions
        try:
            instructions = self.parse_instructions(instructions_text)
            result['instructions'] = {
                'goal': instructions.goal,
                'hints_count': len(instructions.hints),
                'strategies_count': len(instructions.search_strategies)
            }
        except Exception as e:
            self.log(f"Failed to parse instructions: {e}", "ERROR")
            result['logs'] = self.get_logs()
            return result

        # Step 3: Generate config
        config = self.generate_config(schema, instructions, extra_instructions)
        if not config:
            result['logs'] = self.get_logs()
            return result

        result['config'] = config

        # Step 4: Optionally write config
        if auto_write:
            success, msg = self.write_config(config)
            result['config_written'] = success
            self.log(msg)

        result['success'] = True
        result['logs'] = self.get_logs()

        self.log("Agentic setup complete!")
        return result

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _inject_entities(self, config_code: str, entities: List[Dict], ground_truth: Optional[Dict]) -> str:
        """
        Post-process LLM-generated config to ensure ALL entities are included
        and ENTITY_SCHEMA is present. Fixes the common issue of LLM truncating
        or modifying the entity list.
        """
        import re as regex

        # Determine entity field keys from actual data
        entity_keys = list(entities[0].keys()) if entities else []
        primary_key = entity_keys[0] if entity_keys else None

        # Generate the correct TARGET_ENTITIES string
        entities_str = "TARGET_ENTITIES = " + json.dumps(entities, indent=4)

        # Replace the existing TARGET_ENTITIES in the config
        # Match pattern: TARGET_ENTITIES = [...] (handles multi-line)
        pattern = r'TARGET_ENTITIES\s*=\s*\[[\s\S]*?\n\]'
        if regex.search(pattern, config_code):
            config_code = regex.sub(pattern, entities_str, config_code)
            self.log(f"Injected {len(entities)} entities into TARGET_ENTITIES")
        else:
            self.log("Could not find TARGET_ENTITIES pattern to replace", "WARN")

        # Inject/fix ENTITY_SCHEMA
        if entity_keys:
            schema_dict = {"primary_field": primary_key, "fields": entity_keys}
            schema_str = "ENTITY_SCHEMA = " + json.dumps(schema_dict)

            schema_pattern = r'ENTITY_SCHEMA\s*=\s*\{[^}]*\}'
            if regex.search(schema_pattern, config_code):
                config_code = regex.sub(schema_pattern, schema_str, config_code)
                self.log(f"Injected ENTITY_SCHEMA: primary={primary_key}, fields={entity_keys}")
            else:
                # Insert before TARGET_ENTITIES
                config_code = config_code.replace(entities_str, schema_str + "\n\n" + entities_str)
                self.log(f"Added ENTITY_SCHEMA: primary={primary_key}, fields={entity_keys}")

        # Also fix GROUND_TRUTH to include all entities
        if ground_truth and primary_key:
            full_ground_truth = {}
            for entity in entities:
                primary_value = entity.get(primary_key, str(entity))
                if primary_value in ground_truth:
                    full_ground_truth[primary_value] = ground_truth[primary_value]
                else:
                    # Create ground truth entry from all entity fields
                    full_ground_truth[primary_value] = dict(entity)

            ground_truth_str = "GROUND_TRUTH = " + json.dumps(full_ground_truth, indent=4)

            # Replace existing GROUND_TRUTH
            gt_pattern = r'GROUND_TRUTH\s*=\s*\{[\s\S]*?\n\}'
            if regex.search(gt_pattern, config_code):
                config_code = regex.sub(gt_pattern, ground_truth_str, config_code)
                self.log(f"Injected ground truth for {len(full_ground_truth)} entities")

        return config_code

    def _to_snake_case(self, text: str) -> str:
        """Convert text to snake_case"""
        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)
        # Replace spaces with underscores
        text = re.sub(r'\s+', '_', text.strip())
        # Convert to lowercase
        return text.lower()

    def count_csv_rows(self, csv_content: str) -> int:
        """Count total data rows in CSV (excluding header)"""
        lines = csv_content.strip().split('\n')
        return max(0, len(lines) - 1)  # Subtract header row


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """CLI interface for testing"""
    import sys

    print("="*60)
    print("Agentic Setup System")
    print("="*60)

    agent = AgenticSetup()

    # Example: Test with faculty data
    test_csv = """Institution,Name,PhD,Bachelor's,Email,Rank,Gender,Department 1,Department 2
Amherst College,Francis Couvares,University of Michigan,University of Pittsburgh,fgcouvares@amherst.edu,Full,Man,American Studies,History
Amherst College,Solsiree Del Moral,University of Wisconsin,University of Wisconsin,sdelmoral@amherst.edu,Full,Woman,American Studies,Black Studies"""

    test_instructions = """
Research Goal: Collect faculty information from university websites.

Tips for extraction:
- Look for "Degrees" or "Education" sections on profile pages
- Email addresses usually follow pattern: username@institution.edu
- Rank can be found in page titles or headers
- Department affiliations often listed in sidebar or header

Search strategies:
- Search "[Name]" "[Institution]" faculty profile
- Search "[Name]" site:institution.edu
- Look for CV or curriculum vitae links
"""

    result = agent.process(
        dataset_csv=test_csv,
        instructions_text=test_instructions,
        extra_instructions="Focus on American universities",
        auto_write=False  # Don't auto-write in test
    )

    print("\n" + "="*60)
    print("RESULT")
    print("="*60)
    print(f"Success: {result['success']}")
    print(f"Schema: {result['schema']}")
    print(f"Instructions: {result['instructions']}")
    print(f"Config generated: {result['config'] is not None}")

    if result['config']:
        print("\n" + "="*60)
        print("GENERATED CONFIG (first 1000 chars)")
        print("="*60)
        print(result['config'][:1000])


if __name__ == "__main__":
    main()
