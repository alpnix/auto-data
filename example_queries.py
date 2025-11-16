#!/usr/bin/env python3
"""
Example queries for testing the education data pipeline
"""

from education_pipeline import EducationDataPipeline
import time


def run_example_queries():
    """Run several example queries about different educational institutions"""

    # Example queries to test
    queries = [
        {
            "main": "Does MIT have a food bank?",
            "variations": [
                "MIT food pantry",
                "MIT student food assistance",
                "food resources at MIT"
            ]
        },
        {
            "main": "Does UC Berkeley offer mental health counseling?",
            "variations": [
                "Berkeley mental health services",
                "UC Berkeley counseling center",
                "mental health support at Berkeley"
            ]
        },
        {
            "main": "Does Yale have free tutoring services?",
            "variations": [
                "Yale tutoring center",
                "Yale academic support",
                "free tutoring at Yale"
            ]
        }
    ]

    # Initialize pipeline
    pipeline = EducationDataPipeline()
    pipeline.initialize()

    try:
        results = []

        for i, query_set in enumerate(queries, 1):
            print(f"\n{'='*80}")
            print(f"QUERY {i}/{len(queries)}")
            print(f"{'='*80}")

            # Process query
            result = pipeline.process_query(
                query=query_set["main"],
                variations=query_set["variations"]
            )

            results.append(result)

            # Print summary
            print(f"\n[SUMMARY] Query: {result['query']}")
            print(f"[SUMMARY] Results collected: {result['search_results_count']}")
            print(f"[SUMMARY] Top match score: {result['relevant_docs'][0]['score']:.3f}")
            print(f"[SUMMARY] Decision preview: {result['llm_decision'][:150]}...")

            # Rate limiting - wait between queries to avoid API limits
            if i < len(queries):
                print("\n[INFO] Waiting 3 seconds before next query (rate limiting)...")
                time.sleep(3)

        # Print final summary
        print(f"\n\n{'='*80}")
        print("FINAL SUMMARY")
        print(f"{'='*80}\n")

        for i, result in enumerate(results, 1):
            print(f"{i}. {result['query']}")
            print(f"   Answer: {result['llm_decision'][:100]}...")
            print()

    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    print("Running example queries...")
    print("Note: This will make multiple API calls and may take a few minutes.")
    print("Press Ctrl+C to cancel.\n")

    try:
        run_example_queries()
        print("\n[SUCCESS] All queries completed!")
    except KeyboardInterrupt:
        print("\n\n[CANCELLED] Interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] {e}")
