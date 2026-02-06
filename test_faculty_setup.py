#!/usr/bin/env python3
"""
Quick test to verify faculty research pipeline setup
"""

import os
from dotenv import load_dotenv

load_dotenv()

def test_credentials():
    """Check if all required credentials are present"""
    print("="*60)
    print("Faculty Research Pipeline - Setup Check")
    print("="*60)
    print()

    checks = {
        "Google Custom Search Key": os.getenv('GOOGLE_CUSTOM_SEARCH_KEY'),
        "Google CSE ID": os.getenv('GOOGLE_CSE_ID'),
        "DeepSeek API Key": os.getenv('DEEPSEEK_API_KEY') or os.getenv('DEEPSEEK_API')
    }

    all_good = True
    for name, value in checks.items():
        status = "✓ OK" if value else "✗ MISSING"
        color = "\033[92m" if value else "\033[91m"
        reset = "\033[0m"
        print(f"{color}{status}{reset} {name}")
        if not value:
            all_good = False

    print()

    # Test imports
    print("Checking Python packages...")
    packages = [
        ("flask", "Flask"),
        ("sentence_transformers", "Sentence Transformers"),
        ("pymilvus", "Milvus"),
        ("openai", "OpenAI"),
        ("bs4", "BeautifulSoup4")
    ]

    for module, name in packages:
        try:
            __import__(module)
            print(f"\033[92m✓ OK\033[0m {name}")
        except ImportError:
            print(f"\033[91m✗ MISSING\033[0m {name}")
            all_good = False

    print()
    print("="*60)

    if all_good:
        print("\033[92m✓ All checks passed! Ready to run.\033[0m")
        print()
        print("Start the dashboard with:")
        print("  python faculty_dashboard.py")
        print()
        print("Then open: http://localhost:8081")
    else:
        print("\033[91m✗ Setup incomplete. Please fix the issues above.\033[0m")
        print()
        print("Missing credentials? Create a .env file with:")
        print("  GOOGLE_CUSTOM_SEARCH_KEY=your_key")
        print("  GOOGLE_CSE_ID=your_cse_id")
        print("  DEEPSEEK_API_KEY=your_key")
        print()
        print("Missing packages? Install with:")
        print("  pip install flask sentence-transformers pymilvus openai beautifulsoup4 lxml")

    print("="*60)


if __name__ == "__main__":
    test_credentials()
