#!/usr/bin/env python3
"""Test Brave Search API and Google Custom Search as fallback"""

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_brave_search(query="test query"):
    """Test Brave Search API"""
    api_key = os.getenv('BRAVE_KEY')
    if not api_key:
        print("[SKIP] BRAVE_KEY not found in environment")
        return False
    
    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key
    }
    params = {
        "q": query,
        "count": 5
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        
        print(f"[OK] Brave Search API response status: {response.status_code}")
        print(f"[OK] Results count: {len(data.get('web', {}).get('results', []))}")
        
        # Show first result
        results = data.get('web', {}).get('results', [])
        if results:
            first_result = results[0]
            print(f"[OK] First result title: {first_result.get('title', 'N/A')[:50]}...")
            print(f"[OK] First result URL: {first_result.get('url', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Brave Search API failed: {e}")
        return False

def test_google_custom_search(query="test query"):
    """Test Google Custom Search API as fallback"""
    api_key = os.getenv('GOOGLE_API_KEY')
    cse_id = os.getenv('GOOGLE_CSE_ID')
    
    if not api_key or not cse_id:
        print("[SKIP] GOOGLE_API_KEY or GOOGLE_CSE_ID not found in environment")
        return False
    
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": cse_id,
        "q": query,
        "num": 5
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        print(f"[OK] Google CSE API response status: {response.status_code}")
        print(f"[OK] Results count: {len(data.get('items', []))}")
        
        # Show first result
        items = data.get('items', [])
        if items:
            first_item = items[0]
            print(f"[OK] First result title: {first_item.get('title', 'N/A')[:50]}...")
            print(f"[OK] First result URL: {first_item.get('link', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Google CSE API failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing search APIs...")
    
    print("\n=== Testing Brave Search API ===")
    brave_success = test_brave_search("Python programming")
    
    print("\n=== Testing Google Custom Search API ===")
    google_success = test_google_custom_search("Python programming")
    
    if brave_success:
        print("\n[SUCCESS] Brave Search API is working!")
    elif google_success:
        print("\n[SUCCESS] Google CSE API is working as fallback!")
    else:
        print("\n[WARNING] Neither search API is configured. Please set up API keys in .env file.")
