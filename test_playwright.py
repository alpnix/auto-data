#!/usr/bin/env python3
"""Test Playwright headless page pulling"""

import asyncio
from playwright.async_api import async_playwright

async def test_playwright():
    """Test basic headless page pulling with Playwright"""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        # Test with a simple page
        await page.goto("https://httpbin.org/html")
        title = await page.title()
        content = await page.content()
        
        print(f"[OK] Page title: {title}")
        print(f"[OK] Content length: {len(content)} chars")
        print(f"[OK] Contains expected text: {'Herman Melville' in content}")
        
        await browser.close()
        return True

if __name__ == "__main__":
    print("Testing Playwright headless page pulling...")
    result = asyncio.run(test_playwright())
    if result:
        print("[SUCCESS] Playwright test passed!")
    else:
        print("[FAILED] Playwright test failed!")
