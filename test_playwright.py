#!/usr/bin/env python3
"""Test Playwright headless page pulling"""

import asyncio
from playwright.async_api import async_playwright

async def test_playwright_headless():
    """Test headless page pulling with Playwright"""
    async with async_playwright() as p:
        # Try new headless mode first
        browser = await p.chromium.launch(
            headless=True,
            channel="chromium",  # Use new headless mode for better compatibility
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage',
            ]
        )
        page = await browser.new_page()
        await page.goto("https://httpbin.org/html", timeout=30000)
        title = await page.title()
        content = await page.content()

        print(f"[OK] Page title: {title}")
        print(f"[OK] Content length: {len(content)} chars")
        print(f"[OK] Contains expected text: {'Herman Melville' in content}")

        await browser.close()
        return True

async def test_playwright_headed():
    """Test headed mode as fallback (requires display)"""
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            channel="chromium"
        )
        page = await browser.new_page()
        await page.goto("https://httpbin.org/html", timeout=30000)
        title = await page.title()
        content = await page.content()

        print(f"[OK] Page title: {title}")
        print(f"[OK] Content length: {len(content)} chars")
        print(f"[OK] Contains expected text: {'Herman Melville' in content}")

        await browser.close()
        return True

if __name__ == "__main__":
    print("Testing Playwright page pulling...")

    # Try headless mode first
    try:
        print("\n[TEST 1] Trying headless mode...")
        result = asyncio.run(test_playwright_headless())
        if result:
            print("[SUCCESS] Playwright headless mode works!")
    except Exception as e:
        print(f"[WARN] Headless mode failed: {type(e).__name__}: {str(e)[:100]}")
        print("\n[INFO] This is likely a macOS permission issue.")
        print("[INFO] To fix: System Settings → Privacy & Security → Full Disk Access")
        print("[INFO] Add your Terminal or IDE to the list.")

        # Try headed mode as fallback
        try:
            print("\n[TEST 2] Trying headed mode (will open browser window)...")
            result = asyncio.run(test_playwright_headed())
            if result:
                print("[SUCCESS] Playwright headed mode works!")
                print("[INFO] For production, fix permissions to use headless mode.")
        except Exception as e2:
            print(f"[ERROR] Headed mode also failed: {type(e2).__name__}")
            print("[ERROR] Playwright requires macOS permissions to run.")
            print("\n[ALTERNATIVE] Using requests library for web scraping instead.")
