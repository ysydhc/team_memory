#!/usr/bin/env python3
"""Playwright test: workflow viewer - load YAML via file input."""
from pathlib import Path

from playwright.sync_api import sync_playwright

WORKFLOW_PATH = Path(__file__).resolve().parent.parent / ".tm_cursor/plans/workflows/workflow-optimization-workflow.yaml"
BASE_URL = "http://localhost:9111"


def main():
    if not WORKFLOW_PATH.exists():
        print(f"SKIP: {WORKFLOW_PATH} not found")
        return 0
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            page.goto(f"{BASE_URL}/#workflow-viewer", wait_until="networkidle", timeout=15000)
        except Exception as e:
            print(f"ERR: Cannot reach {BASE_URL}: {e}")
            browser.close()
            return 1
        page.wait_for_load_state("networkidle")
        # Wait for workflow-viewer to load
        page.wait_for_selector("#workflow-drop-zone", timeout=5000)
        # Use file input (equivalent to click-then-select)
        page.locator("#workflow-file-input").set_input_files(str(WORKFLOW_PATH))
        page.wait_for_timeout(1500)
        err_el = page.locator("#workflow-error:visible")
        if err_el.count() > 0:
            msg = err_el.text_content()
            print(f"ERR: {msg}")
            browser.close()
            return 1
        graph = page.locator("#workflow-graph-container:visible")
        if graph.count() == 0:
            print("ERR: Graph container not visible")
            browser.close()
            return 1
        print("OK: Workflow loaded and rendered")
        browser.close()
        return 0


if __name__ == "__main__":
    exit(main())
