"""
End-to-End (E2E) UI & Integration Test using Playwright.
Renders and audits the visual dashboard (web_preview.html), verifying
SVG board rendering, vision pipeline images, engine evaluations, and layout integrity.
"""

from pathlib import Path
import pytest
from playwright.sync_api import sync_playwright


def test_dashboard_e2e_playwright():
    html_file = Path(__file__).parent.parent / "web_preview.html"
    assert html_file.exists(), "web_preview.html must exist before running E2E test."

    file_url = f"file://{html_file.resolve()}"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        
        # Navigate to generated visual dashboard
        page.goto(file_url)

        # 1. Assert Title and Header
        title = page.title()
        assert "داشبورد تست و بینایی ماشین" in title

        # 2. Assert Best Move is correctly analyzed as Qxf7#
        best_move_el = page.locator(".best-move-text span").first
        assert "Qxf7#" in best_move_el.text_content()

        # 3. Assert All 12 Vision, AR, Audio, Screen CV, DOM, Overlay, WindMouse & Hybrid Brain cards are rendered
        vision_cards = page.locator(".vision-card")
        assert vision_cards.count() == 12

        # Verify images for visual cards (0-4, 7, 9, 10, 11) loaded without broken links
        for i in [0, 1, 2, 3, 4, 7, 9, 10, 11]:
            img = vision_cards.nth(i).locator("img")
            assert img.is_visible()
            natural_width = img.evaluate("el => el.naturalWidth")
            assert natural_width > 0, f"Vision card {i} image failed to load."

        # Card 5 is the MediaPipe FSM interactive badge
        assert "MediaPipe" in vision_cards.nth(5).text_content()

        # Card 6 is the Phase 5 Persian Voice Assistant badge
        assert "دستیار صوتی هوشمند فارسی" in vision_cards.nth(6).text_content()

        # Card 7 is the Phase 6 Online Screen CV Module card
        assert "فاز ۶: بینایی ماشین شطرنج آنلاین" in vision_cards.nth(7).text_content()

        # Card 8 is the Phase 7 DOM Scraping & Dual-Mode Router badge
        assert "فاز ۷: استخراج DOM و مسیریاب ورودی دوگانه" in vision_cards.nth(8).text_content()

        # Card 9 is the Phase 8 Transparent Screen Overlay badge
        assert "فاز ۸: رابط کاربری شفاف و واقعیت افزوده دسکتاپ" in vision_cards.nth(9).text_content()

        # Card 10 is the Phase 9 WindMouse Anti-Cheat Evasion card
        assert "فاز ۹: اتوماسیون ماوس انسان‌نما و فریب آنتی‌چیت" in vision_cards.nth(10).text_content()

        # Card 11 is the Phase 10 Maia Hybrid Brain card
        assert "فاز ۱۰: مغز هیبریدی (Maia Chess + Polyglot)" in vision_cards.nth(11).text_content()

        # 4. Assert SVG Chessboard is loaded
        svg_el = page.locator("svg").first
        assert svg_el.is_visible()

        # 5. Capture screenshot for audit trail
        screenshot_path = Path(__file__).parent / "fixtures" / "playwright_e2e_audit.png"
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(screenshot_path), full_page=True)
        assert screenshot_path.exists()

        browser.close()
