"""
test_mini_pill_hud.py: Unit tests for Floating Dynamic Island / Mini-Pill HUD (Phase 12).
Verifies initialization, compact/expanded states, hover events, audio mode toggles, and click-through.
"""

import sys
import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QPoint
from ui.mini_pill_hud import MiniPillHUD


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    return app


@pytest.fixture
def pill_hud(qapp):
    hud = MiniPillHUD(headless=True)
    hud.show()
    return hud


def test_mini_pill_init(pill_hud):
    assert pill_hud.width() == MiniPillHUD.COLLAPSED_WIDTH
    assert pill_hud.height() == MiniPillHUD.COLLAPSED_HEIGHT
    assert pill_hud.is_expanded is False
    assert pill_hud.click_through is False


def test_mini_pill_update_state(pill_hud):
    pill_hud.update_state(
        best_move_san="Nf3",
        best_move_uci="g1f3",
        eval_str="+1.25",
        win_chance=69.0,
        concept_title="گسترش اسب",
        concept_explanation="کنترل خانه های مرکزی و آماده سازی قلعه.",
    )
    assert "Nf3" in pill_hud.move_label.text()
    assert pill_hud.eval_label.text() == "+1.25"
    assert pill_hud.win_badge.text() == "69%"
    assert pill_hud.concept_title_label.text() == "🎯 گسترش اسب"


def test_mini_pill_expansion(pill_hud):
    pill_hud.expand_hud()
    assert pill_hud.is_expanded is True
    assert pill_hud.width() == MiniPillHUD.EXPANDED_WIDTH
    assert pill_hud.height() == MiniPillHUD.EXPANDED_HEIGHT
    assert not pill_hud.details_container.isHidden()

    pill_hud.collapse_hud()
    assert pill_hud.is_expanded is False
    assert pill_hud.width() == MiniPillHUD.COLLAPSED_WIDTH
    assert pill_hud.height() == MiniPillHUD.COLLAPSED_HEIGHT
    assert pill_hud.details_container.isHidden()


def test_mini_pill_toggles(pill_hud):
    # Toggle Audio Mode
    initial_mode = pill_hud.stealth_mode_name
    pill_hud._toggle_audio_mode()
    assert pill_hud.stealth_mode_name != initial_mode

    # Toggle Click-Through
    pill_hud._toggle_click_through()
    assert pill_hud.click_through is True
    assert pill_hud.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents) is True
    pill_hud._toggle_click_through()
    assert pill_hud.click_through is False
