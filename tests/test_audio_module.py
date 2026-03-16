"""
tests/test_audio_module.py
───────────────────────────
Unit tests for PriorityTTSQueue — verifies priority ordering and
queue clearing behaviour without requiring TTS hardware/engine.

Run with: pytest tests/test_audio_module.py -v
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.audio_module import PriorityTTSQueue
from config import TTS_PRIORITY_HIGH, TTS_PRIORITY_LOW


@pytest.fixture
def q():
    return PriorityTTSQueue()


class TestPriorityTTSQueue:

    def test_low_priority_item_retrieved(self, q):
        q.put("Hello world", priority=TTS_PRIORITY_LOW)
        result = q.get(timeout=0.5)
        assert result == "Hello world"

    def test_high_priority_comes_before_low(self, q):
        q.put("Scene description here", priority=TTS_PRIORITY_LOW)
        q.put("Caution! Car very close!", priority=TTS_PRIORITY_HIGH)
        first  = q.get(timeout=0.5)
        second = q.get(timeout=0.5)
        assert first == "Caution! Car very close!"
        assert second == "Scene description here"

    def test_multiple_highs_in_insertion_order(self, q):
        q.put("Alert A", priority=TTS_PRIORITY_HIGH)
        q.put("Alert B", priority=TTS_PRIORITY_HIGH)
        first  = q.get(timeout=0.5)
        second = q.get(timeout=0.5)
        # Same priority → FIFO by insertion sequence
        assert first  == "Alert A"
        assert second == "Alert B"

    def test_clear_removes_all_items(self, q):
        for i in range(5):
            q.put(f"item {i}", priority=TTS_PRIORITY_LOW)
        q.clear()
        result = q.get(timeout=0.2)
        assert result is None

    def test_get_returns_none_when_empty(self, q):
        result = q.get(timeout=0.2)
        assert result is None

    def test_empty_property(self, q):
        assert q.empty is True
        q.put("test", priority=TTS_PRIORITY_LOW)
        assert q.empty is False
        q.clear()
        assert q.empty is True

    def test_high_after_low_is_served_first(self, q):
        # Put several low items, then add a high item
        for i in range(3):
            q.put(f"description {i}", priority=TTS_PRIORITY_LOW)
        q.put("DANGER ALERT", priority=TTS_PRIORITY_HIGH)
        first = q.get(timeout=0.5)
        assert first == "DANGER ALERT"
