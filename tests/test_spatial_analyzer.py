"""
tests/test_spatial_analyzer.py
───────────────────────────────
Unit tests for SpatialAnalyzer: clock position, distance word, and
danger zone detection.  Run with: pytest tests/test_spatial_analyzer.py -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.vision_module import SpatialAnalyzer

# Test frame dimensions
W, H = 640, 480


@pytest.fixture
def analyzer():
    return SpatialAnalyzer(frame_w=W, frame_h=H)


# ─────────────────────────────────────────────────────────────────────────────
# Clock Position Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestClockPosition:
    def test_top_center_is_12_oclock(self, analyzer):
        # Centre of frame, top-third
        x1, y1, x2, y2 = 270, 10, 370, 110
        assert analyzer.clock_position(x1, y1, x2, y2) == "12 o'clock"

    def test_top_left_is_10_oclock(self, analyzer):
        x1, y1, x2, y2 = 10, 10, 100, 100
        assert analyzer.clock_position(x1, y1, x2, y2) == "10 o'clock"

    def test_top_right_is_2_oclock(self, analyzer):
        x1, y1, x2, y2 = 540, 10, 630, 100
        assert analyzer.clock_position(x1, y1, x2, y2) == "2 o'clock"

    def test_middle_left_is_9_oclock(self, analyzer):
        x1, y1, x2, y2 = 10, 180, 100, 300
        assert analyzer.clock_position(x1, y1, x2, y2) == "9 o'clock"

    def test_middle_right_is_3_oclock(self, analyzer):
        x1, y1, x2, y2 = 540, 180, 630, 300
        assert analyzer.clock_position(x1, y1, x2, y2) == "3 o'clock"

    def test_middle_center_is_directly_ahead(self, analyzer):
        x1, y1, x2, y2 = 270, 180, 370, 300
        assert analyzer.clock_position(x1, y1, x2, y2) == "directly ahead"

    def test_bottom_center_is_6_oclock(self, analyzer):
        x1, y1, x2, y2 = 270, 370, 370, 470
        assert analyzer.clock_position(x1, y1, x2, y2) == "6 o'clock"

    def test_bottom_left_is_8_oclock(self, analyzer):
        x1, y1, x2, y2 = 10, 370, 100, 470
        assert analyzer.clock_position(x1, y1, x2, y2) == "8 o'clock"

    def test_bottom_right_is_4_oclock(self, analyzer):
        x1, y1, x2, y2 = 540, 370, 630, 470
        assert analyzer.clock_position(x1, y1, x2, y2) == "4 o'clock"


# ─────────────────────────────────────────────────────────────────────────────
# Distance Word Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDistanceWord:
    def test_large_bbox_is_very_close(self, analyzer):
        # ~25% of frame area → "very close"
        x1, y1, x2, y2 = 100, 100, 460, 380
        assert analyzer.distance_word(x1, y1, x2, y2) == "very close"

    def test_medium_bbox_is_nearby(self, analyzer):
        # ~8% of frame area → "nearby"
        x1, y1, x2, y2 = 250, 200, 390, 280
        assert analyzer.distance_word(x1, y1, x2, y2) == "nearby"

    def test_small_bbox_is_in_the_distance(self, analyzer):
        # ~1% of frame area → "in the distance"
        x1, y1, x2, y2 = 310, 230, 340, 250
        assert analyzer.distance_word(x1, y1, x2, y2) == "in the distance"


# ─────────────────────────────────────────────────────────────────────────────
# Danger Zone Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDangerZone:
    def test_central_bbox_in_danger_zone(self, analyzer):
        # Centred bbox well within danger zone
        x1, y1, x2, y2 = 290, 210, 350, 270
        assert analyzer.in_danger_zone(x1, y1, x2, y2) is True

    def test_corner_bbox_not_in_danger_zone(self, analyzer):
        # Top-left corner, away from centre
        x1, y1, x2, y2 = 0, 0, 50, 50
        assert analyzer.in_danger_zone(x1, y1, x2, y2) is False

    def test_partial_overlap_is_in_danger_zone(self, analyzer):
        # Partial overlap with danger zone still counts as danger
        dz = analyzer.get_danger_zone_rect()  # (x1, y1, x2, y2)
        # Box starts at danger zone left edge
        x1, y1, x2, y2 = dz[0] - 30, dz[1] + 10, dz[0] + 30, dz[3] - 10
        assert analyzer.in_danger_zone(x1, y1, x2, y2) is True

    def test_danger_zone_rect_is_centered(self, analyzer):
        x1, y1, x2, y2 = analyzer.get_danger_zone_rect()
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        assert abs(center_x - W / 2) < 5
        assert abs(center_y - H / 2) < 5
