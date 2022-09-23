"""Insights Module"""
from tki.insights.insight import Insight, InsightError, InsightResult
from tki.insights.compound_insight import CompoundInsight, CorrelationInsight
from tki.insights.shape_insight import ShapeInsight, TrendInsight
from tki.insights.point_insight import (PointInsight, OutstandingFirstInsight,
        OutstandingLastInsight, EvennessInsight)