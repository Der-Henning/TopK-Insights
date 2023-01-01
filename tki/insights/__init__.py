"""Insights Module"""
from tki.insights.compound_insight import CompoundInsight, CorrelationInsight
from tki.insights.insight import Insight, InsightError, InsightResult
from tki.insights.point_insight import (EvennessInsight,
                                        OutstandingFirstInsight,
                                        OutstandingLastInsight, PointInsight)
from tki.insights.shape_insight import ShapeInsight, TrendInsight

__all__ = ['Insight', 'InsightError', 'InsightResult', 'CompoundInsight',
           'CorrelationInsight', 'EvennessInsight', 'OutstandingFirstInsight',
           'OutstandingLastInsight', 'PointInsight', 'ShapeInsight',
           'TrendInsight']
