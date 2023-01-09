"""Insights Module"""
from .compound_insight import CompoundInsight, CorrelationInsight
from .insight import Insight, InsightError, InsightResult
from .point_insight import (EvennessInsight, OutstandingFirstInsight,
                            OutstandingLastInsight, PointInsight)
from .shape_insight import ShapeInsight, TrendInsight

__all__ = ['Insight', 'InsightError', 'InsightResult', 'CompoundInsight',
           'CorrelationInsight', 'EvennessInsight', 'OutstandingFirstInsight',
           'OutstandingLastInsight', 'PointInsight', 'ShapeInsight',
           'TrendInsight']
