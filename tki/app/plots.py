"""Module contains Plot Object"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from ..insights import InsightResult
from ..insights.compound_insight import CompoundInsight
from ..insights.point_insight import OutstandingInsight, PointInsight
from ..insights.shape_insight import ShapeInsight, TrendInsight


class Plot(go.Figure):
    """Graph Object Figure containing plot based on InsightResult

    Parameters
    ----------
    insight_result : InsightResult
        Insight Result that will be visualized
    *args, **kwargs:
        Further parameters see plotly.graph_object.Figure
    """

    def __init__(self, insight_result: InsightResult, *args, **kwargs):
        if isinstance(insight_result.insight, PointInsight):
            self._point_insight(insight_result, *args, **kwargs)
        elif isinstance(insight_result.insight, ShapeInsight):
            self._shape_insight(insight_result, *args, **kwargs)
        elif isinstance(insight_result.insight, CompoundInsight):
            self._compound_insight(insight_result, *args, **kwargs)
        else:
            super().__init__(*args, **kwargs)

    def _point_insight(self, insight_result: InsightResult, *args, **kwargs
                       ) -> None:
        derived_measure = len(
            insight_result.composite_extractor.extractors) > 0
        measurement = insight_result.composite_extractor.aggregator.measurement
        super().__init__(
            layout=go.Layout(
                xaxis={
                    'tickmode': 'array',
                    'tickvals': list(range(1, len(insight_result.data.values
                                                  ) + 1)),
                    'ticktext': insight_result.data.index.get_level_values(
                        insight_result.data.index.names[-1]),
                    'title': insight_result.data.index.names[-1]},
                yaxis={
                    'title': f"{'Derived measure ' if derived_measure else ''}"
                    f"{measurement}"},
                legend={
                    'orientation': "h",
                    'yanchor': "bottom", 'y': 1.02,
                    'xanchor': "right", 'x': 1}),
            *args, **kwargs)
        self.add_trace(go.Scatter(
            name=str(insight_result.sibling_group.subspace),
            x=list(range(1, len(insight_result.data.values) + 1)),
            y=insight_result.data.values, showlegend=True))
        if isinstance(insight_result.insight, OutstandingInsight):
            self.add_trace(go.Scatter(
                name="Null-Hypothesis",
                x=list(range(1, len(insight_result.data.values) + 1)),
                y=insight_result.prediction,
                mode="lines", line={'color': "orange"}))

    def _shape_insight(self, insight_result: InsightResult, *args, **kwargs
                       ) -> None:
        derived_measure = len(
            insight_result.composite_extractor.extractors) > 0
        measurement = insight_result.composite_extractor.aggregator.measurement
        super().__init__(
            layout=go.Layout(
                xaxis={
                    'tickmode': 'array',
                    'tickvals': list(range(1, len(insight_result.data.values
                                                  ) + 1)),
                    'ticktext': insight_result.data.index.get_level_values(
                        insight_result.data.index.names[-1]),
                    'title': insight_result.data.index.names[-1]},
                yaxis={
                    'title': f"{'Derived measure ' if derived_measure else ''}"
                    f"{measurement}"},
                legend={
                    'orientation': "h",
                    'yanchor': "bottom", 'y': 1.02,
                    'xanchor': "right", 'x': 1}),
            *args, **kwargs)
        self.add_trace(go.Scatter(
            name=str(insight_result.sibling_group.subspace),
            x=list(range(1, len(insight_result.data.values) + 1)),
            y=insight_result.data.values, showlegend=True))
        if isinstance(insight_result.insight, TrendInsight):
            x_data = np.arange(1, len(insight_result.data.values) + 1)
            self.add_trace(go.Scatter(
                name="regression", x=x_data,
                y=insight_result.intercept + (x_data - 1
                                              ) * insight_result.slope,
                mode="lines", line={'color': "orange"}))

    def _compound_insight(self, insight_result: InsightResult, *args, **kwargs
                          ) -> None:
        derived_measure = len(
            insight_result.composite_extractor.extractors) > 0
        measurement = insight_result.composite_extractor.aggregator.measurement
        x_data = list(range(1, insight_result.data.columns.size + 1))
        super().__init__(
            layout=go.Layout(
                xaxis={
                    'tickmode': 'array',
                    'tickvals': x_data,
                    'ticktext': insight_result.data.columns.get_level_values(
                        insight_result.data.columns.names[-1]),
                    'title': insight_result.data.columns.names[-1]},
                yaxis={
                    'title': f"{'Derived measure ' if derived_measure else ''}"
                    f"{measurement}"},
                legend={
                    'orientation': "h",
                    'yanchor': "bottom", 'y': 1.02,
                    'xanchor': "right", 'x': 1}),
            *args, **kwargs)
        for loc, row in insight_result.data.iterrows():
            insight_result.sibling_group.subspace.set(
                insight_result.sibling_group.dividing_dimension, loc)
            self.add_trace(go.Scatter(
                name=str(insight_result.sibling_group.subspace),
                x=x_data, y=row.values))
        insight_result.sibling_group.subspace.set(
            insight_result.sibling_group.dividing_dimension, '*')
