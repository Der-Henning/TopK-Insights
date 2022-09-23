"""Module containing the Compound Insight classes"""
from __future__ import annotations
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np

from tki.composite_extractor import ExtractionResult
from tki.insights import Insight, InsightError, InsightResult


class CompoundInsight(Insight):
    """Parent class for Compound Insights"""
    def _check_validity(self, extraction_result: ExtractionResult) -> None:
        super()._check_validity(extraction_result)

        # Data must be finite
        if not np.all(np.isfinite(extraction_result["data"].values)):
            raise InsightError(self, "Data is not finite!")

    def plot(self, result: InsightResult) -> None:
        plt.xticks(
            range(result.data.columns.size),
            result.data.columns.get_level_values(
                result.data.columns.names[-1]),
            rotation=90)
        for loc, row in result.data.iterrows():
            result.sibling_group.subspace.set(
                result.sibling_group.dividing_dimension, loc)
            plt.plot(row.values, label=result.sibling_group.subspace)
        result.sibling_group.subspace.set(
            result.sibling_group.dividing_dimension, '*')
        plt.xlabel(result.data.columns.names[-1])
        plt.ylabel(result.composite_extractor.aggregator.measurement.name)
        plt.title(f"{type(self).__name__} - score: {result.score:.2f}")
        plt.legend()


class CorrelationInsight(CompoundInsight):
    """Correlation Insights measure the correlation between two series.
    The pearson correlation coefficient is used as corelation measure.
    To calculate the correlation coefficient and the p-value
    scipy.stats.pearsonr is used.
    """

    def _check_validity(self, extraction_result: ExtractionResult) -> None:
        super()._check_validity(extraction_result)

        # CorrelationInsight only viable for ordinal data
        if extraction_result["sibling_group"] and \
            not extraction_result["sibling_group"].dividing_dimension.is_ordinal:
            raise InsightError(self, "Dividing dimension is not ordinal")

        # If an input array is constant the correlation coefficient is not defined.
        data = extraction_result["data"]
        if (data.iloc[0] == data.iloc[0].iloc[0]).all() or \
                (data.iloc[1] == data.iloc[1].iloc[0]).all():
            raise InsightError(self, "Constant input array")

    def _calc_insight(self, extraction_result: ExtractionResult) -> InsightResult:
        data = extraction_result["data"]

        # Sort by column labels
        data.sort_index(axis=1, inplace=True)

        # Calculate correlation and p-value
        r_value, p_value = pearsonr(data.iloc[0], data.iloc[1])

        # Calculate significance
        significance = 1 - p_value

        # return Insight Result object
        return InsightResult(
            extraction_result=extraction_result,
            data=data,
            insight=self,
            significance=significance,
            p_value=p_value,
            r_value=r_value)
