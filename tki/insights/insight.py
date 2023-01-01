"""Module containing the Base Insight classes"""
from __future__ import annotations

from functools import total_ordering
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd

from tki.composite_extractor import ExtractionResult


class InsightError(Exception):
    """Error calculation Insight"""

    def __init__(self, insight: Insight, message="") -> None:
        super().__init__(f"{insight.name} - {message}")


@total_ordering
class InsightResult():
    """Insight Result

    Parameters
    ----------
    extraction_result : ExtractionResult
        Result of Data Extraction containing derived Data and origin
    data : pd.Series | pd.DataFrame
        Series or DataFrame (compound Insight) to calculate insights score for
    insight : Insight
        Insight Object containing type and configuration
    significance : float
        Significance measure between 0.0 and 1.0
    kwargs : dict
        Additional insight type specific information
    """

    def __init__(self,
                 extraction_result: ExtractionResult,
                 data: Union[pd.Series, pd.DataFrame],
                 insight: Insight,
                 significance: float = 0.0,
                 **kwargs):
        self.impact = extraction_result["impact"]
        self.sibling_group = extraction_result["sibling_group"]
        self.composite_extractor = extraction_result["composite_extractor"]
        self.data = data
        self.insight = insight
        self.significance = significance
        self.score = extraction_result["impact"] * significance
        self.__dict__.update(kwargs)

    def __lt__(self, other: InsightResult) -> bool:
        return self.score < other.score

    def __eq__(self, other: InsightResult) -> bool:
        return self.score == other.score

    def __repr__(self) -> str:
        origin = (self.sibling_group, self.composite_extractor)
        return (f"{type(self.insight).__name__} - "
                f"score: {self.score:.2f}, {origin}")

    def plot(self) -> None:
        """Visualizes the insight result using matplotlib"""
        self.insight.plot(self)


class Insight():
    """Parent class for Insights"""

    def calc_insight(self, extraction_result: ExtractionResult
                     ) -> InsightResult:
        """Calculate Insight score

        Arguments
        ---------
        extraction_result : ExtractionResult
            Result of Data Extraction containing derived Data and origin

        Raises
        ------
            InsightError

        Returns
        -------
            InsightResult
        """
        self._check_validity(extraction_result)
        return self._calc_insight(extraction_result)

    def _check_validity(self, extraction_result: ExtractionResult) -> None:
        # This function will be called before the calculation
        # of the insight.
        # Implement to add any tests if the data is valid and
        # raise InsightError if not valid.
        pass

    def _calc_insight(self, extraction_result: ExtractionResult
                      ) -> InsightResult:
        # Must be implemented from any child class.
        raise NotImplementedError

    @property
    def name(self):
        return type(self).__name__

    def plot(self, result: InsightResult) -> None:
        """Visualizes Insight Result using matplotlib

        Arguments
        ---------
        result : InsightResult
            Insight Result
        """
        plt.xticks(
            range(result.data.index.size),
            result.data.index.get_level_values(
                result.data.index.names[-1]),
            rotation=90)
        plt.plot(result.data.values, label=result.sibling_group.subspace)
        plt.xlabel(result.data.index.names[-1])
        derived_measure = len(result.composite_extractor.extractors) > 0
        plt.ylabel(
            f"{'Derived measure ' if derived_measure else ''}"
            f"{result.composite_extractor.aggregator.measurement}")
        plt.title(
            f"{type(self).__name__} - score: {result.score:.2f}\n"
            f"{(result.sibling_group, result.composite_extractor)}")
        plt.legend()
