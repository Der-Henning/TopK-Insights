"""Module containing the Shape Insight classes"""
from __future__ import annotations
from scipy.stats import rv_continuous, logistic, linregress
import matplotlib.pyplot as plt
import numpy as np

from tki.composite_extractor import ExtractionResult
from tki.insights import Insight, InsightError, InsightResult


class ShapeInsight(Insight):
    """Parent class for Shape Insights"""
    def calc_insight(self, extraction_result: ExtractionResult) -> InsightResult:
        # Sort Data by index
        extraction_result["data"] = extraction_result["data"].sort_index()

        return super().calc_insight(extraction_result)

    def _check_validity(self, extraction_result: ExtractionResult) -> None:
        super()._check_validity(extraction_result)

        # Data must be finite
        if not np.all(np.isfinite(extraction_result["data"].values)):
            raise InsightError(self, "Data is not finite!")
        
        # Dividing Dimension must be temporal
        if extraction_result["sibling_group"] and \
            not extraction_result["sibling_group"].dividing_dimension.is_temporal:
            raise InsightError(self, "Dividing dimension is not temporal")


class TrendInsight(ShapeInsight):
    """Trend Insights create a higher score for significant trends.
    The score is calculated by multiplying the impact factor with the slope
    and the rvalue**2 of a linear regression.

    TODO: Use and compare the Results using the p-value provided by
    scipy.stats.linregress
    I would prefer using the scipy implementation

    Parameters
    ----------
    stat_distribution : scipy.stats.rv_continuous
        Statistical distribution function describing the distribution of slopes.
        Defaults to scipy.stats.logistic
    slope_mean : float
        Position of the distribution of slopes
        Defaults to 0.0
    slope_std : float
        Standard derivation of the distribution of slopes
        Defaults to 0.2
    """
    def __init__(self,
                 stat_distribution: rv_continuous = logistic,
                 slope_mean: float = 0.2,
                 slope_std: float = 2.0):
        self.stat_dist = stat_distribution
        self.dist_params = {'loc': slope_mean, 'scale': slope_std}

    def _calc_insight(self, extraction_result: ExtractionResult) -> InsightResult:
        data = extraction_result["data"]

        # Fit linear regression on the data
        result = linregress(
            x=range(data.values.size),
            y=data.values)
        slope = result.slope
        intercept = result.intercept
        r_value = result.rvalue

        # Calculate the p-value for the slope on the given distribution
        # A higher slope than assumed creates a higher significance score.
        p_value = self.stat_dist.sf(
            abs(slope), **self.dist_params)

        # rvalue**2 is a measurement for the precision of the linear regression.
        # It is used to scale the final score.
        significance = (1 - p_value) * r_value**2

        # return Insight Result object
        return InsightResult(
            extraction_result=extraction_result,
            data=data,
            insight=self,
            significance=significance,
            p_value=p_value,
            r_value=r_value,
            slope=slope,
            intercept=intercept)

    def plot(self, result: InsightResult) -> None:
        super().plot(result)
        x_data = np.arange(result.data.values.size)
        y_data = result.intercept + x_data * result.slope
        plt.plot(y_data, label="regression")
        plt.legend()
