"""Module containing the Point Insight classes"""
from __future__ import annotations

import math
from functools import partial
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm, rv_continuous, t

from ..composite_extractor import ExtractionResult
from ..insights import Insight, InsightError, InsightResult


def power_dist(arr: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Power distribution"""
    return alpha * np.power(arr, -beta)


def power_dist_fix_beta(
        beta: float = 0.7) -> Callable[[np.ndarray, float], np.ndarray]:
    """Power distribution with fixed beta"""
    return partial(power_dist, beta=beta)


def linear_dist(arr: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Linear distribution"""
    return alpha * arr + beta


def quadratic_dist(arr: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Quadratic distribution"""
    return alpha + beta * np.power(arr, 2)


def cubic_dist(arr: np.ndarray, alpha: float,
               beta: float, gamma: float) -> np.ndarray:
    """Cubic distribution"""
    return alpha + beta * np.power(arr, 2) + gamma * np.power(arr, 3)


class PointInsight(Insight):
    """Parent class for Point Insights"""

    def _check_validity(self, extraction_result: ExtractionResult) -> None:
        super()._check_validity(extraction_result)

        # Data must be finite
        if not np.all(np.isfinite(extraction_result["data"].values)):
            raise InsightError(self, "Data is not finite!")


class OutstandingInsight(PointInsight):
    """Parent class for Outstanding Point Insights"""

    def __init__(self,
                 distribution_law: Callable[..., np.ndarray]
                 = power_dist_fix_beta(0.7),
                 stat_distribution: rv_continuous = norm):
        self.dist_law = distribution_law
        self.stat_dist = stat_distribution

    def calc_insight(self, extraction_result: ExtractionResult
                     ) -> InsightResult:
        # Sort Data descending by value
        extraction_result["data"] = extraction_result["data"].sort_values(
            ascending=False)

        return super().calc_insight(extraction_result)

    def _check_validity(self, extraction_result: ExtractionResult) -> None:
        super()._check_validity(extraction_result)

        # Linear distributions lead to a perfect fit and a high score
        # Insight is meaningless -> raise InsightError
        if np.unique(np.diff(extraction_result["data"].values)).size == 1:
            raise InsightError(self, "Linear data")

    def plot(self, result: InsightResult) -> None:
        super().plot(result)
        plt.plot(result.prediction, label="null hypothesis")
        plt.legend()


class OutstandingFirstInsight(OutstandingInsight):
    """By predicting a given distribution for the values sorted descending
    this insight calculates the likelihood of the first (highest) value
    given the later.\n
    The score is calculated by multiplying the impact factor with the
    p-value.

    Parameters
    ----------
    distribution_law : fn(np.ndarray, ...) -> np.ndarray
        Function describing a distribution to fit the data
        Defaults to power_dist_fix_beta(0.7)
    stat_distribution : scipy.stats.rv_continuous
        Statistic distribution to describe the distribution of residuals
        Defaults to scipy.stats.norm
    """

    def _calc_insight(self, extraction_result: ExtractionResult
                      ) -> InsightResult:
        data = extraction_result["data"]
        ydata = data.values
        xdata = range(1, ydata.size + 1)

        # move all values so that the minimum is 0
        offset = np.min(ydata)

        # Fit data to distribution law
        # pylint: disable=unbalanced-tuple-unpacking
        law_params, _ = curve_fit(
            self.dist_law,
            xdata[1:], ydata[1:] - offset,
            maxfev=5000)
        prediction = self.dist_law(xdata, *law_params) + offset

        # Calculate residuals
        residuals = ydata - prediction

        # Fit location parameters of the distribution of residuals
        # TODO:
        # scipy >= 1.9.0 provides a new fit function
        # scipy.stats.fit(dist, data) -> FitResult function
        # Resulting params don't make sense though...
        # fitResult = stats.fit(self.stat_dist, residuals[1:])
        loc, scale = self.stat_dist.fit(residuals[1:])
        dist_params = {'loc': loc, 'scale': scale}

        # Calculate the the probability of the first residual
        # being bigger than it is
        p_value = self.stat_dist.sf(residuals[0], **dist_params)

        # return Insight Result object
        return InsightResult(
            extraction_result=extraction_result,
            data=data,
            insight=self,
            significance=1 - p_value,
            p_value=p_value,
            law_params=law_params,
            dist_params=dist_params,
            residuals=residuals,
            prediction=prediction)


class OutstandingLastInsight(OutstandingInsight):
    """By predicting a given distribution for the values sorted descending
    this insight calculates the likelihood of the last (lowest) value
    given the later.\n
    The score is calculated by multiplying the impact factor with the
    p-value.

    Parameters
    ----------
    distribution_law : fn(np.ndarray, ...) -> np.ndarray
        Function describing a distribution to fit the data
        Defaults to power_dist_fix_beta(0.7)
    stat_distribution : scipy.stats.rv_continuous
        Statistic distribution to describe the distribution of residuals
        Defaults to scipy.stats.norm
    """

    def _check_validity(self, extraction_result: ExtractionResult) -> None:
        super()._check_validity(extraction_result)

        data = extraction_result["data"]
        # Check for low variance
        diff = data.diff()
        if diff.sum() == 0:
            raise InsightError(self, "No variance")
        if diff[-1] / diff.sum() < 1 / data.size:
            raise InsightError(self, "Too low variance on tail")

    def _calc_insight(self, extraction_result: ExtractionResult
                      ) -> InsightResult:
        data = extraction_result["data"]
        ydata = data.values
        xdata = range(1, ydata.size + 1)

        # move all values so that the minimum is 0
        # Excluding the minimum of ydata
        offset = np.sort(ydata)[1]

        # Fit data to distribution law
        # pylint: disable=unbalanced-tuple-unpacking
        law_params, _ = curve_fit(
            self.dist_law,
            xdata[:-1], ydata[:-1] - offset,
            maxfev=5000)
        prediction = self.dist_law(xdata, *law_params) + offset

        # Calculate residuals
        residuals = ydata - prediction

        # Fit location parameters of the distribution of residuals
        loc, scale = self.stat_dist.fit(residuals[:-1])
        dist_params = {'loc': loc, 'scale': scale}

        # Calculate the the probability of the first residual
        # being bigger than it is
        p_value = self.stat_dist.cdf(residuals[-1], **dist_params)

        # return Insight Result object
        return InsightResult(
            extraction_result=extraction_result,
            data=data,
            insight=self,
            significance=1 - p_value,
            p_value=p_value,
            law_params=law_params,
            dist_params=dist_params,
            residuals=residuals,
            prediction=prediction)


class EvennessInsight(PointInsight):
    """The Evenness Insight returns a score describing the evenness of
    the distribution of the values.
    It is based on the Shannon-Index. A high score stands for a very
    even distribution.
    """

    def _check_validity(self, extraction_result: ExtractionResult) -> None:
        super()._check_validity(extraction_result)

        # The Shannon-Index only works only for values
        # greater or equal to zero.
        if np.any(extraction_result["data"].values < 0):
            raise InsightError(self, "Data contains values < 0")

        # Prevent dividing by zero
        data_sum = np.sum(extraction_result["data"].values)
        if data_sum == 0:
            raise InsightError(self, "Sum of values is zero")

    def _calc_insight(self, extraction_result: ExtractionResult
                      ) -> InsightResult:
        data = extraction_result["data"]

        # Calculate the Shannon-Index
        p = data.values / np.sum(extraction_result["data"].values)
        shannon = -np.sum(p * np.log(p, out=np.zeros_like(p),
                                     where=(p != 0))) / np.log(data.size)

        # Calculate the test statistic
        if shannon >= 1:
            shannon = 1.0
            test = np.inf
        else:
            test = shannon * math.sqrt(0.001 / (1 - shannon**2))

        # Calculate the p_value
        p_value = t.sf(test, data.size - 2) * 2

        # Calculate the significance
        significance = 1 - p_value

        # return Insight Result object
        return InsightResult(
            extraction_result=extraction_result,
            data=data,
            insight=self,
            significance=significance,
            shannon_index=shannon,
            test_statistic=test,
            p_value=p_value)

    def plot(self, result: InsightResult) -> None:
        super().plot(result)
        plt.ylim(0, result.data.max() * 1.1)
