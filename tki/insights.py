"""Module containing the Insight classes"""
from __future__ import annotations
import math
from functools import total_ordering, partial
from typing import Union, Callable
from scipy.stats import rv_continuous, norm, logistic, t, linregress, pearsonr
from scipy.optimize import curve_fit
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from tki.composite_extractor import ExtractionResult


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


class InsightError(Exception):
    """Error calculation Insight"""


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

    @property
    def fig(self) -> go.Figure:
        """Returns a plotly.graph_objects.Figure containing
        a visualization of the Insight Result"""
        return self.insight.fig(self)


class Insight():
    """Parent class for Insights"""

    def calc_insight(self, extraction_result: ExtractionResult) -> InsightResult:
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
        raise NotImplementedError

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

    def fig(self, result: InsightResult) -> go.Figure:
        """Creates a Visualization of the Insight Result

        Arguments
        ---------
        result : InsightResult

        Returns
        -------
        plotly.graph_objects.Figure
        """
        derived_measure = len(result.composite_extractor.extractors) > 0
        fig = go.Figure(
            layout=go.Layout(
                xaxis = {
                    'tickmode': 'array',
                    'tickvals': list(range(1, len(result.data.values) + 1)),
                    'ticktext': result.data.index.get_level_values(
                        result.data.index.names[-1]),
                    'title': result.data.index.names[-1]},
                yaxis = {
                    'title': f"{'Derived measure ' if derived_measure else ''}"
                        f"{result.composite_extractor.aggregator.measurement}"},
                legend={
                    'orientation': "h",
                    'yanchor': "bottom", 'y': 1.02,
                    'xanchor': "right", 'x': 1 } ))
        fig.add_trace(go.Scatter(
            name=str(result.sibling_group.subspace),
            x=list(range(1, len(result.data.values) + 1)),
            y=result.data.values, showlegend=True))
        return fig


class PointInsight(Insight):
    """Parent class for Point Insights"""


class ShapeInsight(Insight):
    """Parent class for Shape Insights"""


class CompoundInsight(Insight):
    """Parent class for Compound Insights"""

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

    def fig(self, result: InsightResult) -> go.Figure:
        derived_measure = len(result.composite_extractor.extractors) > 0
        x_data = list(range(1, result.data.columns.size + 1))
        fig = go.Figure(
            layout=go.Layout(
                xaxis = {
                    'tickmode': 'array',
                    'tickvals': x_data,
                    'ticktext': result.data.columns.get_level_values(
                        result.data.columns.names[-1]),
                    'title': result.data.columns.names[-1]},
                yaxis = {
                    'title': f"{'Derived measure ' if derived_measure else ''}"
                        f"{result.composite_extractor.aggregator.measurement}"},
                legend={
                    'orientation': "h",
                    'yanchor': "bottom", 'y': 1.02,
                    'xanchor': "right", 'x': 1} ))
        for loc, row in result.data.iterrows():
            result.sibling_group.subspace.set(
                result.sibling_group.dividing_dimension, loc)
            fig.add_trace(go.Scatter(
                name=str(result.sibling_group.subspace),
                x=x_data, y=row.values))
        result.sibling_group.subspace.set(
            result.sibling_group.dividing_dimension, '*')
        return fig


class OutstandingFirstInsight(PointInsight):
    """By predicting a given distribution for the values sorted descending
    this insight calculates the likelihood of the first (highest) value
    given the later.\n
    The score is calculated by multiplying the impact factor with the
    p-value.

    Parameters
    ----------
    distribution_law : fn(np.ndarray, ...) -> np.ndarray
        Funktion describing a distribution to fit the data
        Defaults to power_dist_fix_beta(0.7)
    stat_distribution : scipy.stats.rv_continuous
        Statistic distribution to describe the distribution of residuals
        Defaults to scipy.stats.norm
    """

    def __init__(self,
                 distribution_law: Callable[..., np.ndarray]
                    = power_dist_fix_beta(0.7),
                 stat_distribution: rv_continuous = norm):
        self.dist_law = distribution_law
        self.stat_dist = stat_distribution

    def calc_insight(self, extraction_result: ExtractionResult) -> InsightResult:
        # Sort values descending
        data = extraction_result["data"].sort_values(ascending=False)
        ydata = data.values
        xdata = range(1, ydata.size + 1)

        # Linear distributions lead to a perfect fit and a high score
        # Insight is meaningless -> raise InsightError
        if (np.unique(np.diff(ydata)).size == 1) or \
                (not np.all(np.isfinite(ydata))):
            raise InsightError("OutstandingFirstInsight: Linear data")

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
        # Resulting params don't make sence though...
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

    def plot(self, result: InsightResult) -> None:
        super().plot(result)
        plt.plot(result.prediction, label="null hypothesis")
        plt.legend()

    def fig(self, result: InsightResult) -> go.Figure:
        fig = super().fig(result)
        fig.add_trace(go.Scatter(
            name="Null-Hypothesis",
            x=list(range(1, len(result.data.values) + 1)),
            y=result.prediction,
            mode="lines", line={'color': "orange"}))
        return fig


class OutstandingLastInsight(PointInsight):
    """By predicting a given distribution for the values sorted descending
    this insight calculates the likelihood of the last (lowest) value
    given the later.\n
    The score is calculated by multiplying the impact factor with the
    p-value.

    Parameters
    ----------
    distribution_law : fn(np.ndarray, ...) -> np.ndarray
        Funktion describing a distribution to fit the data
        Defaults to power_dist_fix_beta(0.7)
    stat_distribution : scipy.stats.rv_continuous
        Statistic distribution to describe the distribution of residuals
        Defaults to scipy.stats.norm
    """

    def __init__(self,
                 distribution_law: Callable[..., np.ndarray]
                 = power_dist_fix_beta(0.7),
                 stat_distribution: rv_continuous = norm):
        self.dist_law = distribution_law
        self.stat_dist = stat_distribution

    def calc_insight(self, extraction_result: ExtractionResult) -> InsightResult:
        # Sort values descending
        data = extraction_result["data"].sort_values(ascending=False)
        ydata = data.values
        xdata = range(1, ydata.size + 1)

        diff = data.diff()
        if diff.sum() == 0:
            raise InsightError("OutstandingLastInsight: No variance")
        if diff[-1] / diff.sum() < 1 / data.size:
            raise InsightError("OutstandingLastInsight: Too low variance on tail")

        # Linear distributions lead to a perfect fit and a high score
        # Insight is meaningless -> return None
        if (np.unique(np.diff(ydata)).size == 1) or \
                (not np.all(np.isfinite(ydata))):
            raise InsightError("OutstandingLastInsight: Linear data")

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

    def plot(self, result: InsightResult) -> None:
        super().plot(result)
        plt.plot(result.prediction, label="null hypothesis")
        plt.legend()

    def fig(self, result: InsightResult) -> go.Figure:
        fig = super().fig(result)
        fig.add_trace(go.Scatter(
            name="Null-Hypothesis",
            x=list(range(1, len(result.data.values) + 1)),
            y=result.prediction,
            mode="lines", line={'color': "orange"}))
        return fig


class EvennessInsight(PointInsight):
    """The Evenness Insight returns a score describing the evenness of
    the distribution of the values.
    It is based on the Shannon-Index. A high score stands for a very
    even distribution.
    """

    def calc_insight(self, extraction_result: ExtractionResult) -> InsightResult:
        data = extraction_result["data"]

        # Insight is meaningless if there are any NaN or Inf values
        # -> raise InsightError
        if not np.all(np.isfinite(data.values)):
            raise InsightError("EvennessInsight: data not finite")

        # The Shannon-Index only works only for values greater or equal to zero.
        if np.any(data.values < 0):
            raise InsightError("EvennessInsight: data contains values < 0")

        # Prevent dividing by zero
        data_sum = np.sum(data.values)
        if data_sum == 0:
            raise InsightError("EvennessInsight: Sum of values is zero")

        # Calculate the Shannon-Index
        p = data.values / data_sum
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

    def calc_insight(self, extraction_result: ExtractionResult) -> InsightResult:
        data = extraction_result["data"]

        # All values must be real
        if not np.isfinite(data.values).all():
            raise InsightError("TrendInsight: data not finite")

        # TrendInsight only viable for time based data
        if extraction_result["sibling_group"] and \
            not extraction_result["sibling_group"].dividing_dimension.is_temporal:
            raise InsightError("TrendInsight: dimension is not temporal")

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

    def fig(self, result: InsightResult) -> go.Figure:
        fig = super().fig(result)
        x_data = np.arange(1, len(result.data.values) + 1)
        fig.add_trace(go.Scatter(
            name="regression", x=x_data,
            y=result.intercept + (x_data - 1) * result.slope,
            mode="lines", line={'color': "orange"}))
        return fig


class CorrelationInsight(CompoundInsight):
    """Correlation Insights measure the correlation between two series.
    The pearson correlation coefficient is used as corelation measure.
    To calculate the correlation coefficient and the p-value
    scipy.stats.pearsonr is used.
    """

    def calc_insight(self, extraction_result: ExtractionResult) -> InsightResult:
        data = extraction_result["data"]

        # CorrelationInsight only viable for ordinal data
        if extraction_result["sibling_group"] and \
            not extraction_result["sibling_group"].dividing_dimension.is_ordinal:
            raise InsightError(
                "CorrelationInsight: dividing dimension is not ordinal")

        # If an input array is constant the correlation coefficient is not defined.
        # -> raise InsightError
        if (data.iloc[0] == data.iloc[0].iloc[0]).all() or \
                (data.iloc[1] == data.iloc[1].iloc[0]).all():
            raise InsightError("CorrelationInsight: constant input array")
        if not np.isfinite(data.iloc[0]).all() or \
                not np.isfinite(data.iloc[1]).all():
            raise InsightError("CorrelationInsight: data is not finite")

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
