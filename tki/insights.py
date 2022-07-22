from __future__ import annotations
import math
from functools import total_ordering, partial
from typing import Union, Callable
from scipy.stats import rv_continuous, norm, logistic, t, linregress, pearsonr
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from .spaces import SiblingGroup
from .compositeExtractor import CompositeExtractor


def power_dist(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * np.power(x, -b)


def power_dist_fix_beta(
        b: float = 0.7) -> Callable[[np.ndarray, float], np.ndarray]:
    return partial(power_dist, b=b)


def linear_dist(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * x + b


def quadratic_dist(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return a + b * np.power(x, 2)


def cubic_dist(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a + b * np.power(x, 2) + c * np.power(x, 3)


@total_ordering
class InsightResult():
    """Insight Result

    Parameters
    ----------
    data : pd.Series | pd.DataFrame
        Series or DataFrame (compound Insight) to calculate insights score for
    insight : Insight
        Insight Object containing type and configuration
    impact : float
        Impact factor to scale insight score between 0.0 and 1.0
    significance : float
        Significance measure between 0.0 and 1.0
    sibling_group : SiblingGroup
        Sibling Group Object describing the observed part of the data cube
    composite_extractor : CompositeExtractor
        Composite Extractor Object describing the used Extractors
    details : dict
        Additional insight type specific information
    """

    def __init__(self,
                 data: Union[pd.Series, pd.DataFrame],
                 insight: Insight,
                 impact: float = 0.0,
                 significance: float = 0.0,
                 sibling_group: SiblingGroup = None,
                 composite_extractor: CompositeExtractor = None,
                 details: dict = None):
        self.data = data
        self.insight = insight
        self.impact = impact
        self.significance = significance
        self.sibling_group = sibling_group
        self.composite_extractor = composite_extractor
        self.details = details if details else {}
        self.score = impact * significance

    def __lt__(self, other: InsightResult) -> bool:
        return (self.score < other.score)

    def __eq__(self, other: InsightResult) -> bool:
        return (self.score == other.score)

    def __repr__(self) -> str:
        origin = (self.sibling_group, self.composite_extractor)
        return f"{type(self.insight).__name__} - score: {self.score:.2f}, {origin}"

    def plot(self) -> None:
        """Visualizes the insight result using matplotlib
        """
        self.insight.plot(self)


class Insight():
    """Generic insight class for subclassing

    Parameters
    ----------
    data : pd.Series | pd.DataFrame
        Series/DataFrame to calculate insights score for
    impact : float
        Impact factor to scale insight score between 0.0 and 1.0
    """

    def calc_insight(self,
                     data: Union[pd.Series, pd.DataFrame],
                     impact: float = 1.0,
                     sibling_group: SiblingGroup = None,
                     composite_extractor: CompositeExtractor = None
                     ) -> Union[InsightResult, None]:
        raise NotImplementedError

    def plot(self, result: InsightResult) -> None:
        plt.xticks(
            range(result.data.index.size),
            result.data.index.get_level_values(
                result.data.index.names[-1]),
            rotation=90
        )
        plt.plot(result.data.values, label=result.sibling_group.subspace)
        plt.xlabel(result.data.index.names[-1])
        plt.ylabel(result.composite_extractor.aggregator.measurement)
        plt.title(f"{type(self).__name__} - score: {result.score:.2f}")
        plt.legend()


class PointInsight(Insight):
    """Generic class for Point Insights
    """


class ShapeInsight(Insight):
    """Generic class for Shape Insights
    """


class CompoundInsight(Insight):
    """Generic class for Compound Insights
    """

    def plot(self, result: InsightResult) -> None:
        plt.xticks(
            range(result.data.columns.size),
            result.data.columns.get_level_values(
                result.data.columns.names[-1]),
            rotation=90
        )
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


class OutstandingFirstInsight(PointInsight):
    """By predicting a given distribution for the values sorted descending
    this insight calculates the likelyhood of the first (highest) value
    given the later.\n
    The score is calculated by multiplying the impact factor with the
    p-value.
    """

    def __init__(self,
                 dist_law: Callable[..., np.ndarray] = power_dist_fix_beta(0.7),
                 stat_distribution: rv_continuous = norm):
        self.dist_law = dist_law
        self.stat_dist = stat_distribution

    def calc_insight(self,
                     data: pd.Series,
                     impact: float = 1.0,
                     sibling_group: SiblingGroup = None,
                     composite_extractor: CompositeExtractor = None
                     ) -> Union[InsightResult, None]:

        # Sort values descending
        data = data.sort_values(ascending=False)
        ydata = data.values
        xdata = range(1, ydata.size + 1)

        # Linear distributions lead to a perfect fit and a high score
        # Insight is meaningless -> return None
        if (np.unique(np.diff(ydata)).size == 1) or \
                (not np.all(np.isfinite(ydata))):
            return None

        # move all values so that the minimum is 0
        offset = np.min(ydata)

        # Fit data to distribution law
        # pylint: disable=unbalanced-tuple-unpacking
        law_params, _ = curve_fit(
            self.dist_law,
            xdata[1:], ydata[1:] - offset,
            maxfev=5000
        )
        prediction = self.dist_law(xdata, *law_params) + offset

        # Calculate residuals
        residuals = ydata - prediction

        # Fit location parameters of the distribution of residuals
        loc, scale = self.stat_dist.fit(residuals[1:])
        dist_params = {'loc': loc, 'scale': scale}

        # Calculate the the probability of the first residual
        # being bigger than it is
        p_value = self.stat_dist.sf(residuals[0], **dist_params)

        # return Insight Result object
        return InsightResult(
            data=data,
            insight=self,
            impact=impact,
            significance=1 - p_value,
            sibling_group=sibling_group,
            composite_extractor=composite_extractor,
            details={
                'p_value': p_value,
                'law_params': law_params,
                'dist_params': dist_params,
                'residuals': residuals,
                'prediction': prediction
            }
        )

    def plot(self, result: InsightResult) -> None:
        super().plot(result)
        plt.plot(result.details["prediction"],
                 label="null hypothesis")
        plt.legend()


class OutstandingLastInsight(PointInsight):
    """By predicting a given distribution for the values sorted descending
    this insight calculates the likelyhood of the last (lowest) value
    given the later.\n
    The score is calculated by multiplying the impact factor with the
    p-value.
    """

    def __init__(self,
                 dist_law: Callable[..., np.ndarray] = power_dist_fix_beta(0.7),
                 stat_distribution: rv_continuous = norm):
        self.dist_law = dist_law
        self.stat_dist = stat_distribution

    def calc_insight(self,
                     data: pd.Series,
                     impact: float = 1.0,
                     sibling_group: SiblingGroup = None,
                     composite_extractor: CompositeExtractor = None
                     ) -> Union[InsightResult, None]:
        # Sort values descending
        data = data.sort_values(ascending=False)
        ydata = data.values
        xdata = range(1, ydata.size + 1)

        # Linear distributions lead to a perfect fit and a high score
        # Insight is meaningless -> return None
        if (np.unique(np.diff(ydata)).size == 1) or \
                (not np.all(np.isfinite(ydata))):
            return None

        # move all values so that the minimum is 0
        offset = np.min(ydata)

        # Fit data to distribution law
        # pylint: disable=unbalanced-tuple-unpacking
        law_params, _ = curve_fit(
            self.dist_law,
            xdata[:-1], ydata[:-1] - offset,
            maxfev=5000
        )
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
            data=data,
            insight=self,
            impact=impact,
            significance=1 - p_value,
            sibling_group=sibling_group,
            composite_extractor=composite_extractor,
            details={
                'p_value': p_value,
                'law_params': law_params,
                'dist_params': dist_params,
                'residuals': residuals,
                'prediction': prediction
            }
        )

    def plot(self, result: InsightResult) -> None:
        super().plot(result)
        plt.plot(result.details["prediction"],
                 label="null hypothesis")
        plt.legend()


class EvennessInsight(PointInsight):
    """The Evenness Insight returns a score describing the evenness of
    the distribution of the values.
    It is based on the Shannon-Index. A high score stands for a very
    even distribution.
    """

    def calc_insight(self,
                     data: pd.Series,
                     impact: float = 1.0,
                     sibling_group: SiblingGroup = None,
                     composite_extractor: CompositeExtractor = None
                     ) -> Union[InsightResult, None]:

        # Insight is meaningless if there are any NaN or Inf values
        # -> return None
        if not np.all(np.isfinite(data.values)):
            return None

        # The Shannon-Index only works only for values greater or equal to zero.
        if np.any(data.values < 0):
            return None

        # Calculate the Shannon-Index
        p = data.values / np.sum(data.values)
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
            data=data,
            insight=self,
            impact=impact,
            significance=significance,
            sibling_group=sibling_group,
            composite_extractor=composite_extractor,
            details={
                'shannon_index': shannon,
                'test_statistic': test,
                'p_value': p_value
            }
        )

    def plot(self, result: InsightResult) -> None:
        super().plot(result)
        plt.ylim(0, result.data.max() * 1.1)


class TrendInsight(ShapeInsight):
    """Trend Insights create a higher score for significant trends.
    The score is calculated by multiplying the impact factor with the slope
    and the rvalue**2 of a linear regression.

    TODO: Use and compare the Results using the p-value provided by
    scipy.stats.linregress

    Parameters
    ----------
    stat_distribution : scipy.stats.rv_continuous
        Statistical distribution function describing the distribution of slopes.
        Defaults to scipy.stats.logistic
    slope_mean : float
        Position of the distribution of slopes
        Defaults to 0.0
    slope_str : float
        Standard derivation of the distribution of slopes
        Defaults to 1.0
    """

    def __init__(self,
                 stat_distribution: rv_continuous = logistic,
                 slope_mean: float = 0.0,
                 slope_std: float = 0.2):
        self.stat_dist = stat_distribution
        self.dist_params = {'loc': slope_mean, 'scale': slope_std}

    def calc_insight(self,
                     data: pd.Series,
                     impact: float = 1.0,
                     sibling_group: SiblingGroup = None,
                     composite_extractor: CompositeExtractor = None
                     ) -> Union[InsightResult, None]:

        # All values must be real
        if not np.isfinite(data.values).all():
            return None

        # TrendInsight only viable for time based data
        if sibling_group and not sibling_group.dividing_dimension.is_temporal:
            return None

        # Scaling factor for normalization
        scale = (np.max(data.values) - np.min(data.values)) / data.values.size

        # Fit linear regression on the data
        result = linregress(
            x=range(data.values.size),
            y=data.values if scale == 0 else data.values / scale
        )
        slope = result.slope * scale
        intercept = result.intercept * scale
        r_value = result.rvalue

        # Calculate the p-value for the slope on the given distribution
        # A higher slope than assumed creates a higher significance score.
        p_value = self.stat_dist.sf(
            abs(slope), **self.dist_params
        )
        # rvalue**2 is a measurement for the precision of the linear regression.
        # It is used to scale the final score.
        significance = (1 - p_value) * r_value**2

        # return Insight Result object
        return InsightResult(
            data=data,
            insight=self,
            impact=impact,
            significance=significance,
            sibling_group=sibling_group,
            composite_extractor=composite_extractor,
            details={
                'p_value': p_value,
                'r_value': r_value,
                'slope': slope,
                'intercept': intercept
            }
        )

    def plot(self, result: InsightResult) -> None:
        super().plot(result)
        x_data = np.arange(result.data.values.size)
        y_data = result.details["intercept"] + x_data * result.details["slope"]
        plt.plot(y_data, label="regression")
        plt.legend()


class CorrelationInsight(CompoundInsight):
    """Correlation Insights measure the correlation between two series.
    The pearson correlation coefficient is used as corelation measure.
    To calculate the correlation coefficient and the p-value
    scipy.stats.pearsonr is used.
    """

    def calc_insight(self,
                     data: pd.Series,
                     impact: float = 1.0,
                     sibling_group: SiblingGroup = None,
                     composite_extractor: CompositeExtractor = None
                     ) -> Union[InsightResult, None]:

        # CorrelationInsight only viable for ordinal data
        if sibling_group and not sibling_group.dividing_dimension.is_ordinal:
            return None

        # If an input array is constant the correlation coefficient is not defined.
        # -> return None
        if (data.iloc[0] == data.iloc[0].iloc[0]).all() or \
                (data.iloc[1] == data.iloc[1].iloc[0]).all():
            return None
        if not np.isfinite(data.iloc[0]).all() or \
                not np.isfinite(data.iloc[1]).all():
            return None

        # Calculate correlation and p-value
        r_value, p_value = pearsonr(
            data.iloc[0], data.iloc[1]
        )

        # Calculate significance
        significance = 1 - p_value

        # return Insight Result object
        return InsightResult(
            data=data,
            insight=self,
            impact=impact,
            significance=significance,
            sibling_group=sibling_group,
            composite_extractor=composite_extractor,
            details={
                'p_value': p_value,
                'r_value': r_value
            }
        )
