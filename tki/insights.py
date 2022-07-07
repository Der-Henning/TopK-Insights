import math
from functools import total_ordering
from typing import Tuple, Union, Callable
from scipy.stats import rv_continuous, norm, logistic, t, linregress, pearsonr
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from .spaces import SiblingGroup
from .dimensions import Dimension, TemporalDimension
from .compositeExtractor import CompositeExtractor


def power_dist(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * np.power(x, -b)


def power_dist_fix_beta(
    b: float = 0.7) -> Callable[[np.ndarray, float], np.ndarray]:
    return lambda x, a: power_dist(x, a, b)


def linear_dist(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * x + b


def quadratic_dist(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return a + b * np.power(x, 2)


def cubic_dist(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a + b * np.power(x, 2) + c * np.power(x, 3)


distribution_laws = {
    'power': power_dist,
    'power-0.7': power_dist_fix_beta(0.7),
    'linear': linear_dist,
    'quadratic': quadratic_dist,
    'cubic': cubic_dist
}


@total_ordering
class Insight():
    """Generic insight class for subclassing

    Parameters
    ----------
    data : pd.Series
        Series to calculate insights score for
    impact : float
        Impact factor to scale insight score between 0.0 and 1.0
    """

    def __init__(self,
                 data: Union[pd.Series, pd.DataFrame],
                 dividing_dimension: Dimension,
                 impact: float = 1.0,
                 sibling_group: SiblingGroup = None,
                 composite_extractor: CompositeExtractor = None):
        self.data = data.copy()
        self.impact = impact
        self.significance = 0.0
        self.sibling_group = sibling_group
        self.composite_extractor = composite_extractor
        self.dividing_dimension = dividing_dimension

    @property
    def score(self) -> float:
        return self.impact * self.significance

    def __lt__(self, other) -> bool:
        return (self.score < other.score)

    def __eq__(self, other) -> bool:
        return (self.score == other.score)

    def __repr__(self) -> str:
        origin = (self.sibling_group, self.composite_extractor)
        return f"{type(self).__name__} - score: {self.score:.2f}, {origin}"

    def plot(self) -> None:
        plt.xticks(
            range(self.data.index.size),
            self.data.index.get_level_values(self.data.index.names[-1]),
            rotation=90
        )
        plt.plot(self.data.values, label=self.sibling_group.subspace)
        plt.xlabel(self.data.index.names[-1])
        plt.ylabel(self.composite_extractor.aggregator.measurement)
        plt.title(f"{type(self).__name__} - score: {self.score:.2f}")
        plt.legend()


class PointInsight(Insight):
    """Generic class for Point Insights
    """

    def __init__(self, data: pd.Series,
                 dividing_dimension: Dimension,
                 impact: float = 1.0,
                 sibling_group: SiblingGroup = None,
                 composite_extractor: Tuple = None):
        super().__init__(data, dividing_dimension, impact,
                         sibling_group, composite_extractor)


class ShapeInsight(Insight):
    """Generic class for Shape Insights
    """

    def __init__(self, data: pd.Series,
                 dividing_dimension: Dimension,
                 impact: float = 1.0,
                 sibling_group: SiblingGroup = None,
                 composite_extractor: Tuple = None):
        super().__init__(data, dividing_dimension, impact,
                         sibling_group, composite_extractor)


class CompoundInsight(Insight):
    """Generic class for Compound Insights
    """

    def __init__(self, data: pd.DataFrame,
                 dividing_dimension: Dimension,
                 impact: float = 1.0,
                 sibling_group: SiblingGroup = None,
                 composite_extractor: Tuple = None):
        super().__init__(data, dividing_dimension, impact,
                         sibling_group, composite_extractor)

    def plot(self) -> None:
        plt.xticks(
            range(self.data.columns.size),
            self.data.columns.get_level_values(self.data.columns.names[-1]),
            rotation=90
        )
        for loc, row in self.data.iterrows():
            self.sibling_group.subspace.set(self.sibling_group.dividing_dimension, loc)
            plt.plot(row.values, label=self.sibling_group.subspace)
        self.sibling_group.subspace.set(self.sibling_group.dividing_dimension, '*')
        plt.xlabel(self.data.columns.names[-1])
        plt.ylabel(self.composite_extractor.aggregator.measurement.name)
        plt.title(f"{type(self).__name__} - score: {self.score:.2f}")
        plt.legend()


class OutstandingFirstInsight(PointInsight):
    """By predicting a given distribution for the values sorted descending
    this insight calculates the likelyhood of the first (highest) value
    given the later.\n
    The score is calculated by multiplying the impact factor with the 
    p-value.
    """

    def __init__(self,
                 data: pd.Series,
                 dividing_dimension: Dimension,
                 impact: float = 1.0,
                 sibling_group: SiblingGroup = None,
                 composite_extractor: Tuple = None,
                 dist_law: str = 'power-0.7',
                 stat_distribution: rv_continuous = norm):
        super().__init__(data, dividing_dimension, impact,
                         sibling_group, composite_extractor)
        self.dist_law_name = dist_law
        self.dist_law = distribution_laws[dist_law]
        self.stat_dist = stat_distribution

        # Sort values descending
        self.data.sort_values(ascending=False, inplace=True)
        self.ydata = self.data.values
        self.xdata = range(1, self.ydata.size + 1)

        # Linear distributions lead to a perfect fit and a high score
        # Insight is meaningless
        if np.unique(np.diff(self.ydata)).size == 1:
            self.significance = 0.0
            return

        if not np.all(np.isfinite(self.ydata)):
            self.significance = 0.0
            return

        # move all values so that the minimum is 0
        self.offset = np.min(self.ydata)

        # Fit data to distribution law
        self.law_params, _ = curve_fit(
            self.dist_law,
            self.xdata[1:],
            self.ydata[1:] - self.offset,
            maxfev=5000
        )
        self.prediction = self.dist_law(
            self.xdata, *self.law_params) + self.offset

        # Calculate residuals
        self.residuals = self.ydata - self.prediction

        # Fit location parameters of the distribution of residuals
        loc, scale = self.stat_dist.fit(self.residuals[1:])
        self.dist_params = {'loc': loc, 'scale': scale}

        # Calculate the the probability of the first residual
        # being bigger than it is
        self.p_value = self.stat_dist.sf(
            self.residuals[0], **self.dist_params)

        # Significance
        self.significance = 1 - self.p_value

    def plot(self) -> None:
        super().plot()
        plt.plot(self.prediction, label=f"{self.dist_law_name}-law")
        plt.legend()


class OutstandingLastInsight(PointInsight):
    """By predicting a given distribution for the values sorted descending
    this insight calculates the likelyhood of the last (lowest) value
    given the later.\n
    The score is calculated by multiplying the impact factor with the 
    p-value.
    """

    def __init__(self,
                 data: pd.Series,
                 dividing_dimension: Dimension,
                 impact: float = 1.0,
                 sibling_group: SiblingGroup = None,
                 composite_extractor: Tuple = None,
                 dist_law: str = 'power-0.7',
                 stat_distribution: rv_continuous = norm):
        super().__init__(data, dividing_dimension, impact,
                         sibling_group, composite_extractor)
        self.dist_law_name = dist_law
        self.dist_law = distribution_laws[dist_law]
        self.stat_dist = stat_distribution

        # Sort values descending
        self.data.sort_values(ascending=False, inplace=True)
        self.ydata = self.data.values
        self.xdata = range(1, self.ydata.size + 1)

        # Linear distributions lead to a perfect fit and a high score
        # Insight is meaningless
        if np.unique(np.diff(self.ydata)).size == 1:
            self.significance = 0.0
            return

        if not np.all(np.isfinite(self.ydata)):
            self.significance = 0.0
            return

        # move all values so that the minimum is 0
        self.offset = np.min(self.ydata)

        # Fit data to distribution law
        self.law_params, _ = curve_fit(
            self.dist_law,
            self.xdata[:-1],
            self.ydata[:-1] - self.offset,
            maxfev=5000
        )
        self.prediction = self.dist_law(
            self.xdata, *self.law_params) + self.offset

        # Calculate residuals
        self.residuals = self.ydata - self.prediction

        # Fit location parameters of the distribution of residuals
        loc, scale = self.stat_dist.fit(self.residuals[:-1])
        self.dist_params = {'loc': loc, 'scale': scale}

        # Calculate the the probability of the last residual
        # being smaller than it is
        self.p_value = self.stat_dist.cdf(
            self.residuals[-1], **self.dist_params)

        # Significance
        self.significance = 1 - self.p_value

    def plot(self) -> None:
        super().plot()
        plt.plot(self.prediction, label=f"{self.dist_law_name}-law")
        plt.legend()


class EvennessInsight(PointInsight):
    """The Evenness Insight returns a score describing the evenness of
    the distribution of the values.
    It is based on the Shannon-Index. A high score stands for a very
    even distribution.
    """

    def __init__(self, data: pd.Series,
                 dividing_dimension: Dimension,
                 impact: float = 1.0,
                 sibling_group: SiblingGroup = None,
                 composite_extractor: Tuple = None):
        super().__init__(data, dividing_dimension, impact,
                         sibling_group, composite_extractor)

        if not np.all(np.isfinite(self.data.values)):
            self.significance = 0.0
            return

        # The Shannon-Index only works only for values greater or equal to zero.
        if np.any(self.data.values < 0):
            self.significance = 0
        else:
            p = self.data.values / np.sum(self.data.values)
            self.shannon = - \
                np.sum(p * np.log(p, out=np.zeros_like(p),
                       where=(p != 0))) / np.log(self.data.size)
            # Scaled Shannon-Index with the power of 100 to reduce the score
            # for not so even distributions.
            self.significance = np.power(self.shannon, 100)

    def plot(self) -> None:
        super().plot()
        plt.ylim(0)


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
                 data: pd.Series,
                 dividing_dimension: Dimension,
                 impact: float = 1.0,
                 sibling_group: SiblingGroup = None,
                 composite_extractor: Tuple = None,
                 stat_distribution: rv_continuous = logistic,
                 slope_mean: float = 0.0,
                 slope_std: float = 0.2):
        super().__init__(data, dividing_dimension, impact,
                         sibling_group, composite_extractor)

        # All values must be real
        if not np.isfinite(self.data.values).all():
            self.significance = 0.0
            return

        # TrendInsight only viable for time based data
        if not self.dividing_dimension.is_temporal:
            self.significance = 0.0
            return

        # Scaling factor for normalization
        self.scale = (np.max(self.data.values) - np.min(self.data.values)) /\
            self.data.values.size

        # Scale == 0 corresponds to a slope of 0
        if self.scale == 0:
            self.significance = 0.0
            return

        self.stat_dist = stat_distribution
        # Fit linear regression on the data
        result = linregress(
            x=range(self.data.values.size),
            y=self.data.values / self.scale
        )
        self.slope = result.slope
        self.rvalue = result.rvalue
        self.intercept = result.intercept
        self.dist_params = {'loc': slope_mean, 'scale': slope_std}
        # Calculate the p-value for the slope on the given distribution
        # A higher slope than assumed creates a higher significance score.
        self.p_value = self.stat_dist.sf(
            abs(self.slope),
            **self.dist_params
        )
        # rvalue**2 is a measurement for the precision of the linear regression.
        # It is used to scale the final score.
        self.significance = (1 - self.p_value) * self.rvalue**2

    def plot(self) -> None:
        super().plot()
        x = np.arange(self.data.values.size)
        y = (self.intercept + x * self.slope) * self.scale
        plt.plot(y, label="regression")
        plt.legend()


class CorrelationInsight(CompoundInsight):
    """Correlation Insights measure the correlation between two series.
    The pearson correlation coefficient is used as corelation measure.
    To calculate the correlation coefficient and the p-value 
    scipy.stats.pearsonr is used.
    """

    def __init__(self,
                 data: pd.DataFrame,
                 dividing_dimension: Dimension,
                 impact: float = 1.0,
                 sibling_group: SiblingGroup = None,
                 composite_extractor: Tuple = None
                 ):
        super().__init__(data, dividing_dimension, impact,
                         sibling_group, composite_extractor)

        # CorrelationInsight only viable for ordinal data
        if not self.dividing_dimension.is_ordinal:
            self.significance = 0.0
            return

        # Calculate correlation and p-value
        self.r_value, self.p_value = pearsonr(
            self.data.iloc[0], self.data.iloc[1]
        )

        # Calculate significance
        self.significance = 1 - self.p_value
