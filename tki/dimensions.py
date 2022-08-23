"""Module containing all Dimension classes"""
from __future__ import annotations
from typing import List
import pandas as pd


class Dimension():
    """Parent Dimension class.

    Parameters
    ----------
    name : str
        Name of column in Dataset
    dependent_dimensions : List[str]
        List of dependent dimensions. e.g. Country -> Region
    value : str
        Current value reducing the Subspace.
        Defaults to '*' aka all
    """
    is_ordinal: bool = False
    is_nominal: bool = False
    is_cardinal: bool = False
    is_temporal: bool = False

    def __init__(self,
                 name: str,
                 dependent_dimensions: List[str] = None,
                 value: str = '*'):
        self.name = name
        self.dependent_dimensions = \
            dependent_dimensions if dependent_dimensions else []
        self.value = value

    def preprocess(self, data: pd.Series) -> pd.Series:
        """Applies preprocessing steps to the values of the dimension.

        Arguments
        ---------
        data : pd.Series
            Data Series

        Returns
        -------
        pd.Series: processed Data Series
        """
        return data

    @property
    def grouper(self) -> pd.Grouper:
        """Returns pandas.Grouper to use for the dimension."""
        return pd.Grouper(key=('dimensions', self.name))

    def __eq__(self, __o: Dimension) -> bool:
        return isinstance(__o, Dimension) and self.name == __o.name

    def __repr__(self) -> str:
        return self.name


class NominalDimension(Dimension):
    """Dimension for nominal data.

    Parameters
    ----------
    name : str
        Name of column in Dataset
    dependent_dimensions : List[str]
        List of dependent dimensions. e.g. Country -> Region
    value : str
        Current value reducing the Subspace.
        Defaults to '*' aka all
    """
    is_nominal: bool = True


class OrdinalDimension(NominalDimension):
    """Dimension for ordinal data.

    Parameters
    ----------
    name : str
        Name of column in Dataset
    dependent_dimensions : List[str]
        List of dependent dimensions. e.g. Country -> Region
    value : str
        Current value reducing the Subspace.
        Defaults to '*' aka all
    """
    is_ordinal: bool = True


class CardinalDimension(OrdinalDimension):
    """Dimension for cardinal data.

    Parameters
    ----------
    name : str
        Name of column in Dataset
    dependent_dimensions : List[str]
        List of dependent dimensions. e.g. Country -> Region
    value : str
        Current value reducing the Subspace.
        Defaults to '*' aka all
    bins : int
        Number of bins to split the data. 0 means no splitting.
        Defaults to 10
    """
    is_cardinal: bool = True

    def __init__(self,
                 name: str,
                 dependent_dimensions: List[str] = None,
                 value: str = '*',
                 bins: int = 10):
        super().__init__(name, dependent_dimensions, value)
        self.bins = bins

    def preprocess(self, data: pd.Series) -> pd.Series:
        if self.bins == 0:
            return data
        return pd.cut(data, bins=self.bins)


class TemporalDimension(CardinalDimension):
    """Dimension for temporal data.

    Parameters
    ----------
    name : str
        Name of column in Dataset
    dependent_dimensions : List[str]
        List of dependent dimensions. e.g. Country -> Region
    value : str
        Current value reducing the Subspace.
        Defaults to '*' aka all
    bins : int
        Number of bins to split the data. 0 means no splitting.
        Defaults to 10
    date_format : str
        Will be used to convert dates as strings to Datetime object.
        Defaults to None
    freq : str
        Frequency string to group the data. E.g. '1Y' or 'M'
        Defaults to None
    """
    is_temporal: bool = True

    def __init__(self,
                 name: str,
                 dependent_dimensions: List[str] = None,
                 value: str = '*',
                 bins: int = 0,
                 date_format: 'str' = None,
                 freq: str = None):
        super().__init__(name, dependent_dimensions, value, bins)
        self.date_format = date_format
        self.freq = freq

    def preprocess(self, data: pd.Series) -> pd.Series:
        # if not self.date_format:
        #     return super().preprocess(data)
        return super().preprocess(
            pd.to_datetime(data, format=self.date_format))

    @property
    def grouper(self) -> pd.Grouper:
        if self.bins > 0:
            return super().grouper
        if not self.freq:
            return super().grouper
        return pd.Grouper(key=('dimensions', self.name), freq=self.freq)
