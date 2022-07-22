from __future__ import annotations
from typing import List
import pandas as pd


class Dimension():
    is_ordinal: bool = False
    is_nominal: bool = False
    is_cardinal: bool = False
    is_temporal: bool = False

    def __init__(self,
                 name: str,
                 dependend_dimensions: List[Dimension] = None,
                 value: str = '*'):
        self.name = name
        self.dependend_dimensions = \
            dependend_dimensions if dependend_dimensions else []
        self.value = value

    def preprocess(self, data: pd.Series) -> pd.Series:
        return data

    @property
    def grouper(self) -> pd.Grouper:
        return pd.Grouper(key=('dimensions', self.name))

    def __eq__(self, __o: Dimension) -> bool:
        return isinstance(__o, Dimension) and self.name == __o.name

    def __repr__(self) -> str:
        return self.name


class NominalDimension(Dimension):
    is_nominal: bool = True


class OrdinalDimension(NominalDimension):
    is_ordinal: bool = True


class CardinalDimension(OrdinalDimension):
    is_cardinal: bool = True

    def __init__(self,
                 name: str,
                 dependend_dimensions: List[Dimension] = None,
                 value: str = '*',
                 bins: int = 10):
        super().__init__(name, dependend_dimensions, value)
        self.bins = bins

    def preprocess(self, data: pd.Series) -> pd.Series:
        if self.bins == 0:
            return data
        return pd.cut(data, bins=self.bins)


class TemporalDimension(CardinalDimension):
    is_temporal: bool = True

    def __init__(self,
                 name: str,
                 dependend_dimensions: List[Dimension] = None,
                 value: str = '*',
                 bins: int = 0,
                 date_format: 'str' = None,
                 freq: str = None):
        super().__init__(name, dependend_dimensions, value, bins)
        self.date_format = date_format
        self.freq = freq

    def preprocess(self, data: pd.Series) -> pd.Series:
        if not self.date_format:
            return super().preprocess(data)
        return super().preprocess(
            pd.to_datetime(data, format=self.date_format))

    @property
    def grouper(self) -> pd.Grouper:
        if self.bins > 0:
            return super().grouper
        if not self.freq:
            return super().grouper
        return pd.Grouper(key=('dimensions', self.name), freq=self.freq)
