import pandas as pd

from .dimensions import Dimension


class MetaAggregator(type):
    def __repr__(cls):
        return getattr(cls, '_class_repr')()


class Aggregator(metaclass=MetaAggregator):
    name: str = ''

    def __init__(self, measurement: Dimension):
        self.measurement = measurement

    def agg(self, group: pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
        raise NotImplementedError

    @classmethod
    def _class_repr(cls):
        return cls.name

    def __repr__(self) -> str:
        return str((type(self).name, self.measurement))


class SumAggregator(Aggregator):
    name = 'SUM'

    def agg(self, group: pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
        return group.sum()[self.measurement.name]


class MeanAggregator(Aggregator):
    name = 'MEAN'

    def agg(self, group: pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
        return group.mean()[self.measurement.name]


class CountAggregator(Aggregator):
    name = 'COUNT'

    def agg(self, group: pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
        return group.count()[self.measurement.name]
