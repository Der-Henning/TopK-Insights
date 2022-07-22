import pandas as pd

from .dimensions import Dimension


class AggregationError(Exception):
    pass


class MetaAggregator(type):
    def __repr__(cls):
        return getattr(cls, '_class_repr')()


class Aggregator(metaclass=MetaAggregator):
    name: str = ''

    def __init__(self, measurement: Dimension):
        self.measurement = measurement

    def agg(self, group: pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
        raise NotImplementedError

    @property
    def _col(self):
        return ('measurements', self.measurement.name)

    @classmethod
    def _class_repr(cls):
        return cls.name

    def __repr__(self) -> str:
        return str((type(self).name, self.measurement))


class SumAggregator(Aggregator):
    name = 'SUM'

    def agg(self, group: pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
        if not self.measurement.is_ordinal:
            raise AggregationError("SumAggregator requires an ordinal Measurement!")
        return group[[self._col]].sum()[self._col]


class MeanAggregator(Aggregator):
    name = 'MEAN'

    def agg(self, group: pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
        if not self.measurement.is_ordinal:
            raise AggregationError("MeanAggregator requires an ordinal Measurement!")
        return group[[self._col]].mean()[self._col]


class CountAggregator(Aggregator):
    name = 'COUNT'

    def agg(self, group: pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
        return group[[self._col]].count()[self._col]
