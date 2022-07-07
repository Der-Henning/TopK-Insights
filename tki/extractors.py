from __future__ import annotations
from typing import Union
import pandas as pd

from .dimensions import Dimension
from .spaces import SiblingGroup


class MetaExtractor(type):
    def __repr__(cls):
        return getattr(cls, '_class_repr')()


class Extractor(metaclass=MetaExtractor):
    name = ''

    def __init__(self, dimension: Dimension):
        self.dimension = dimension

    def is_valid(self, sibling_group: SiblingGroup,
        prev_extractor: Extractor = None) -> bool:
        for dim in sibling_group.subspace.dimensions:
            if dim == self.dimension:
                if dim.value != '*':
                    return False
        return True

    def axis(self, cube: pd.DataFrame) -> int:
        for i, x in enumerate(cube.axes):
            if x.names[-1] == self.dimension.name:
                return i

    def extract(self, cube: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        raise NotImplementedError

    @classmethod
    def _class_repr(cls):
        return cls.name

    def __repr__(self) -> str:
        return str((type(self).name, self.dimension))


class RankExtractor(Extractor):
    name = 'Rank'

    def is_valid(self, sibling_group: SiblingGroup,
        prev_extractor: Extractor = None) -> bool:
        return super().is_valid(sibling_group)

    def extract(self, cube: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        if isinstance(cube, pd.Series):
            return cube.rank(ascending=False)
        return cube.rank(axis=self.axis(cube), ascending=False)


class DeltaPrevExtractor(Extractor):
    name = 'DeltaPrev'

    def is_valid(self, sibling_group: SiblingGroup,
        prev_extractor: Extractor = None) -> bool:
        return super().is_valid(sibling_group) and self.dimension.is_ordinal

    def extract(self, cube: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        if isinstance(cube, pd.Series):
            return cube.diff(1).dropna()
        axis = self.axis(cube)
        return cube.diff(1, axis=axis).dropna(axis=axis)


class DeltaMeanExtractor(Extractor):
    name = 'DeltaMean'

    def is_valid(self, sibling_group: SiblingGroup,
        prev_extractor: Extractor = None) -> bool:
        return super().is_valid(sibling_group) and \
            not (
                not isinstance(prev_extractor, Extractor) and
                self.dimension == sibling_group.dividing_dimension
            )

    def extract(self, cube: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        if isinstance(cube, pd.Series):
            return cube.subtract(cube.mean())
        axis = self.axis(cube)
        return cube.subtract(cube.mean(axis=axis), axis=0 if axis == 1 else 1)


class ProportionExtractor(Extractor):
    name = 'Proportion'

    def is_valid(self, sibling_group: SiblingGroup,
        prev_extractor: Extractor = None) -> bool:
        return super().is_valid(sibling_group) and \
            not isinstance(prev_extractor, Extractor) and \
            self.dimension != sibling_group.dividing_dimension

    def extract(self, cube: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        if isinstance(cube, pd.Series):
            return cube.divide(cube.sum())
        axis = self.axis(cube)
        return cube.divide(cube.sum(axis=axis), axis=0 if axis == 1 else 1)
