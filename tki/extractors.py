"""Module containing all Extractor classes"""
from __future__ import annotations
from typing import Union
import pandas as pd

from tki.dimensions import Dimension
from tki.spaces import SiblingGroup, Subspace


class MetaExtractor(type):
    """Meta Extractor class"""
    def __repr__(cls):
        return getattr(cls, '_class_repr')()


class Extractor(metaclass=MetaExtractor):
    """Parent Extractor class.

    Parameters
    ----------
    dimension : Dimension
        Dimension to apply Extractor to
    """
    name = ''

    def __init__(self, dimension: Dimension):
        self.dimension = dimension

    def is_valid(self, subspace: Subspace,
        prev_extractor: Extractor = None) -> bool:
        """Checks if Extractor is valid for a given Sibling Group
        and Composite Extractor.

        Arguments
        ---------
        sibling_group : SiblingGroup
            Parent Sibling Group
        prev_extractor : Extractor, optional
            Previous Extractor. Defaults to None.

        Returns
        -------
        bool
        """
        for dim in subspace.dimensions:
            if dim == self.dimension and dim.value != '*':
                return False
        if isinstance(prev_extractor, type(self)) and \
            prev_extractor.dimension == self.dimension:
            return False
        return True

    def is_useful(self, sibling_group: SiblingGroup) -> bool:
        """Checks if the Extractor returns a useful Result
        for a given SiblingGroup

        Arguments
        ---------
        sibling_group : SiblingGroup

        Returns
        -------
        bool
        """
        return True

    def _axis(self, cube: pd.DataFrame) -> int:
        for idx, axis in enumerate(cube.axes):
            if axis.names[-1] == self.dimension.name:
                return idx

    def extract(self, cube: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """Executes Extractor on a given data cube.

        Arguments
        ---------
        cube : Union[pd.DataFrame, pd.Series]
            Data Cube

        Returns
        -------
        pd.DataFrame : Derived data cube
        """
        raise NotImplementedError

    @classmethod
    def _class_repr(cls):
        return cls.name

    def __repr__(self) -> str:
        return str((type(self).name, self.dimension))


class RankExtractor(Extractor):
    """Rank Extractor class.

    Parameters
    ----------
    dimension : Dimension
        Dimension to apply Extractor to
    """
    name = 'Rank'

    def is_valid(self, subspace: Subspace, prev_extractor: Extractor = None) -> bool:
        return super().is_valid(subspace, prev_extractor) and \
            not (isinstance(prev_extractor, (DeltaMeanExtractor, ProportionExtractor)) and
            prev_extractor.dimension == self.dimension)

    def extract(self, cube: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        if cube.empty:
            return cube
        if isinstance(cube, pd.Series):
            return cube.rank(ascending=False)
        return cube.rank(axis=self._axis(cube), ascending=False)


class DeltaPrevExtractor(Extractor):
    """Delta Previous Extractor class.

    Parameters
    ----------
    dimension : Dimension
        Dimension to apply Extractor to
    """
    name = 'DeltaPrev'

    def is_valid(self, subspace: Subspace,
        prev_extractor: Extractor = None) -> bool:
        return super().is_valid(subspace, prev_extractor) and \
            self.dimension.is_ordinal

    def extract(self, cube: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        if cube.empty:
            return cube
        if isinstance(cube, pd.Series):
            return cube.diff(1).dropna()
        axis = self._axis(cube)
        return cube.diff(1, axis=axis).dropna(axis=axis)


class DeltaMeanExtractor(Extractor):
    """Delta Mean Extractor class.

    Parameters
    ----------
    dimension : Dimension
        Dimension to apply Extractor to
    """
    name = 'DeltaMean'

    def is_valid(self, subspace: Subspace,
        prev_extractor: Extractor = None) -> bool:
        return super().is_valid(subspace, prev_extractor) and \
            not isinstance(prev_extractor, DeltaMeanExtractor) and \
            not isinstance(prev_extractor, ProportionExtractor)

    def is_useful(self, sibling_group: SiblingGroup):
        return super().is_useful(sibling_group) and \
            self.dimension != sibling_group.dividing_dimension

    def extract(self, cube: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        if cube.empty:
            return cube
        if isinstance(cube, pd.Series):
            return cube.subtract(cube.mean())
        axis = self._axis(cube)
        return cube.subtract(cube.mean(axis=axis), axis=0 if axis == 1 else 1)


class ProportionExtractor(Extractor):
    """Proportion Extractor class.

    Parameters
    ----------
    dimension : Dimension
        Dimension to apply Extractor to
    """
    name = 'Proportion'

    def is_valid(self, subspace: Subspace,
        prev_extractor: Extractor = None) -> bool:
        return super().is_valid(subspace) and \
            not isinstance(prev_extractor, Extractor)

    def is_useful(self, sibling_group: SiblingGroup):
        return super().is_useful(sibling_group) and \
            self.dimension != sibling_group.dividing_dimension

    def extract(self, cube: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        if cube.empty:
            return cube
        if isinstance(cube, pd.Series):
            return cube.divide(cube.sum())
        axis = self._axis(cube)
        return cube.divide(cube.sum(axis=axis), axis=0 if axis == 1 else 1)
