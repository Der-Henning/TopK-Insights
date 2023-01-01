"""Module containing the Subspace and SiblingGroup classes"""
from copy import deepcopy
from functools import cached_property
from typing import Any, List, Union

import numpy as np
import pandas as pd

from tki.aggregators import Aggregator
from tki.dimensions import Dimension


class Subspace():
    """Subspace

    Parameters
    ----------
    dataset : pandas.Dataset
        Dataset
    dimensions : List[Dimension]
        List of Dimensions with set values
    measurements : List[Dimension]
        List of Measurements
    """

    def __init__(
            self,
            dataset: pd.DataFrame,
            dimensions: List[Dimension],
            measurements: List[Dimension]):
        self._dataset = dataset
        self.dimensions = dimensions
        self.measurements = measurements

    @cached_property
    def sums(self) -> pd.DataFrame:
        """Measurement absolute Sums for the current Data Set"""
        return self.dataset['measurements'].abs().sum()

    @cached_property
    def dataset(self) -> pd.DataFrame:
        """Filtered Data Set"""
        dim_filter = [
            (self._dataset['dimensions', dimension.name] == dimension.value)
            for dimension in self.dimensions if dimension.value != '*']
        if len(dim_filter) > 0:
            return self._dataset[np.logical_and.reduce(dim_filter)]
        return self._dataset

    def cube(self,
             aggregator: Aggregator,
             dimensions: List[Dimension]
             ) -> Union[pd.Series, pd.DataFrame]:
        """Aggregate Data cube

        Arguments
        ---------
        aggregator : Aggregator
            Aggregator to apply to Dataset
        dimensions : List[Dimension]
            Dimension pair

        Returns
        -------
        pd.Series | pd.DataFrame : Aggregated Data Cube
        """
        cube = aggregator.agg(
            self.dataset.groupby([
                dimension.grouper for dimension in dimensions
            ])
        )
        cube.index.names = [name[1] for name in cube.index.names]
        if len(dimensions) > 1:
            cube = cube.unstack(dimensions[1].name)
        for axis, dimension in enumerate(dimensions):
            if dimension.is_ordinal:
                cube.sort_index(axis=axis, inplace=True)
                if dimension.is_temporal and dimension.freq is not None:
                    cube = cube.reindex(
                        pd.date_range(
                            cube.axes[axis][0],
                            cube.axes[axis][-1],
                            freq=dimension.freq,
                            name=cube.axes[axis].name
                        ), axis=axis)
                    cube.fillna(0.0, inplace=True)
        return cube

    def set(self, dimension: Dimension, value: Any) -> None:
        """Set value of a dimension.\n
        CAUTION: Does not update dataset and sums.\n
        To update use `del subspace.dataset` or `del subspace.sums`

        Arguments
        ---------
        dimension : Dimension
        value : Any
        """
        for dim in self.dimensions:
            if dim == dimension:
                dim.value = value
                break

    def values(self, dimension: Dimension) -> np.ndarray:
        """Returns unique values of a given dimension.

        Arguments
        ---------
        dimension : Dimension

        Returns
        -------
        numpy.ndarray: Array of unique values
        """
        return self.dataset['dimensions', dimension.name].unique()

    def __repr__(self) -> str:
        return str(tuple((str(dim.value) for dim in self.dimensions)))


class SiblingGroup():
    """Sibling Group

    Parameters
    ----------
    subspace : Subspace
        Subspace for the Sibling Group
    dividing_dimension : Dimension
        Dividing Dimension of the sibling group
    """

    def __init__(self, subspace: Subspace, dividing_dimension: Dimension):
        self.subspace = subspace
        self.dividing_dimension = dividing_dimension

    @property
    def subspaces(self) -> List[Subspace]:
        """Generate list of subspaces. Sorted by impact score."""
        subs: List[Subspace] = []
        for val in self.subspace.values(self.dividing_dimension):
            dimensions = deepcopy(self.subspace.dimensions)
            for dim in dimensions:
                if dim == self.dividing_dimension:
                    dim.value = val
                    break
            subs.append(
                Subspace(self.subspace.dataset, dimensions,
                         self.subspace.measurements)
            )
        return sorted(subs, key=lambda x: x.sums.max(), reverse=True)

    def __repr__(self):
        return str(tuple((self.subspace, self.dividing_dimension)))
