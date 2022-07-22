from typing import List, Any, Union
from copy import deepcopy
import pandas as pd
import numpy as np

from .dimensions import Dimension
from .aggregators import Aggregator


class Subspace():
    def __init__(
            self,
            dataset: pd.DataFrame,
            dimensions: List[Dimension],
            measurements: List[Dimension]):
        self.dataset = dataset
        self.dimensions = dimensions
        self.measurements = measurements
        self.dimension_names = [
            dimension.name for dimension in self.dimensions
            if dimension.value == '*']
        self.measurement_names = [
            measurement.name for measurement in self.measurements]
        for dimension in self.dimensions:
            if dimension.value != '*':
                self.dataset = self.dataset[
                    self.dataset['dimensions', dimension.name] == dimension.value]
        self.sums = self.dataset['measurements'].sum()

    def cube(self,
        aggregator: Aggregator,
        dimensions: List[Dimension]
    ) -> Union[pd.Series, pd.DataFrame]:
        cube = aggregator.agg(
            self.dataset.groupby([
                dimension.grouper for dimension in dimensions
            ])
        )
        cube.index.names = [name[1] for name in cube.index.names]
        if len(dimensions) > 1:
            return cube.unstack(dimensions[0].name)
        return cube

    def set(self, dimension: Dimension, value: Any) -> None:
        for dim in self.dimensions:
            if dim == dimension:
                dim.value = value
                break

    def values(self, dimension: Dimension) -> np.ndarray:
        return self.dataset['dimensions', dimension.name].unique()

    def __repr__(self) -> str:
        return str(tuple((str(dim.value) for dim in self.dimensions)))


class SiblingGroup():
    def __init__(self, subspace: Subspace, dividing_dimension: Dimension):
        self.subspace = subspace
        self.dividing_dimension = dividing_dimension

    @property
    def subspaces(self) -> List[Subspace]:
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
