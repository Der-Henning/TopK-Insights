"""Module containing the Composite Extractor class"""
from typing import List, Union, Generator, TypedDict
from itertools import permutations, product
import pandas as pd

from tki.spaces import SiblingGroup, Subspace
from tki.aggregators import Aggregator
from tki.extractors import Extractor
from tki.dimensions import Dimension


class CompositeExtractor():
    """Composite Extractor

    Parameters
    ----------
    aggregator : Aggregator
        Takes exactly one Aggregator
    extractors : List[Extractor]
        List of used Extractors
    """

    def __init__(self,
                 aggregator: Aggregator,
                 extractors: List[Extractor]
                 ):
        self.aggregator = aggregator
        self.extractors = extractors

    def __getitem__(self, level: Union[int, slice]
        ) -> Union[Aggregator, Extractor]:
        if isinstance(level, slice):
            comp = [self[idx] for idx in range(*level.indices(len(self)))]
            return CompositeExtractor(comp[0], comp[1:])
        if level == 0:
            return self.aggregator
        return self.extractors[level - 1]

    def __len__(self):
        return len(self.extractors) + 1

    def is_valid(self, subspace: Subspace) -> bool:
        """Checks if the Composite Extractor is valid for a given Subspace.

        Arguments
        ---------
        subspace : Subspace

        Returns
        -------
        bool
        """
        for idx, extractor in enumerate(self.extractors):
            prev_extractor = None
            if idx > 0:
                prev_extractor = self.extractors[idx - 1]
            if not extractor.is_valid(subspace, prev_extractor):
                return False
        return True

    def is_useful(self, sibling_group: SiblingGroup) -> bool:
        """Checks if the Composite Extractor returns a useful Result
        for a given SiblingGroup

        Arguments
        ---------
        sibling_group : SiblingGroup

        Returns
        -------
        bool:
        """
        for extractor in self.extractors:
            if not extractor.is_useful(sibling_group):
                return False
        return True

    def __repr__(self) -> str:
        return str(tuple([self.aggregator, *self.extractors]))


class ExtractionResult(TypedDict):
    """Typed Dict for Extraction results.

    Parameters
    ----------
    data : pandas.Series | pandas.DataFrame
        Data set
    impact : float
        Impact score
    sibling_group : SiblingGroup
        Sibling Group
    composite_extractor : CompositeExtractor
        Used Composite Extractor
    """
    data: Union[pd.Series, pd.DataFrame]
    impact: float
    sibling_group: SiblingGroup
    composite_extractor: CompositeExtractor


def generate_composite_extractors(
    aggregators: List[Aggregator],
    extractors: List[Extractor],
    measurements: List[Dimension],
    dimensions: List[Dimension],
    depth: int = 2
) -> Generator[CompositeExtractor, None, None]:
    """Generator for possible Composite Extractors

    Args:
        aggregators (List[Aggregator]): _description_
        extractors (List[Extractor]): _description_
        measurements (List[Dimension]): _description_
        dimensions (List[Dimension]): _description_
        depth (int, optional): _description_. Defaults to 2.

    Yields:
        CompositeExtractor
    """
    for comp in product(product(aggregators, measurements),
        permutations(product(extractors, dimensions), depth - 1)):
        yield CompositeExtractor(
            aggregator=comp[0][0](comp[0][1]),
            extractors=[c[0](c[1]) for c in comp[1]])
