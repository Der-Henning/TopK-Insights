from typing import List, Union, Generator
from itertools import permutations, product

from .spaces import SiblingGroup, Subspace
from .aggregators import Aggregator
from .extractors import Extractor
from .dimensions import Dimension


class CompositeExtractor():
    def __init__(self,
        aggregator: Aggregator,
        extractors: List[Extractor],
        dimensions: List[Dimension]
    ):
        self.aggregator = aggregator
        self.extractors = extractors
        self.dimensions = dimensions

    def __getitem__(self,
        level: Union[int, slice]) -> Union[Aggregator, Extractor]:
        if isinstance(level, slice):
            comp = [self[idx] for idx in range(*level.indices(len(self)))]
            return CompositeExtractor(comp[0], comp[1:], self.dimensions)
        if level == 0:
            return self.aggregator
        return self.extractors[level - 1]

    def __len__(self):
        return len(self.extractors) + 1

    def is_valid(self, subspace: Subspace) -> bool:
        for idx, extractor in enumerate(self.extractors):
            prev_extractor = None
            if idx > 0:
                prev_extractor = self.extractors[idx - 1]
            if not extractor.is_valid(
                SiblingGroup(subspace, extractor.dimension),
                prev_extractor
            ):
                return False
        return True

    def __repr__(self) -> str:
        return str(tuple([self.aggregator, *self.extractors]))


def generate_composite_extractors(
    aggregators: List[Aggregator],
    extractors: List[Extractor],
    measurements: List[Dimension],
    dimensions: List[Dimension],
    depth: int = 2
) -> Generator[CompositeExtractor, None, None]:
    for comp in product(
        product(aggregators, measurements),
        permutations(product(extractors, dimensions), depth - 1)
    ):
        yield CompositeExtractor(
            aggregator=comp[0][0](comp[0][1]),
            extractors=[c[0](c[1]) for c in comp[1]],
            dimensions=dimensions
        )
