import logging
import signal
from contextlib import contextmanager
from itertools import product, combinations, chain
from typing import List, Set, TypedDict, Generator, Union
from copy import deepcopy
import pandas as pd

from .dimensions import Dimension
from .extractors import Extractor
from .aggregators import Aggregator
from .insights import Insight, CompoundInsight
from .spaces import Subspace, SiblingGroup
from .compositeExtractor import CompositeExtractor
from .heap import HeapMemory

log = logging.getLogger('tki')


class TimeBoxException(Exception):
    pass


class Result(TypedDict):
    data: Union[pd.Series, pd.DataFrame]
    impact: float
    sibling_group: SiblingGroup
    composite_extractor: CompositeExtractor


class TKI():
    def __init__(self,
                 dataset: pd.DataFrame,
                 dimensions: List[Dimension],
                 measurements: List[Dimension],
                 extractors: Set[Extractor],
                 aggregators: Set[Aggregator],
                 insight_types: Set[Insight],
                 depth: int = 2,
                 result_size: int = 12,
                 time_limit: int = 10
                 ):
        self.heap = HeapMemory(result_size)
        self.dimensions = dimensions
        self.measurements = measurements
        self.extractors = extractors
        self.aggregators = aggregators
        self.insight_types = insight_types
        self.depth = depth
        self.time_limit = time_limit
        self.dimension_names = [
            dimension.name for dimension in self.dimensions]
        self.measurement_names = [
            measurement.name for measurement in self.measurements]
        self.dataset = dataset
        self.sums = self.dataset[self.measurement_names].sum()

    def run(self) -> None:
        try:
            with self._timebox():
                self._runner()
        except TimeBoxException:
            log.warning("Time limit exceeded! Aborting!")

    def _runner(self) -> None:
        subspace = Subspace(self.dataset, self.dimensions, self.measurements)
        self._enumerate_insights(subspace)

    @contextmanager
    def _timebox(self):
        def signal_handler(signum, frame):
            raise TimeBoxException("TimeBox Limit reached")
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(self.time_limit)
        try:
            yield
        finally:
            signal.alarm(0)

    def _enumerate_insights(self, subspace: Subspace) -> None:
        # Phase I
        max_impact = (subspace.sums / self.sums).max()
        for result in self._extract(subspace, compound=any(
            issubclass(insight_type, CompoundInsight) 
                for insight_type in self.insight_types
        )):
            for insight_type in self.insight_types:
                try:
                    if issubclass(insight_type, CompoundInsight) and \
                       isinstance(result['data'], pd.DataFrame):
                        self.heap.add(insight_type(
                            data=result['data'],
                            dividing_dimension=result['sibling_group'].dividing_dimension,
                            impact=result['impact'],
                            sibling_group=result['sibling_group'],
                            composite_extractor=result['composite_extractor']
                        ))
                    if not issubclass(insight_type, CompoundInsight) and \
                       not isinstance(result['data'], pd.DataFrame):
                        self.heap.add(insight_type(
                            data=result['data'],
                            dividing_dimension=result['sibling_group'].dividing_dimension,
                            impact=result['impact'],
                            sibling_group=result['sibling_group'],
                            composite_extractor=result['composite_extractor']
                        ))
                except Exception as exc:
                    log.warning(exc)
                if max_impact < self.heap.upper_bound:
                    log.debug(
                        'Interrupting extraction of %s due to low impact.',
                        subspace
                    )
                    break
        # Phase II
        for dimension in [dimension for dimension in sorted(
            subspace.dimensions, key=lambda x: subspace.values(x).size
        ) if dimension.value == '*']:
            sibling_group = SiblingGroup(subspace, dimension)
            for sub_subspace in sibling_group.subspaces:
                impact = sub_subspace.sums / self.sums
                if impact.max() < self.heap.upper_bound:
                    log.debug('Skipping %s due to low impact.', sub_subspace)
                    continue
                self._enumerate_insights(sub_subspace)

    def _extract(self, subspace: Subspace,
        compound: bool = False) -> Generator[Result, None, None]:
        dimensions = [dimension for dimension in sorted(
            subspace.dimensions, key=lambda dimension: subspace.values(dimension).size
            ) if dimension.value == '*']
        dimension_pairs = combinations(dimensions, 2)
        if all(dimension.value == '*' for dimension in subspace.dimensions):
            dimension_pairs = chain(
                [[dimension] for dimension in dimensions],
                dimension_pairs
            )
        for aggregator_type, measurement, dimension_pair in product(
                self.aggregators, self.measurements, dimension_pairs):
            if (len(dimension_pair) > 1 and
               (dimension_pair[0] in dimension_pair[1].dependend_dimensions or
                dimension_pair[1] in dimension_pair[0].dependend_dimensions)):
                log.debug('Skipping dependent Dimensions %s', dimension_pair)
                continue
            aggregator = aggregator_type(measurement)
            extractions = self._recure_extract(
                subspace, self.depth - 1, aggregator, dimension_pair)
            for extraction in extractions:
                if isinstance(extractions[0][0]['data'], pd.Series):
                    for result in extraction:
                        yield Result({
                            **result,
                            'impact': extractions[0][0]['data'].sum() /
                                self.sums[result['composite_extractor']
                                    .aggregator.measurement.name],
                            'sibling_group': SiblingGroup(subspace, dimension_pair[0])
                        })
                else:
                    for result in extraction:
                        for loc, row in result['data'].iterrows():
                            subspace.set(dimension_pair[1], loc)
                            yield Result({
                                **result,
                                'data': row,
                                'impact': extractions[0][0]['data'].loc[loc].sum() /
                                    self.sums[result['composite_extractor']
                                        .aggregator.measurement.name],
                                'sibling_group': SiblingGroup(
                                    deepcopy(subspace), dimension_pair[0]
                                )
                            })
                        subspace.set(dimension_pair[1], '*')
                        for loc, row in result['data'].T.iterrows():
                            subspace.set(dimension_pair[0], loc)
                            yield Result({
                                **result,
                                'data': row,
                                'impact': extractions[0][0]['data'].T.loc[loc].sum() /
                                    self.sums[result['composite_extractor']
                                        .aggregator.measurement.name],
                                'sibling_group': SiblingGroup(
                                    deepcopy(subspace), dimension_pair[1]
                                )
                            })
                        subspace.set(dimension_pair[0], '*')
                        if compound:
                            ## TODO: Combinations between '*' and subspace
                            for comb in combinations(result['data'].index.values, 2):
                                yield Result({
                                    **result,
                                    'data': result['data'].loc[[comb[0], comb[1]]],
                                    'impact': result['data']
                                        .loc[[comb[0], comb[1]]].sum().sum() /
                                            self.sums[result['composite_extractor']
                                                .aggregator.measurement.name],
                                    'sibling_group': SiblingGroup(
                                        deepcopy(subspace), dimension_pair[0]
                                    )
                                })
                            for comb in combinations(result['data'].columns.values, 2):
                                yield Result({
                                    **result,
                                    'data': result['data'].T.loc[[comb[0], comb[1]]],
                                    'impact': result['data']
                                        .T.loc[[comb[0], comb[1]]].sum().sum() /
                                            self.sums[result['composite_extractor']
                                                .aggregator.measurement.name],
                                    'sibling_group': SiblingGroup(
                                        deepcopy(subspace), dimension_pair[1]
                                    )
                                })

    def _recure_extract(self,
        subspace: Subspace,
        level: int,
        aggregator: Aggregator,
        dimensions: List[Dimension]
    ) -> List[List[Result]]:
        if level > 0:
            resultset = self._recure_extract(
                subspace, level - 1, aggregator, dimensions)
            resultset_level = []
            for extractor_type, dimension in product(self.extractors, dimensions):
                extractor = extractor_type(dimension)
                for result in resultset[level - 1]:
                    composite_extractor = CompositeExtractor(
                        aggregator,
                        [*result['composite_extractor'].extractors, extractor],
                        dimensions
                    )
                    if composite_extractor.is_valid(subspace):
                        resultset_level.append(Result({
                            'data': extractor.extract(result['data']),
                            'composite_extractor': composite_extractor
                        }))
            return [*resultset, resultset_level]
        return [[Result({
            'data': subspace.cube(aggregator, dimensions),
            'composite_extractor': CompositeExtractor(aggregator, [], dimensions)
        })]]
