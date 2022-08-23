"""Module containing the main TKI class"""
import logging
import signal
from contextlib import contextmanager
from itertools import product, combinations, chain
from typing import List, Set, Generator
from copy import deepcopy
import pickle
import pandas as pd

from tki.dimensions import Dimension
from tki.extractors import Extractor
from tki.aggregators import AggregationError, Aggregator
from tki.insights import Insight, CompoundInsight, InsightError, InsightResult
from tki.spaces import Subspace, SiblingGroup
from tki.composite_extractor import CompositeExtractor, ExtractionResult
from tki.heap import HeapMemory

log = logging.getLogger('tki')
log.setLevel(logging.WARNING)


class TimeBoxException(Exception):
    """Exception to raise on timeout"""


class TKI():
    """Top-K Insights

    Parameters
    ----------
    dataset : pandas.DataFrame
        Dataset to retrieve Insights from
    dimensions : List[Dimension]
        List of Dimension objects
    measurements : List[Dimension]
        List of Dimension objects
    extractors : Set[Extractor]
        Set of Extractor classes to use
    aggregators : Set[Aggregator]
        Set of Aggregator classes to use
    insights : Set[Insight]
        Set of Insight objects
    depth : int
        Depth of composite Extractors. Defaults to 2.
    result_size : int
        Number of Insights to retrieve. Defaults to 12.
    time_limit : int
        Time Limit for execution in seconds. Defaults to 10.
    """

    def __init__(self,
                 dataset: pd.DataFrame,
                 dimensions: List[Dimension],
                 measurements: List[Dimension],
                 extractors: Set[Extractor],
                 aggregators: Set[Aggregator],
                 insights: Set[Insight],
                 depth: int = 2,
                 result_size: int = 12,
                 time_limit: int = 10
                 ):
        self.dimensions = dimensions
        self.measurements = measurements
        self.extractors = extractors
        self.aggregators = aggregators
        self.insights = insights
        self.depth = depth
        self.time_limit = time_limit
        # Initialize Heap Memory
        self.heap = HeapMemory(result_size)
        self.dimension_names = [
            dimension.name for dimension in self.dimensions]
        self.measurement_names = [
            measurement.name for measurement in self.measurements]
        # prepare dataset
        self.dataset = pd.concat(
            [dataset[self.dimension_names], dataset[self.measurement_names]],
            axis=1, keys=['dimensions', 'measurements'])
        # calculate measurement sums for impact calculation
        self.sums = self.dataset['measurements'].sum()
        # preprocess dimension data
        for dimension in self.dimensions:
            self.dataset['dimensions', dimension.name] = dimension.preprocess(
                self.dataset['dimensions'][dimension.name])

    def run(self) -> None:
        """Start insight extraction.

        Stops either when all possible combinations are checked
        or the time limit is reached
        """
        try:
            with self._timebox():
                self._runner()
        except TimeBoxException:
            log.warning("Time limit exceeded! Aborting!")

    def save(self, filename: str) -> None:
        """Saves Insights Results to file

        Arguments
        ---------
        filename : str
        """
        with open(filename, 'wb') as file:
            pickle.dump(self.heap.insights, file)

    @staticmethod
    def load(filename: str) -> List[InsightResult]:
        """Loads Insights Results from file

        Arguments
        ---------
        filename : str

        Returns
        -------
        List[InsightResult]
        """
        with open(filename, 'rb') as file:
            return pickle.load(file)

    def _runner(self) -> None:
        # Initialize Subspace and start insight extraction
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
        # TODO:
        # Use multiprocessing for extraction and insight calculation
        # -> Queues to communicate between procs

        # Phase I
        # Generate ExtractionResults and calculate Insights
        log.info("Current Subspace: %s", subspace)
        max_impact = (subspace.sums / self.sums).max()
        for result in self._extract(subspace, compound=any(
            issubclass(type(insight), CompoundInsight) for insight in self.insights
        )):
            self._calculate_insights(result)
            if max_impact < self.heap.upper_bound:
                log.debug(
                    'Interrupting extraction of %s due to low impact.',
                    subspace
                )
                break

        # Phase II
        # Generate SiblingGroups and recursively call _enumerate_insights
        # on subspaces
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

    def _extract(self, subspace: Subspace, compound: bool = False
                ) -> Generator[ExtractionResult, None, None]:
        # TODO:
        # This needs some refactoring ...

        # sorted list of Dimensions by number of possible values
        dimensions = [dimension for dimension in sorted(
            subspace.dimensions, key=lambda dimension: subspace.values(
                dimension).size
        ) if dimension.value == '*']

        # generate all possible dimension pairs
        dimension_pairs = combinations(dimensions, 2)
        if all(dimension.value == '*' for dimension in subspace.dimensions):
            dimension_pairs = chain(
                [[dimension] for dimension in dimensions],
                dimension_pairs
            )

        # iterate over all combinations of aggregator, measurements
        # and dimension_pairs
        for aggregator_type, measurement, dimension_pair in product(
                self.aggregators, self.measurements, dimension_pairs):

            # block dependent dimensions
            if (len(dimension_pair) > 1 and
               (dimension_pair[0].name in dimension_pair[1].dependent_dimensions or
                dimension_pair[1].name in dimension_pair[0].dependent_dimensions)):
                log.debug('Skipping dependent Dimensions %s', dimension_pair)
                continue
            try:
                aggregator = aggregator_type(measurement)
            except AggregationError as err:
                log.warning(err)
                continue

            # call recurring extract and iterate over resultset
            # to generate Extraction Results
            # calculate impact score as early as possible to
            # skip low impact ExtractionResults
            extractions = self._recure_extract(
                subspace, self.depth - 1, aggregator, dimension_pair)
            for extraction in extractions:
                # 1-dimensional array
                if isinstance(extractions[0][0]['data'], pd.Series):
                    for result in extraction:
                        sibling_group = SiblingGroup(
                                deepcopy(subspace), dimension_pair[0])
                        if result['composite_extractor'].is_useful(sibling_group):
                            yield ExtractionResult({
                                **result,
                                'impact': (subspace.sums / self.sums)
                                [result['composite_extractor']
                                    .aggregator.measurement.name],
                                'sibling_group': sibling_group
                            })
                # 2-dimensional array
                else:
                    for result in extraction:
                        # use rows of array
                        for loc, row in result['data'].iterrows():
                            impact = extractions[0][0]['data'].loc[loc].sum() / \
                                self.sums[result['composite_extractor']
                                .aggregator.measurement.name]
                            if impact > self.heap.upper_bound:
                                subspace.set(dimension_pair[1], loc)
                                subsubspace = Subspace(
                                    subspace.dataset,
                                    deepcopy(subspace.dimensions),
                                    subspace.measurements)
                                subspace.set(dimension_pair[1], '*')
                                sibling_group = SiblingGroup(
                                        subsubspace, dimension_pair[0]
                                    )
                                if result['composite_extractor'].is_useful(sibling_group):
                                    yield ExtractionResult({
                                        **result,
                                        'data': row,
                                        'impact': impact,
                                        'sibling_group': sibling_group
                                    })
                        # use columns of array
                        for loc, row in result['data'].T.iterrows():
                            impact = extractions[0][0]['data'].T.loc[loc].sum() / \
                                self.sums[result['composite_extractor']
                                .aggregator.measurement.name]
                            if impact > self.heap.upper_bound:
                                subspace.set(dimension_pair[0], loc)
                                subsubspace = Subspace(
                                    subspace.dataset,
                                    deepcopy(subspace.dimensions),
                                    subspace.measurements)
                                subspace.set(dimension_pair[0], '*')
                                sibling_group = SiblingGroup(
                                        subsubspace, dimension_pair[1]
                                    )
                                if result['composite_extractor'].is_useful(sibling_group):
                                    yield ExtractionResult({
                                        **result,
                                        'data': row,
                                        'impact': impact,
                                        'sibling_group': sibling_group
                                    })
                        # generate ExtractionResults for compound Insights
                        # TODO:
                        # Combinations between '*' and subspace
                        if compound:
                            for comb in combinations(result['data'].index.values, 2):
                                impact = (extractions[0][0]['data'].loc[comb[0]].sum() +
                                    extractions[0][0]['data'].loc[comb[1]].sum()) / \
                                    self.sums[result['composite_extractor']
                                    .aggregator.measurement.name] / 2
                                if impact > self.heap.upper_bound:
                                    sibling_group = SiblingGroup(
                                            deepcopy(subspace),
                                            dimension_pair[0]
                                        )
                                    if result['composite_extractor'].is_useful(sibling_group):
                                        yield ExtractionResult({
                                            **result,
                                            'data': result['data'].loc[[comb[0], comb[1]]],
                                            'impact' : impact,
                                            'sibling_group': sibling_group
                                        })
                            for comb in combinations(result['data'].columns.values, 2):
                                impact = (extractions[0][0]['data'].T.loc[comb[0]].sum() +
                                    extractions[0][0]['data'].T.loc[comb[1]].sum()) / \
                                    self.sums[result['composite_extractor']
                                    .aggregator.measurement.name] / 2
                                if impact > self.heap.upper_bound:
                                    sibling_group = SiblingGroup(
                                            deepcopy(subspace),
                                            dimension_pair[1]
                                        )
                                    if result['composite_extractor'].is_useful(sibling_group):
                                        yield ExtractionResult({
                                            **result,
                                            'data': result['data'].T.loc[[comb[0], comb[1]]],
                                            'impact' : impact,
                                            'sibling_group': sibling_group
                                        })

    def _recure_extract(self,
                        subspace: Subspace,
                        level: int,
                        aggregator: Aggregator,
                        dimensions: List[Dimension]
                        ) -> List[List[ExtractionResult]]:
        if level > 0:
            # Use Extractor on level x
            resultset = self._recure_extract(
                subspace, level - 1, aggregator, dimensions)
            resultset_level = []
            for extractor_type, dimension in product(self.extractors, dimensions):
                extractor = extractor_type(dimension)
                for result in resultset[level - 1]:
                    composite_extractor = CompositeExtractor(
                        aggregator,
                        [*result['composite_extractor'].extractors, extractor]
                    )
                    if composite_extractor.is_valid(subspace):
                        resultset_level.append(ExtractionResult({
                            'data': extractor.extract(result['data']),
                            'composite_extractor': composite_extractor
                        }))
            return [*resultset, resultset_level]

        # Use Aggregator on level 0
        return [[ExtractionResult({
            'data': subspace.cube(aggregator, dimensions),
            'composite_extractor': CompositeExtractor(aggregator, [])
        })]]

    def _calculate_insights(self, result: ExtractionResult) -> None:
        # cleanse data and calculate insight when
        # there is sufficient data left
        for insight in self.insights:
            if issubclass(type(insight), CompoundInsight) and \
                    isinstance(result['data'], pd.DataFrame):
                result['data'].dropna(axis=1, inplace=True)
                if result['data'].size > 3:
                    self._calculate_insight(result, insight)
            elif not issubclass(type(insight), CompoundInsight) and \
                    isinstance(result['data'], pd.Series):
                result['data'].dropna(axis=0, inplace=True)
                if result['data'].size > 1:
                    self._calculate_insight(result, insight)

    def _calculate_insight(self, result: ExtractionResult, insight: Insight) -> None:
        try:
            self._save_insight(insight.calc_insight(result))
        except InsightError as err:
            log.debug(err)
        except Exception as exc:
            log.warning(exc)

    def _save_insight(self, result: InsightResult) -> None:
        self.heap.add(result)
