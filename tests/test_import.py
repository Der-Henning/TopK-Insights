from typing import List

import pandas as pd
import pytest

from tki import TKI
from tki.aggregators import SumAggregator
from tki.dimensions import (CardinalDimension, NominalDimension,
                            TemporalDimension)
from tki.extractors import (DeltaMeanExtractor, DeltaPrevExtractor,
                            ProportionExtractor, RankExtractor)
from tki.insights import (CorrelationInsight, EvennessInsight,
                          OutstandingFirstInsight, OutstandingLastInsight,
                          TrendInsight)


@pytest.fixture
def example_data() -> List[list]:
    return [
        ['H', 2010, 40], ['T', 2010, 38], ['F', 2010, 13], ['B', 2010, 20],
        ['H', 2011, 35], ['T', 2011, 34], ['F', 2011, 10], ['B', 2011, 18],
        ['H', 2012, 36], ['T', 2012, 34], ['F', 2012, 14], ['B', 2012, 20],
        ['H', 2013, 43], ['T', 2013, 29], ['F', 2013, 23], ['B', 2013, 17],
        ['H', 2014, 58], ['T', 2014, 36], ['F', 2014, 27], ['B', 2014, 19]
    ]


def test_import(example_data: List[list]):
    extractors = {
        RankExtractor,
        DeltaPrevExtractor,
        DeltaMeanExtractor,
        ProportionExtractor
    }
    aggregators = {
        SumAggregator
    }
    insights = {
        OutstandingFirstInsight(),
        OutstandingLastInsight(),
        TrendInsight(),
        EvennessInsight(),
        CorrelationInsight()
    }
    TKI(
        pd.DataFrame(example_data, columns=['Brand', 'year', 'Cars Sold']),
        dimensions=[
            NominalDimension('Brand'),
            TemporalDimension('year', date_format='%Y', freq='1Y')],
        measurements=[CardinalDimension('Cars Sold')],
        extractors=extractors,
        aggregators=aggregators,
        insights=insights,
        depth=3,
        result_size=21)
