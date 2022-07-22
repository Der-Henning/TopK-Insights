
# Top-K Insights

Pandas based Python package to extract Top-K Insights from multi-dimensional data

The package contains an implementation of the article 'Extracting Top-K Insights from Multi-dimensional Data' (2017) by Tang, B., Han, S., Yiu, M.L., Ding, R. and Zhang, D. and their subsequent work.

## Requirements

- Python3.9+

## Setup

````
python -m pip install git+https://github.com/Der-Henning/TopK-Insights
````

## Example

````Python
import pandas as pd
import matplotlib.pyplot as plt

from tki import TKI
from tki.insights import OutstandingFirstInsight, OutstandingLastInsight, TrendInsight, EvennessInsight, CorrelationInsight
from tki.extractors import RankExtractor, DeltaPrevExtractor, DeltaMeanExtractor, ProportionExtractor
from tki.aggregators import SumAggregator
from tki.dimensions import TemporalDimension, OrdinalDimension, NominalDimension

data = [
    ['H', 2010, 40], ['T', 2010, 38], ['F', 2010, 13], ['B', 2010, 20],
    ['H', 2011, 35], ['T', 2011, 34], ['F', 2011, 10], ['B', 2011, 18],
    ['H', 2012, 36], ['T', 2012, 34], ['F', 2012, 14], ['B', 2012, 20],
    ['H', 2013, 43], ['T', 2013, 29], ['F', 2013, 23], ['B', 2013, 17],
    ['H', 2014, 58], ['T', 2014, 36], ['F', 2014, 27], ['B', 2014, 19]
]

extractors = {
    RankExtractor,
    DeltaPrevExtractor,
    DeltaMeanExtractor,
    ProportionExtractor
}
aggregators = {
    SumAggregator
}
insight_types = {
    OutstandingFirstInsight(),
    OutstandingLastInsight(),
    TrendInsight(),
    EvennessInsight(),
    CorrelationInsight()
}
tki = TKI(
    pd.DataFrame(data, columns=['Brand', 'year', 'Cars Sold']),
    dimensions=[NominalDimension('Brand'), TemporalDimension('year')],
    measurements=[OrdinalDimension('Cars Sold')],
    extractors=extractors,
    aggregators=aggregators,
    insight_types=insight_types,
    depth=2,
    result_size=21)
tki.run()

_, axes = plt.subplots(7, 3, figsize=(25, 40), dpi=80)
plt.subplots_adjust(hspace=0.3)
for idx, i in enumerate(tki.heap.insights):
    plt.axes(axes[int(idx/3)][idx % 3])
    i.plot()
    plt.title(
        f"{type(i.insight).__name__} score: {i.impact:.2f} * {i.significance:.2f} = \
            {i.score:.2f} \n{(i.sibling_group, i.composite_extractor)}")
    plt.xticks(rotation=0)
plt.savefig('insights.svg')

````

### Result

![Insights](./insights.svg)
