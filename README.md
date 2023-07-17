[![Tests](https://github.com/Der-Henning/TopK-Insights/actions/workflows/tests.yml/badge.svg)](https://github.com/Der-Henning/TopK-Insights/actions/workflows/tests.yml)
[![codecov](https://codecov.io/github/Der-Henning/TopK-Insights/branch/main/graph/badge.svg?token=UhCvn9aoVm)](https://codecov.io/github/Der-Henning/TopK-Insights)

# Top-K Insights

Pandas based Python package to extract Top-K Insights from multi-dimensional data.

The package contains an implementation of the article 'Extracting Top-K Insights from Multi-dimensional Data' (2017) by Tang, B., Han, S., Yiu, M.L., Ding, R. and Zhang, D. and their subsequent work based on my Bachelor Thesis.

- Documentation: https://topk-insights.readthedocs.io
- Repository: https://github.com/Der-Henning/TopK-Insights
- License: MIT

## Requirements

- Python3.9+

## Setup

````bash
python -m pip install git+https://github.com/Der-Henning/TopK-Insights
````

## Example

````Python
import logging

import matplotlib.pyplot as plt
import pandas as pd

from tki import TKI
from tki.aggregators import SumAggregator
from tki.dimensions import (CardinalDimension, NominalDimension,
                            TemporalDimension)
from tki.extractors import (DeltaMeanExtractor, DeltaPrevExtractor,
                            ProportionExtractor, RankExtractor)
from tki.insights import (CorrelationInsight, EvennessInsight,
                          OutstandingFirstInsight, OutstandingLastInsight,
                          TrendInsight)

logging.basicConfig()
logging.getLogger('tki').setLevel(logging.INFO)

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
insights = {
    OutstandingFirstInsight(),
    OutstandingLastInsight(),
    TrendInsight(),
    EvennessInsight(),
    CorrelationInsight()
}
tki = TKI(
    pd.DataFrame(data, columns=['Brand', 'year', 'Cars Sold']),
    dimensions=[
        NominalDimension('Brand'),
        TemporalDimension('year', date_format='%Y', freq='1Y')],
    measurements=[CardinalDimension('Cars Sold')],
    extractors=extractors,
    aggregators=aggregators,
    insights=insights,
    depth=3,
    result_size=21)
tki.run()

fig, axes = plt.subplots(7, 3, figsize=(25, 40), dpi=80)
for idx, i in enumerate(tki.heap.insights):
    plt.axes(axes[int(idx / 3)][idx % 3])
    i.plot()
    plt.title(
        f"{idx + 1}) {type(i.insight).__name__} "
        f"score: {i.impact:.2f} * {i.significance:.2f} = {i.score:.2f}\n"
        f"{(i.sibling_group, i.composite_extractor)}")
    x_index = i.data.index.get_level_values(i.data.index.names[-1])
    plt.xticks(rotation=0)
    if isinstance(x_index, pd.DatetimeIndex):
        plt.xticks(
            range(i.data.index.size),
            x_index.to_series().dt.year)
fig.tight_layout()
plt.savefig('insights.svg')
tki.save('insights.pkl')
````

### Result

![Insights](./insights.svg)

## User Interface

The App will provide a web based user interface for the tki package in the future.
At the moment it is possible to visualize saved insights in the browser.

Start the web server with

````bash
python -m tki.app
````

The project will be accessible via `http://127.0.0.1:8050/` in your web browser.
To display your previously generated results open the `Results` tab and upload your previously generated `insights.pkl` file.
