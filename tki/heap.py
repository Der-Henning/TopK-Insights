"""Module containing the Heap Memory class"""
from typing import List, Union

from tki.insights import InsightResult


class HeapMemory():
    """Heap memory to store the Top-K Insights.

    Parameters
    ----------
    size : int
        Maximum size of the heap memory.
        Defaults to 12
    """

    def __init__(self, size: int = 12):
        self.size: int = size
        self.insights: List[InsightResult] = []
        self.counter: int = 0

    def add(self, insights: Union[InsightResult, List[InsightResult]]) -> None:
        """Adds one or more insights to the heap memory.

        Arguments
        ---------
        insights : InsightResult | List[InsightResult]
            One or more Insight Results
        """
        # TODO: Maybe it is more efficient to insert the
        # new insight directly at the right spot in the list.
        # This way sorting could be avoided.
        if insights:
            if isinstance(insights, list):
                self.insights = [*self.insights, *insights]
                self.counter += len(insights)
            else:
                self.insights = [*self.insights, insights]
                self.counter += 1
            self.insights.sort()
            self.insights.reverse()
            self.insights = list(
                filter(lambda x: x.score > 0, self.insights)
            )[:self.size]

    @property
    def upper_bound(self) -> float:
        """Returns the lowest score in heap memory.
        If the heap memory is not full 0.0 is returned.
        """
        if len(self.insights) == self.size:
            return self.insights[-1].score
        return 0.0
