from __future__ import annotations
from typing import List


class Dimension():
    is_ordinal: bool = False
    is_nominal: bool = False
    is_cardinal: bool = False
    is_temporal: bool = False

    def __init__(self,
                 name: str,
                 dependend_dimensions: List[Dimension] = None,
                 value: str = '*'
                 ):
        self.name = name
        self.dependend_dimensions = \
            dependend_dimensions if dependend_dimensions else []
        self.value = value

    def __eq__(self, __o: Dimension) -> bool:
        return isinstance(__o, Dimension) and self.name == __o.name

    def __repr__(self) -> str:
        return self.name


class NominalDimension(Dimension):
    is_nominal: bool = True


class OrdinalDimension(NominalDimension):
    is_ordinal: bool = True


class CardinalDimension(OrdinalDimension):
    is_cardinal: bool = True


class TemporalDimension(CardinalDimension):
    is_temporal: bool = True
