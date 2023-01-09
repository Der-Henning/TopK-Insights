"""Top-K Insights Library.

For Documentation and Issues please refer to
https://github.com/Der-Henning/TopK-Insights
"""
from .__version__ import (__author__, __copyright__, __description__,
                          __version__)
from .app import App
from .tki import TKI

__all__ = ['App', 'TKI', '__author__',
           '__copyright__', '__description__',
           '__version__']
