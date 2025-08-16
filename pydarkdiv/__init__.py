"""
PyDarkDiv - Python implementation of the DarkDiv R package

A package for estimating dark diversity from species co-occurrence data.
Based on the DarkDiv R package by Carmona & Pärtel (2021).

Basic usage:
    >>> import pydarkdiv as pdd
    >>> import pandas as pd
    >>> 
    >>> data = pd.read_csv('species_data.csv', index_col=0)
    >>> result = pdd.DarkDiv(data, method='Hypergeometric')
    >>> dfs = result.to_dataframes()
    >>> dark_diversity = dfs['dark']

Reference:
    Carmona, C.P. & Pärtel, M. (2021). Estimating probabilistic site-specific 
    species pools and dark diversity from co-occurrence data. Global Ecology 
    and Biogeography, 30(1), 316-326.
"""

from .darkdiv import DarkDiv, DarkDiv_calc
from .utility_functions import (
    data_prep,
    cooc_prep,
    hypergeometric,
    beals,
    beals_threshold,
    favorability,
    dd_weighting,
)

__version__ = "0.1.0"
__author__ = "davidyshen"
__license__ = "MIT"
__email__ = "davidyshen@example.com"

__all__ = [
    "DarkDiv",
    "DarkDiv_calc",
    "data_prep",
    "cooc_prep",
    "hypergeometric",
    "beals",
    "beals_threshold",
    "favorability",
    "dd_weighting",
]
