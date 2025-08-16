# PyDarkDiv

Python implementation of the DarkDiv R package for calculating dark diversity from species co-occurrence data.

## Overview

PyDarkDiv estimates dark diversity - the unobserved portion of a site's species pool consisting of species that could potentially occur but are currently absent. This package implements the methods described in Carmona & Pärtel (2021).

**Based on:** DarkDiv R package by Carlos P. Carmona and Meelis Pärtel

## Installation

```bash
pip install pydarkdiv
```

## Quick Start

```python
import pandas as pd
import pydarkdiv as pdd

# Load your species data (sites × species matrix)
data = pd.read_csv('species_data.csv', index_col=0)

# Calculate dark diversity
dd = pdd.DarkDiv(data, method='Hypergeometric')

# Get results as DataFrames
dfs = dd.to_dataframes()
dark_diversity = dfs['dark']
species_pool = dfs['pool']

# Summary statistics
dark_richness = (dark_diversity > 0.5).sum(axis=1)
print(f"Average dark diversity: {dark_richness.mean():.1f} species")
```

## Methods

- **Hypergeometric** (recommended): Compares observed vs. expected co-occurrences
- **RawBeals**: Beals smoothing for occurrence probabilities  
- **Favorability**: Favorability-corrected Beals values
- **ThresholdBeals**: Binary predictions using thresholds

## Usage Examples

See `usage_examples.py` for detailed examples including:

- Basic usage
- Method comparisons
- Reference data usage
- Abundance weighting

## Requirements

- Python ≥ 3.8
- numpy ≥ 1.21.0
- pandas ≥ 1.3.0
- scipy ≥ 1.7.0

## Reference

Carmona, C.P. & Pärtel, M. (2021). Estimating probabilistic site-specific species pools and dark diversity from co-occurrence data. *Global Ecology and Biogeography*, 30(1), 316-326. [DOI link](https://doi.org/10.1111/geb.13203)

## License

MIT License
