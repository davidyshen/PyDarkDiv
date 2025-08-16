"""
Main DarkDiv class for calculating dark diversity from species co-occurrence patterns.

Based on the DarkDiv R package by Carmona & Pärtel (2021).
"""

import numpy as np
import pandas as pd
from typing import Optional, Union
from .utility_functions import (
    data_prep,
    cooc_prep,
    hypergeometric,
    beals,
    beals_threshold,
    favorability,
    dd_weighting,
)


class DarkDiv:
    """
    Estimate dark diversity based on species co-occurrence patterns.

    Based on the DarkDiv R package methods described in Carmona & Pärtel (2021).

    Parameters
    ----------
    x : pandas.DataFrame or numpy.ndarray
        Study data with sites in rows and species in columns.
    r : pandas.DataFrame, numpy.ndarray, or None, optional
        Reference dataset. If None, uses x. Default is None.
    method : str, optional
        Method: 'Hypergeometric' (default), 'RawBeals', 'Favorability', or 'ThresholdBeals'.
    limit : str, optional
        Threshold method for ThresholdBeals: 'min', 'quantile', 'const', 'outlier'. Default 'min'.
    const : float, optional
        Constant for threshold methods. Default 0.01.
    remove_absent : bool, optional
        Remove species with zero occurrences. Default True.
    wa : bool, optional
        Use abundance weighting. Default False.
    weights : pandas.DataFrame, numpy.ndarray, or None, optional
        Weight matrix for abundance weighting. Default None.

    Attributes
    ----------
    indication : numpy.ndarray
        Species × species indication matrix
    pool : numpy.ndarray
        Species pool probabilities (sites × species)
    dark : numpy.ndarray
        Dark diversity probabilities (sites × species)
    species_names : list
        Species names
    site_names : list
        Site names

    Examples
    --------
    >>> import pandas as pd
    >>> import pydarkdiv as pdd
    >>> 
    >>> data = pd.read_csv('species_data.csv', index_col=0)
    >>> dd = pdd.DarkDiv(data, method='Hypergeometric')
    >>> dfs = dd.to_dataframes()
    >>> dark_diversity = dfs['dark']

    References
    ----------
    Carmona, C.P. & Pärtel, M. (2021). Estimating probabilistic site-specific 
    species pools and dark diversity from co-occurrence data. Global Ecology 
    and Biogeography, 30(1), 316-326.
    """

    def __init__(
        self,
        x: Union[np.ndarray, pd.DataFrame],
        r: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        method: str = "Hypergeometric",
        limit: str = "min",
        const: float = 0.01,
        remove_absent: bool = True,
        wa: bool = False,
        weights: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    ):
        """
        Initialize DarkDiv calculator.

        Parameters:
        -----------
        x : array-like
            Study data with sites in rows and species in columns
        r : array-like, optional
            Reference dataset. If None, uses x
        method : str
            Method to use: "Hypergeometric", "RawBeals", "ThresholdBeals", "Favorability"
        limit : str
            Limit type for ThresholdBeals: "min", "quantile", "const", "outlier"
        const : float
            Constant for limit in ThresholdBeals method
        remove_absent : bool
            Whether to remove species with zero occurrences
        wa : bool
            Whether to use abundance weighting
        weights : array-like, optional
            Weight matrix for abundance weighting
        """

        # Validate method
        valid_methods = ["Hypergeometric", "RawBeals", "ThresholdBeals", "Favorability"]
        if method not in valid_methods:
            raise ValueError(f"Method must be one of: {valid_methods}")

        # Validate limit for ThresholdBeals
        if method == "ThresholdBeals":
            valid_limits = ["quantile", "min", "const", "outlier"]
            if limit not in valid_limits:
                raise ValueError(
                    f"For ThresholdBeals, limit must be one of: {valid_limits}"
                )

            if wa:
                raise ValueError(
                    "Weighted abundances not yet implemented for ThresholdBeals method"
                )

        # Validate weights if using abundance weighting
        if wa and weights is not None:
            if isinstance(x, pd.DataFrame) and isinstance(weights, pd.DataFrame):
                if not x.index.equals(weights.index) or not x.columns.equals(
                    weights.columns
                ):
                    raise ValueError(
                        "x and weights must have same index and column names"
                    )
            elif isinstance(x, np.ndarray) and isinstance(weights, np.ndarray):
                if x.shape != weights.shape:
                    raise ValueError("x and weights must have same dimensions")

        # Store parameters
        self.method = method
        self.limit = limit
        self.const = const
        self.remove_absent = remove_absent
        self.wa = wa
        self.weights = weights

        # Store original data
        self.x_orig = x
        self.r_orig = r if r is not None else x

        # Calculate dark diversity
        self.results = self._calculate_dark_diversity(x, r)

    def _calculate_dark_diversity(self, x, r):
        """Calculate dark diversity using the specified method."""

        # Check if r is already an indication matrix
        r_is_indication = self._is_indication_matrix(r)

        if not r_is_indication:
            # Prepare data
            prep_data = data_prep(x, r, self.remove_absent)
            x_proc = prep_data["x"]
            r_proc = prep_data["r"]

            # Convert to binary presence/absence
            r_binary = np.where(r_proc > 0, 1, 0)
            x_binary = np.where(x_proc > 0, 1, 0)

            # Create co-occurrence matrix
            M = cooc_prep(r_binary)

            # Apply the selected method
            if self.method == "Hypergeometric":
                results = hypergeometric(M, x_binary, r_binary)

            elif self.method in ["RawBeals", "ThresholdBeals", "Favorability"]:
                results = beals(M, x_binary, r_binary)

                if self.method == "RawBeals":
                    # Remove the 't' key for RawBeals
                    results.pop("t", None)

                elif self.method == "ThresholdBeals":
                    results = beals_threshold(
                        results, self.limit, self.const, r_binary, x_binary
                    )

                elif self.method == "Favorability":
                    results = favorability(results, x_binary)

            # Store processed data info
            results["x_cols"] = prep_data["x_cols"]
            results["r_cols"] = prep_data["r_cols"]
            results["x_processed"] = x_binary
            results["r_processed"] = r_binary

        else:
            # r is already an indication matrix
            results = self._use_indication_matrix(x, r)

        # Apply abundance weighting if requested
        if self.wa:
            weights_to_use = self.weights if self.weights is not None else self.x_orig
            results = dd_weighting(
                np.array(self.x_orig),
                results["indication"],
                np.array(weights_to_use) if weights_to_use is not None else None,
                self.method,
            )

        return results

    def _is_indication_matrix(self, r):
        """Check if r is already an indication matrix."""
        if r is None:
            return False

        # Convert to array for checking
        r_array = np.array(r)

        # Check if it's square
        if r_array.shape[0] != r_array.shape[1]:
            return False

        # For DataFrame, check if row and column names suggest indication matrix
        if isinstance(r, pd.DataFrame):
            row_names = r.index.tolist()
            col_names = r.columns.tolist()

            # Check if names follow the pattern of indication matrices
            row_target = [name.startswith("T.") for name in row_names]
            col_indicator = [name.startswith("I.") for name in col_names]

            if all(row_target) and all(col_indicator):
                return True

            # Also check if base names match
            row_base = [name.replace("T.", "") for name in row_names]
            col_base = [name.replace("I.", "") for name in col_names]

            if row_base == col_base:
                return True

        return False

    def _use_indication_matrix(self, x, r):
        """Use pre-calculated indication matrix."""
        x_binary = np.where(np.array(x) > 0, 1, 0)
        M = np.array(r)

        # Get species names
        if isinstance(x, pd.DataFrame):
            x_cols = x.columns.tolist()
        else:
            x_cols = [f"Species_{i}" for i in range(x.shape[1])]

        if isinstance(r, pd.DataFrame):
            r_cols = r.columns.tolist()
            # Handle indication matrix column names
            r_cols = [col.replace("I.", "") for col in r_cols]
        else:
            r_cols = x_cols

        # Match species between x and indication matrix
        common_species = list(set(x_cols) & set(r_cols))
        if not common_species:
            raise ValueError("No common species between x and indication matrix")

        # Filter to common species
        x_indices = [i for i, col in enumerate(x_cols) if col in common_species]
        r_indices = [i for i, col in enumerate(r_cols) if col in common_species]

        x_filtered = x_binary[:, x_indices]
        M_filtered = M[np.ix_(r_indices, r_indices)]

        # Calculate probabilities based on method
        if self.method == "Hypergeometric":
            # Calculate weighted averages for hypergeometric
            row_sums = np.sum(x_filtered, axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)
            b = np.dot(x_filtered, M_filtered) / row_sums

        else:
            # Calculate for Beals-based methods
            S = np.sum(x_filtered, axis=1)
            b = np.zeros_like(x_filtered, dtype=float)

            for i in range(x_filtered.shape[0]):
                b[i, :] = np.sum(M_filtered * x_filtered[i, :], axis=1)

            SM = np.tile(S, (x_filtered.shape[1], 1)).T - x_filtered
            SM = np.where(SM == 0, 1, SM)
            b = b / SM

        # Create Pool matrix (set present species to 1)
        Pool = b.copy()
        Pool[x_filtered > 0] = 1

        # Create Dark matrix (set present species to NaN)
        Dark = b.copy()
        Dark[x_filtered > 0] = np.nan

        # Apply method-specific transformations
        if self.method == "Favorability":
            favorability_result = favorability(
                {"AllProbs": b, "indication": M_filtered}, x_filtered
            )
            b = favorability_result["AllProbs"]
            Pool = favorability_result["Pool"]
            Dark = favorability_result["Dark"]

        return {
            "indication": M_filtered,
            "AllProbs": b,
            "Pool": Pool,
            "Dark": Dark,
            "x_cols": [x_cols[i] for i in x_indices],
            "r_cols": [r_cols[i] for i in r_indices],
        }

    @property
    def indication(self):
        """Get the indication matrix."""
        return self.results["indication"]

    @property
    def all_probs(self):
        """Get probabilities for all species in all sites."""
        return self.results["AllProbs"]

    @property
    def pool(self):
        """Get species pool probabilities (present species set to 1)."""
        return self.results["Pool"]

    @property
    def dark(self):
        """Get dark diversity probabilities (present species set to NaN)."""
        return self.results["Dark"]

    def to_dataframes(self):
        """
        Convert results to pandas DataFrames with proper column/row names.

        Returns:
        --------
        dict
            Dictionary of DataFrames with results
        """
        if "x_cols" not in self.results:
            # Generate default column names
            n_species = self.results["AllProbs"].shape[1]
            species_names = [f"Species_{i}" for i in range(n_species)]
        else:
            species_names = self.results["x_cols"]

        n_sites = self.results["AllProbs"].shape[0]
        site_names = [f"Site_{i}" for i in range(n_sites)]

        # Create indication matrix column/row names
        indication_rows = [f"T.{name}" for name in species_names]
        indication_cols = [f"I.{name}" for name in species_names]

        return {
            "indication": pd.DataFrame(
                self.results["indication"],
                index=indication_rows,
                columns=indication_cols,
            ),
            "AllProbs": pd.DataFrame(
                self.results["AllProbs"], index=site_names, columns=species_names
            ),
            "Pool": pd.DataFrame(
                self.results["Pool"], index=site_names, columns=species_names
            ),
            "Dark": pd.DataFrame(
                self.results["Dark"], index=site_names, columns=species_names
            ),
        }


# Convenience function to match R interface
def DarkDiv_calc(
    x,
    r=None,
    method="Hypergeometric",
    limit="min",
    const=0.01,
    remove_absent=True,
    wa=False,
    weights=None,
):
    """
    Convenience function that matches the R DarkDiv interface.

    Returns the results dictionary directly.
    """
    calculator = DarkDiv(x, r, method, limit, const, remove_absent, wa, weights)
    return calculator.to_dataframes()
