"""
Utility functions for dark diversity calculations
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Dict, Optional, Any, Union


def data_prep(
    x: Union[np.ndarray, pd.DataFrame],
    r: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    remove_absent: bool = True,
) -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
    """
    Check the validity of data provided for dark diversity estimations.

    Parameters:
    -----------
    x : array-like
        Study data with sites in rows and species in columns
    r : array-like, optional
        Reference dataset with sites in rows and species in columns.
        If not provided, x is used by default
    remove_absent : bool
        Whether to remove species with zero occurrences

    Returns:
    --------
    dict
        Dictionary containing processed x, r, and original versions
    """
    # Convert to numpy arrays if needed
    x_orig = np.array(x) if not isinstance(x, np.ndarray) else x.copy()
    r_orig = np.array(r) if r is not None else x_orig.copy()

    x = x_orig.copy()
    r = r_orig.copy() if r is not None else x_orig.copy()

    # Get column names if working with DataFrames
    if isinstance(x, pd.DataFrame):
        x_cols = x.columns.tolist()
        x = x.values
    else:
        x_cols = [f"Species_{i}" for i in range(x.shape[1])]

    if isinstance(r, pd.DataFrame):
        r_cols = r.columns.tolist()
        r = r.values
    else:
        r_cols = [f"Species_{i}" for i in range(r.shape[1])]

    # Check that species names exist
    if x_cols is None or r_cols is None:
        raise ValueError("x and/or r must have column names (species names)")

    # Check for common species between x and r
    if not np.array_equal(x, r):
        common_species = list(set(x_cols) & set(r_cols))
        if not common_species:
            raise ValueError("x and r do not have common species names")

        if len(common_species) != len(x_cols):
            print("Warning: x does not contain exactly the same species as r.")
            print("Only those species present in r have been kept in x")

            # Filter to common species
            x_indices = [i for i, col in enumerate(x_cols) if col in r_cols]
            r_indices = [i for i, col in enumerate(r_cols) if col in x_cols]

            x = x[:, x_indices]
            r = r[:, r_indices]
            x_cols = [x_cols[i] for i in x_indices]
            r_cols = [r_cols[i] for i in r_indices]

    # Check for species with zero occurrences in r
    zero_occ_species = np.where(np.sum(r > 0, axis=0) == 0)[0]
    if len(zero_occ_species) > 0:
        if remove_absent:
            print(
                f"Warning: r included {len(zero_occ_species)} species with zero occurrences."
            )
            print("They have been removed from both r and x")
            keep_indices = np.where(np.sum(r > 0, axis=0) > 0)[0]
            r = r[:, keep_indices]
            x = x[:, keep_indices]
            x_cols = [x_cols[i] for i in keep_indices]
            r_cols = [r_cols[i] for i in keep_indices]
        else:
            print(
                f"Warning: r included {len(zero_occ_species)} species with zero occurrences."
            )
            print("They will be filled with NA")

    return {
        "x": x,
        "r": r,
        "x_orig": x_orig,
        "r_orig": r_orig,
        "x_cols": x_cols,
        "r_cols": r_cols,
    }


def cooc_prep(r: np.ndarray) -> np.ndarray:
    """
    Create co-occurrence matrix needed for dark diversity methods.

    Parameters:
    -----------
    r : array-like
        Reference dataset with sites in rows and species in columns

    Returns:
    --------
    numpy.ndarray
        Co-occurrence matrix
    """
    # Convert to binary presence/absence
    r_binary = np.where(r > 0, 1, 0)

    # Calculate co-occurrence matrix using matrix multiplication
    M = np.dot(r_binary.T, r_binary)

    return M


def hypergeometric(
    M: np.ndarray, x: np.ndarray, r: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Estimate dark diversity probability based on hypergeometric distribution.

    Parameters:
    -----------
    M : numpy.ndarray
        Co-occurrence matrix
    x : numpy.ndarray
        Study data
    r : numpy.ndarray
        Reference data

    Returns:
    --------
    dict
        Dictionary containing indication matrix and probability matrices
    """
    # Get diagonal (species frequencies)
    C = np.diag(M)
    N = r.shape[0]  # total number of plots
    S = len(C)  # total number of species

    # Create matrices for expected calculations
    M1 = np.tile(C, (S, 1))  # Species frequencies in rows
    M2 = M1.T  # Species frequencies in columns

    # Expected number of co-occurrences
    M_hat = (M1 * M2) / N

    # Variance calculation
    variance = (M1 * M2 / N) * (N - M1) / N * (N - M2) / (N - 1)

    # Standard deviation
    sdev = np.sqrt(variance)

    # Standardized effect size
    M_std = np.where(sdev == 0, 0, (M - M_hat) / sdev)

    # Zero species cannot have indication 1, replacing by 0
    M_std = np.where((M1 * M2) == 0, 0, M_std)

    # Set diagonal to 0
    np.fill_diagonal(M_std, 0)

    # Calculate probabilities for study sites
    row_sums = np.sum(x, axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)  # Avoid division by zero

    # Matrix multiplication to get indication values
    b = np.dot(x, M_std) / row_sums

    # Convert to probabilities using normal CDF
    b = stats.norm.cdf(b, loc=0, scale=1)

    # Create Pool matrix (set present species to 1)
    Pool = b.copy()
    Pool[x > 0] = 1

    # Create Dark matrix (set present species to NaN)
    Dark = b.copy()
    Dark[x > 0] = np.nan

    return {"indication": M_std, "AllProbs": b, "Pool": Pool, "Dark": Dark}


def beals(M: np.ndarray, x: np.ndarray, r: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Calculate Beals index for dark diversity estimation.

    Parameters:
    -----------
    M : numpy.ndarray
        Co-occurrence matrix
    x : numpy.ndarray
        Study data
    r : numpy.ndarray
        Reference data

    Returns:
    --------
    dict
        Dictionary containing indication matrix and probability matrices
    """
    # Get diagonal (species frequencies)
    C = np.diag(M)

    # Create indication matrix
    M_beals = M / np.where(C == 0, 1, C)[:, np.newaxis]
    np.fill_diagonal(M_beals, 0)

    # Calculate for reference dataset
    S = np.sum(r, axis=1)  # Species richness per site
    b_ref = np.zeros_like(r, dtype=float)

    for i in range(r.shape[0]):
        b_ref[i, :] = np.sum(M_beals * r[i, :], axis=1)

    # Normalize
    SM = np.tile(S, (r.shape[1], 1)).T - r
    SM = np.where(SM == 0, 1, SM)
    b_ref = b_ref / SM

    # Calculate for study dataset if different
    if not np.array_equal(x, r):
        S = np.sum(x, axis=1)
        b = np.zeros_like(x, dtype=float)

        for i in range(x.shape[0]):
            b[i, :] = np.sum(M_beals * x[i, :], axis=1)

        SM = np.tile(S, (x.shape[1], 1)).T - x
        SM = np.where(SM == 0, 1, SM)
        b = b / SM
    else:
        b = b_ref.copy()

    # Create Pool matrix (set present species to 1)
    Pool = b.copy()
    Pool[x > 0] = 1

    # Create Dark matrix (set present species to NaN)
    Dark = b.copy()
    Dark[x > 0] = np.nan

    return {
        "indication": M_beals,
        "AllProbs": b,
        "Pool": Pool,
        "Dark": Dark,
        "t": b_ref,  # Beals values for reference dataset
    }


def beals_threshold(
    beals_result: Dict[str, np.ndarray],
    limit: str = "min",
    const: float = 0.01,
    r: np.ndarray = None,
    x: np.ndarray = None,
) -> Dict[str, np.ndarray]:
    """
    Apply thresholds to Beals values to create binary presence/absence indications.

    Parameters:
    -----------
    beals_result : dict
        Results from beals function
    limit : str
        Method for threshold: "min", "quantile", "const", "outlier"
    const : float
        Constant for quantile or const methods
    r : numpy.ndarray
        Reference data
    x : numpy.ndarray
        Study data

    Returns:
    --------
    dict
        Dictionary with binary indication values
    """
    if limit not in ["quantile", "min", "const", "outlier"]:
        raise ValueError("limit must be one of: 'quantile', 'min', 'const', 'outlier'")

    if limit in ["quantile", "const"]:
        if const < 0 or const > 1:
            raise ValueError(f"For limit type {limit}, const must be between 0 and 1")

    # Get thresholds for each species
    q = np.ones(r.shape[1])
    t = beals_result["t"]
    b = beals_result["AllProbs"]

    for j in range(r.shape[1]):
        if np.sum(r[:, j]) > 0:  # Species is present in reference data
            present_sites = r[:, j] > 0
            species_vals = t[present_sites, j]

            if limit == "min":
                q[j] = np.min(species_vals)
            elif limit == "outlier":
                q25 = np.percentile(species_vals, 25)
                iqr = np.percentile(species_vals, 75) - q25
                threshold = q25 - 1.5 * iqr
                q[j] = max(threshold, np.min(species_vals))
            elif limit == "quantile":
                q[j] = np.percentile(species_vals, const * 100)
            elif limit == "const":
                q[j] = const

    # Create binary dark diversity matrix
    DD = np.zeros_like(x)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if b[i, j] >= q[j] and x[i, j] == 0:
                DD[i, j] = 1

    # AllProbs and Pool are binary (DD + present species)
    AllProbs_Pool = DD + (x > 0).astype(int)

    # Dark matrix (set present species to NaN)
    Dark = DD.astype(float)
    Dark[x > 0] = np.nan

    return {
        "indication": beals_result["indication"],
        "AllProbs": AllProbs_Pool,
        "Pool": AllProbs_Pool,
        "Dark": Dark,
    }


def favorability(
    beals_result: Dict[str, np.ndarray], x: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Apply favorability correction to Beals values.

    Parameters:
    -----------
    beals_result : dict
        Results from beals function
    x : numpy.ndarray
        Study data

    Returns:
    --------
    dict
        Dictionary with favorability-corrected values
    """
    # Get Beals probabilities
    P = beals_result["AllProbs"]

    # Calculate species frequencies
    n1 = np.sum(x > 0, axis=0)  # Number of presences per species
    n0 = x.shape[0] - n1  # Number of absences per species

    # Expand to match dimensions
    n1_matrix = np.tile(n1, (x.shape[0], 1))
    n0_matrix = np.tile(n0, (x.shape[0], 1))

    # Calculate favorability
    # Avoid division by zero
    denominator = (n1_matrix / n0_matrix) + (P / (1 - P + 1e-10))
    DD = (P / (1 - P + 1e-10)) / denominator

    # Handle edge cases
    DD = np.where(np.isnan(DD) | np.isinf(DD), 0, DD)

    # Create Pool matrix (set present species to 1)
    Pool = DD.copy()
    Pool[x > 0] = 1

    # Create Dark matrix (set present species to NaN)
    Dark = DD.copy()
    Dark[x > 0] = np.nan

    return {
        "indication": beals_result["indication"],
        "AllProbs": DD,
        "Pool": Pool,
        "Dark": Dark,
    }


def dd_weighting(
    x: np.ndarray,
    indication: np.ndarray,
    weights: Optional[np.ndarray] = None,
    method: str = "Hypergeometric",
) -> Dict[str, np.ndarray]:
    """
    Apply abundance weighting to dark diversity calculations.

    Parameters:
    -----------
    x : numpy.ndarray
        Study data with abundances
    indication : numpy.ndarray
        Indication matrix
    weights : numpy.ndarray, optional
        Weight matrix. If None, uses abundances from x
    method : str
        Method used ("Hypergeometric", "Favorability", etc.)

    Returns:
    --------
    dict
        Dictionary with weighted results
    """
    if weights is None:
        weights = x.copy()

    # Convert to relative weights
    row_sums = np.sum(weights, axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    weights_rel = weights / row_sums

    # Calculate weighted indication values
    out = np.dot(weights_rel, indication)

    # Apply method-specific transformations
    if method == "Hypergeometric":
        out = stats.norm.cdf(out, loc=0, scale=1)

    # Create Pool matrix (set present species to 1)
    Pool = out.copy()
    Pool[x > 0] = 1

    # Create Dark matrix (set present species to NaN)
    Dark = out.copy()
    Dark[x > 0] = np.nan

    result = {"indication": indication, "AllProbs": out, "Pool": Pool, "Dark": Dark}

    # Apply favorability correction if needed
    if method == "Favorability":
        result = favorability(result, x)

    return result
