# PyDarkDiv Usage Examples

"""
Simple usage examples for PyDarkDiv package.

PyDarkDiv is a Python implementation of the DarkDiv R package for calculating
dark diversity from species co-occurrence data, based on methods described in:

Carmona, C.P. & Pärtel, M. (2021). Estimating probabilistic site-specific
species pools and dark diversity from co-occurrence data. Global Ecology
and Biogeography, 30(1), 316-326.
"""

import pandas as pd
import numpy as np
import pydarkdiv as pdd


# Example 1: Basic usage with CSV data
def basic_example():
    """Basic dark diversity calculation."""
    # Load your species data (sites × species matrix)
    # data = pd.read_csv('your_data.csv', index_col=0)

    # For demonstration, create sample data
    np.random.seed(42)
    sites = [f"Site_{i}" for i in range(1, 11)]
    species = [f"Species_{i}" for i in range(1, 21)]
    data = pd.DataFrame(np.random.poisson(1, (10, 20)), index=sites, columns=species)

    print("Sample data shape:", data.shape)
    print("Species richness per site:", (data > 0).sum(axis=1).values)

    # Calculate dark diversity using hypergeometric method (recommended)
    dd = pdd.DarkDiv(data, method="Hypergeometric")

    # Get results as labeled DataFrames
    dfs = dd.to_dataframes()

    print("\nResults available:", list(dfs.keys()))
    print("Dark diversity matrix shape:", dfs["dark"].shape)

    # Calculate summary statistics
    dark_richness = (dfs["dark"] > 0.5).sum(axis=1)  # Species with >50% probability
    pool_richness = dfs["pool"].sum(axis=1)  # Total species pool size
    realized_richness = (data > 0).sum(axis=1)  # Currently present species

    print(f"\nSummary statistics:")
    print(f"Average realized diversity: {realized_richness.mean():.1f}")
    print(f"Average dark diversity (>50% prob): {dark_richness.mean():.1f}")
    print(f"Average total pool size: {pool_richness.mean():.1f}")

    return dd, dfs


# Example 2: Compare different methods
def method_comparison():
    """Compare different dark diversity calculation methods."""
    # Create sample data
    np.random.seed(123)
    data = pd.DataFrame(
        np.random.poisson(1, (15, 25)),
        index=[f"Site_{i}" for i in range(1, 16)],
        columns=[f"Sp_{i}" for i in range(1, 26)],
    )

    methods = ["Hypergeometric", "RawBeals", "Favorability", "ThresholdBeals"]
    results = {}

    print("Comparing methods...")
    for method in methods:
        dd = pdd.DarkDiv(data, method=method)
        dfs = dd.to_dataframes()

        if method == "ThresholdBeals":
            dark_richness = dfs["dark"].sum(axis=1)  # Binary method
        else:
            dark_richness = (dfs["dark"] > 0.5).sum(axis=1)  # Probabilistic methods

        results[method] = dark_richness
        print(f"{method}: Mean dark richness = {dark_richness.mean():.1f}")

    # Create comparison DataFrame
    comparison = pd.DataFrame(results)
    print("\nMethod correlations:")
    print(comparison.corr().round(3))

    return comparison


# Example 3: Using reference data
def reference_example():
    """Use separate reference dataset for indication matrix."""
    # Study data (smaller)
    study_data = pd.DataFrame(
        np.random.poisson(1, (8, 15)),
        index=[f"StudySite_{i}" for i in range(1, 9)],
        columns=[f"Species_{i}" for i in range(1, 16)],
    )

    # Reference data (larger, regional dataset)
    reference_data = pd.DataFrame(
        np.random.poisson(1, (20, 15)),
        index=[f"RefSite_{i}" for i in range(1, 21)],
        columns=[f"Species_{i}" for i in range(1, 16)],
    )

    print("Study data shape:", study_data.shape)
    print("Reference data shape:", reference_data.shape)

    # Calculate dark diversity using reference data
    dd = pdd.DarkDiv(x=study_data, r=reference_data, method="Hypergeometric")
    dfs = dd.to_dataframes()

    dark_richness = (dfs["dark"] > 0.5).sum(axis=1)
    print(f"Mean dark diversity with reference data: {dark_richness.mean():.1f}")

    return dd


# Example 4: Abundance weighting
def abundance_weighting_example():
    """Use abundance weighting in calculations."""
    # Create abundance data (not just presence/absence)
    np.random.seed(456)
    data = pd.DataFrame(
        np.random.lognormal(0, 1, (12, 18)),
        index=[f"Site_{i}" for i in range(1, 13)],
        columns=[f"Species_{i}" for i in range(1, 19)],
    )

    # Set some values to zero to create realistic presence/absence pattern
    data[data < 0.5] = 0

    print("Abundance data summary:")
    print(f"Mean abundance: {data[data > 0].mean().mean():.2f}")
    print(f"Species occurrence frequency: {(data > 0).mean(axis=0).mean():.2f}")

    # Compare with and without abundance weighting
    dd_no_weight = pdd.DarkDiv(data, method="Hypergeometric", wa=False)
    dd_weighted = pdd.DarkDiv(data, method="Hypergeometric", wa=True)

    dfs_no_weight = dd_no_weight.to_dataframes()
    dfs_weighted = dd_weighted.to_dataframes()

    dark_no_weight = (dfs_no_weight["dark"] > 0.5).sum(axis=1)
    dark_weighted = (dfs_weighted["dark"] > 0.5).sum(axis=1)

    print(f"\nMean dark diversity without weighting: {dark_no_weight.mean():.1f}")
    print(f"Mean dark diversity with abundance weighting: {dark_weighted.mean():.1f}")
    print(f"Correlation between methods: {dark_no_weight.corr(dark_weighted):.3f}")

    return dd_no_weight, dd_weighted


def main():
    """Run all examples."""
    print("=== PyDarkDiv Usage Examples ===\n")

    print("Example 1: Basic usage")
    print("-" * 30)
    dd, dfs = basic_example()

    print("\n\nExample 2: Method comparison")
    print("-" * 30)
    comparison = method_comparison()

    print("\n\nExample 3: Reference data")
    print("-" * 30)
    dd_ref = reference_example()

    print("\n\nExample 4: Abundance weighting")
    print("-" * 30)
    dd_no_weight, dd_weighted = abundance_weighting_example()

    print("\n=== Examples completed ===")


if __name__ == "__main__":
    main()
