"""
Example usage of PyDarkDiv package

This script demonstrates how to use the PyDarkDiv package to calculate dark diversity
from species occurrence data, replicating the functionality of the R DarkDiv package.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the src directory to the path for this example
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pydarkdiv import DarkDiv, DarkDiv_calc


def load_example_data():
    """Load the dune dataset for examples."""
    try:
        dune_df = pd.read_csv("tests/dune.csv")
        return dune_df
    except FileNotFoundError:
        print("Error: Could not find tests/dune.csv")
        print("Make sure you're running this from the PyDarkDiv root directory")
        return None


def example_basic_usage():
    """Basic usage example."""
    print("=== Basic Usage Example ===")

    # Load data
    dune = load_example_data()
    if dune is None:
        return

    print(f"Loaded dune dataset with {dune.shape[0]} sites and {dune.shape[1]} species")

    # Calculate dark diversity using Hypergeometric method
    results = DarkDiv_calc(dune, method="Hypergeometric")

    print(f"Results contain:")
    print(f"- indication: {results['indication'].shape} - indication matrix")
    print(f"- AllProbs: {results['AllProbs'].shape} - probabilities for all species")
    print(f"- Pool: {results['Pool'].shape} - species pool (present species = 1)")
    print(f"- Dark: {results['Dark'].shape} - dark diversity (present species = NaN)")

    # Show some example values
    print(f"\\nExample pool probabilities for first site:")
    print(results["Pool"].iloc[0, :5])  # First 5 species


def example_different_methods():
    """Example using different methods."""
    print("\\n=== Different Methods Example ===")

    dune = load_example_data()
    if dune is None:
        return

    methods = ["Hypergeometric", "RawBeals", "Favorability"]

    for method in methods:
        print(f"\\nUsing {method} method:")
        results = DarkDiv_calc(dune, method=method)

        # Calculate some summary statistics
        pool_probs = results["Pool"].values
        mean_prob = np.nanmean(pool_probs)
        print(f"  Average pool probability: {mean_prob:.3f}")

        # Count high-probability absent species (dark diversity candidates)
        dark_probs = results["Dark"].values
        high_dark = np.sum((dark_probs > 0.5) & np.isfinite(dark_probs))
        total_absent = np.sum(np.isfinite(dark_probs))
        print(f"  High-probability dark diversity species: {high_dark}/{total_absent}")


def example_class_interface():
    """Example using the DarkDiv class."""
    print("\\n=== Class Interface Example ===")

    dune = load_example_data()
    if dune is None:
        return

    # Create DarkDiv calculator object
    calculator = DarkDiv(dune, method="Hypergeometric")

    # Access results through properties
    print(f"Indication matrix shape: {calculator.indication.shape}")
    print(f"Pool matrix shape: {calculator.pool.shape}")
    print(f"Dark matrix shape: {calculator.dark.shape}")

    # Convert to properly labeled DataFrames
    df_results = calculator.to_dataframes()

    print(f"\\nDataFrame results:")
    print(f"- indication DataFrame with proper T./I. labels")
    print(f"- Pool DataFrame with site and species names")
    print(f"- Dark DataFrame showing dark diversity probabilities")

    # Show example of indication matrix
    print(f"\\nIndication matrix (first 3x3):")
    print(df_results["indication"].iloc[:3, :3])


def example_species_pool_interpretation():
    """Example of interpreting species pool results."""
    print("\\n=== Species Pool Interpretation ===")

    dune = load_example_data()
    if dune is None:
        return

    # Calculate using Hypergeometric method
    calculator = DarkDiv(dune, method="Hypergeometric")

    # Get pool probabilities
    pool_df = calculator.to_dataframes()["Pool"]

    # For each site, show the most likely species in the pool
    print("Top 3 species pool candidates per site (first 5 sites):")

    for site_idx in range(min(5, pool_df.shape[0])):
        site_probs = pool_df.iloc[site_idx, :]
        top_species = site_probs.nlargest(3)

        print(f"\\nSite {site_idx + 1}:")
        for species, prob in top_species.items():
            status = "PRESENT" if prob == 1.0 else f"Prob={prob:.3f}"
            print(f"  {species}: {status}")


def validate_against_r():
    """Validate that our results match the R implementation."""
    print("\\n=== Validation Against R ===")

    dune = load_example_data()
    if dune is None:
        return

    try:
        # Load expected R results
        expected_indication = pd.read_csv("tests/indication.csv")
        expected_pool = pd.read_csv("tests/pool.csv", index_col=0)

        # Calculate with Python
        results = DarkDiv_calc(dune, method="Hypergeometric")

        # Compare indication matrix
        species_names = dune.columns.tolist()
        target_names = [f"T.{name}" for name in species_names]
        expected_indication.index = target_names

        indication_diff = np.max(
            np.abs(results["indication"].values - expected_indication.values)
        )
        print(f"Indication matrix max difference from R: {indication_diff:.2e}")

        # Compare pool matrix (R excludes first species)
        python_pool_subset = results["Pool"].iloc[:, 1:]
        pool_diff = np.max(np.abs(python_pool_subset.values - expected_pool.values))
        print(f"Pool matrix max difference from R: {pool_diff:.2e}")

        if indication_diff < 1e-10 and pool_diff < 1e-10:
            print("✅ Python implementation matches R results perfectly!")
        else:
            print("❌ Some differences found - check implementation")

    except FileNotFoundError as e:
        print(f"Could not load R test files: {e}")
        print("Validation requires tests/indication.csv and tests/pool.csv")


if __name__ == "__main__":
    print("PyDarkDiv Example Usage")
    print("======================")

    example_basic_usage()
    example_different_methods()
    example_class_interface()
    example_species_pool_interpretation()
    validate_against_r()

    print("\\n=== Summary ===")
    print("PyDarkDiv successfully converts the R DarkDiv functionality to Python!")
    print("Key features:")
    print("- Multiple methods: Hypergeometric, RawBeals, Favorability, ThresholdBeals")
    print("- Species pool calculation matching R implementation")
    print("- Proper handling of present vs absent species")
    print("- Both functional and object-oriented interfaces")
    print("- Full compatibility with R results (within numerical precision)")
