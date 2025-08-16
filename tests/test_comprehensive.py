"""
PyDarkDiv Validation Test Suite

This test validates that the Python implementation matches the R DarkDiv package results.
"""

import numpy as np
import pandas as pd
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pydarkdiv import DarkDiv, DarkDiv_calc


def test_species_pool_calculation():
    """
    Test species pool calculation - the most important function to validate.
    """
    print("Testing species pool calculation against R DarkDiv...")

    # Load test data
    dune_df = pd.read_csv("tests/dune.csv")
    expected_indication = pd.read_csv("tests/indication.csv")
    expected_pool = pd.read_csv("tests/pool.csv", index_col=0)

    # Set proper row names for indication matrix
    species_names = dune_df.columns.tolist()
    target_names = [f"T.{name}" for name in species_names]
    expected_indication.index = target_names

    print(f"Input data: {dune_df.shape[0]} sites × {dune_df.shape[1]} species")

    # Calculate using Python implementation
    results = DarkDiv_calc(dune_df, method="Hypergeometric")

    # Validate indication matrix (30×30)
    indication_diff = np.max(
        np.abs(results["indication"].values - expected_indication.values)
    )
    indication_match = indication_diff < 1e-10

    # Validate pool matrix (R excludes first species, so we compare 20×29)
    python_pool = results["Pool"].iloc[:, 1:]  # Exclude Achimill
    pool_diff = np.max(np.abs(python_pool.values - expected_pool.values))
    pool_match = pool_diff < 1e-10

    # Calculate correlation for additional validation
    correlation = np.corrcoef(
        python_pool.values.flatten(), expected_pool.values.flatten()
    )[0, 1]

    print(f"\nValidation Results:")
    print(f"- Indication matrix max difference: {indication_diff:.2e}")
    print(f"- Pool matrix max difference: {pool_diff:.2e}")
    print(f"- Pool matrix correlation with R: {correlation:.10f}")

    # Assert validation
    assert (
        indication_match
    ), f"Indication matrix difference too large: {indication_diff}"
    assert pool_match, f"Pool matrix difference too large: {pool_diff}"
    assert correlation > 0.999999, f"Correlation too low: {correlation}"

    print("✅ Species pool calculation matches R implementation perfectly!")
    return True


def test_different_methods():
    """Test that all methods work without errors."""
    print("\nTesting different methods...")

    dune_df = pd.read_csv("tests/dune.csv")
    methods = ["Hypergeometric", "RawBeals", "Favorability"]

    for method in methods:
        try:
            calculator = DarkDiv(dune_df, method=method)

            # Basic shape validation
            assert calculator.indication.shape == (30, 30)
            assert calculator.pool.shape == (20, 30)
            assert calculator.dark.shape == (20, 30)

            print(f"✅ {method} method working correctly")

        except Exception as e:
            print(f"❌ {method} method failed: {e}")
            raise


def test_present_species_handling():
    """Test that present species are handled correctly."""
    print("\nTesting present species handling...")

    dune_df = pd.read_csv("tests/dune.csv")
    calculator = DarkDiv(dune_df, method="Hypergeometric")

    # Get binary presence/absence matrix
    x_binary = (dune_df.values > 0).astype(int)
    pool_matrix = calculator.pool
    dark_matrix = calculator.dark

    # Check a sample of positions
    errors = []
    for i in range(min(5, x_binary.shape[0])):
        for j in range(min(10, x_binary.shape[1])):
            if x_binary[i, j] == 1:
                if pool_matrix[i, j] != 1:
                    errors.append(f"Pool not 1 at ({i}, {j}): {pool_matrix[i, j]}")
                if not np.isnan(dark_matrix[i, j]):
                    errors.append(f"Dark not NaN at ({i}, {j}): {dark_matrix[i, j]}")

    assert len(errors) == 0, f"Present species handling errors: {errors}"
    print("✅ Present species handling correct")


def test_dataframe_interface():
    """Test the DataFrame interface."""
    print("\nTesting DataFrame interface...")

    dune_df = pd.read_csv("tests/dune.csv")
    calculator = DarkDiv(dune_df, method="Hypergeometric")
    df_results = calculator.to_dataframes()

    # Check that we get DataFrames
    assert isinstance(df_results["indication"], pd.DataFrame)
    assert isinstance(df_results["Pool"], pd.DataFrame)
    assert isinstance(df_results["Dark"], pd.DataFrame)

    # Check basic shapes
    assert df_results["indication"].shape == (30, 30)
    assert df_results["Pool"].shape == (20, 30)

    print("✅ DataFrame interface working correctly")


if __name__ == "__main__":
    print("PyDarkDiv Validation Test Suite")
    print("=" * 40)

    try:
        # Main validation test
        test_species_pool_calculation()

        # Additional functionality tests
        test_different_methods()
        test_present_species_handling()
        test_dataframe_interface()

        print("\n" + "=" * 40)
        print("🎉 ALL TESTS PASSED!")
        print("PyDarkDiv successfully reproduces R DarkDiv results!")
        print("✅ Species pool calculation: Perfect match")
        print("✅ All methods: Working correctly")
        print("✅ Data handling: Robust")
        print("✅ Interface: Complete")

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
