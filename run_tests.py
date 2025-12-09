"""
Quick test runner for loan default prediction project
======================================================

Usage:
    python run_tests.py              # Run all Part 1 tests
    python run_tests.py -v           # Verbose output
    python run_tests.py -k missing   # Run only tests matching "missing"
"""

import sys
import pytest
from pathlib import Path


def main():
    """Run Part 1 data quality tests"""

    # Check if cleaned data exists
    cleaned_csv = Path("processed_data/cleaned_data_part1.csv")
    if not cleaned_csv.exists():
        print("=" * 80)
        print("ERROR: Cleaned data not found!")
        print("=" * 80)
        print(f"\nExpected file: {cleaned_csv}")
        print("\nPlease run your Part 1 EDA notebook first:")
        print("  1. Open part-1-eda.ipynb")
        print("  2. Run all cells")
        print("  3. Ensure cleaned_data_part1.csv is saved")
        print("\nThen re-run this test script.")
        print("=" * 80)
        return 1

    # Run pytest with arguments
    test_path = "tests/test_part1_data_quality.py"

    # Default args
    args = [test_path, "-v", "--tb=short"]

    # Add any command line arguments
    if len(sys.argv) > 1:
        # Remove default -v if user provided their own
        if any(arg in sys.argv[1:] for arg in ['-v', '-vv', '-q']):
            args.remove('-v')
        args.extend(sys.argv[1:])

    print("=" * 80)
    print("RUNNING PART 1 DATA QUALITY TESTS")
    print("=" * 80)
    print()

    # Run tests
    exit_code = pytest.main(args)

    # Additional summary if tests passed
    if exit_code == 0:
        print("\n" + "=" * 80)
        print("🎉 SUCCESS! All Part 1 data quality tests passed!")
        print("=" * 80)
        print("\nYour data is ready for:")
        print("  → Part 2: Visualization & Bivariate Analysis")
        print("  → Part 3: Categorical & Temporal Data Processing")
        print("  → Part 4: Feature Engineering")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("❌ TESTS FAILED!")
        print("=" * 80)
        print("\nPlease review the failures above and:")
        print("  1. Identify which validation failed")
        print("  2. Fix the corresponding code in part-1-eda.ipynb")
        print("  3. Re-run Part 1 EDA")
        print("  4. Re-run these tests")
        print("\nFor help, see: tests/README_TESTING.md")
        print("=" * 80)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())