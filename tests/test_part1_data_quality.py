"""
Part 1 Data Quality Tests
==========================
Comprehensive test suite to validate data quality after Part 1 EDA processing.

Run with:
    pytest tests/test_part1_data_quality.py -v
    pytest tests/test_part1_data_quality.py -v --tb=short  # Short traceback
    pytest tests/test_part1_data_quality.py -v -k "test_missing"  # Run specific test
"""

import pytest
import pandas as pd
import numpy as np
import json
from pathlib import Path


# ============================================================================
# FIXTURES - Load data once for all tests
# ============================================================================

@pytest.fixture(scope="module")
def cleaned_data():
    """Load cleaned data from Part 1"""
    data_path = Path("processed_data/cleaned_data_part1.csv")
    if not data_path.exists():
        pytest.skip(f"Cleaned data not found at {data_path}")
    return pd.read_csv(data_path)


@pytest.fixture(scope="module")
def metadata():
    """Load metadata from Part 1"""
    metadata_path = Path("artifacts/part1_metadata.json")
    if not metadata_path.exists():
        pytest.skip(f"Metadata not found at {metadata_path}")
    with open(metadata_path, 'r') as f:
        return json.load(f)


@pytest.fixture(scope="module")
def feature_lists():
    """Load feature lists from Part 1"""
    feature_path = Path("artifacts/part1_feature_lists.json")
    if not feature_path.exists():
        pytest.skip(f"Feature lists not found at {feature_path}")
    with open(feature_path, 'r') as f:
        return json.load(f)


@pytest.fixture(scope="module")
def cardinality_info():
    """Load cardinality info from Part 1"""
    cardinality_path = Path("artifacts/cardinality_info.json")
    if not cardinality_path.exists():
        pytest.skip(f"Cardinality info not found at {cardinality_path}")
    with open(cardinality_path, 'r') as f:
        return json.load(f)


# ============================================================================
# TEST CLASS 1: Missing Values
# ============================================================================

class TestMissingValues:
    """Test suite for missing value handling"""

    def test_no_missing_values(self, cleaned_data):
        """Test that there are no missing values in the cleaned dataset"""
        missing_count = cleaned_data.isnull().sum().sum()
        assert missing_count == 0, f"Found {missing_count} missing values in cleaned data"

    def test_missing_by_column(self, cleaned_data):
        """Test that no individual column has missing values"""
        missing_by_col = cleaned_data.isnull().sum()
        columns_with_missing = missing_by_col[missing_by_col > 0]
        assert len(columns_with_missing) == 0, \
            f"Columns with missing values: {columns_with_missing.to_dict()}"

    def test_metadata_missing_count(self, metadata):
        """Test that metadata reports 0 missing values"""
        assert metadata['missing_values'] == 0, \
            f"Metadata reports {metadata['missing_values']} missing values"


# ============================================================================
# TEST CLASS 2: Data Shape and Structure
# ============================================================================

class TestDataStructure:
    """Test suite for data shape and structure"""

    def test_minimum_rows(self, cleaned_data):
        """Test that we have at least 19,000 rows (allowing for potential removals)"""
        assert len(cleaned_data) >= 19000, \
            f"Too few rows: {len(cleaned_data)} (expected >= 19,000)"

    def test_target_column_exists(self, cleaned_data):
        """Test that target column exists"""
        assert 'target' in cleaned_data.columns, "Target column not found"

    def test_target_binary(self, cleaned_data):
        """Test that target is binary (0 or 1)"""
        unique_values = cleaned_data['target'].unique()
        assert set(unique_values).issubset({0, 1}), \
            f"Target has non-binary values: {unique_values}"

    def test_feature_lists_match(self, cleaned_data, feature_lists):
        """Test that feature lists in metadata match actual columns"""
        expected_cols = set(feature_lists['numerical_features'] +
                          feature_lists['categorical_features'] +
                          [feature_lists['target']])
        actual_cols = set(cleaned_data.columns)

        missing_cols = expected_cols - actual_cols
        extra_cols = actual_cols - expected_cols

        assert len(missing_cols) == 0, f"Missing columns: {missing_cols}"
        assert len(extra_cols) == 0, f"Extra columns: {extra_cols}"

    def test_no_duplicate_rows(self, cleaned_data):
        """Test that there are no duplicate rows"""
        duplicate_count = cleaned_data.duplicated().sum()
        assert duplicate_count == 0, f"Found {duplicate_count} duplicate rows"


# ============================================================================
# TEST CLASS 3: Temporal Variables
# ============================================================================

class TestTemporalVariables:
    """Test suite for temporal variable transformations"""

    def test_temporal_vars_exist(self, cleaned_data):
        """Test that converted temporal variables exist"""
        required_temporal = ['age_years', 'employment_years']
        missing = [col for col in required_temporal if col not in cleaned_data.columns]
        assert len(missing) == 0, f"Missing temporal variables: {missing}"

    def test_age_years_range(self, cleaned_data):
        """Test that age_years is within realistic bounds"""
        age = cleaned_data['age_years']
        assert age.min() >= 18, f"Minimum age too low: {age.min()}"
        assert age.max() <= 100, f"Maximum age too high: {age.max()}"

    def test_employment_years_range(self, cleaned_data):
        """Test that employment_years is within realistic bounds (no 1000+ years bug)"""
        if 'employment_years' in cleaned_data.columns:
            employment = cleaned_data['employment_years']
            assert employment.max() <= 50, \
                f"Employment years too high: {employment.max()} (unemployment bug not fixed!)"
            assert employment.min() >= 0, \
                f"Employment years negative: {employment.min()}"

    def test_no_original_days_columns(self, cleaned_data):
        """Test that original DAYS_* columns were dropped"""
        days_cols = [col for col in cleaned_data.columns if col.startswith('days_')]
        assert len(days_cols) == 0, \
            f"Original DAYS_* columns still present: {days_cols}"


# ============================================================================
# TEST CLASS 4: Outlier Treatment
# ============================================================================

class TestOutlierTreatment:
    """Test suite for outlier handling"""

    def test_financial_vars_capped(self, cleaned_data):
        """Test that financial variables have been capped (winsorized)"""
        financial_vars = [col for col in cleaned_data.columns if col.startswith('amt_')]

        for col in financial_vars:
            if col in cleaned_data.columns:
                # Check if max value is reasonable (not in millions)
                max_val = cleaned_data[col].max()
                assert max_val < 10_000_000, \
                    f"{col} has extreme value: ${max_val:,.2f} (outliers not capped?)"

    def test_no_extreme_outliers(self, cleaned_data, feature_lists):
        """Test that extreme outliers (>99.9th percentile) are capped"""
        numerical_features = feature_lists['numerical_features']

        for col in numerical_features:
            if col in cleaned_data.columns:
                series = cleaned_data[col].dropna()
                if len(series) > 0:
                    p999 = series.quantile(0.999)
                    extreme_count = (series > p999 * 1.5).sum()

                    # Allow some extreme values but not many
                    extreme_pct = (extreme_count / len(series)) * 100
                    assert extreme_pct < 1.0, \
                        f"{col} has {extreme_pct:.2f}% extreme outliers (expected <1%)"


# ============================================================================
# TEST CLASS 5: Categorical Variables
# ============================================================================

class TestCategoricalVariables:
    """Test suite for categorical variable processing"""

    def test_no_high_cardinality(self, cleaned_data, cardinality_info):
        """Test that high cardinality variables have been reduced"""
        high_card_threshold = cardinality_info['high_cardinality_threshold']
        high_card_cols = cardinality_info['high_cardinality_cols']

        for col in high_card_cols:
            if col in cleaned_data.columns:
                unique_count = cleaned_data[col].nunique()
                # After binning, should be reduced
                assert unique_count <= 20, \
                    f"{col} still has high cardinality: {unique_count} unique values"

    def test_categorical_lowercase(self, cleaned_data, feature_lists):
        """Test that categorical variables are lowercase"""
        categorical_features = feature_lists['categorical_features']

        for col in categorical_features:
            if col in cleaned_data.columns:
                # Check if any uppercase values exist
                uppercase_count = cleaned_data[col].str.isupper().sum()
                assert uppercase_count == 0, \
                    f"{col} has {uppercase_count} uppercase values"

    def test_occupation_type_no_missing(self, cleaned_data):
        """Test that occupation_type has no missing values (should be 'unknown')"""
        if 'occupation_type' in cleaned_data.columns:
            missing = cleaned_data['occupation_type'].isnull().sum()
            assert missing == 0, \
                f"occupation_type has {missing} missing values (should be 'unknown')"

            # Check that 'unknown' category exists
            assert 'unknown' in cleaned_data['occupation_type'].values, \
                "occupation_type should have 'unknown' category for missing values"


# ============================================================================
# TEST CLASS 6: Data Types
# ============================================================================

class TestDataTypes:
    """Test suite for data type consistency"""

    def test_numerical_features_numeric(self, cleaned_data, feature_lists):
        """Test that numerical features are numeric types"""
        numerical_features = feature_lists['numerical_features']

        for col in numerical_features:
            if col in cleaned_data.columns:
                dtype = cleaned_data[col].dtype
                assert pd.api.types.is_numeric_dtype(dtype), \
                    f"{col} is not numeric (dtype: {dtype})"

    def test_categorical_features_object(self, cleaned_data, feature_lists):
        """Test that categorical features are object type"""
        categorical_features = feature_lists['categorical_features']

        for col in categorical_features:
            if col in cleaned_data.columns:
                dtype = cleaned_data[col].dtype
                assert dtype == 'object' or pd.api.types.is_categorical_dtype(dtype), \
                    f"{col} is not categorical (dtype: {dtype})"

    def test_target_numeric(self, cleaned_data):
        """Test that target is numeric"""
        assert pd.api.types.is_numeric_dtype(cleaned_data['target'].dtype), \
            f"Target is not numeric (dtype: {cleaned_data['target'].dtype})"


# ============================================================================
# TEST CLASS 7: Statistical Properties
# ============================================================================

class TestStatisticalProperties:
    """Test suite for statistical properties of the data"""

    def test_target_imbalance(self, cleaned_data):
        """Test that target variable has expected imbalance (around 8-10% default rate)"""
        default_rate = cleaned_data['target'].mean() * 100
        assert 5 <= default_rate <= 15, \
            f"Default rate {default_rate:.1f}% is outside expected range (5-15%)"

    def test_no_constant_columns(self, cleaned_data):
        """Test that no columns are constant (0 variance)"""
        for col in cleaned_data.columns:
            if pd.api.types.is_numeric_dtype(cleaned_data[col].dtype):
                unique_count = cleaned_data[col].nunique()
                assert unique_count > 1, f"{col} is constant (only 1 unique value)"

    def test_correlation_with_target_exists(self, cleaned_data, feature_lists):
        """Test that at least some features correlate with target"""
        numerical_features = feature_lists['numerical_features']

        correlations = []
        for col in numerical_features:
            if col in cleaned_data.columns:
                corr = abs(cleaned_data[col].corr(cleaned_data['target']))
                correlations.append(corr)

        # At least 3 features should have |correlation| > 0.05
        strong_corr_count = sum(1 for c in correlations if c > 0.05)
        assert strong_corr_count >= 3, \
            f"Only {strong_corr_count} features have correlation > 0.05 with target"


# ============================================================================
# TEST CLASS 8: Artifacts and Metadata
# ============================================================================

class TestArtifacts:
    """Test suite for saved artifacts and metadata"""

    def test_cleaned_csv_exists(self):
        """Test that cleaned CSV file exists"""
        csv_path = Path("processed_data/cleaned_data_part1.csv")
        assert csv_path.exists(), f"Cleaned CSV not found at {csv_path}"

    def test_metadata_json_exists(self):
        """Test that metadata JSON exists"""
        metadata_path = Path("artifacts/part1_metadata.json")
        assert metadata_path.exists(), f"Metadata not found at {metadata_path}"

    def test_feature_lists_json_exists(self):
        """Test that feature lists JSON exists"""
        feature_path = Path("artifacts/part1_feature_lists.json")
        assert feature_path.exists(), f"Feature lists not found at {feature_path}"

    def test_cardinality_json_exists(self):
        """Test that cardinality info JSON exists"""
        cardinality_path = Path("artifacts/cardinality_info.json")
        assert cardinality_path.exists(), f"Cardinality info not found at {cardinality_path}"

    def test_metadata_has_required_keys(self, metadata):
        """Test that metadata contains all required keys"""
        required_keys = ['timestamp', 'original_shape', 'cleaned_shape',
                        'missing_values', 'numerical_features', 'categorical_features',
                        'transformations_applied']

        missing_keys = [key for key in required_keys if key not in metadata]
        assert len(missing_keys) == 0, f"Metadata missing keys: {missing_keys}"

    def test_transformations_documented(self, metadata):
        """Test that all key transformations are documented"""
        transformations = metadata['transformations_applied']

        required_transformations = [
            'temporal', 'missing', 'outlier', 'rare', 'employment'
        ]

        for required in required_transformations:
            found = any(required.lower() in t.lower() for t in transformations)
            assert found, f"Transformation '{required}' not documented in metadata"


# ============================================================================
# SUMMARY REPORT
# ============================================================================

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Generate a summary report after all tests complete"""

    # Only run if tests passed
    if exitstatus == 0:
        terminalreporter.write_sep("=", "PART 1 DATA QUALITY REPORT")

        try:
            # Load data for summary
            df = pd.read_csv("processed_data/cleaned_data_part1.csv")
            with open("artifacts/part1_metadata.json") as f:
                metadata = json.load(f)

            terminalreporter.write_line("ALL DATA QUALITY TESTS PASSED!\n")

            terminalreporter.write_line("Dataset Summary:")
            terminalreporter.write_line(f"  Original shape: {tuple(metadata['original_shape'])}")
            terminalreporter.write_line(f"  Cleaned shape: {tuple(metadata['cleaned_shape'])}")
            terminalreporter.write_line(f"  Missing values: {metadata['missing_values']}")
            terminalreporter.write_line(f"  Numerical features: {len(metadata['numerical_features'])}")
            terminalreporter.write_line(f"  Categorical features: {len(metadata['categorical_features'])}")

            terminalreporter.write_line("\nTarget Distribution:")
            default_rate = df['target'].mean() * 100
            terminalreporter.write_line(f"  Default rate: {default_rate:.2f}%")
            terminalreporter.write_line(f"  Class balance: {(100-default_rate):.1f}% / {default_rate:.1f}%")

            terminalreporter.write_line("\n✓ Data is ready for Part 2: VISUALIZATION & BIVARIATE ANALYSIS")
            terminalreporter.write_line("="*80)

        except Exception as e:
            terminalreporter.write_line(f"\nNote: Could not generate summary report: {e}")


if __name__ == "__main__":
    # Allow running with: python test_part1_data_quality.py
    pytest.main([__file__, "-v", "--tb=short"])