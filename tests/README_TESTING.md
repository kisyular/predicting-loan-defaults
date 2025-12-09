# Part 1 Data Quality Testing

This directory contains comprehensive pytest tests to validate the data quality after Part 1 EDA processing.

## Installation

First, install pytest:

```bash
pip install pytest pytest-cov
```

## Running Tests

### Run all tests:
```bash
pytest tests/test_part1_data_quality.py -v
```

### Run with detailed output:
```bash
pytest tests/test_part1_data_quality.py -v --tb=long
```

### Run specific test class:
```bash
# Test only missing values
pytest tests/test_part1_data_quality.py::TestMissingValues -v

# Test only temporal variables
pytest tests/test_part1_data_quality.py::TestTemporalVariables -v

# Test only outliers
pytest tests/test_part1_data_quality.py::TestOutlierTreatment -v
```

### Run specific test:
```bash
pytest tests/test_part1_data_quality.py::TestMissingValues::test_no_missing_values -v
```

### Run tests matching pattern:
```bash
# All tests with "missing" in name
pytest tests/test_part1_data_quality.py -v -k "missing"

# All tests with "temporal" in name
pytest tests/test_part1_data_quality.py -v -k "temporal"
```

### Generate HTML coverage report:
```bash
pytest tests/test_part1_data_quality.py --cov=. --cov-report=html
```

## Test Structure

### Test Classes

1. **TestMissingValues**: Validates that all missing values have been properly imputed
2. **TestDataStructure**: Checks data shape, target variable, and column consistency
3. **TestTemporalVariables**: Validates temporal variable conversions and ranges
4. **TestOutlierTreatment**: Ensures outliers have been properly capped/winsorized
5. **TestCategoricalVariables**: Checks cardinality reduction and rare category binning
6. **TestDataTypes**: Validates correct data types for all features
7. **TestStatisticalProperties**: Checks statistical properties like target balance and correlations
8. **TestArtifacts**: Verifies all required files and metadata have been saved

## What Gets Tested

### Critical Validations

- **No missing values** (0 NaN in entire dataset)
- **Temporal variables** converted correctly (age 18-100 years, employment 0-50 years)
- **No unemployment bug** (365243 days code properly handled)
- **Outliers capped** (winsorization applied to financial variables)
- **Cardinality reduced** (high-cardinality categoricals binned)
- **All lowercase** (categorical variables standardized)
- **Target binary** (only 0 and 1 values)
- **No duplicates** (0 duplicate rows)
- **Artifacts saved** (CSV + JSON metadata files exist)

### tatistical Validations

- Target imbalance (5-15% default rate expected)
- Feature correlations with target (at least 3 features with |r| > 0.05)
- No constant columns (all features have variance)
- Reasonable data ranges (no extreme outliers)

## Expected Output

### All tests pass:

```
============================== test session starts ===============================
collected 35 items

tests/test_part1_data_quality.py::TestMissingValues::test_no_missing_values PASSED
tests/test_part1_data_quality.py::TestMissingValues::test_missing_by_column PASSED
tests/test_part1_data_quality.py::TestMissingValues::test_metadata_missing_count PASSED
tests/test_part1_data_quality.py::TestDataStructure::test_minimum_rows PASSED
tests/test_part1_data_quality.py::TestDataStructure::test_target_column_exists PASSED
tests/test_part1_data_quality.py::TestDataStructure::test_target_binary PASSED
...

=============================== 35 passed in 5.23s ==============================

================================ PART 1 DATA QUALITY REPORT ====================

ALL DATA QUALITY TESTS PASSED!

Dataset Summary:
  Original shape: (20000, 68)
  Cleaned shape: (20000, 73)
  Missing values: 0
  Numerical features: 61
  Categorical features: 12

Target Distribution:
  Default rate: 8.17%
  Class balance: 91.8% / 8.2%

Data is ready for Part 2: VISUALIZATION & BIVARIATE ANALYSIS
================================================================================
```

### If tests fail:

```
FAILED tests/test_part1_data_quality.py::TestMissingValues::test_no_missing_values
AssertionError: Found 243 missing values in cleaned data
```

This indicates missing values weren't properly imputed - check your imputation code.

## Troubleshooting

### FileNotFoundError

```
pytest.skip: Cleaned data not found at processed_data/cleaned_data_part1.csv
```

**Solution**: Run your Part 1 EDA notebook first to generate the cleaned data.

### Test failures

If tests fail:

1. Check the error message for which validation failed
2. Review your Part 1 EDA code for that specific transformation
3. Re-run Part 1 EDA with fixes
4. Re-run tests

### Common Issues

| Error | Cause | Fix |
|-------|-------|-----|
| Missing values test fails | Imputation not applied | Check imputation code cells |
| Employment years > 50 | Unemployment bug not fixed | Fix 365243 code before conversion |
| Artifacts not found | Part 1 not run | Run part-1-eda.ipynb completely |
| Correlation test fails | corr_df not created | Add correlation calculation code |

## Integrating with CI/CD

Add to your `.github/workflows/test.yml`:

```yaml
name: Data Quality Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install pytest pandas numpy
      - name: Run tests
        run: |
          pytest tests/test_part1_data_quality.py -v
```

## Best Practices

1. **Run tests after every Part 1 modification**
   ```bash
   pytest tests/test_part1_data_quality.py -v
   ```

2. **Check specific concerns quickly**
   ```bash
   # Just check missing values
   pytest tests/ -k "missing" -v

   # Just check temporal variables
   pytest tests/ -k "temporal" -v
   ```

3. **Use in pre-commit hooks**
   ```bash
   # .git/hooks/pre-commit
   #!/bin/bash
   pytest tests/test_part1_data_quality.py -v
   ```

## Next Steps

After all tests pass:

1. Part 1 data quality validated
2. → Proceed to Part 2: Visualization & Bivariate Analysis
3. → Create similar test suites for Part 2, 3, 4

## Questions?

- Tests failing? Check the error messages for specific validation failures
- Need to add tests? Add new test methods to the appropriate class
- Want to skip tests? Use `@pytest.mark.skip("reason")`

---

**Remember**: Tests are your safety net. Green tests = production-ready data!