# %% [markdown]
# # LOAN DEFAULT PREDICTION

# %% [markdown]
# ### **PART 1 - DATA QUALITY**
# Exploratory Data Analysis and foundational data understanding
# - Overall dataset inspection and structure analysis
# - Target variable distribution analysis (critical for imbalanced data)
# - Missing value pattern identification across all variables
# - Data quality assessment (duplicates, outliers, anomalies)
# - Baseline statistics and distributions
# - Visualization of key patterns
# - Foundation insights that guide team preprocessing decisions

# %% [markdown]
# ### 1. Setup and Imports

# %%
# !pip install pandas numpy matplotlib plotnine joblib seaborn

# %%
# Imports
import pandas as pd
import numpy as np
from plotnine import *
import matplotlib.pyplot as plt
import warnings, os, json
from IPython.display import display
from pathlib import Path
from pprint import pprint

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

# %%
# Plot theme
theme_set(
    theme_minimal()
    + theme(
        figure_size=(18, 8),
        text=element_text(size=11),
        plot_title=element_text(size=16, fontweight="bold"),
        plot_subtitle=element_text(size=12, fontstyle="italic"),
        axis_title=element_text(size=12),
        axis_text=element_text(size=10),
        legend_text=element_text(size=10),
        legend_position="bottom",
    )
)

# %%
# Paths
RAW_CSV = "case_data.csv"
OUT_DIR = Path("processed_data")
OUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ### 2. Data Exploration (EDA)

# %% [markdown]
# #### Data Loading

# %%
df = pd.read_csv(RAW_CSV)
print("Dataset shape:", df.shape)

# %% [markdown]
# #### General Overview of the Dataset

# %%
# Quick peek
display(df.head())

# LINE 68: display(df.head())

# ============================================================================
# Data Types Check
# ============================================================================
print("\nData Types Summary:")
print(df.dtypes.value_counts())
print("\nSample data types:")
display(df.dtypes.head(20))

# %% [markdown]
# ### Humanized Column Names
# We can use janitor to clean up the column names for better readability. You can install janitor via pip if you haven't already:
# 
# ```bash
# pip install pyjanitor
# ```
# 
# ```python
# from janitor import clean_names
# df = df.clean_names()
# df.head().T
# ```

# %%
# humanize column names for better readability using janitor
import janitor
df = df.clean_names()  # Convert column names to snake_case
print("Columns after cleaning:")
pprint(df.columns.tolist())

# %%
# Calculate basic dataset dimensions
n_rows, n_cols = df.shape          # Number of loan applications and variables
n_predictors = n_cols - 1            # All variables except 'target'

# Display the dataset size information
print("Dataset Dimensions:")
print(f"- Total observations (rows): {n_rows:,}")
print(f"- Total variables (columns): {n_cols}")
print(f"- Predictor variables: {n_predictors}")
print(f"- Response variable: 1 (target)")

# %%
# Classify variables by type
all_cols = df.columns.tolist()

# Get numeric and categorical variables
numeric_vars = df.select_dtypes(include=[np.number]).columns.tolist()
if 'target' in numeric_vars:
    numeric_vars.remove('target')

categorical_vars = df.select_dtypes(include=['object']).columns.tolist()

# Specific variable types based on naming patterns
date_vars = [col for col in numeric_vars if 'days_' in col]
flag_vars = [col for col in all_cols if 'flag_' in col]
amt_vars = [col for col in all_cols if 'amt_' in col]
cnt_vars = [col for col in all_cols if 'cnt_' in col]
ext_vars = [col for col in all_cols if 'ext_source' in col]

other_numeric = [col for col in numeric_vars 
                if col not in date_vars + amt_vars + cnt_vars + ext_vars]

# Create classification summary
var_classification = {
    'Categorical': categorical_vars,
    'Financial Amounts': amt_vars,
    'Date Variables': date_vars,
    'Flag Variables': flag_vars,
    'Count Variables': cnt_vars,
    'External Scores': ext_vars,
    'Other Numeric': other_numeric
}

print("Variable Classification Summary:")
for var_type, vars_list in var_classification.items():
    print(f"  {var_type}: {len(vars_list)} variables")
    if vars_list:
        examples = ', '.join(vars_list[:3])
        if len(vars_list) > 3:
            examples += f" (and {len(vars_list)-3} more)"
        print(f"    Examples: {examples} \n")

# LINE 157: print(f"    Examples: {examples} \n")

# ============================================================================
# Descriptive Statistics for Numerical Variables
# ============================================================================
print("\nNumerical Variables - Descriptive Statistics:")
print(df[numeric_vars].describe().T)

# Key financial variables summary
print("\nKey Financial Variables Summary:")
for col in ['amt_income_total', 'amt_credit', 'amt_annuity', 'amt_goods_price']:
    if col in df.columns:
        print(f"\n{col}:")
        print(f"  Mean: ${df[col].mean():,.2f}")
        print(f"  Median: ${df[col].median():,.2f}")
        print(f"  Std: ${df[col].std():,.2f}")

# %%
# Duplicates
duplicate_count = df.duplicated().sum()
display(pd.DataFrame({'Duplicate_Rows':[duplicate_count]}))

# %%
# Analyze our target variable - who had payment difficulties vs who didn't
target_counts = df['target'].value_counts().sort_index()
target_pct = df['target'].value_counts(normalize=True).sort_index() * 100

print("target Variable Distribution:")
print(f"  0 (No Payment Difficulties): {target_counts[0]:,} ({target_pct[0]:.1f}%)")
print(f"  1 (Payment Difficulties):    {target_counts[1]:,} ({target_pct[1]:.1f}%)")

# Prepare data for visualization
target_summary = (df.groupby('target')
                 .size()
                 .reset_index(name='count')
                 .assign(
                     percentage=lambda x: round(x['count'] / x['count'].sum() * 100, 1),
                     label=lambda x: x['target'].map({0: 'No Payment Difficulties', 1: 'Payment Difficulties'})
                 ))

# Add formatted labels to dataframe for display on chart
target_summary['label_text'] = (target_summary['count'].astype(str) + 
                               ' (' + target_summary['percentage'].astype(str) + '%)')

# Create a bar chart to visualize the balance of our outcome variable
plot = (ggplot(target_summary, aes(x='factor(target)', y='count', fill='label')) +
        geom_col(width=0.6) +
        # Add labels showing counts and percentages on the bars
        geom_text(aes(label='label_text'), 
                 va='bottom', nudge_y=target_summary['count'].max() * 0.01, 
                 size=11, fontweight='bold') +
        # Use green for good outcomes, red for payment difficulties
        scale_fill_manual(values={'No Payment Difficulties': '#2E8B57', 
                                'Payment Difficulties': '#DC143C'}) +
        labs(title='Distribution of target Variable',
             subtitle='Payment Difficulties vs No Payment Difficulties',
             x='target Value',
             y='Number of Clients',
             fill='Category'))

plot

# %%
# Missing values analysis
missing_data = df.isnull().sum()
missing_pct = (missing_data / len(df)) * 100
missing_summary = (pd.DataFrame({
    'Variable': missing_data.index,
    'Missing_Count': missing_data.values,
    'Missing_Percentage': missing_pct.values
}).query('Missing_Count > 0')
  .sort_values('Missing_Count', ascending=False))

print("Missing Values Summary:")
print(f"Total variables with missing values: {len(missing_summary)}")
print(f"Variables with >50% missing: {sum(missing_summary['Missing_Percentage'] > 50)}")
print(f"Variables with >20% missing: {sum(missing_summary['Missing_Percentage'] > 20)}")
print()

# %%
if len(missing_summary) > 0:
    print("Top 10 variables with most missing values:")
    display(missing_summary.head(10).round(2))
else:
    print("No missing values found in the dataset.")

# %%
full_data_missing_values=df.isnull().sum().sum()
print("Total missing values in full dataset:", full_data_missing_values)

# %%
# Visualize missing values using plotnine (FIXED)
if len(missing_summary) > 0:
    top_missing = missing_summary.head(15)  # Top 15
    
    plot = (ggplot(top_missing, aes(x='reorder(Variable, Missing_Percentage)', y='Missing_Percentage')) +
            geom_col(fill='#DC143C', alpha=0.7) +
            coord_flip() +
            labs(title='Missing Values by Variable',
                 subtitle='Top 15 variables with highest missing percentage',
                 x='Variable',
                 y='Missing Percentage (%)'))
else:
    print("No missing values to visualize.")
plot


# %% [markdown]
# #### Check for negative values in employment

# %%
# Simple outliers for key money fields
key_money = [c for c in ['amt_income_total','amt_credit','amt_annuity'] if c in df.columns]
out_rows = []
for col in key_money:
    s = df[col].dropna()
    if s.empty:
        continue
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5*iqr, q3 + 1.5*iqr
    out_count = int(((s < low) | (s > high)).sum())
    out_pct = round(out_count / len(s) * 100, 2)
    out_rows.append({'column': col, 'outliers': out_count, 'pct': out_pct})
    display(pd.DataFrame(out_rows))

# %% [markdown]
# ### Data Quality Summary
# 
# **Key Findings:**
# 
# 1. **Missing Values**: The dataset shows varying levels of missing data across variables, with some variables having substantial missing rates that may require to drop the variable or impute the missing values.
# 2. **Outliers**: Financial variables show expected outliers in income and credit amounts, which is typical in financial data and may represent legitimate high-value cases.
# 3. **Data Consistency**: Most variables show expected ranges and formats, with date variables properly encoded as days before application.
# 
# **Implications for Analysis:**
# 
# - The data seems to be clean and consistent, with no major issues found. The two variables with missing data  are
# 
# | Variable        | Missing Count | Missing Percentage |
# |:----------------|:--------------|:-------------------|
# | occupation_type | 6290         | 31.45              |
# | ext_source_3    | 3945         | 19.73              |
# 
# 
# - Outliers in financial variables may need treatment depending on analysis goals. For example
# 
# | Variable         | Outliers | Percentage |
# |:-----------------|:---------|:-----------|
# | amt_income_total | 882      | 4.41%      |
# | amt_credit       | 451      | 2.26%      |
# | amt_annuity      | 498      | 2.49%      |

# %% [markdown]
# ## Data Cleaning

# %%
# step 1: data preparation
# create a working copy of the original data to avoid modifying the raw dataset
working_dataframe = df.copy()

# remove any exact duplicate rows (though we found 0 duplicates earlier)
working_dataframe = working_dataframe.drop_duplicates().reset_index(drop=True)

# %%
# find all columns that start with 'days_' first convert them to positive years
days_cols = [c for c in working_dataframe.columns if c.startswith('days_')]
days_cols

# %%
# Check the original days_employed for the unemployment code (365243)
print("Checking original DAYS_EMPLOYED values:")
print(working_dataframe['days_employed'].describe())

# %%
# The unemployment code is 365243 (a specific value, not just any large number)
unemployment_code = 365243
unemployed_count = (working_dataframe['days_employed'] == unemployment_code).sum()
print(f"Unemployed individuals (code 365243): {unemployed_count}")

# %%
# Replace unemployment code with NaN BEFORE conversion
working_dataframe.loc[working_dataframe['days_employed'] == unemployment_code, 'days_employed'] = np.nan

# %%
# step 2: temporal variable transformation
# convert negative days into positive years for better human interpretation
# days_birth: negative days from birth → positive age in years
# days_employed: negative days from employment start → positive employment years

# convert birthdays to age: divide by 365 to get years, round to 1 decimal place
working_dataframe['age_years'] = (-working_dataframe['days_birth'] / 365).round(3)

# convert employment days to years: divide by 365 to get years, round to 1 decimal place
working_dataframe['employment_years'] = (-working_dataframe['days_employed'] / 365).round(3)

# convert registration days to years: divide by 365 to get years, round to 2 decimal places
working_dataframe['registration_years'] = (-working_dataframe['days_registration'] / 365).round(3)

# convert days since identification to years: divide by 365 to get years, round to 2 decimal places
working_dataframe['id_years'] = (-working_dataframe['days_id_publish'] / 365).round(3)

# convert days since last phone change to years: divide by 365 to get years, round to 2 decimal places
working_dataframe['phone_change_years'] = (-working_dataframe['days_last_phone_change'] / 365).round(3)


# %%
# check for anomalies in the new temporal variables
temporal_vars = ['age_years',  'employment_years', 'registration_years',
                    'id_years', 'phone_change_years']
working_dataframe[temporal_vars].describe()

# %%
# see the converted days converted id, phone change, registration
working_dataframe[['days_birth', 'age_years', 'days_employed', 'employment_years', 'days_registration', 'registration_years',
                    'days_id_publish', 'id_years', 'days_last_phone_change', 'phone_change_years']].head()

# %%
# drop original days columns to avoid confusion
working_dataframe = working_dataframe.drop(columns=days_cols)

# %%
# lets check the missing values again after our transformations
missing_data_after = working_dataframe.isnull().sum()
missing_summary_after = (pd.DataFrame({
    'variable': missing_data_after.index,
    'missing_count': missing_data_after.values,
    'missing_percentage': (missing_data_after.values / len(working_dataframe)) * 100
}).query('missing_count > 0')
  .sort_values('missing_count', ascending=False))

missing_summary_after

# %%
# Missing Value Patterns by Target Class (as DataFrame)
missing_analysis = missing_summary_after[missing_summary_after['missing_count'] > 0].sort_values('missing_percentage', ascending=False).reset_index(drop=True)

missing_by_class_list = []

for var in missing_analysis['variable'].head(5):  # Top 5 variables with most missing
    missing_by_target = working_dataframe.groupby('target')[var].apply(lambda x: x.isna().sum())
    total_by_target = working_dataframe['target'].value_counts()
    pct_by_target = (missing_by_target / total_by_target * 100).round(2)
    
    # Calculate difference
    difference = abs(pct_by_target[0] - pct_by_target[1])
    significant = "Yes" if difference > 5 else "No"
    
    missing_by_class_list.append({
        'variable': var,
        'no_default_missing_pct': pct_by_target[0],
        'default_missing_pct': pct_by_target[1],
        'difference_pct': round(difference, 2),
        'significant_diff': significant
    })

# Create DataFrame
missing_by_class_df = pd.DataFrame(missing_by_class_list)

# Optional: Style it for better visualization
missing_by_class_df.style.background_gradient(subset=['difference_pct'], cmap='YlOrRd')

# %%
# step 3: column type identification
# after adding new temporal features (age_years, employment_years), we need to 
# re-identify which columns are categorical vs numeric for proper preprocessing

# identify categorical columns (text-based data like gender, occupation, etc.)
categorical_features = working_dataframe.select_dtypes(include=['object', 'category']).columns.tolist()

# identify numeric columns (numbers like income, credit amount, age, etc.) excluding target
numerical_features = working_dataframe.select_dtypes(include=['number']).columns.tolist()

# remove target variable from feature lists since it's our prediction target, not a feature
target = 'target'
if target in categorical_features:
    categorical_features.remove(target)
if target in numerical_features:
    numerical_features.remove(target)

# display summary of column types for verification
print(f"categorical columns: {len(categorical_features)}")
print(f"numeric columns: {len(numerical_features)} (now includes years variables)")
print(f"sample numeric columns: {numerical_features[:5]}")

print(f"\nsample categorical columns: {categorical_features[:5]}")

# %%
# see if target variable is still present
target in categorical_features

# %%
# we need to lowercase all values in categorical/string columns
for col in categorical_features:
    working_dataframe[col] = working_dataframe[col].str.lower()

# %%
# check a sample of the modified categorical columns
working_dataframe[categorical_features].head()

# %%
# handle missing data
# Missing Value Imputation Strategy
# handle missing data
from sklearn.impute import SimpleImputer

# Track what we're imputing
numeric_with_missing = [col for col in numerical_features if working_dataframe[col].isna().sum() > 0]
print(f"Imputing {len(numeric_with_missing)} numeric variables with median...")

# %%
# Impute
num_imputer = SimpleImputer(strategy='median')
working_dataframe[numerical_features] = num_imputer.fit_transform(working_dataframe[numerical_features])

# Create summary DataFrame
imputation_summary = pd.DataFrame({
    'variable': numeric_with_missing,
    'method': 'median',
    'imputed_value': num_imputer.statistics_[:len(numeric_with_missing)]
}).round(3)

display(imputation_summary)


# %%
# verify no missing values remain in numeric features
working_dataframe[numerical_features].isnull().sum().sum()

# %%
# handle occupation_type - missing likely means "unemployed" or "unknown"
working_dataframe['occupation_type'] = working_dataframe['occupation_type'].fillna('unknown')
working_dataframe['occupation_type'].isnull().sum()

# %%
# handle missing values in categorical columns by imputing
categorical_with_missing = [col for col in categorical_features
                            if working_dataframe[col].isna().sum() > 0]
from sklearn.impute import SimpleImputer
cat_imputer = SimpleImputer(strategy='most_frequent')
working_dataframe[categorical_with_missing] = cat_imputer.fit_transform(working_dataframe[categorical_with_missing])
# verify no missing values remain in categorical features
working_dataframe[categorical_features].isnull().sum().sum()

# %%
# handle the outliers in numeric columns

def detect_outliers_iqr(series, multiplier=1.5):
    """Detect outliers using IQR method"""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    outliers = (series < lower_bound) | (series > upper_bound)
    return outliers, lower_bound, upper_bound

# %%
outlier_report = []
for col in numerical_features:
    if col in working_dataframe.columns:
        series = working_dataframe[col].dropna()
        outliers, lower, upper = detect_outliers_iqr(series)

        outlier_count = outliers.sum()
        outlier_pct = (outlier_count / len(series)) * 100

        outlier_report.append({
            'variable': col,
            'total_values': len(series),
            'outliers_count': outlier_count,
            'outliers_pct': round(outlier_pct, 2),
            'lower_bound': round(lower, 2),
            'upper_bound': round(upper, 2),
            'min_value': round(series.min(), 2),
            'max_value': round(series.max(), 2)
        })

outlier_df = pd.DataFrame(outlier_report).sort_values('outliers_pct', ascending=False)
display(outlier_df[outlier_df['outliers_count'] > 0])

# %%
# identify which variables to treat based on type
financial_vars = [col for col in numerical_features if col.startswith('amt_')]
temporal_vars = [col for col in numerical_features if col in ['age_years', 'employment_years']]

# Variables that should be treated (high outliers matter for model)
vars_to_treat = financial_vars + temporal_vars

print(f"Variables flagged for outlier treatment: {len(vars_to_treat)}")
print(f"Financial: {financial_vars}")
print(f"Temporal: {temporal_vars}")


# %%
# show outlier impact on target
outlier_impact = []
for col in vars_to_treat:
    if col in working_dataframe.columns:
        series = working_dataframe[col].dropna()
        outliers, _, _ = detect_outliers_iqr(series)
        
        # Create temporary outlier indicator
        temp_outlier_col = f'{col}_is_outlier'
        working_dataframe[temp_outlier_col] = False
        working_dataframe.loc[series.index[outliers], temp_outlier_col] = True
        
        # Calculate default rates
        default_rate_outliers = working_dataframe[working_dataframe[temp_outlier_col]]['target'].mean() * 100
        default_rate_normal = working_dataframe[~working_dataframe[temp_outlier_col]]['target'].mean() * 100
        
        outlier_impact.append({
            'variable': col,
            'default_rate_outliers_%': round(default_rate_outliers, 2),
            'default_rate_normal_%': round(default_rate_normal, 2),
            'difference_%': round(default_rate_outliers - default_rate_normal, 2)
        })
        
        # Drop temporary column
        working_dataframe.drop(columns=[temp_outlier_col], inplace=True)

outlier_impact_df = pd.DataFrame(outlier_impact)
display(outlier_impact_df)

# %%
# STEP 4: Apply winsorization ONLY to variables that need it

def winsorize_column(df, column, lower_percentile=0.01, upper_percentile=0.99):
    """Cap outliers at specified percentiles"""
    lower_bound = df[column].quantile(lower_percentile)
    upper_bound = df[column].quantile(upper_percentile)
    
    original_min = df[column].min()
    original_max = df[column].max()
    
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    
    return {
        'variable': column,
        'original_min': round(original_min, 2),
        'original_max': round(original_max, 2),
        'new_min': round(df[column].min(), 2),
        'new_max': round(df[column].max(), 2),
        'capped_at_1st': round(lower_bound, 2),
        'capped_at_99th': round(upper_bound, 2)
    }

print("Applying Winsorization (1st to 99th percentile)")

winsorization_report = []
for col in vars_to_treat:  # Only treat selected variables
    if col in working_dataframe.columns:
        result = winsorize_column(working_dataframe, col, 0.01, 0.99)
        winsorization_report.append(result)

winsor_df = pd.DataFrame(winsorization_report)
display(winsor_df)

print("Outliers treated using winsorization for financial and temporal variables")
print(f"{len(vars_to_treat)} variables processed")
print(f"Other variables left unchanged (valid variation)")

# %%
# after cleaning, check missing values again
final_missing = working_dataframe.isnull().sum().sum()

# %%
# after cleaning, check shape again
working_dataframe.shape


# LINE 605: working_dataframe.shape

# ============================================================================
# Numerical Feature-Target Relationship Analysis
# ============================================================================
print("Numerical Features Correlation with Target:\n")

# Calculate correlations
correlations = []
for col in numerical_features:
    if col in working_dataframe.columns and col != 'target':
        corr = working_dataframe[col].corr(working_dataframe['target'])
        correlations.append({
            'feature': col,
            'correlation': corr,
            'abs_correlation': abs(corr)
        })

corr_df = pd.DataFrame(correlations).sort_values('abs_correlation', ascending=False)
print("Top 15 Numerical Predictors by Absolute Correlation:")
display(corr_df.head(15))

# LINE 610: # %% [markdown]
# LINE 611: # ### Statistical Feature-Target Relationship Analysis
# %% [markdown]
# ### Statistical Feature-Target Relationship Analysis
# This section would include statistical tests and visualizations to analyze the relationship between features and the target variable

# %%
# Identify top predictors as DataFrame
top_predictors_df = corr_df.head(10).copy()

# Add direction column
top_predictors_df['direction'] = top_predictors_df['correlation'].apply(
    lambda x: "increases" if x > 0 else "decreases"
)

# Add interpretation column
top_predictors_df['interpretation'] = top_predictors_df.apply(
    lambda row: f"{row['direction']} default risk", axis=1
)

# Reorder columns for clarity
top_predictors_df = top_predictors_df[['feature', 'correlation', 'abs_correlation', 'direction', 'interpretation']]

display(top_predictors_df)

# %%
# Categorical default rate analysis - single DataFrame
categorical_analysis = []

for col in categorical_features:
    if col in working_dataframe.columns:
        default_rates = working_dataframe.groupby(col)['target'].agg(['mean', 'count'])
        default_rates = default_rates[default_rates['count'] >= 50]

        if len(default_rates) > 0:
            max_default_rate = default_rates['mean'].max() * 100
            min_default_rate = default_rates['mean'].min() * 100
            spread = max_default_rate - min_default_rate

            categorical_analysis.append({
                'feature': col,
                'categories': len(default_rates),
                'min_default_%': round(min_default_rate, 2),
                'max_default_%': round(max_default_rate, 2),
                'spread_%': round(spread, 2)
            })

# Create and sort DataFrame
cat_analysis_df = (pd.DataFrame(categorical_analysis)
                   .sort_values('spread_%', ascending=False)
                   .reset_index(drop=True))

# Add rank column
cat_analysis_df.insert(0, 'rank', range(1, len(cat_analysis_df) + 1))

print("\nCategorical Features - Default Rate Analysis (sorted by spread):")
display(cat_analysis_df)


# LINE 658: display(cat_analysis_df)

# ============================================================================
# Distribution Analysis - Skewness and Kurtosis
# ============================================================================
distribution_analysis = []
for col in ['amt_income_total', 'amt_credit', 'amt_annuity', 'age_years', 'employment_years']:
    if col in working_dataframe.columns:
        skew = working_dataframe[col].skew()
        kurt = working_dataframe[col].kurtosis()
        
        distribution_analysis.append({
            'variable': col,
            'skewness': round(skew, 3),
            'kurtosis': round(kurt, 3),
            'interpretation': 'Right-skewed' if skew > 0.5 else 'Left-skewed' if skew < -0.5 else 'Symmetric'
        })

dist_df = pd.DataFrame(distribution_analysis)
print("\nDistribution Analysis:")
display(dist_df)

# LINE 660: # %% [markdown]
# LINE 661: # ### Multicollinearity Check

# %% [markdown]
# ### Multicollinearity Check

# %%
# Multicollinearity Check (document only, don't drop)

# Calculate correlation matrix
numerical_for_corr = [col for col in numerical_features if col in working_dataframe.columns]
corr_matrix = working_dataframe[numerical_for_corr].corr()

# Find highly correlated pairs
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr_value = corr_matrix.iloc[i, j]
        if abs(corr_value) > 0.8:
            high_corr_pairs.append({
                'feature_1': corr_matrix.columns[i],
                'feature_2': corr_matrix.columns[j],
                'correlation': round(corr_value, 3)
            })

if high_corr_pairs:
    high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('correlation', ascending=False, key=abs)
    print(f"Highly Correlated Feature Pairs (|r| > 0.8):")
    display(high_corr_df)
    print("Features documented but NOT dropped - will be evaluated in Part 3")
else:
    print("No highly correlated pairs found (|r| > 0.8)")

print(f"Final Part 1 Dataset:")
print(f"Shape: {working_dataframe.shape}")
print(f"Features retained: {working_dataframe.shape[1] - 1}")

# LINE 694: print(f"Features retained: {working_dataframe.shape[1] - 1}")

# ============================================================================
# Correlation Heatmap (Preview for Part 2 Visualization)
# ============================================================================
import seaborn as sns
import matplotlib.pyplot as plt

# Select top correlated features for visualization
top_features = corr_df.head(15)['feature'].tolist() + ['target']
corr_subset = working_dataframe[top_features].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_subset, annot=True, cmap='coolwarm', center=0, 
            fmt='.2f', square=True, linewidths=1)
plt.title('Correlation Heatmap - Top 15 Features + Target')
plt.tight_layout()
plt.savefig(ARTIFACT_DIR / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Correlation heatmap saved to artifacts/correlation_heatmap.png")
plt.show()

# LINE 696: # %%
# LINE 697: # create a summary of categorical variables

# %%
# create a summary of categorical variables showing the most frequent values for each column.
def categorical_summary(dframe: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:

    # identify all categorical columns (text-based data like gender, occupation, etc.)
    categorical_columns = dframe.select_dtypes(include=["object", "category"])
    
    rows = []
    
    # step 3: loop through each categorical column to analyze its values
    for column in categorical_columns.columns:
        vc = categorical_columns[column].value_counts(dropna=False).head(top_n)
        
        # for each value-count pair, create a dictionary with column name, value, and count
        for k, v in vc.items():
            rows.append({
                "column": column,    # which categorical variable this is
                "value": k,         # the actual category value (e.g., "male", "manager")
                "count": int(v)     # how many times this value appears (convert to regular int)
            })
    
    # step 5: convert list of dictionaries to DataFrame for easy viewing and analysis
    return pd.DataFrame(rows)

# apply the function to analyze categorical variables in our working dataset
# top_n=8 means we'll see the 8 most frequent values for each categorical column
categorical_cols_summary = categorical_summary(working_dataframe, top_n=8)

# display the results - this shows the most common categories across all categorical variables
categorical_cols_summary


# %%
RARE_THRESHOLD = 0.02
HIGH_CARDINALITY_VALUE = 8
TARGET = 'target'

# %%
def categorical_unique_counts(df_to_check: pd.DataFrame, top_n: int = None) -> pd.DataFrame:
    categorical_cols = df_to_check.select_dtypes(include=['object', 'category']).columns.tolist()
    counts_df = pd.DataFrame({
        "column": categorical_cols,
        "unique_count": [df_to_check[col].nunique(dropna=False) for col in categorical_cols]
    }).sort_values("unique_count", ascending=False).reset_index(drop=True)
    if top_n is not None:
        counts_df = counts_df.head(top_n)
    return counts_df

# %%
# count unique values for each categorical column
# create and sort the DataFrame by unique_count descending
cat_unique_counts_df = categorical_unique_counts(working_dataframe)

cat_unique_counts_df

# %%
# step 4: identify high-cardinality categories
# find categorical columns that have too many unique values (> 8)
# these need special treatment to avoid creating too many dummy variables
# high cardinality = many unique categories (e.g., 50+ job types)
# low cardinality = few unique categories (e.g., male/female)

high_card_cols = cat_unique_counts_df[
    cat_unique_counts_df['unique_count'] > HIGH_CARDINALITY_VALUE
]['column'].tolist()
print(f"High-cardinality categorical columns (> {HIGH_CARDINALITY_VALUE} unique values):")
print(high_card_cols)

# %%
# step 4b: identify low-cardinality categories  
# find categorical columns with few unique values (<= 8)
# these can safely be one-hot encoded without creating too many features
# examples: gender (m/f), car ownership (y/n), education level (5 types)

low_card_cols = cat_unique_counts_df[cat_unique_counts_df['unique_count'] <= HIGH_CARDINALITY_VALUE]['column'].tolist()
print(f"Low-cardinality categorical columns (<= {HIGH_CARDINALITY_VALUE} unique values):")
print(low_card_cols)

# %%
# step 5: rare category binning
# for high-cardinality columns only: combine rare categories into "other"
# this prevents the model from trying to learn patterns from categories with very few examples
# rare_threshold = 0.02 means any category appearing in < 2% of data becomes "other"
# example: if only 50 out of 20,000 people are "astronauts", group them as "other"

for col in high_card_cols:
    # Calculate frequency of each category as proportion of total data
    freq = working_dataframe[col].value_counts(normalize=True, dropna=False)
    
    # Find categories that appear less than 2% of the time
    rare_values = freq[freq < RARE_THRESHOLD].index
    
    # Replace rare values with "Other" to reduce cardinality while preserving information
    working_dataframe[col] = working_dataframe[col].where(~working_dataframe[col].isin(rare_values), "Other")

# %%
# CHECK: Show updated unique counts after binning
updated_cat_unique_counts_df = categorical_unique_counts(working_dataframe)
updated_cat_unique_counts_df

# %%
# After binning, show comparison
print("Cardinality Reduction Summary:")

comparison_df = pd.merge(
    cat_unique_counts_df, 
    updated_cat_unique_counts_df, 
    on='column', 
    suffixes=('_before', '_after')
)

comparison_df['reduction'] = comparison_df['unique_count_before'] - comparison_df['unique_count_after']
comparison_df['reduction_%'] = (comparison_df['reduction'] / comparison_df['unique_count_before'] * 100).round(1)

display(comparison_df)

# Show how many rare categories were binned
print("Rare Category Binning Results:")
for col in high_card_cols:
    before = cat_unique_counts_df[cat_unique_counts_df['column'] == col]['unique_count'].values[0]
    after = updated_cat_unique_counts_df[updated_cat_unique_counts_df['column'] == col]['unique_count'].values[0]
    reduced = before - after
    print(f"{col}: {before} → {after} categories (binned {reduced} rare values)")

# Save for Part 2
cardinality_info = {
    'high_cardinality_cols': high_card_cols,
    'low_cardinality_cols': low_card_cols,
    'rare_threshold': RARE_THRESHOLD,
    'high_cardinality_threshold': HIGH_CARDINALITY_VALUE
}

import json
with open(ARTIFACT_DIR / 'cardinality_info.json', 'w') as f:
    json.dump(cardinality_info, f, indent=2)

print(f"Cardinality info saved to artifacts/cardinality_info.json")

# %% [markdown]
# ### Save the cleaned dataframe for Part 2

# %%
output_path = OUT_DIR / "cleaned_data_part1.csv"
working_dataframe.to_csv(output_path, index=False)
print(f"Cleaned dataset saved to: {output_path}")

# %%
# Save metadata
metadata = {
    'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'original_shape': (20000, 68),
    'cleaned_shape': working_dataframe.shape,
    'missing_values': int(working_dataframe.isnull().sum().sum()),
    'numerical_features': numerical_features,
    'categorical_features': categorical_features,
    'transformations_applied': [
        'Temporal variables converted to years',
        'Missing values imputed (median for numeric, mode for categorical)',
        'Outliers winsorized (1st-99th percentile)',
        'Rare categories binned (2% threshold)',
        'Employment anomalies fixed',
        'Strings lowercased'
    ]
}

# %%
metadata_path = ARTIFACT_DIR / "part1_metadata.json"
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"Metadata saved to: {metadata_path}")

# %%
# Save feature lists
feature_lists = {
    'numerical_features': numerical_features,
    'categorical_features': categorical_features,
    'target': 'target'
}
feature_lists_path = ARTIFACT_DIR / "part1_feature_lists.json"
with open(feature_lists_path, 'w') as f:
    json.dump(feature_lists, f, indent=2)
print(f"Feature lists saved to: {feature_lists_path}")


