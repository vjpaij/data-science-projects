import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Initial Setup and Data Generation ---

def generate_ab_test_data(
    n_control=5000,
    n_treatment=5000,
    control_conversion_rate=0.10,
    treatment_conversion_rate=0.11, # Slightly higher for treatment
    random_seed=42
):
    """
    Generates a synthetic dataset for A/B testing.

    Args:
        n_control (int): Number of users in the control group.
        n_treatment (int): Number of users in the treatment group.
        control_conversion_rate (float): Baseline conversion rate for the control group.
        treatment_conversion_rate (float): Conversion rate for the treatment group.
        random_seed (int): Seed for reproducibility.

    Returns:
        pd.DataFrame: A DataFrame containing the simulated A/B test data.
    """
    np.random.seed(random_seed)

    # Generate data for Control Group
    control_users = pd.DataFrame({
        'user_id': range(1, n_control + 1),
        'group': 'Control',
        'conversion': np.random.choice([0, 1], size=n_control, p=[1 - control_conversion_rate, control_conversion_rate])
    })

    # Generate data for Treatment Group
    treatment_users = pd.DataFrame({
        'user_id': range(n_control + 1, n_control + n_treatment + 1),
        'group': 'Treatment',
        'conversion': np.random.choice([0, 1], size=n_treatment, p=[1 - treatment_conversion_rate, treatment_conversion_rate])
    })

    # Concatenate the dataframes
    df = pd.concat([control_users, treatment_users], ignore_index=True)

    # Add a timestamp column (optional, but good for real-world context)
    start_date = pd.to_datetime('2024-05-01')
    end_date = pd.to_datetime('2024-05-15')
    df['timestamp'] = pd.to_datetime(np.random.uniform(start_date.value, end_date.value, len(df)))

    # Shuffle the DataFrame to mix control and treatment users
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # Save the data to a CSV file
    csv_path = 'ab_test_data.csv'
    df.to_csv(csv_path, index=False)
    print(f"Synthetic A/B test data generated and saved to '{csv_path}'")
    return df


# --- Data Loading ---

def load_data(file_path='ab_test_data.csv'):
    """
    Loads the A/B test data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame, or None if file not found.
    """
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found. Please ensure data is generated or available.")
        return None
    
    df = pd.read_csv(file_path)
    print(f"\nData loaded successfully from '{file_path}'. Shape: {df.shape}")
    return df


# --- Data Cleaning and Preprocessing ---

def clean_data(df):
    """
    Performs data cleaning and preprocessing for the A/B test dataset.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    if df is None:
        print("No DataFrame provided for cleaning.")
        return None

    print("\n--- Data Cleaning and Preprocessing ---")

    # Check for missing values
    print("\nMissing values before cleaning:")
    print(df.isnull().sum())
    # In this synthetic dataset, we expect no missing values.
    # For real data, you might use df.dropna() or df.fillna()

    # Check for duplicate user_ids
    initial_rows = len(df)
    df.drop_duplicates(subset='user_id', inplace=True)
    if len(df) < initial_rows:
        print(f"\nRemoved {initial_rows - len(df)} duplicate user_ids.")
    else:
        print("\nNo duplicate user_ids found.")

    # Convert 'group' to categorical type for efficiency and clarity
    df['group'] = df['group'].astype('category')
    print("\n'group' column converted to categorical type.")

    # Sanity check: Ensure only 'Control' and 'Treatment' groups exist
    unique_groups = df['group'].unique()
    expected_groups = ['Control', 'Treatment']
    if not all(group in unique_groups for group in expected_groups) or len(unique_groups) != len(expected_groups):
        print(f"\nWarning: Unexpected groups found. Expected {expected_groups}, found {unique_groups}")
        # Filter out unexpected groups if necessary, or raise an error
        df = df[df['group'].isin(expected_groups)]
        print(f"Filtered data to include only {expected_groups} groups.")
    else:
        print(f"\nGroups confirmed: {unique_groups}")

    # Ensure 'conversion' is an integer type (0 or 1)
    if not pd.api.types.is_integer_dtype(df['conversion']):
        df['conversion'] = df['conversion'].astype(int)
        print("\n'conversion' column converted to integer type.")
    
    print("\nData cleaning complete. Cleaned data info:")
    df.info()
    return df


# --- Initial Observations and Sanity Checks ---

def perform_sanity_checks(df):
    """
    Performs sanity checks on the A/B test data, including Sample Ratio Mismatch (SRM).

    Args:
        df (pd.DataFrame): The cleaned DataFrame.

    Returns:
        None
    """
    if df is None:
        print("No DataFrame provided for sanity checks.")
        return

    print("\n--- Initial Observations and Sanity Checks ---")

    # Check group distribution
    group_counts = df['group'].value_counts()
    group_proportions = df['group'].value_counts(normalize=True)

    print("\nGroup Distribution:")
    print(group_counts)
    print("\nGroup Proportions:")
    print(group_proportions)

    # Sample Ratio Mismatch (SRM) Check
    # Null Hypothesis (H0): The observed group proportions are equal to the expected proportions (e.g., 50/50).
    # Alternative Hypothesis (H1): The observed group proportions are significantly different from the expected.

    # For a 50/50 split, expected counts would be total_users / 2 for each group.
    total_users = len(df)
    expected_control = total_users / 2
    expected_treatment = total_users / 2

    # Observed counts
    obs_control = group_counts.get('Control', 0)
    obs_treatment = group_counts.get('Treatment', 0)

    # Perform Chi-squared test for goodness of fit
    # We are testing if the observed frequencies match the expected frequencies.
    # degrees of freedom = number of categories - 1 = 2 - 1 = 1
    chi2_stat, p_value_srm = stats.chisquare(f_obs=[obs_control, obs_treatment], 
                                             f_exp=[expected_control, expected_treatment])

    alpha_srm = 0.05 # Significance level for SRM check

    print(f"\nSRM Check (Chi-squared test):")
    print(f"Observed Control: {obs_control}, Observed Treatment: {obs_treatment}")
    print(f"Expected Control: {expected_control}, Expected Treatment: {expected_treatment}")
    print(f"Chi-squared Statistic: {chi2_stat:.4f}")
    print(f"P-value for SRM: {p_value_srm:.4f}")

    if p_value_srm < alpha_srm:
        print(f"Conclusion: P-value ({p_value_srm:.4f}) < Alpha ({alpha_srm}).")
        print("There is a statistically significant Sample Ratio Mismatch (SRM).")
        print("This indicates a potential issue with traffic allocation or data collection.")
        print("Proceed with caution, as results might be biased. Investigate the cause of SRM.")
    else:
        print(f"Conclusion: P-value ({p_value_srm:.4f}) >= Alpha ({alpha_srm}).")
        print("No statistically significant Sample Ratio Mismatch (SRM) detected.")
        print("The group distribution appears to be as expected.")

    # Check conversion rates per group (initial glance)
    conversion_rates = df.groupby('group')['conversion'].mean()
    print("\nInitial Conversion Rates per Group:")
    print(conversion_rates)


# --- Exploratory Data Analysis (EDA) and Visualization ---

def perform_eda_and_visualize(df):
    """
    Performs EDA and generates visualizations for the A/B test data.

    Args:
        df (pd.DataFrame): The cleaned DataFrame.

    Returns:
        None
    """
    if df is None:
        print("No DataFrame provided for EDA.")
        return

    print("\n--- Exploratory Data Analysis (EDA) ---")

    # Summary statistics for conversion
    conversion_summary = df.groupby('group')['conversion'].agg(['count', 'sum', 'mean']).rename(
        columns={'count': 'total_users', 'sum': 'conversions', 'mean': 'conversion_rate'}
    )
    print("\nConversion Summary per Group:")
    print(conversion_summary)

    # Visualization: Bar plot of Conversion Rates
    plt.figure(figsize=(8, 6))
    sns.barplot(x='group', y='conversion', data=df, errorbar=('ci', 95)) # errorbar shows 95% confidence interval
    plt.title('Conversion Rate by Group (with 95% CI)', fontsize=16)
    plt.xlabel('Group', fontsize=12)
    plt.ylabel('Conversion Rate', fontsize=12)
    plt.ylim(0, df['conversion'].mean() * 1.5) # Set y-limit for better visualization
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Optional: Distribution of timestamps (if relevant for time-based effects)
    # This can show if traffic was consistent over time or if there were spikes/drops.
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='timestamp', hue='group', kde=True, bins=20)
    plt.title('Distribution of User Timestamps by Group', fontsize=16)
    plt.xlabel('Timestamp', fontsize=12)
    plt.ylabel('Number of Users', fontsize=12)
    plt.tight_layout()
    plt.show()

    print("\nEDA and visualizations complete.")


# --- Step 5: Hypothesis Formulation and Statistical Test ---

def perform_ab_test(df, alpha=0.05):
    """
    Performs the A/B test using a Z-test for proportions.

    Args:
        df (pd.DataFrame): The cleaned DataFrame.
        alpha (float): The significance level for the test (e.g., 0.05).

    Returns:
        dict: A dictionary containing test results (z_stat, p_value, ci_lower, ci_upper, conclusion).
    """
    if df is None:
        print("No DataFrame provided for A/B test.")
        return {}

    print("\n--- Hypothesis Formulation and Statistical Test ---")

    # Define groups
    control_group = df[df['group'] == 'Control']
    treatment_group = df[df['group'] == 'Treatment']

    # Calculate conversions and total users for each group
    conversions_control = control_group['conversion'].sum()
    n_control = len(control_group)
    conversion_rate_control = conversions_control / n_control if n_control > 0 else 0

    conversions_treatment = treatment_group['conversion'].sum()
    n_treatment = len(treatment_group)
    conversion_rate_treatment = conversions_treatment / n_treatment if n_treatment > 0 else 0

    print(f"\nControl Group: {n_control} users, {conversions_control} conversions, Rate: {conversion_rate_control:.4f}")
    print(f"Treatment Group: {n_treatment} users, {conversions_treatment} conversions, Rate: {conversion_rate_treatment:.4f}")

    # Null Hypothesis (H0): p_control = p_treatment (No difference in conversion rates)
    # Alternative Hypothesis (H1): p_control != p_treatment (There is a difference)

    # Perform Z-test for two proportions
    # `count`: array-like, number of successes in each group
    # `nobs`: array-like, number of trials (observations) in each group
    z_stat, p_value = proportions_ztest(
        count=[conversions_treatment, conversions_control],
        nobs=[n_treatment, n_control],
        alternative='two-sided' # We are interested if treatment is better OR worse
    )

    # Calculate confidence interval for the difference in proportions
    # This requires calculating the standard error of the difference
    # and then using the Z-score for the desired confidence level.
    # A common way to calculate the CI for difference in proportions is:
    # (p1 - p2) +/- Z * sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
    # For a 95% CI, Z-score is approx 1.96
    
    # Pooled proportion (used for Z-test, but not directly for CI of difference)
    # p_pooled = (conversions_control + conversions_treatment) / (n_control + n_treatment)
    
    # Standard error of the difference in proportions (for CI)
    se_diff = np.sqrt(
        (conversion_rate_control * (1 - conversion_rate_control) / n_control) +
        (conversion_rate_treatment * (1 - conversion_rate_treatment) / n_treatment)
    )
    
    # Z-score for 95% confidence interval
    z_score_ci = stats.norm.ppf(1 - alpha / 2) # For alpha = 0.05, this is 1.96

    # Difference in rates
    diff_conversion_rates = conversion_rate_treatment - conversion_rate_control

    # Confidence Interval
    ci_lower = diff_conversion_rates - z_score_ci * se_diff
    ci_upper = diff_conversion_rates + z_score_ci * se_diff

    print(f"\nZ-statistic: {z_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Significance Level (alpha): {alpha}")
    print(f"Observed Difference (Treatment - Control): {diff_conversion_rates:.4f}")
    print(f"95% Confidence Interval for Difference: [{ci_lower:.4f}, {ci_upper:.4f}]")

    # Conclusion based on p-value
    conclusion = ""
    if p_value < alpha:
        conclusion = f"Reject the Null Hypothesis. P-value ({p_value:.4f}) < Alpha ({alpha}).\n" \
                     "There is a statistically significant difference in conversion rates between the Treatment and Control groups."
        if diff_conversion_rates > 0:
            conclusion += "\nThe Treatment group performed significantly better."
        else:
            conclusion += "\nThe Treatment group performed significantly worse."
    else:
        conclusion = f"Fail to Reject the Null Hypothesis. P-value ({p_value:.4f}) >= Alpha ({alpha}).\n" \
                     "There is no statistically significant difference in conversion rates between the Treatment and Control groups."

    print(f"\nConclusion:\n{conclusion}")

    results = {
        'control_conversions': conversions_control,
        'control_n': n_control,
        'control_rate': conversion_rate_control,
        'treatment_conversions': conversions_treatment,
        'treatment_n': n_treatment,
        'treatment_rate': conversion_rate_treatment,
        'z_statistic': z_stat,
        'p_value': p_value,
        'diff_rate': diff_conversion_rates,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'conclusion': conclusion
    }
    return results


# --- Drawing Insights ---
"""
Based on the p_value and confidence interval, we will formulate a recommendation.

If p<α and Treatment Rate > Control Rate: Recommend launching the treatment. Quantify the expected uplift using the confidence interval.
If p<α and Treatment Rate < Control Rate: Recommend not launching the treatment and investigating why it performed worse.
If p≥α: Recommend no significant difference. Suggest further testing, iterating on the design, or considering other factors.

"""


# Call the function to generate data when the script is run directly
if __name__ == "__main__":
    df = generate_ab_test_data()
    df = load_data()
    if df is not None:
        df_cleaned = clean_data(df.copy()) # Use a copy to avoid modifying original df for subsequent runs
        if df_cleaned is not None:
            perform_sanity_checks(df_cleaned)
            perform_eda_and_visualize(df_cleaned)
            ab_test_results = perform_ab_test(df_cleaned)
            print("\nFull A/B Test Results Dictionary:")
            for k, v in ab_test_results.items():
                if isinstance(v, float):
                    print(f"{k}: {v:.4f}")
                else:
                    print(f"{k}: {v}")
