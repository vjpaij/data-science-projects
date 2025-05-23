import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Import functions from ab_test_analysis.py
# In a real project, you might structure this as a package,
# but for demonstration, we'll assume ab_test_analysis.py is in the same directory.

# Re-define functions from ab_test_analysis.py or import them.
# For simplicity in a single Streamlit app file, we'll include them here.
# In a larger project, you would put these in a separate utility file and import.

def generate_ab_test_data(
    n_control=5000,
    n_treatment=5000,
    control_conversion_rate=0.10,
    treatment_conversion_rate=0.11, # Slightly higher for treatment
    random_seed=42
):
    """
    Generates a synthetic dataset for A/B testing.
    """
    np.random.seed(random_seed)
    control_users = pd.DataFrame({
        'user_id': range(1, n_control + 1),
        'group': 'Control',
        'conversion': np.random.choice([0, 1], size=n_control, p=[1 - control_conversion_rate, control_conversion_rate])
    })
    treatment_users = pd.DataFrame({
        'user_id': range(n_control + 1, n_control + n_treatment + 1),
        'group': 'Treatment',
        'conversion': np.random.choice([0, 1], size=n_treatment, p=[1 - treatment_conversion_rate, treatment_conversion_rate])
    })
    df = pd.concat([control_users, treatment_users], ignore_index=True)
    start_date = pd.to_datetime('2024-05-01')
    end_date = pd.to_datetime('2024-05-15')
    df['timestamp'] = pd.to_datetime(np.random.uniform(start_date.value, end_date.value, len(df)))
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    return df

def clean_data(df):
    """
    Performs data cleaning and preprocessing for the A/B test dataset.
    """
    if df is None:
        st.error("No DataFrame provided for cleaning.")
        return None

    # Check for missing values
    # st.write("Missing values before cleaning:")
    # st.dataframe(df.isnull().sum().to_frame().T) # Display as horizontal table

    initial_rows = len(df)
    df.drop_duplicates(subset='user_id', inplace=True)
    if len(df) < initial_rows:
        st.warning(f"Removed {initial_rows - len(df)} duplicate user_ids.")
    
    df['group'] = df['group'].astype('category')
    
    unique_groups = df['group'].unique()
    expected_groups = ['Control', 'Treatment']
    if not all(group in unique_groups for group in expected_groups) or len(unique_groups) != len(expected_groups):
        st.warning(f"Unexpected groups found. Expected {expected_groups}, found {unique_groups}. Filtering data.")
        df = df[df['group'].isin(expected_groups)]
    
    if not pd.api.types.is_integer_dtype(df['conversion']):
        df['conversion'] = df['conversion'].astype(int)
    
    return df

def perform_sanity_checks(df):
    """
    Performs sanity checks on the A/B test data, including Sample Ratio Mismatch (SRM).
    """
    if df is None:
        st.error("No DataFrame provided for sanity checks.")
        return None, None, None

    group_counts = df['group'].value_counts()
    group_proportions = df['group'].value_counts(normalize=True)

    total_users = len(df)
    expected_control = total_users / 2
    expected_treatment = total_users / 2

    obs_control = group_counts.get('Control', 0)
    obs_treatment = group_counts.get('Treatment', 0)

    chi2_stat, p_value_srm = stats.chisquare(f_obs=[obs_control, obs_treatment], 
                                             f_exp=[expected_control, expected_treatment])

    alpha_srm = 0.05 
    srm_detected = p_value_srm < alpha_srm

    return srm_detected, p_value_srm, group_proportions

def perform_ab_test(df, alpha=0.05):
    """
    Performs the A/B test using a Z-test for proportions.
    """
    if df is None:
        st.error("No DataFrame provided for A/B test.")
        return {}

    control_group = df[df['group'] == 'Control']
    treatment_group = df[df['group'] == 'Treatment']

    conversions_control = control_group['conversion'].sum()
    n_control = len(control_group)
    conversion_rate_control = conversions_control / n_control if n_control > 0 else 0

    conversions_treatment = treatment_group['conversion'].sum()
    n_treatment = len(treatment_group)
    conversion_rate_treatment = conversions_treatment / n_treatment if n_treatment > 0 else 0

    z_stat, p_value = proportions_ztest(
        count=[conversions_treatment, conversions_control],
        nobs=[n_treatment, n_control],
        alternative='two-sided'
    )
    
    se_diff = np.sqrt(
        (conversion_rate_control * (1 - conversion_rate_control) / n_control) +
        (conversion_rate_treatment * (1 - conversion_rate_treatment) / n_treatment)
    )
    
    z_score_ci = stats.norm.ppf(1 - alpha / 2)

    diff_conversion_rates = conversion_rate_treatment - conversion_rate_control

    ci_lower = diff_conversion_rates - z_score_ci * se_diff
    ci_upper = diff_conversion_rates + z_score_ci * se_diff

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

# --- Streamlit App Layout ---

st.set_page_config(layout="wide", page_title="A/B Testing Framework")

st.title("📊 A/B Testing Framework for Conversion Rate Optimization")
st.markdown("""
This application provides a framework to analyze A/B test results and determine if a new variation
(Treatment group) has a statistically significant impact on conversion rates compared to the original
(Control group).
""")

st.sidebar.header("Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
use_synthetic_data = st.sidebar.checkbox("Use Synthetic Data", value=True)

df_raw = None
if use_synthetic_data:
    st.sidebar.info("Using pre-generated synthetic data for demonstration.")
    df_raw = generate_ab_test_data()
elif uploaded_file is not None:
    st.sidebar.success("CSV file uploaded successfully!")
    df_raw = pd.read_csv(uploaded_file)
else:
    st.info("Please upload a CSV file or choose to use synthetic data.")

if df_raw is not None:
    st.header("1. Data Overview")
    st.write("First 5 rows of the raw data:")
    st.dataframe(df_raw.head())
    st.write(f"Dataset shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
    
    st.subheader("Data Information:")
    buffer = io.StringIO()
    df_raw.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    st.header("2. Data Cleaning & Preprocessing")
    df_cleaned = clean_data(df_raw.copy())
    if df_cleaned is not None:
        st.success("Data cleaning complete!")
        st.write("First 5 rows of the cleaned data:")
        st.dataframe(df_cleaned.head())
        st.write(f"Cleaned dataset shape: {df_cleaned.shape[0]} rows, {df_cleaned.shape[1]} columns")

        st.header("3. Initial Observations & Sanity Checks")
        srm_detected, p_value_srm, group_proportions = perform_sanity_checks(df_cleaned)
        
        st.subheader("Group Distribution:")
        st.dataframe(group_proportions.to_frame(name='Proportion'))

        st.subheader("Sample Ratio Mismatch (SRM) Check:")
        if srm_detected:
            st.error(f"🔴 SRM Detected! P-value: {p_value_srm:.4f}. This indicates a potential issue with traffic allocation or data collection. Proceed with caution.")
        else:
            st.success(f"🟢 No significant SRM detected. P-value: {p_value_srm:.4f}. Group distribution appears as expected.")

        st.header("4. A/B Test Analysis")
        ab_test_results = perform_ab_test(df_cleaned)

        if ab_test_results:
            st.subheader("Conversion Rates:")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Control Group Conversion Rate", value=f"{ab_test_results['control_rate']:.2%}")
                st.write(f"({ab_test_results['control_conversions']} conversions / {ab_test_results['control_n']} users)")
            with col2:
                st.metric(label="Treatment Group Conversion Rate", value=f"{ab_test_results['treatment_rate']:.2%}")
                st.write(f"({ab_test_results['treatment_conversions']} conversions / {ab_test_results['treatment_n']} users)")

            st.subheader("Statistical Test Results (Z-test for Proportions):")
            st.write(f"**Z-statistic:** {ab_test_results['z_statistic']:.4f}")
            st.write(f"**P-value:** {ab_test_results['p_value']:.4f}")
            st.write(f"**Observed Difference (Treatment - Control):** {ab_test_results['diff_rate']:.4f} ({ab_test_results['diff_rate']:.2%})")
            st.write(f"**95% Confidence Interval for Difference:** [{ab_test_results['ci_lower']:.4f}, {ab_test_results['ci_upper']:.4f}]")

            st.subheader("Conclusion:")
            st.write(ab_test_results['conclusion'])

            st.header("5. Visualization of Results")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(x='group', y='conversion', data=df_cleaned, errorbar=('ci', 95), ax=ax)
            ax.set_title('Conversion Rate by Group (with 95% CI)', fontsize=16)
            ax.set_xlabel('Group', fontsize=12)
            ax.set_ylabel('Conversion Rate', fontsize=12)
            ax.set_ylim(0, df_cleaned['conversion'].mean() * 1.5)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            st.pyplot(fig)

            st.header("6. Recommendations")
            if ab_test_results['p_value'] < 0.05 and ab_test_results['diff_rate'] > 0:
                st.success("🎉 **Recommendation:** Launch the new Treatment design! It shows a statistically significant uplift in conversion rate.")
                st.write(f"The estimated uplift is between {ab_test_results['ci_lower']:.2%} and {ab_test_results['ci_upper']:.2%}.")
            elif ab_test_results['p_value'] < 0.05 and ab_test_results['diff_rate'] < 0:
                st.error("🛑 **Recommendation:** Do NOT launch the new Treatment design. It shows a statistically significant *decrease* in conversion rate.")
                st.write("Investigate why the new design performed worse than the control.")
            else:
                st.info("ℹ️ **Recommendation:** The A/B test did not show a statistically significant difference between the groups.")
                st.write("Consider:")
                st.write("- Running the test for a longer duration if sample size was small.")
                st.write("- Iterating on the Treatment design and re-testing.")
                st.write("- The observed difference might be due to random chance.")
        else:
            st.error("A/B test could not be performed. Please check data.")