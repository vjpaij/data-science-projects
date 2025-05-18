"""
Explanation of the Complete Code:

Import Libraries: Imports all necessary libraries for data manipulation, preprocessing, AutoML, and Streamlit.
generate_synthetic_data():
Generates a synthetic dataset with numerical and categorical features for binary classification.
Uses make_classification from scikit-learn to create the base data.
Adds a few categorical features.
Returns a Pandas DataFrame.
preprocess_data():
Splits the data into features (X) and target variable (y) for both training and testing sets.
Identifies numerical and categorical features.
Creates separate pipelines for numerical and categorical feature preprocessing:
Numerical: Imputes missing values with the mean and scales the features using StandardScaler.
Categorical: Imputes missing values with the most frequent value and performs one-hot encoding using OneHotEncoder.
Uses ColumnTransformer to apply the appropriate pipeline to the corresponding features.
Fits the preprocessing steps on the training data and transforms both the training and testing data.  Critically, the same fitted transformer is used on the test data to prevent data leakage.
Returns the preprocessed training and testing features (X_train, X_test) and the training and testing target variables (y_train, y_test).
train_automl_model():
Initializes an AutoSklearnClassifier with a specified time limit.  The time limit is a crucial parameter.  A longer time limit will generally lead to better performance, but will also take longer to run.
Fits the Auto-sklearn model to the preprocessed training data.  Auto-sklearn will automatically search for the best model and hyperparameters within the given time limit.
Returns the trained Auto-sklearn model.
evaluate_model():
Takes the trained model and the test data as input.
Makes predictions on the test set using the model.
Calculates and prints the following evaluation metrics: accuracy, precision, recall, F1-score, and ROC AUC.
Returns a dictionary containing the evaluation metrics.
create_streamlit_app():
Creates a Streamlit web application.
Displays a title and instructions.
Creates input fields for each feature (numerical and categorical) using st.number_input() and st.selectbox().
When the "Predict" button is clicked:
Creates a Pandas DataFrame from the user-provided input values.
Preprocesses the input data using the same ColumnTransformer that was fitted on the training data. This is essential to ensure consistency in the data format.
Makes a prediction using the trained AutoML model.
Displays the prediction result (0 or 1) using st.write().
Displays the probability of the positive class.
if __name__ == '__main__'::
This block of code is executed when the script is run directly.
It performs the following steps:
Loads the data using generate_synthetic_data().
Preprocesses the data using preprocess_data().
Trains the AutoML model using train_automl_model().
Evaluates the trained model using evaluate_model().
Creates and runs the Streamlit app using create_streamlit_app().

"""


import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import autosklearn.classification
import sklearn.metrics
import streamlit as st

def generate_synthetic_data(n_samples=1000, n_features=20):
    """
    Generates a synthetic dataset for binary classification.

    Args:
        n_samples: The number of samples to generate.
        n_features: The number of features to generate.

    Returns:
        A Pandas DataFrame containing the synthetic dataset.
    """
    # Generate the synthetic dataset
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=15,  # Number of informative features
        n_redundant=5,    # Number of redundant features
        random_state=42,
        n_classes=2,
        weights=[0.8, 0.2], #Imbalanced classes
        flip_y=0.05 # Add some noise to the labels
    )

    # Convert to Pandas DataFrame
    df = pd.DataFrame(X, columns=[f'Feature_{i}' for i in range(1, n_features + 1)])
    df['Target'] = y

    # Add categorical features
    df['Feature_11'] = np.random.choice(['A', 'B', 'C'], size=n_samples)
    df['Feature_12'] = np.random.choice(['X', 'Y', 'Z', 'W'], size=n_samples)
    df['Feature_13'] = np.random.choice(['P', 'Q'], size=n_samples)

    return df

def preprocess_data(train_data, test_data):
    """
    Preprocesses the training and testing data.

    Args:
        train_data: Pandas DataFrame containing the training data.
        test_data: Pandas DataFrame containing the testing data.

    Returns:
        Tuple of preprocessed training and testing data (X_train, X_test, y_train, y_test).
    """
    # Separate features and target variable
    X_train = train_data.drop('Target', axis=1)
    y_train = train_data['Target']
    X_test = test_data.drop('Target', axis=1)
    y_test = test_data['Target']

    # Identify numerical and categorical features
    numerical_features = X_train.select_dtypes(include=np.number).columns
    categorical_features = X_train.select_dtypes(include='object').columns

    # Create transformers for preprocessing
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Use ColumnTransformer to apply transformers to the correct columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Fit and transform the training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_train_processed = pd.DataFrame(X_train_processed) # Convert back to dataframe
    
    # Transform the testing data using the same preprocessor
    X_test_processed = preprocessor.transform(X_test)
    X_test_processed = pd.DataFrame(X_test_processed)

    return X_train_processed, X_test_processed, y_train, y_test

def train_automl_model(X_train, y_train, time_limit=60):
    """
    Trains an AutoML model using auto-sklearn.

    Args:
        X_train: Preprocessed training features.
        y_train: Training target variable.
        time_limit: The time limit for the AutoML search in seconds.

    Returns:
        The trained Auto-sklearn model.
    """
    # Create an Auto-sklearn classification object
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=time_limit,  # Time limit in seconds
        per_run_time_limit=15,            # Time limit for each individual run
        memory_limit=4096,               # Memory limit in MB
        n_jobs=-1,                       # Use all available CPU cores
        # Resampling strategy
        resampling_strategy="cv",
        resampling_strategy_arguments={'folds': 3},
        
        #Enables logging
        #delete_tmp_folder_after_terminate=False, # Keep the temp files.
        #tmp_folder="tmp"
    )

    # Fit the AutoML model to the training data
    automl.fit(X_train, y_train, dataset_name='synthetic_data')

return automl

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model on the test data.

    Args:
        model: The trained machine learning model.
        X_test: Preprocessed testing features.
        y_test: Testing target variable.

    Returns:
        A dictionary of evaluation metrics.
    """
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Get probabilities for ROC AUC

    # Calculate evaluation metrics
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    precision = sklearn.metrics.precision_score(y_test, y_pred)
    recall = sklearn.metrics.recall_score(y_test, y_pred)
    f1 = sklearn.metrics.f1_score(y_test, y_pred)
    roc_auc = sklearn.metrics.roc_auc_score(y_test, y_prob)

    # Print the evaluation metrics
    print("Test Set Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }

def create_streamlit_app(model, numerical_features, categorical_features):
    """
    Creates a Streamlit web application for making predictions with the trained model.

    Args:
        model: The trained machine learning model.
        numerical_features: A list of numerical feature names.
        categorical_features: A list of categorical feature names.
    """
    st.title('Automated ML Prediction App')
    st.write('Enter the feature values to get a prediction.')

    # Create input fields for numerical features
    numerical_inputs = {}
    for feature in numerical_features:
        numerical_inputs[feature] = st.number_input(feature)

    # Create input fields for categorical features
    categorical_inputs = {}
    for feature in categorical_features:
        if feature == 'Feature_11':
            categorical_inputs[feature] = st.selectbox(feature, ['A', 'B', 'C'])
        elif feature == 'Feature_12':
            categorical_inputs[feature] = st.selectbox(feature, ['X', 'Y', 'Z', 'W'])
        elif feature == 'Feature_13':
             categorical_inputs[feature] = st.selectbox(feature, ['P', 'Q'])

    # Create a button to make predictions
    if st.button('Predict'):
        # Create a Pandas DataFrame from the input values
        input_data = pd.DataFrame([numerical_inputs | categorical_inputs]) # Merged the dictionaries

        # Preprocess the input data using the same transformer used for training
        # Get the column order used during training
        input_data_processed = pd.DataFrame(preprocessor.transform(input_data))

        # Make a prediction using the trained model
        prediction = model.predict(input_data_processed)[0]
        probability = model.predict_proba(input_data_processed)[0][1] #added probability

        # Display the prediction
        st.subheader('Prediction:')
        if prediction == 1:
            st.write(f'<span style="color:red">Positive (1)</span>', unsafe_allow_html=True)
        else:
            st.write(f'<span style="color:blue">Negative (0)</span>', unsafe_allow_html=True)
        
        st.subheader('Probability of Positive Class:') # Display probability
        st.write(f'{probability:.4f}')

if __name__ == '__main__':
    # Load the data
    train_data, test_data = train_test_split(generate_synthetic_data(), test_size=0.2, random_state=42)

    # Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(train_data, test_data)
    
    # Get the list of numerical and categorical features.
    numerical_features = X_train.select_dtypes(include=np.number).columns.to_list()
    categorical_features = X_train.select_dtypes(include='object').columns.to_list()

    # Train the AutoML model
    automl_model = train_automl_model(X_train, y_train, time_limit=120)

    # Evaluate the model
    evaluation_metrics = evaluate_model(automl_model, X_test, y_test)

    # Create the Streamlit app
    create_streamlit_app(automl_model, numerical_features, categorical_features)
