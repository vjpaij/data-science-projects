import streamlit as st
import pandas as pd
import numpy as np

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

#Get the list of numerical and categorical features.
numerical_features = X_train.select_dtypes(include=np.number).columns.to_list()
categorical_features = X_train.select_dtypes(include='object').columns.to_list()

# Create the Streamlit app
create_streamlit_app(automl_model, numerical_features, categorical_features)