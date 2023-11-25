import numpy as np
import pickle
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler 

# Load the trained model
loaded_model = pickle.load(open('./trained_model.sav', 'rb'))

# Function to predict depression diagnosis
def depression_diagnosis(user_input_values):
    # Create a DataFrame from the user input
    cat_col = ['gender', 'ethnicity', 'marital_status', 'education_level', 'employment_status', 'smoking_status']
    num_col = ['age', 'income', 'social_support', 'stress_level', 'sleep_quality', 'exercise_frequency', 'alcohol_consumption', 'self_esteem_level', 'loneliness_level', 'life_satisfaction', 'positive_emotions', 'negative_emotions']
    bol_col = ['family_history', 'drug_use', 'depression_diagnosis', 'anxiety_diagnosis', 'suicidal_thoughts', 'therapy_attendance', 'medication_usage']
    user_input_df = pd.DataFrame([user_input_values], columns=num_col + cat_col + bol_col)

    # Apply Label Encoding to categorical columns using the fitted label encoders
    label_encoder_dict = {}
    for col in cat_col:
        label_encoder = LabelEncoder()
        user_input_df[col] = label_encoder_dict[col].transform(user_input_df[col])

    # Standardize numerical features using the fitted scaler
    scaler = StandardScaler()
    user_input_df[num_col] = scaler.transform(user_input_df[num_col])

    # Make predictions for the user input
    user_prediction = loaded_model.predict(user_input_df)

    # Display the prediction
    print(f'Predicted Depression for User Input: {user_prediction[0]}')

    if user_prediction[0] == 0:
        return 'The person is not diagnosed with depression'
    else:
        return 'The person is diagnosed with depression'



# Streamlit app
def main():
    cat_col = ['gender', 'ethnicity', 'marital_status', 'education_level', 'employment_status', 'smoking_status']
    num_col = ['age', 'income', 'social_support', 'stress_level', 'sleep_quality', 'exercise_frequency', 'alcohol_consumption', 'self_esteem_level', 'loneliness_level', 'life_satisfaction', 'positive_emotions', 'negative_emotions']
    bol_col = ['family_history', 'drug_use', 'depression_diagnosis', 'anxiety_diagnosis', 'suicidal_thoughts', 'therapy_attendance', 'medication_usage']
    st.title('Depression Analyzer Web App')

    # List of column names
    column_names = num_col + cat_col + bol_col

    # Dictionary to store user input
    user_input = {}

    # Generate input fields for each column
    for column_name in column_names:
        # Determine the input type based on the column
        if column_name in num_col:
            user_input[column_name] = st.number_input(f'Enter {column_name.capitalize()}')
        elif column_name in cat_col:
            user_input[column_name] = st.selectbox(f'Select {column_name.capitalize()}',
                                                   options=get_unique_values(column_name))
        elif column_name in bol_col:
            user_input[column_name] = st.checkbox(f'{column_name.replace("_", " ").capitalize()}')

    # Display the user input dictionary
    st.write('User Input:', user_input)

    diagnosis = ''

    if st.button('Depression Test Result'):
        diagnosis = depression_diagnosis(list(user_input.values()))

        st.success(diagnosis)

