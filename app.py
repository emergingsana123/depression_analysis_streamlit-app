import numpy as np
import pickle
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler 

loaded_model = pickle.load(open('trained_model.sav', 'rb'))

cat_col = [ 'gender', 'ethnicity', 'marital_status', 'education_level',
       'employment_status', 'smoking_status']

num_col = ['age', 'income', 'social_support',
       'stress_level', 'sleep_quality', 'exercise_frequency',
       'alcohol_consumption', 'self_esteem_level',
       'loneliness_level', 'life_satisfaction', 'positive_emotions',
       'negative_emotions']

bol_col = ['family_history' ,'drug_use',
       'depression_diagnosis', 'anxiety_diagnosis', 'suicidal_thoughts',
       'therapy_attendance', 'medication_usage']

def depression_diagnosis(user_input_values):

  user_input_values = (74,'Female','Caucasian','Single','College','Student',999360.15,1,4,1,10,2,3,'Occasional smoker',0,1,1,0,1,2,2,3,10,10)

# Create a DataFrame from the user input
  user_input_df = pd.DataFrame([user_input_values], columns=X.columns)

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

  if(user_prediction[0]==0):
    return 'The person is not diagonised with depression'
  else:
    return 'The person is  diagonised with depression'

def main():
  
    st.title('Depression Analyzer Web App')
  
    import numpy as np
    import pickle
    import streamlit as st
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import StandardScaler 

    loaded_model = pickle.load(open('C:\Users\sansk\OneDrive\Desktop\New folder (3)', 'rb'))

    cat_col = [ 'gender', 'ethnicity', 'marital_status', 'education_level',
       'employment_status', 'smoking_status']

    num_col = ['age', 'income', 'social_support',
       'stress_level', 'sleep_quality', 'exercise_frequency',
       'alcohol_consumption', 'self_esteem_level',
       'loneliness_level', 'life_satisfaction', 'positive_emotions',
       'negative_emotions']

    bol_col = ['family_history' ,'drug_use',
       'depression_diagnosis', 'anxiety_diagnosis', 'suicidal_thoughts',
       'therapy_attendance', 'medication_usage']

def depression_diagnosis(user_input_values):

    user_input_values = (74,'Female','Caucasian','Single','College','Student',999360.15,1,4,1,10,2,3,'Occasional smoker',0,1,1,0,1,2,2,3,10,10)

# Create a DataFrame from the user input
    user_input_df = pd.DataFrame([user_input_values], columns=X.columns)

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

    if(user_prediction[0]==0):
      return 'The person is not diagonised with depression'
    else:
      return 'The person is  diagonised with depression'

def main():
  
    st.title('Depression Analyzer Web App')
  
    import streamlit as st

# List of column names
    column_names = ['age', 'gender', 'ethnicity', 'marital_status', 'education_level',
    'employment_status', 'income', 'family_history', 'social_support',
    'stress_level', 'sleep_quality', 'exercise_frequency',
    'alcohol_consumption', 'smoking_status', 'drug_use',
    'suicidal_thoughts', 'anxiety_diagnosis', 
    'therapy_attendance', 'medication_usage', 'self_esteem_level',
    'loneliness_level', 'life_satisfaction', 'positive_emotions',
    'negative_emotions']

# Dictionary to store user input
    user_input = {}

# Generate input fields for each column
    for column_name in column_names:
    # Determine the input type based on the column
      if column_name in ['age', 'income', 'social_support','stress_level', 'sleep_quality', 'exercise_frequency','alcohol_consumption', 'self_esteem_level','loneliness_level', 'life_satisfaction', 'positive_emotions','negative_emotions']:
        user_input[column_name] = st.number_input(f'Enter {column_name.capitalize()}')

        elif column_name in ['gender', 'ethnicity', 'marital_status', 'education_level','employment_status', 'smoking_status']:
        user_input[column_name] = st.selectbox(f'Select {column_name.capitalize()}',
                                               options=get_unique_values(column_name))  # Define get_unique_values function

        elif column_name in ['family_history' ,'drug_use','depression_diagnosis', 'anxiety_diagnosis', 'suicidal_thoughts','therapy_attendance', 'medication_usage']:
        user_input[column_name] = st.checkbox(f'{column_name.replace("_", " ").capitalize()}')

# Display the user input dictionary
st.write('User Input:', user_input)

diagnosis =''
    
if st.button('Depression Test Result'):
    diagnosis = depression_diagnosis([age,gender,ethnicity,marital_status,education_level,employment_status,income,family_history,social_support,stress_level,sleep_quality,exercise_frequency,alcohol_consumption,smoking_status,drug_use,depression_diagnosis,anxiety_diagnosis,suicidal_thoughts,therapy_attendance,medication_usage,self_esteem_level,loneliness_level,life_satisfaction,positive_emotions,negative_emotions])
    
    st.success(diagnosis)
    
    
    
if__name =='__main__'
    main()
