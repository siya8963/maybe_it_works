
import streamlit as st
import pandas as pd
import joblib

# Load the model and scaler
model = joblib.load('rf_model_best.pkl')
scaler = joblib.load('scaler.pkl')

# Title of the app
st.title('Credit Risk Assessment')

# Input fields for user to fill in
age = st.number_input('Age (numeric)', min_value=0)
sex = st.selectbox('Sex', options=['male', 'female'])
job = st.selectbox('Job', options=[0, 1, 2, 3])
housing = st.selectbox('Housing', options=['own', 'rent', 'free'])
saving_accounts = st.selectbox('Saving Accounts', options=['little', 'moderate', 'quite rich', 'rich'])
checking_account = st.number_input('Checking Account (in DM)', min_value=0)
credit_amount = st.number_input('Credit Amount (in DM)', min_value=0)
duration = st.number_input('Duration (in months)', min_value=1)
purpose = st.selectbox('Purpose', options=['car', 'furniture/equipment', 'radio/TV', 'domestic appliances', 'repairs', 'education', 'business', 'vacation/others'])

# Button to trigger prediction
if st.button('Predict'):
    # Preprocess input data
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex_male': [1 if sex == 'male' else 0],
        'Job_0': [1 if job == 0 else 0],
        'Job_1': [1 if job == 1 else 0],
        'Job_2': [1 if job == 2 else 0],
        'Job_3': [1 if job == 3 else 0],
        'Housing_own': [1 if housing == 'own' else 0],
        'Housing_rent': [1 if housing == 'rent' else 0],
        'Housing_free': [1 if housing == 'free' else 0],
        'Saving Accounts_little': [1 if saving_accounts == 'little' else 0],
        'Saving Accounts_moderate': [1 if saving_accounts == 'moderate' else 0],
        'Saving Accounts_quite rich': [1 if saving_accounts == 'quite rich' else 0],
        'Saving Accounts_rich': [1 if saving_accounts == 'rich' else 0],
        'Checking Account': [checking_account],
        'Credit Amount': [credit_amount],
        'Duration': [duration],
        'Purpose_car': [1 if purpose == 'car' else 0],
        'Purpose_furniture/equipment': [1 if purpose == 'furniture/equipment' else 0],
        'Purpose_radio/TV': [1 if purpose == 'radio/TV' else 0],
        'Purpose_domestic appliances': [1 if purpose == 'domestic appliances' else 0],
        'Purpose_repairs': [1 if purpose == 'repairs' else 0],
        'Purpose_education': [1 if purpose == 'education' else 0],
        'Purpose_business': [1 if purpose == 'business' else 0],
        'Purpose_vacation/others': [1 if purpose == 'vacation/others' else 0]
    })

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_scaled)

    # Display the result
    risk = 'Low Risk' if prediction[0] == 0 else 'High Risk'
    st.success(f'The predicted credit risk is: {risk}')
