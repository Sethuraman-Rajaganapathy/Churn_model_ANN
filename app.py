import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

### Load the model
model = tf.keras.models.load_model('model.h5')

### Load the encoder and scaler
with open('ohe_geo_encoder.pkl', 'rb') as file:
    label_geo_encoder = pickle.load(file)
with open('gender_label_encoder.pkl', 'rb') as file:
    gender_label_encoder = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

### Streamlit app
st.title('Customer Churn Prediction')

### User input
geography = st.selectbox('Geography', label_geo_encoder.categories_[0])
gender = st.selectbox('Gender', gender_label_encoder.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of products', 0, 10)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

### Prepare input data
input_data = pd.DataFrame(
    {
        'CreditScore': [credit_score],
        'Gender': [gender_label_encoder.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    }
)

### One-hot encoding 'Geography'
geo_encoded = label_geo_encoder.transform([[geography]]).toarray()  # Use the geography selected earlier
geo_encoded_df = pd.DataFrame(geo_encoded, columns=label_geo_encoder.get_feature_names_out(['Geography']))

### Combine 'Geography' encoding with other input data
input_data = pd.concat([input_data, geo_encoded_df], axis=1)

### Scaling input data
scaled_input = scaler.transform(input_data)

### Prediction
prediction = model.predict(scaled_input)
predict_proba = prediction[0][0]
st.write(f'Churn Probability: {predict_proba:.2f}')

if prediction[0][0] > 0.5:
    st.write('The customer is likely to churn')
else:
    st.write('The customer is not likely to churn')
