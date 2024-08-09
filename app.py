import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Define log transformation function
def log_transform(x):
    return np.log1p(x)

# Load the models
def load_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
        st.write(f"Loaded model type: {type(model)}")  # Debugging line
        return model

selling_price_model_path = 'G:/Data Science Projects/Industrial Copper/Model Files/selling_price_prediction_elasticnet.pkl'

try:
    selling_price_model = load_model(selling_price_model_path)
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Define feature names for selling price model
selling_price_features = [
    'item_date_year', 'item_date_month', 'item_date_day',
    'delivery_date_year', 'delivery_date_month', 'delivery_date_day',
    'date_interaction', 'delivery_lag',
    'quantity tons', 'application', 'thickness', 'width',
    'material_ref', 'product_ref', 'country', 'customer'
]

# Define the tabs
tab1 = st.tabs(["Predict Selling Price"])

with tab1[0]:
    st.header('Predict Selling Price')

    # Input fields for selling price
    item_date = st.text_input('Item Date (YYYYMMDD)', key='item_date_sp')
    quantity_tons = st.number_input('Quantity (tons)', key='quantity_tons_sp')
    customer = st.text_input('Customer', key='customer_sp')
    country = st.text_input('Country', key='country_sp')
    item_type = st.text_input('Item Type', key='item_type_sp')
    application = st.number_input('Application', key='application_sp')
    thickness = st.number_input('Thickness', key='thickness_sp')
    width = st.number_input('Width', key='width_sp')
    material_ref = st.text_input('Material Reference', key='material_ref_sp')
    product_ref = st.text_input('Product Reference', key='product_ref_sp')
    delivery_date = st.text_input('Delivery Date (YYYYMMDD)', key='delivery_date_sp')

    if st.button('Predict Selling Price'):
        if item_date and delivery_date:
            try:
                item_date = pd.to_datetime(item_date, format='%Y%m%d')
                delivery_date = pd.to_datetime(delivery_date, format='%Y%m%d')
                input_data = {
                    'item_date': item_date,
                    'quantity tons': quantity_tons,
                    'customer': customer,
                    'country': country,
                    'item_type': item_type,
                    'application': application,
                    'thickness': thickness,
                    'width': width,
                    'material_ref': material_ref,
                    'product_ref': product_ref,
                    'delivery_date': delivery_date
                }
                input_df = pd.DataFrame([input_data])

                # Feature engineering for selling price
                input_df['item_date_year'] = input_df['item_date'].dt.year
                input_df['item_date_month'] = input_df['item_date'].dt.month
                input_df['item_date_day'] = input_df['item_date'].dt.day
                input_df['delivery_date_year'] = input_df['delivery_date'].dt.year
                input_df['delivery_date_month'] = input_df['delivery_date'].dt.month
                input_df['delivery_date_day'] = input_df['delivery_date'].dt.day
                input_df['date_interaction'] = input_df['item_date_year'] * input_df['delivery_date_year']
                input_df['delivery_lag'] = (input_df['delivery_date'] - input_df['item_date']).dt.days

                # Handle categorical features
                categorical_features = ['country', 'item_type', 'material_ref', 'product_ref']
                for feature in categorical_features:
                    if feature in input_df.columns:
                        input_df[feature] = input_df[feature].astype(str)

                # Convert categorical features to numeric if necessary
                input_df = pd.get_dummies(input_df, columns=categorical_features, drop_first=True)

                # Ensure all required features are present
                for feature in selling_price_features:
                    if feature not in input_df.columns:
                        input_df[feature] = 0

                # Ensure all features are numeric and in the right order
                input_df = input_df.reindex(columns=selling_price_features, fill_value=0)

                # Predict using the selling price model
                try:
                    if hasattr(selling_price_model, 'predict'):
                        selling_price_prediction = selling_price_model.predict(input_df[selling_price_features])
                        st.write(f'Predicted Selling Price: {selling_price_prediction[0]}')
                    else:
                        st.error("Loaded model does not have a predict method.")
                except Exception as e:
                    st.error(f"Prediction error: {e}")
            except Exception as e:
                st.error(f"Input error: {e}")
        else:
            st.error('Please fill in both Item Date and Delivery Date.')
