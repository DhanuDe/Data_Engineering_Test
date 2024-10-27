#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np


# In[2]:


DATABASE_URL = "mysql+pymysql://user:User%40cc98!@localhost/customer_orders_db"


# In[6]:


from sqlalchemy import create_engine


# In[7]:

from datetime import datetime, date

def load():
    engine = create_engine(DATABASE_URL)
    query = """
    
    SELECT
        o.id, o.customer_id, c.name,
        o.total_amount, o.created_at
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    
    """
    df = pd.read_sql(query, engine)
    df['created_at'] = pd.to_datetime(df['created_at'])
    return df
    
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

def prepare_ml_data(df):
    customer_metrics = df.groupby('customer_id').agg({
        'id': 'count',
        'total_amount': ['sum', 'mean', 'std']  # revenue metrics
    }).reset_index()
    
    
    customer_metrics.columns = ['customer_id', 'order_count', 'total_revenue', 'avg_order_value', 'order_std']
    
    
    customer_metrics['order_std'] = customer_metrics['order_std'].fillna(0)
    
    
    customer_metrics['is_repeat'] = (customer_metrics['order_count'] > 1).astype(int)
    
    return customer_metrics
    
def train_customer_model(df):

    customer_metrics = prepare_ml_data(df)
    
    
    if len(customer_metrics) < 50:
        return None, None, None, "Insufficient data for training"
    
    if customer_metrics['is_repeat'].nunique() < 2:
        return None, None, None, "Need both repeat and non-repeat customers for training"
    
    
    features = ['order_count', 'total_revenue', 'avg_order_value', 'order_std']
    X = customer_metrics[features]
    y = customer_metrics['is_repeat']
    
   
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    except ValueError as e:
        return None, None, None, f"Error splitting data: {str(e)}"
    
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, scaler, features, f"Model accuracy: {accuracy:.2f}"


def add_ml_section(df):
    st.subheader("Customer Prediction Model")
    
    
    model, scaler, features, message = train_customer_model(df)
    
    if model is None:
        st.warning(message)
        return
    
    st.success(message)
    
    
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': np.abs(model.coef_[0])
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    st.write("Feature Importance:")
    st.bar_chart(feature_importance.set_index('Feature'))
    
    
    st.subheader("Predict Customer Repeat Purchase")
    
    
    col1, col2 = st.columns(2)
    with col1:
        order_count = st.number_input("Number of Orders", min_value=1, value=1)
        total_revenue = st.number_input("Total Revenue ($)", min_value=0.0, value=100.0)
    with col2:
        avg_order = st.number_input("Average Order Value ($)", min_value=0.0, value=100.0)
        order_std = st.number_input("Order Value Standard Deviation", min_value=0.0, value=0.0)
    
    if st.button("Predict"):
        
        input_data = np.array([[order_count, total_revenue, avg_order, order_std]])
        input_scaled = scaler.transform(input_data)
        
       
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        
        if prediction == 1:
            st.success(f"This customer is likely to be a repeat purchaser! (Probability: {probability[1]:.2f})")
        else:
            st.warning(f"This customer is likely to be a one-time purchaser. (Probability of repeat: {probability[1]:.2f})")



def main():
    st.title("Welcome fellows 😉")
    df = load()
    st.sidebar.header("Filter:")
    
    min_date = df['created_at'].min().date()
    max_date = df['created_at'].max().date()
    date_range = st.sidebar.date_input("Choose Date Range" ,value = (min_date , max_date) , min_value =min_date , max_value = max_date )
    date_range[0]
    date_range[1]
    
    if len(date_range) == 2:
        start_date = datetime.combine(date_range[0], datetime.min.time())
        end_date = datetime.combine(date_range[1], datetime.max.time())
    else:
        start_date = datetime.combine(date_range[0], datetime.min.time())
        end_date = datetime.combine(date_range[0], datetime.max.time())
    
    filter_frame = (df['created_at'] >= start_date) & (df['created_at'] <= end_date)
    filter_df = df.loc[filter_frame]
    
    
    min_amount = st.sidebar.slider(
        "Minimum Spent($)",
        min_value=int(df['total_amount'].min()),
        max_value=int(df['total_amount'].max())
    )
    filter_mask = (filter_df['total_amount'] >= min_amount)
    filter_df = filter_df.loc[filter_mask]
    
    order_count_customer = df.groupby('customer_id').size().max()
    
    minimum_orders = st.sidebar.selectbox("Minimum Number Orders" , options=list(range(1, order_count_customer + 1)) ,index =0)
    
    
    costomer_order_count = df.groupby('customer_id').size()
    customer_min_order =costomer_order_count[costomer_order_count >=minimum_orders].index
    filter_df = filter_df[filter_df['customer_id'].isin(customer_min_order)]
    
    st.dataframe(filter_df)
    
    st.subheader("Top 10 Customers by Total Revenue")
    top_customers = filter_df.groupby('customer_id')['total_amount'].sum().nlargest(10)
    st.bar_chart(top_customers)
    
    st.subheader("Total Revenue Over the Time")
    revenue = filter_df.groupby(filter_df['created_at'].dt.date)['total_amount'].sum()
    st.line_chart(revenue)
    
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Revenue", f"${filter_df['total_amount'].sum():,.2f}")
    
    with col2:
        st.metric("Identity Customers", filter_df['customer_id'].nunique())
    
    with col3:
        st.metric("Total Orders", len(filter_df))
    
    
        add_ml_section(df)


# In[9]:


if __name__ == "__main__":
    main()
    

# In[10]:


