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


def load():
    engine = create_engine(DATABASE_URL)
    query = """
    
    SELECT
        o.id, o.customer_id, c.name,
        o.total_amount, o.created_at
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    
    """
    return pd.read_sql(query, engine)


# In[8]:


def main():
    st.title("Welcome fellows")
    df = load()
    st.sidebar.header("Filter:")
    df


# In[9]:


if __name__ == "__main__":
    main()


# In[10]:


