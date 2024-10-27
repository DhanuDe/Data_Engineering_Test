#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[15]:


import pandas as pd

from sqlalchemy.orm import sessionmaker
from datetime import datetime
import logging


# Setting Up Logging , |
# logging Used to log information, warnings, and errors for tracking the program’s status.

# In[5]:


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# In[29]:


DATABASE_URL = "mysql+pymysql://user:User%40cc98!@localhost/customer_orders_db"


# In[39]:


from sqlalchemy.orm  import declarative_base


# In[16]:


#Create database model this initializes the base class for the ORM models
Base = declarative_base()


# In[40]:


from sqlalchemy import create_engine, Column, Integer, String, Float, Date


# In[17]:


class Customer(Base):
    __tablename__ = 'customers'
    
    #add attribites 
    customer_id = Column(Integer , primary_key = True)
    customer_name = Column(String(150) ,nullable = False)
    


# In[18]:


class Order(Base):
    __tablename__ = 'orders'
    #Attributes 
    order_id = Column(Integer, primary_key=True)
    customer_id = Column(Integer, nullable=False)
    total_amount = Column(Float, nullable=False)
    order_date = Column(Date, nullable=False)


# In[25]:


#Create Database Connection 
def create_connection():
    try:
        engine = create_engine(DATABASE_URL)
        Base.metadata.create_all(engine)
        return engine
    except Exception as e:
        logger.error(f"Error {str(e)}")
        raise 


# In[36]:


def add_data(engine):
    try:
        customers_data = pd.read_csv('customers.csv')
        orders_data  = pd.read_csv('order.csv')
        customers_data.to_sql('customers' , engine , if_exists='replace', index=False)
        orders_data.to_sql('orders', engine, if_exists='replace', index=False)
        logger.info("Data imported Successfully")
    except Exception as e:
        logger.error(f"Error in  importing data: {str(e)}")
        raise
    
        


# In[37]:


def main():
    try:
        engine = create_connection()
        add_data(engine)
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise


# In[38]:


if __name__ == "__main__":
    main()


# In[ ]:




