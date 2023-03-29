#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[16]:


df=pd.read_excel(r'C:\Project\Flight-Price-Prediction1-main\flight_price.xlsx')
df.head()


# In[17]:


df.info()


# In[18]:


df['Airline'].unique()


# In[19]:


import pandas as pd


# In[20]:


df['Date_of_Journey']=pd.to_datetime(df['Date_of_Journey'],infer_datetime_format=True)


# In[21]:


df.info()


# In[22]:


df['Day']=df['Date_of_Journey'].dt.day
df['Month']=df['Date_of_Journey'].dt.month
df['Year']=df['Date_of_Journey'].dt.year


# In[23]:


df.head()


# In[24]:


df.drop('Date_of_Journey',axis=1,inplace=True)


# In[25]:


df.head()

## Extract day,month and year from the string
df["Date"]=df['Date_of_Journey'].apply(lambda x:x.split("/")[0])
df["Month"]=df['Date_of_Journey'].apply(lambda x:x.split("/")[1])
df["Year"]=df['Date_of_Journey'].apply(lambda x:x.split("/")[2])
# ##Fetaure Engineering Process
# df['Date']=df['Date_of_Journey'].str.split('/').str[0]
# final_df['Month']=df['Date_of_Journey'].str.split('/').str[1]
# df['Year']=df['Date_of_Journey'].str.split('/').str[2]

# In[29]:


df.head(2)


# In[30]:


df['Dept_Hour']=df['Dep_Time'].str.split(':').str[0]
df['Dept_Min']=df['Dep_Time'].str.split(':').str[1]
df.drop('Dep_Time',axis=1,inplace=True)


# In[31]:


df.info()


# In[32]:


df['Dept_Hour']=df['Dept_Hour'].astype(int)
df['Dept_Min']=df['Dept_Min'].astype(int)


# In[36]:


df['Arrival_Time']=df['Arrival_Time'].apply(lambda x: x.split(' ')[0])


# In[37]:


df['Arrival_hour']=df['Arrival_Time'].str.split(':').str[0]
df['Arrival_min']=df['Arrival_Time'].str.split(':').str[1]


# In[38]:


df['Arrival_hour']=df['Arrival_hour'].astype(int)
df['Arrival_min']=df['Arrival_min'].astype(int)


# In[39]:


df.drop('Arrival_Time',axis=1,inplace=True)


# In[40]:


df.info()


# In[43]:


df['Total_Stops'].unique()


# In[44]:


df['Total_Stops'].value_counts()


# In[45]:


df[df['Total_Stops'].isnull()]


# In[47]:


df['Total_Stops']=df['Total_Stops'].map({'non-stop':0,'1 stop':1,'2 stops':2,'3 stops':3,'4 stops':4,'nan':1})


# In[48]:


df.head()


# In[49]:


df['duration_hour']=df['Duration'].str.split(' ').str[0].str.split('h').str[0]


# In[51]:


df[df['duration_hour']=='5m']


# In[52]:


df.drop(6474,axis=0,inplace=True)


# In[54]:


df['duration_min']=df['Duration'].str.split(' ').str[1].str.split('m').str[0]


# In[55]:


df['duration_min']=df['duration_min'].fillna(0)

# Drop Duration
df.drop('Duration',axis=1,inplace=True)
# In[59]:


df['duration_hour']=df['duration_hour'].astype(int)
df['duration_min']=df['duration_min'].astype(int)


# In[63]:


"Krish Naik".split(" ")[0]


# In[64]:


df.info()


# In[65]:


df.Airline.unique()


# In[66]:


df.head()


# In[67]:


df.groupby('Airline')['Price'].mean().sort_values()


# In[68]:


from sklearn.preprocessing import OneHotEncoder


# In[69]:


ohe=OneHotEncoder()


# In[70]:


ohe.fit_transform(df[['Airline']]).toarray()

## Replacing target guided ordinal encoding
def replace_airline_with_mean(df):
    mean_prices = df.groupby('Airline')['Price'].mean().sort_values()
    df['Airline'] = df['Airline'].apply(lambda x: mean_prices[x])
    return df

df = replace_airline_with_mean(df)
df.head()
# In[71]:


pd.DataFrame(ohe.fit_transform(df[['Airline']]).toarray(),columns=ohe.get_feature_names())


# In[72]:


ohe.transform([['Air Asia']]).toarray()


# In[ ]:




