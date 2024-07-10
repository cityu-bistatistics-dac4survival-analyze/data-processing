#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
df = pd.read_csv('latestdata.csv')


# In[3]:


ColNames = df.columns.tolist()
print(ColNames)


# In[4]:


df.shape


# In[5]:


df_full_age = df.dropna(subset=['age'], inplace=False)


# In[6]:


df_full_age.shape


# In[7]:


df_full_age_sex = df_full_age.dropna(subset=['sex'], inplace=False)


# In[8]:


df_full_age_sex.shape


# In[9]:


df_full_age_sex_onset = df_full_age_sex.dropna(subset=['date_onset_symptoms'], inplace=False)


# In[10]:


df_full_age_sex_onset.shape


# In[11]:


df_full_age_sex_onset_confi = df_full_age_sex_onset.dropna(subset=['date_confirmation'], inplace=False)


# In[12]:


df_full_age_sex_onset_confi.shape


# In[13]:


df_full_age_sex_onset_confi["date_death_or_discharge"]


# In[14]:


df_full_age_sex_onset_confi.date_death_or_discharge.isnull().sum()


# In[15]:


df_full_age_sex_onset_confi['date_confirmation'] = df_full_age_sex_onset_confi['date_confirmation'].str[:10]
df_full_age_sex_onset_confi['date_confirmation'] = pd.to_datetime(df_full_age_sex_onset_confi['date_confirmation'], format="%d.%m.%Y")


# In[16]:


df_full_age_sex_onset_confi['date_onset_symptoms'] = df_full_age_sex_onset_confi['date_onset_symptoms'].str[:10]
df_full_age_sex_onset_confi['date_onset_symptoms'] = pd.to_datetime(df_full_age_sex_onset_confi['date_onset_symptoms'], format="%d.%m.%Y")


# In[17]:


df_full_age_sex_onset_confi['date_death_or_discharge'] = df_full_age_sex_onset_confi['date_death_or_discharge'].str[:10]
df_full_age_sex_onset_confi['date_death_or_discharge'] = pd.to_datetime(df_full_age_sex_onset_confi['date_death_or_discharge'], format="%d.%m.%Y")


# In[18]:


df_full_age_sex_onset_confi['censor_time'] = (df_full_age_sex_onset_confi['date_confirmation'] - df_full_age_sex_onset_confi['date_onset_symptoms']).dt.days


# In[19]:


df_full_age_sex_onset_confi['event_time'] = (df_full_age_sex_onset_confi['date_death_or_discharge'] - df_full_age_sex_onset_confi['date_onset_symptoms']).dt.days


# In[20]:


df_full_age_sex_onset_confi['event_time']


# In[21]:


df_last = df_full_age_sex_onset_confi


# In[22]:


df_last['time'] = df_last['event_time'].fillna(df_last['censor_time']).where(df_last['event_time'].notnull() & (df_last['event_time'] >= df_last['censor_time']), df_last['censor_time'])


# In[23]:


df_last['time']


# In[24]:


df_last.time.isnull().sum()


# In[25]:


df_last['status'] = df_last['event_time'].notnull().astype(int)


# In[26]:


df_last['status']


# In[27]:


df_last.date_death_or_discharge.isnull().sum()


# In[28]:


(df_last['status'] == 0).sum()


# In[29]:


print(ColNames)
len(ColNames)


# In[30]:


df_last_toreg = df_last[['ID','age','status','time', 'country', 'latitude', 'longitude', 'symptoms','chronic_disease_binary', 'chronic_disease', 'outcome']]


# In[31]:


df_last_toreg.head()


# In[32]:


df_last_toreg['chronic_disease'].fillna(0, inplace=True)


# In[33]:


df_last_toreg['symptoms'].fillna(0, inplace=True)


# In[34]:


df_last_toreg.head(5)


# In[35]:


df_last_toreg[df_last_toreg['chronic_disease'] != 0].head(5)


# In[36]:


uni_counts = df_last_toreg['chronic_disease'].value_counts()

uni_counts


# In[37]:


split_values = df_last_toreg['chronic_disease'].str.split(r'[;:,]', expand=True)


# In[38]:


df_without_missing = split_values[split_values[0].notna()]


# In[39]:


df_without_missing.head()


# In[40]:


all_elements = split_values.values.ravel()


# In[41]:


all_elements_series = pd.Series(all_elements)


# In[42]:


chronic_counts = all_elements_series.value_counts()
print(chronic_counts)


# In[43]:


max_columns = split_values.shape[1]


# In[44]:


new_columns = [f'B_{i+1}' for i in range(max_columns)]


# In[45]:


split_values.columns = new_columns


# In[46]:


split_values.columns


# In[47]:


df_without_missing = split_values[split_values['B_1'].notna()]
df_without_missing.head()


# In[48]:


chronic_index = chronic_counts.index.tolist()
max_columns = len(chronic_index)
print(chronic_index)


# In[49]:


for i in range(max_columns):
    split_values["chronic_"+chronic_index[i]] = (
           (split_values['B_1'].str.contains(chronic_index[i], case=False, na=False)) | 
           (split_values['B_2'].str.contains(chronic_index[i], case=False, na=False)) | 
           (split_values['B_3'].str.contains(chronic_index[i], case=False, na=False)) | 
           (split_values['B_4'].str.contains(chronic_index[i], case=False, na=False)) |
           (split_values['B_5'].str.contains(chronic_index[i], case=False, na=False))).astype(int)


# In[50]:


split_values[split_values['chronic_hypertension'] != 0].head(5)


# In[51]:


one_hot_code_chronic = split_values.drop(['B_1', 'B_2', 'B_3', 'B_4', 'B_5'],axis=1)


# In[52]:


one_hot_code_chronic.head()


# In[53]:


df_last_toreg[df_last_toreg['symptoms'] != 0].head(5)


# In[54]:


symptoms_names = uni_counts.index.tolist()
max_columns = len(symptoms_names)
print(symptoms_names)


# In[55]:


symptoms_split_values = df_last_toreg['symptoms'].str.split(r'[;:,]', expand=True)


# In[56]:


symptoms_split_values.head()


# In[57]:


df_without_missing = symptoms_split_values[symptoms_split_values[0].notna()]
df_without_missing.head()


# In[58]:


max_columns = symptoms_split_values.shape[1]


# In[59]:


new_columns = [f'S_{i+1}' for i in range(max_columns)]


# In[60]:


symptoms_split_values.columns = new_columns


# In[61]:


df_without_missing = symptoms_split_values[symptoms_split_values['S_1'].notna()]
df_without_missing.head()


# In[62]:


r'\d'


# In[63]:


columns_to_process = new_columns

def replace_fever(text):
    if pd.isna(text):  
        return text
    return 'fever' if re.search(r'fever', str(text), re.IGNORECASE) else text

# 对每一列进行处理
for col in columns_to_process:
    symptoms_split_values[col] = symptoms_split_values[col].apply(replace_fever)


# In[64]:


def replace_fever(text):
    if pd.isna(text):  
        return text
    return 'fever' if re.search(r'\d', str(text), re.IGNORECASE) else text

for col in columns_to_process:
    symptoms_split_values[col] = symptoms_split_values[col].apply(replace_fever)


# In[65]:


all_elements = symptoms_split_values.values.ravel()


# In[66]:


all_elements_series = pd.Series(all_elements)


# In[67]:


symptoms_counts = all_elements_series.value_counts()
print(symptoms_counts)


# In[68]:


symptoms_index = symptoms_counts.index.tolist()
max_columns = len(symptoms_index)
print(symptoms_index)


# In[69]:


symptoms_split_values.head()


# In[70]:


max_columns


# In[71]:


for i in range(max_columns):
    symptoms_split_values["symptoms_"+symptoms_index[i]] = (
           (symptoms_split_values['S_1'].str.contains(symptoms_index[i], case=False, na=False)) | 
           (symptoms_split_values['S_2'].str.contains(symptoms_index[i], case=False, na=False)) | 
           (symptoms_split_values['S_3'].str.contains(symptoms_index[i], case=False, na=False)) |
           (symptoms_split_values['S_4'].str.contains(symptoms_index[i], case=False, na=False)) | 
           (symptoms_split_values['S_5'].str.contains(symptoms_index[i], case=False, na=False)) | 
           (symptoms_split_values['S_6'].str.contains(symptoms_index[i], case=False, na=False)) | 
           (symptoms_split_values['S_7'].str.contains(symptoms_index[i], case=False, na=False))).astype(int)


# In[72]:


symptoms_split_values.head()


# In[73]:


one_hot_code_symptoms = symptoms_split_values.drop(['S_1', 'S_2', 'S_3', 'S_4', 'S_5','S_6','S_7'],axis=1)


# In[74]:


one_hot_code_chronic.head()


# In[75]:


one_hot_code_symptoms.head()


# In[76]:


df_tocsv = pd.concat([df_last_toreg, one_hot_code_chronic], axis=1)


# In[77]:


df_tocsv = pd.concat([df_last_toreg, one_hot_code_symptoms], axis=1)


# In[78]:


df_tocsv.head()


# In[79]:


df_tocsv.shape


# In[80]:


uni_counts = df_last_toreg['outcome'].value_counts()

uni_counts


# df_tocsv.to_csv('covid-19.csv', index = True)

# In[82]:


df_tocsv.to_csv('covid-19.csv', index = False, encoding='utf-8-sig')


# In[ ]:




