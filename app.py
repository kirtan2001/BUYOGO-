#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd


# In[2]:


df = pd.read_csv("hotel_bookings.csv")


# In[5]:


df['country'].fillna("Unknown", inplace=True)


# In[6]:


df['agent'].fillna(0, inplace=True)
df['company'].fillna(0, inplace=True)
df['children'].fillna(0, inplace=True)


# In[7]:


df.isnull().sum().sum()


# In[3]:


model = SentenceTransformer("all-MiniLM-L6-v2")


# In[8]:


# Ensure 'arrival_date' is created
df['arrival_date'] = pd.to_datetime(df['arrival_date_year'].astype(str) + '-' +
                                    df['arrival_date_month'] + '-' +
                                    df['arrival_date_day_of_month'].astype(str),
                                    errors='coerce')

# Verify the column exists
print(df[['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month', 'arrival_date']].head())


# In[10]:


df['revenue'] = df['adr'] * (df['stays_in_week_nights'] + df['stays_in_weekend_nights'])
monthly_revenue = df.groupby('arrival_date_month')['revenue'].sum()


# In[11]:


# Text data for embedding 
df['text_data'] = df['hotel'] + " " + df['arrival_date'].astype(str) + " " + \
                  "Revenue: " + df['revenue'].astype(str) + " " + \
                  "Cancellation: " + df['is_canceled'].astype(str)


# In[ ]:





# In[12]:


embeddings = model.encode(df['text_data'].tolist(), convert_to_numpy=True)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)


# In[13]:


app = FastAPI()


# In[14]:


class QueryModel(BaseModel):
    question: str


# In[15]:


@app.post("/analytics")
def get_analytics():
    cancellation_rate = df['is_canceled'].mean() * 100
    avg_revenue = df['revenue'].mean()
    avg_adr = df['adr'].mean()

    return {
        "cancellation_rate": f"{cancellation_rate:.2f}%",
        "average_revenue": f"{avg_revenue:.2f}",
        "average_adr": f"{avg_adr:.2f}"
    }


# In[16]:


def answer_question(query):
    query_embedding = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_embedding, 1)  # Get the top match
    
    # Extract the first matching row and remove the index
    result = df.iloc[I[0]].to_dict()

    # Convert dictionary values from {index: value} to value
    cleaned_result = {key: list(value.values())[0] if isinstance(value, dict) else value for key, value in result.items()}

    # Convert Timestamp values to string if present
    if 'arrival_date' in cleaned_result and isinstance(cleaned_result['arrival_date'], pd.Timestamp):
        cleaned_result['arrival_date'] = cleaned_result['arrival_date'].strftime('%Y-%m-%d')

    return cleaned_result


# In[17]:


@app.post("/ask")
def ask_question(query: QueryModel):
    answer = answer_question(query.question)
    return {"answer": answer}


# In[18]:


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)


# In[ ]:




