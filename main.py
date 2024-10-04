#!/usr/bin/env python
# coding: utf-8

# In[12]:


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

# Load the saved model and preprocessor
model = joblib.load('best_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

class PredictionRequest(BaseModel):
    age: int = Field(..., ge=0)
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str
    duration: float = Field(..., ge=0)
    campaign: int = Field(..., ge=0)
    pdays: int = Field(..., ge=0)
    previous: int = Field(..., ge=0)
    poutcome: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the prediction API. Use the /predict/ endpoint to make predictions."}

@app.post('/predict')
def predict(input_data: PredictionRequest):
    try:
        # Convert input data to DataFrame
        data = {
            'age': [input_data.age],
            'job': [input_data.job],
            'marital': [input_data.marital],
            'education': [input_data.education],
            'default': [input_data.default],
            'housing': [input_data.housing],
            'loan': [input_data.loan],
            'contact': [input_data.contact],
            'month': [input_data.month],
            'day_of_week': [input_data.day_of_week],
            'duration': [input_data.duration],
            'campaign': [input_data.campaign],
            'pdays': [input_data.pdays],
            'previous': [input_data.previous],
            'poutcome': [input_data.poutcome]
        }
        data_df = pd.DataFrame(data)
        
        # Preprocess the data
        data_preprocessed = preprocessor.transform(data_df)
        
        # Make predictions
        prediction = model.predict(data_preprocessed)
        
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# In[ ]:





# In[ ]:




