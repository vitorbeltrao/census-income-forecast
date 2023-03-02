'''
Author: Vitor Abdo

this file is for creating our inference api with fastapi
'''

# Import necessary packages
from fastapi import FastAPI
from pydantic import BaseModel
import json
import wandb
import logging
import mlflow
import pandas as pd

# basic logs config
logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    format='%(asctime)-15s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Creating a Fastapi object
app = FastAPI()

# Using Pydantic lib, defining the data type for all the inputs
class model_input(BaseModel):
    
    age : int
    workclass : str
    fnlwgt : int
    education : str
    education_num : int
    marital_status : str
    occupation : str
    relationship : str    
    race : str 
    sex : str 
    capital_gain : int 
    capital_loss : int 
    hours_per_week : int
    native_country: str

# loading the saved model
# start a new run at wandb
run = wandb.init(
    project='census-income-forecast',
    entity='vitorabdo',
    job_type='test_model')

# download mlflow model
model_local_path = run.use_artifact(
    'vitorabdo/census-income-forecast/final_model_pipe:prod', type='pickle').download()
sk_pipe = mlflow.sklearn.load_model(model_local_path)
wandb.finish()
logger.info('Downloaded prod mlflow model: SUCCESS')

# creating a GET request to the API
@app.get("/")
def greetings():
    return "Welcome to our model API"

# creating a POST request to the API
@app.post('/income_prediction')
def income_predd(input_parameters: model_input):
    
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)

    input_df = pd.DataFrame(
        input_dictionary, 
        columns=sk_pipe.named_steps['preprocessor'].transformers_[0][2] + sk_pipe.named_steps['preprocessor'].transformers_[1][2], 
        index=[0])

    prediction = sk_pipe.predict(input_df)
    
    if (prediction[0] == 0):
        return 'The person income is less than or equal to 50K'
    else:
        return 'The person income is greater than 50K'
    

if __name__ == '__main__':
    uvicorn.run(app, host = '127.0.0.1', port=8000)
