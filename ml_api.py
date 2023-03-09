'''
Author: Vitor Abdo

this file is for creating our inference api with fastapi
'''

# Import necessary packages
import json
import joblib
import logging
# import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
# import wandb

# basic logs config
logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    format='%(asctime)-15s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Creating a Fastapi object
app = FastAPI()


class ModelInput(BaseModel):
    '''identifying the type of our model features'''
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        schema_extra = {
                        "example": {
                                    "age": 50,
                                    "workclass": " Private", 
                                    "fnlwgt": 234721,
                                    "education": " Doctorate",
                                    "education_num": 16,
                                    "marital_status": " Separated",
                                    "occupation": " Exec-managerial",
                                    "relationship": " Not-in-family",
                                    "race": " Black",
                                    "sex": " Female",
                                    "capital_gain": 0,
                                    "capital_loss": 0,
                                    "hours_per_week": 50,
                                    "native_country": " United-States"
                                    }
                        }

# # loading the saved model
# # start a new run at wandb
# run = wandb.init(
#     project='census-income-forecast',
#     entity='vitorabdo',
#     job_type='get_mlflow_model')

# # download mlflow model
# model_local_path = run.use_artifact(
#     'vitorabdo/census-income-forecast/final_model_pipe:prod',
#     type='pickle').download()
# sk_pipe = mlflow.sklearn.load_model(model_local_path)
# wandb.finish()
# logger.info('Downloaded prod mlflow model: SUCCESS')

# download mlflow model
model_local_path = 'model.pkl'
sk_pipe = joblib.load(model_local_path)


@app.get('/')
def greetings():
    '''get method to to greet a user'''
    return 'Welcome to our model API'


@app.post('/income_prediction')
def income_pred(input_parameters: ModelInput):
    '''post method to our inference'''

    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)

    input_df = pd.DataFrame(
        input_dictionary,
        columns=sk_pipe.named_steps['preprocessor'].transformers_[0][2] +
        sk_pipe.named_steps['preprocessor'].transformers_[1][2],
        index=[0])

    prediction = sk_pipe.predict(input_df)

    if prediction[0] == 0:
        return 'The person income is less than or equal to 50K'

    return 'The person income is greater than 50K'


if __name__ == '__main__':
    pass
