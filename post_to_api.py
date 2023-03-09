'''
Author: Vitor Abdo

Script to post to FastAPI instance for model inference
'''

import requests
import json

url = "https://census-income-forecast.herokuapp.com"

# explicit the sample to perform inference on
sample =  {
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

data = json.dumps(sample)

# post to API and collect response
response = requests.post(url, data=data)

# display output - response will show sample details + model prediction added
print("response status code", response.status_code)
print("response content:")
print(response.json())
