'''
Author: Vitor Abdo
Unit test of ml_api.py API module with pytest
'''

# import necessary packages
import json
import logging
from fastapi.testclient import TestClient
from ml_api import app

# basic logs config
logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    format='%(asctime)-15s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# create a test client instance
client = TestClient(app)


def test_get():
    '''Test welcome message for get at root'''
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "Welcome to our model API"


def test_inference_class1():
    '''Test model inference output for class 1'''
    sample =  {  
        'age':50,
        'workclass':" Private", 
        'fnlgt':234721,
        'education':" Doctorate",
        'education_num':16,
        'marital_status':" Separated",
        'occupation':" Exec-managerial",
        'relationship':" Not-in-family",
        'race':" Black",
        'sex':" Female",
        'capital_gain':0,
        'capital_loss':0,
        'hours_per_week':50,
        'native_country':" United-States"
    }

    data = json.dumps(sample)

    r = client.post("/inference/", data=data)

    # test response and output
    assert r.status_code == 200
    assert r.json()["age"] == 50
    assert r.json()["fnlgt"] == 234721

    # test prediction vs expected label
    logger.info(f'********* prediction = {r.json()["prediction"]} ********')
    assert r.json()["prediction"] == '>50K'


def test_inference_class0():
    '''Test model inference output for class 0'''
    sample =  {
        'age':30,
        'workclass':" Private", 
        'fnlgt':234721,
        'education':" HS-grad",
        'education_num':1,
        'marital_status':" Separated",
        'occupation':" Handlers-cleaners",
        'relationship':" Not-in-family",
        'race':" Black",
        'sex':" Male",
        'capital_gain':0,
        'capital_loss':0,
        'hours_per_week':35,
        'native_country':" United-States"
    }

    data = json.dumps(sample)

    r = client.post("/inference/", data=data )

    # test response and output
    assert r.status_code == 200
    assert r.json()["age"] == 30
    assert r.json()["fnlgt"] == 234721

    # test prediction vs expected label
    logging.info(f'********* prediction = {r.json()["prediction"]} ********')
    assert r.json()["prediction"][0] == '<=50K'