'''
Author: Vitor Abdo
This .py file runs the necessary tests to check our data 
after cleaning it after the "basic_clean" step
'''
# import necessary packages
import pandas as pd
import numpy as np
import scipy.stats

def test_column_names(data):
    '''Tests if the column names are the same as the original 
    file, including in the same order
    '''
    expected_colums = [
       'age', 'workclass', 'fnlwgt', 'education', 'education_num',
       'marital_status', 'occupation', 'relationship', 'race', 'sex',
       'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
       'income']

    these_columns = data.columns.values

    # This also enforces the same order
    assert list(expected_colums) == list(these_columns)

def test_race_names(data):
   '''Tests if the categories of variable "neighbourhood_group" 
   are the same
   '''
   known_names = [" White", " Black", " Asian-Pac-Islander", " Amer-Indian-Eskimo", " Other"]

   neigh = set(data['race'].unique())

   # Unordered check
   assert set(known_names) == set(neigh)

def test_row_count(data):
   '''checks that the size of the dataset is reasonable 
   (not too small, not too large)
   '''
   assert 15000 < data.shape[0] < 1000000