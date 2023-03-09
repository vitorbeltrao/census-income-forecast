'''
Author: Vitor Abdo

This .py file is for creating the fixtures
'''

# import necessary packages
import pytest
import pandas as pd
import wandb


def pytest_addoption(parser):
    '''parser to let the user input the csv file for fixture'''
    parser.addoption('--csv', action='store')


@pytest.fixture(scope='session')
def data(request):
    '''fixture to generate data to our tests'''
    run = wandb.init(
        project='census-income-forecast',
        entity='vitorabdo',
        job_type='data_tests',
        resume=True)

    # Download input artifact. This will also note that this script is using this
    # particular version of the artifact
    data_path = run.use_artifact(request.config.option.csv).file()

    if data_path is None:
        pytest.fail('You must provide the --csv option on the command line')

    df = pd.read_csv(data_path)
    return df
