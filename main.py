'''
Author: Vitor Abdo
This is the main system file that runs all the necessary
components to run the machine learning pipeline
'''

# import necessary packages
import mlflow
import tempfile
import os
import hydra
import json
from omegaconf import DictConfig

_steps = [
    'upload_raw_data',
    'transform_raw_data',
    'basic_clean',
    'data_check',
    'train_model',
    # 'test_model'
]

@hydra.main(version_base=None, config_path=".", config_name="config")
def go(config: DictConfig):
    '''main file that runs the entire pipeline end-to-end using hydra and mlflow
    :param config: (.yaml file)
    file that contains all the default data for the entire machine learning pipeline to run
    '''
    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ['WANDB_PROJECT'] = config['main']['project_name']
    os.environ['WANDB_RUN_GROUP'] = config['main']['experiment_name']

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(',') if steps_par != 'all' else _steps

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        if 'upload_raw_data' in active_steps:
            # Download file from source and load in W&B
            _ = mlflow.run(
                f"{config['main']['components_repository']}/01_upload_raw_data",
                'main',
                version='main',
                parameters={
                    'artifact_name': 'raw_data',
                    'artifact_type': 'dataset',
                    'artifact_description': 'Raw dataset used for the project, pulled directly from UCI - Census income',
                    'input_uri': config['01_upload_raw_data']['input_uri']
                },
            )






            
if __name__ == "__main__":
    go()