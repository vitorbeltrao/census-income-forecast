'''
Author: Vitor Abdo
This is the main system file that runs all the necessary
components to run the machine learning pipeline
'''

# import necessary packages
import os
import hydra
import mlflow
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
    if 'upload_raw_data' in active_steps:
        project_uri = f"{config['main']['components_repository']}/01_upload_raw_data"
        mlflow.run(project_uri)

    if 'transform_raw_data' in active_steps:
        project_uri = f"{config['main']['components_repository']}/02_transform_raw_data"
        mlflow.run(project_uri)

        # if 'basic_clean' in active_steps:
        #     _ = mlflow.run(
        #         f"{config['main']['components_repository']}/04_basic_clean",
        #         'main',
        #         version='main',
        #         parameters={
        #             'input_artifact': config['04_basic_clean']['input_artifact'],
        #             'artifact_name': 'clean_data',
        #             'artifact_type': 'dataset',
        #             'artifact_description': 'Clean dataset after we apply "clean_data" function',
        #             'min_price': config['04_basic_clean']['race'],
        #         },
        #     )

        # if 'data_check' in active_steps:
        #     _ = mlflow.run(
        #         f"{config['main']['components_repository']}/05_data_check",
        #         'main',
        #         version='main',
        #         parameters={
        #             'csv': config['05_data_check']['csv'],
        #         },
        #     )

        # if 'train_model' in active_steps:
        #     rf_config = os.path.abspath('rf_config.json')
        #     with open(rf_config, 'w+') as fp:
        #         json.dump(
        #             dict(
        #                 config['06_train_model']['random_forest'].items()),
        #             fp)

        #     _ = mlflow.run(
        #         f"{config['main']['components_repository']}/06_train_model",
        #         'main',
        #         version='main',
        #         parameters={
        #             'input_artifact': config['06_train_model']['input_artifact'],
        #             'rf_config': rf_config,
        #             'cv': config['06_train_model']['cv'],
        #             'scoring': config['06_train_model']['scoring'],
        #             'artifact_name': 'final_model_pipe',
        #             'artifact_type': 'pickle',
        #             'artifact_description': 'Final model pipeline after training, exported in the correct format for making inferences'
        #         },
        #     )

        # if 'test_model' in active_steps:
        #     _ = mlflow.run(
        #         f"{config['main']['components_repository']}/07_test_model",
        #         'main',
        #         version='main',
        #         parameters={
        #             'mlflow_model': config['07_test_model']['mlflow_model'],
        #             'test_data': config['07_test_model']['test_data'],
        #             'artifact_name': 'aequitas_data',
        #             'artifact_type': 'dataset',
        #             'artifact_description': 'Final dataset for us to use with aequitas'},
        #     )


if __name__ == "__main__":
    go()