'''
Author: Vitor Abdo
This .py file is used to clean up the data,
for example removing outliers.
'''

# import necessary packages
import logging
import pandas as pd
import wandb

# basic logs config
logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    format='%(asctime)-15s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


def clean_data() -> None:
    '''Function to clean up our training dataset to feed the machine
    learning model.
    '''
    # start a new run at wandb
    run = wandb.init(
        project='census-income-forecast',
        entity='vitorabdo',
        job_type='clean_data')
    artifact = run.use_artifact("vitorabdo/census-income-forecast/train_set:latest", type='dataset')
    filepath = artifact.file()
    logger.info('Downloaded raw data artifact: SUCCESS')

    # clean the train dataset
    df_raw = pd.read_csv(filepath)
    df_clean = df_raw.copy()
    logger.info('Train dataset are clean: SUCCESS')

    # upload to W&B
    artifact = wandb.Artifact(
        name='clean_data',
        type='dataset',
        description='Clean dataset after we apply "clean_data" function')

    df_clean.to_csv('df_clean.csv', index=False)
    artifact.add_file('df_clean.csv')
    run.log_artifact(artifact)
    logger.info('Artifact Uploaded: SUCCESS')


if __name__ == "__main__":
    logging.info('About to start executing the clean_data function')
    clean_data()
    logging.info('Done executing the clean_data function')
