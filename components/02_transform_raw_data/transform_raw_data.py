'''
Author: Vitor Abdo
This .py file serves to download the first artifact
uploaded from the last step and split the raw
dataset into train and test set.
'''

# import necessary packages
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
import wandb

# basic logs config
logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    format='%(asctime)-15s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


def transform_raw_data() -> None:
    '''Function that receives the raw data coming from the data source
    and divide the raw data into train and test set.
    '''
    # start a new run at wandb
    run = wandb.init(
        project='census-income-forecast',
        entity='vitorabdo',
        job_type='transform_data')
    artifact = run.use_artifact("vitorabdo/census-income-forecast/raw_data:latest", type='dataset')
    filepath = artifact.file()
    logger.info('Downloaded raw data artifact: SUCCESS')

    # read the file as a pandas dataframe
    df_raw = pd.read_csv(
        filepath,
        names=[
            'age',
            'workclass',
            'fnlwgt',
            'education',
            'education_num',
            'marital_status',
            'occupation',
            'relationship',
            'race',
            'sex',
            'capital_gain',
            'capital_loss',
            'hours_per_week',
            'native_country',
            'income'])

    # transform the string label to bool
    df_raw['income'] = df_raw['income'].apply(
        lambda val: 0 if val == ' <=50K' else 1)

    # divide the dataset into train and test
    train_set, test_set = train_test_split(
        df_raw,
        test_size=0.2,
        random_state=42)
    logger.info('Splitted raw data into train and test: SUCCESS')

    # upload to W&B
    for df, name in zip([train_set, test_set], ['train_set', 'test_set']):
        logger.info(f'Uploading {name}.csv dataset')
        artifact = wandb.Artifact(
            name=name,
            type='dataset',
            description='Raw dataset transformed with some necessary functions and then divided between training and testing to start the data science pipeline')

        df.to_csv(name + '.csv', index=False)
        artifact.add_file(name + '.csv')
        run.log_artifact(artifact)
        logger.info(f'Uploaded {name}.csv: SUCCESS')


if __name__ == "__main__":
    logging.info('About to start executing the transform_raw_data function')
    transform_raw_data()
    logging.info('Done executing the transform_raw_data function')
