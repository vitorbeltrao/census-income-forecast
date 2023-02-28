'''
Author: Vitor Abdo
This .py file is used to clean up the data,
for example removing outliers.
'''

# import necessary packages
import logging
import argparse
import wandb
import pandas as pd

# basic logs config
logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    format='%(asctime)-15s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


def clean_data(args) -> None:
    '''Function to clean up our training dataset to feed the machine
    learning model.
    '''
    # start a new run at wandb
    run = wandb.init(
        project='census-income-forecast',
        entity='vitorabdo',
        job_type='clean_data')
    artifact = run.use_artifact(args.input_artifact, type='dataset')
    filepath = artifact.file()
    logger.info('Downloaded raw data artifact: SUCCESS')

    # clean the train dataset
    df_raw = pd.read_csv(filepath)
    df_clean = df_raw.copy()
    logger.info('Train dataset are clean: SUCCESS')

    # upload to W&B
    artifact = wandb.Artifact(
        name=args.artifact_name,
        type=args.artifact_type,
        description=args.artifact_description)

    df_clean.to_csv('df_clean.csv', index=False)
    artifact.add_file('df_clean.csv')
    run.log_artifact(artifact)
    logger.info('Artifact Uploaded: SUCCESS')


if __name__ == "__main__":
    logging.info('About to start executing the clean_data function')

    parser = argparse.ArgumentParser(
        description='Upload an artifact to W&B. Adds a reference denoted by a csv to the artifact.')

    parser.add_argument(
        '--input_artifact',
        type=str,
        help='String referring to the W&B directory where the csv with the train set to be transformed is located.',
        required=True)

    parser.add_argument(
        '--artifact_name',
        type=str,
        help='A human-readable name for this artifact which is how you can identify this artifact.',
        required=True)

    parser.add_argument(
        '--artifact_type',
        type=str,
        help='The type of the artifact, which is used to organize and differentiate artifacts.',
        required=True)

    parser.add_argument(
        '--artifact_description',
        type=str,
        help='Free text that offers a description of the artifact.',
        required=False,
        default='Clean dataset after we apply "clean_data" function')

    parser.add_argument(
        '--race',
        type=str,
        help='Category of variable "race" that we dont want to keep in the dataset.',
        required=False,
        default=' Other')

    args = parser.parse_args()
    clean_data(args)
    logging.info('Done executing the clean_data function')