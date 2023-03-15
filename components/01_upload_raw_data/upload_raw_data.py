'''
Author: Vitor Abdo
This .py file serves to upload the raw data
in W&B extracted from the data source
'''

# import necessary packages
import logging
import wandb

# basic logs config
logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    format='%(asctime)-15s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


def upload_raw_data(args) -> None:
    '''Function that upload an artifact, in this
    case a raw dataset for weights and biases
    '''
    # start a new run at wandb
    run = wandb.init(
        project='census-income-forecast',
        entity='vitorabdo',
        job_type='upload_raw_data')
    logger.info('Creating run for census income forecast: SUCCESS')

    artifact = wandb.Artifact(
        name='raw_data',
        type='dataset',
        description='Raw dataset used for the project, pulled directly from UCI - Census income'
    )
    artifact.add_reference('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data')  # add a HTTP link to the artifact
    # save the artifact version to W&B and mark it as the output of this run
    run.log_artifact(artifact)
    logger.info('artifact uploaded to the wandb: SUCCESS')


if __name__ == "__main__":
    logging.info('About to start executing the upload_raw_data function')
    upload_raw_data()
    logging.info('Done executing the upload_raw_data function')
