'''
Author: Vitor Abdo
This file is for testing the final model with the "prod" tag in the test data
'''

# Import necessary packages
import logging
import sys
import pandas as pd
import mlflow

from sklearn.metrics import f1_score

import wandb

# basic logs config
logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    format='%(asctime)-15s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


def evaluate_model():
    '''Function to test the model listed for production on the test dataset'''
    # start a new run at wandb
    run = wandb.init(
        project='census-income-forecast',
        entity='vitorabdo',
        job_type='test_model')

    # download mlflow model
    model_local_path = run.use_artifact(
        "vitorabdo/census-income-forecast/final_model_pipe:prod", type='pickle').download()
    logger.info('Downloaded prod mlflow model: SUCCESS')

    # download test dataset
    test_data = run.use_artifact("vitorabdo/census-income-forecast/test_set:latest").file()
    logger.info('Downloaded test dataset artifact: SUCCESS')

    # Read test dataset
    test_data = pd.read_csv(test_data)
    X_test = test_data.drop(['income'], axis=1)
    y_test = test_data['income']

    # making inference on test set
    logger.info("Loading model and performing inference on test set")
    sk_pipe = mlflow.sklearn.load_model(model_local_path)
    y_pred = sk_pipe.predict(X_test)

    # scoring the results
    logger.info("Scoring...")
    f1score = f1_score(y_test, y_pred)
    logger.info(f"Test_f1: {f1score}")

    # lets save and upload all metrics to wandb
    run.summary['Test_f1'] = f1score
    logger.info('Metric Uploaded: SUCCESS')

    # lets see some metrics in our data sliced by sex
    filename = open('slice_output', 'w')
    sys.stdout = filename

    # sliced data for categorical values
    for columns in X_test[['workclass', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex']]:
        print(f'\nF1 score on {columns} slices:')
        slice_options = X_test[columns].unique().tolist()
        for option in slice_options:
            row_slice = X_test[columns] == option
            print(
                f'{option}',
                f1_score(
                    y_test[row_slice],
                    sk_pipe.predict(
                        X_test[row_slice])))

    filename.close()

    # create a dataframe to use for fairness
    df_aq = test_data.copy()
    df_aq.rename(columns={'income': 'label_value'}, inplace=True)  # real label
    df_aq['score'] = y_pred  # prediction

    # upload to W&B
    artifact = wandb.Artifact(
        name='aequitas_data',
        type='dataset',
        description='Final dataset for us to use with aequitas')

    df_aq.to_csv('df_aq.csv', index=False)
    artifact.add_file('df_aq.csv')
    run.log_artifact(artifact)
    logger.info('Artifact Uploaded: SUCCESS')


if __name__ == "__main__":
    logging.info('About to start executing the test_model function')
    evaluate_model()
    logging.info('Done executing the test_model function')
