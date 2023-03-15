'''
Author: Vitor Abdo
This .py file is for training, saving the best model and
get the feature importance for model
'''

# Import necessary packages
import json
import logging
import os
import tempfile
import math
import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

import wandb

# basic logs config
logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    format='%(asctime)-15s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


def get_inference_pipeline() -> Pipeline:
    '''function that creates the entire inference pipeline'''
    # preprocessing step
    # categorical values
    non_ordinal_categorical = [
        'workclass', 'education', 'marital_status', 'occupation',
        'relationship', 'race', 'sex']

    # numerical values
    zero_imputed = [
        'age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss',
        'hours_per_week']

    # categorical preprocessing
    non_ordinal_categorical_preproc = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        OneHotEncoder(drop='first'))

    # numerical preprocessing
    zero_imputer = SimpleImputer(strategy='constant', fill_value=0)

    # apply the respective transformations with columntransformer method
    preprocessor = ColumnTransformer([
        ('non_ordinal_cat', non_ordinal_categorical_preproc, non_ordinal_categorical),
        ('impute_zero', zero_imputer, zero_imputed)],
        remainder='drop')

    processed_features = non_ordinal_categorical + zero_imputed

    # instantiate the final model
    final_model = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('scaling', StandardScaler(with_mean=False)),
            ('rf', RandomForestClassifier(random_state=42))
        ]
    )
    return final_model, processed_features


def plot_feature_importance(pipe, feat_names) -> plt.figure:
    '''Function to generate the graph of the
    most important variables for the model

    :param model: (Pipeline)
    The pipeline that made the final model
    :param feat_names: (list)
    List with the name of the variables used in your model
    :return: (figure)
    Returns the figure with the graph of the most
    important variables for the model
    '''
    # We collect the feature importance for all non-nlp features first
    feat_imp = pipe["rf"].feature_importances_[: len(feat_names)]

    # plot the figure
    fig_feat_imp, sub_feat_imp = plt.subplots(figsize=(10, 10))
    sub_feat_imp.bar(
        range(
            feat_imp.shape[0]),
        feat_imp,
        color="r",
        align="center")
    _ = sub_feat_imp.set_xticks(range(feat_imp.shape[0]))
    _ = sub_feat_imp.set_xticklabels(np.array(feat_names), rotation=90)
    fig_feat_imp.tight_layout()
    return fig_feat_imp


def train_model():
    '''Function to train the model, tune the hyperparameters
    and save the best final model
    '''
    # start a new run at wandb
    run = wandb.init(
        project='census-income-forecast',
        entity='vitorabdo',
        job_type='train_data')
    artifact = run.use_artifact("vitorabdo/census-income-forecast/clean_data:latest", type='dataset')
    filepath = artifact.file()
    logger.info('Downloaded cleaned data artifact: SUCCESS')

    # Get the Random Forest configuration and update W&B
    try:
        with open({'n_estimators': 50, 'max_depth': 10}) as fp:
            rf_config = json.load(fp)
        run.config.update(rf_config)
    except BaseException:
        rf_config = {}

    # select only the features that we are going to use
    df_clean = pd.read_csv(filepath)
    X = df_clean.drop(['income'], axis=1)
    y = df_clean['income']
    logger.info(f"Numbers of unique incomes: {y.value_counts()}")

    # training the model
    logger.info('Preparing sklearn pipeline')
    sk_pipe, processed_features = get_inference_pipeline()

    # hyperparameter interval to be trained and tested
    logger.info('Fitting...')
    param_grid = rf_config

    grid_search = GridSearchCV(
        sk_pipe,
        param_grid,
        cv=5,
        scoring="f1",
        return_train_score=True)
    grid_search.fit(X, y)

    # instantiate best model
    final_model = grid_search.best_estimator_

    # scoring
    logger.info('Scoring...')
    cvres = grid_search.cv_results_

    cvres = [(mean_test_score,
              mean_train_score) for mean_test_score,
             mean_train_score in sorted(zip(cvres['mean_test_score'],
                                            cvres['mean_train_score']),
                                        reverse=True) if (math.isnan(mean_test_score) != True)]

    logger.info(
        f"The mean val score and mean train score of f1 is, respectively: {cvres[0]}")

    # exporting the model: save model package in the MLFlow sklearn format
    logger.info('Exporting model')

    with tempfile.TemporaryDirectory() as temp_dir:
        export_path = os.path.join(temp_dir, 'model_export')

    mlflow.sklearn.save_model(
        final_model,
        export_path,
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)

    # upload the model artifact into wandb
    artifact = wandb.Artifact(
        name='final_model_pipe',
        type='pickle',
        description='Final model pipeline after training, exported in the correct format for making inferences')

    artifact.add_dir(export_path)
    run.log_artifact(artifact)
    artifact.wait()
    logger.info('Artifact Uploaded: SUCCESS')

    # Plot feature importance
    fig_feat_imp = plot_feature_importance(final_model, processed_features)

    # lets save and upload all metrics to wandb
    run.summary['Train_score'] = cvres[0][1]
    run.summary['Val_score'] = cvres[0][0]

    run.log(
        {
            'feature_importance': wandb.Image(fig_feat_imp)
        }
    )


if __name__ == "__main__":
    logging.info('About to start executing the train_model function')
    train_model()
    logging.info('Done executing the train_model function')
