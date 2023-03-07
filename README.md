# Census Income Forecast - v1.0.0

## Table of Contents

1. [Project Description](#Description)
2. [Files Description](#files)
3. [Running Files](#running)
4. [Using the API](#api)
5. [Licensing and Authors](#licensingandauthors)
***

## Project Description <a name="Description"></a>

The goal of the project is to create a machine learning model that makes real-time inference of census data. The main idea is to classify whether a person receives a salary greater than 50K or less than or equal to 50K. In this case, the model receives the inputs referring to the characteristics of these people and releases the result classifying in what income range they are.
***

## Files Description <a name="files"></a>

In "census-income-forecast" repository we have:

* **components**: Inside this folder, we have all the files needed to run the entire model pipeline, from raw data collection to final predictions for never-before-seen data. These are the final files for the production environment. Each component is a block in the model that performs some task and in general generates some output artifact to feed the next steps.

* **main.py file**: Main script in Python that runs all the components. All this managed by *MLflow* and *Hydra*.

* **ml_api.py file**: Script that creates the necessary methods for creating the API with the *FastAPI* library.

* **conda.yaml file**: File that contains all the libraries and their respective versions so that the system works perfectly.

* **config.yaml**: This is the file where we have the environment variables necessary for the components to work.

* **environment.yaml**: This file is for creating a virtual *conda* environment. It contains all the necessary libraries and their respective versions to be created in this virtual environment.

* **model_card.md file**: Documentation of the created machine learning model.
***

## Running Files <a name="running"></a>

### Clone the repository

Go to [census-income-forecast](https://github.com/vitorbeltrao/census-income-forecast) and click on Fork in the upper right corner. This will create a fork in your Github account, i.e., a copy of the repository that is under your control. Now clone the repository locally so you can start working on it:

`git clone https://github.com/[your_github_username]/census-income-forecast.git`

and go into the repository:

`cd census-income-forecast`

### Create the environment

Make sure to have conda installed and ready, then create a new environment using the *environment.yaml* file provided in the root of the repository and activate it. This file contain list of module needed to run the project:

`conda env create -f environment.yaml`
`conda activate census-income-forecast`

### Get API key for Weights and Biases

Let's make sure we are logged in to Weights & Biases. Get your API key from W&B by going to https://wandb.ai/authorize and click on the + icon (copy to clipboard), then paste your key into this command:

`wandb login [your API key]`

You should see a message similar to:

`wandb: Appending key for api.wandb.ai to your netrc file: /home/[your username]/.netrc`

### The configuration

The parameters controlling the pipeline are defined in the `config.yaml` file defined in the root of the repository. We will use Hydra to manage this configuration file.

Open this file and get familiar with its content. Remember: this file is only read by the `main.py` script (i.e., the pipeline) and its content is available with the `go` function in `main.py` as the `config` dictionary. For example, the name of the project is contained in the `project_name` key under the `main` section in the configuration file. It can be accessed from the `go` function as `config["main"]["project_name"]`.

### Running the entire pipeline or just a selection of steps

In order to run the pipeline when you are developing, you need to be in the root of the repository, then you can execute this command:

`mlflow run .`

This will run the entire pipeline.

If you want to run a certain steps you can use the examples of command bellow:

`mlflow run . -P steps=upload_raw_data`

This is useful for testing whether steps that have been added or developed can be performed or not.

If you want to run multiple steps (ex: `upload_raw_data` and the `transform_raw_data` steps), you can similarly do:

`mlflow run . -P steps=upload_raw_data,transform_raw_data`

> NOTE: Make sure the previous artifact step is available in W&B. Otherwise we recommend running each step in order.

You can override any other parameter in the configuration file using the Hydra syntax, by providing it as a `hydra_options` parameter. For example, say that we want to set the parameter 06_train_model -> random_forest -> n_estimators to 10:

`mlflow run . -P steps=train_model -P hydra_options="06_train_model.random_forest.n_estimators=10"`

### Run existing pipeline

We can directly use the existing pipeline to do the training process without the need to fork the repository. All it takes to do that is to conda environment with MLflow and wandb already installed and configured. To do so, all we have to do is run the following command:

`mlflow run -v [pipeline_version] https://github.com/vitorbeltrao/census-income-forecast.git`

`[pipeline_version]` is a release version of the pipeline. For example this repository has currently been released for version `1.0.0`. So we need to input `1.0.0` in place of `[pipeline_version]`.
***

## Using the API <a name="api"></a>

To use the app follow the [link](https://census-income-forecast.herokuapp.com/)
***

## Licensing and Author <a name="licensingandauthors"></a>

Vítor Beltrão - Data Scientist

Reach me at: 

- vitorbeltraoo@hotmail.com

- [linkedin](https://www.linkedin.com/in/v%C3%ADtor-beltr%C3%A3o-56a912178/)

- [github](https://github.com/vitorbeltrao)

- [medium](https://pandascouple.medium.com)

Licensing: [MIT LICENSE](https://github.com/vitorbeltrao/census-income-forecast/blob/main/LICENSE)