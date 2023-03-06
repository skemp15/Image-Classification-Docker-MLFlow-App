"""
KNN Model on MNIST Dataset 

Model: KNN
Dataset: MNIST
Artifacts: Confusion Matrix Plot
Metrics: Accuracy
"""

__date__ = "2023-01-30"
__author__ = "SamKemp"



# %% --------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import mlflow
import argparse
import random

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


# %% --------------------------------------------------------------------------
# Set up logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    format = "[%(levelname)s %(module)s] %(asctime)s - %(message)s",
    level=logging.INFO
)

logger=logging.getLogger(__name__)

# %% --------------------------------------------------------------------------
# Set up argument parser
# -----------------------------------------------------------------------------
# Create the parser object
parser = argparse.ArgumentParser(description = 'KNN - MNIST Experiment')

# Add the arguments to the parser
parser.add_argument('--n', help='Number of experiments to run', type=int)
parser.add_argument('--max_k', help='Maximum value of nearst neighbours', type=int)
parser.add_argument('--seed', help='Random seed', type=int)

# Parse the arguments 
args = parser.parse_args()

if not args.n:
    n_experiments = 10
else:
    n_experiments = args.n

if not args.max_k:
    max_k_nbrs = 10
else:
    max_k_nbrs = args.max_k

if not args.seed:
    seed = 123
else:
    seed = args.seed

logger.info(f'Number of experiments: {n_experiments}')
logger.info(f'Max K-neighbours: {max_k_nbrs}')
logger.info(f'Random seed: {seed}')


# %% --------------------------------------------------------------------------
# Set Random State
# -----------------------------------------------------------------------------
rng = np.random.RandomState(seed)
logger.debug(f'Random State initialised with seed {seed}')

# %% --------------------------------------------------------------------------
# Set up MLflow
# -----------------------------------------------------------------------------
# Set the folder to record experiments
tracking_uri = r"file:///C:/mlflow_local/mlruns"
logger.info(f'Tracking uri set to {tracking_uri}')

# Set up MLflow
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment('MNIST - KNN')

# %% --------------------------------------------------------------------------
# Data load and setup 
# -----------------------------------------------------------------------------
# Fetch data
mnist = fetch_openml(name='mnist_784')

# Define X and y
X = mnist.data
y = mnist.target

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=rng
)

# Train/Validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=rng
)

# Save test data to csv
X_test.to_csv(r"../resources/X_test.csv", index=False)
y_test.to_csv(r"../resources/y_test.csv", index=False)

logger.debug(f'{X_train.shape=}')
logger.debug(f'{X_test.shape=}')
logger.debug(f'{X_val.shape=}')
logger.debug(f'{y_train.shape=}')
logger.debug(f'{y_test.shape=}')
logger.debug(f'{y_val.shape=}')

# %% --------------------------------------------------------------------------
# Run experiments
# -----------------------------------------------------------------------------
k_list = random.sample(range(1,max_k_nbrs+1), n_experiments)
for i in range(n_experiments):
    with mlflow.start_run():
        # Get parameters
        k = k_list[i]
        logger.info(f'Run {i+1}/{n_experiments}: k: {k}')

        # Fit the model
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)

        # Log the parameter 
        mlflow.log_param('k-neighbours', k)

        # Evaluate the model
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)

        # Confusion Matrix
        cm = confusion_matrix(y_val, y_pred)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(cmap='OrRd').figure_.savefig('../resources/confusion_matrices/confusion_matrix_train')

        # Log the results
        mlflow.log_metric('accuracy', acc)
        mlflow.sklearn.log_model(model, 'model')
        mlflow.log_artifact('../resources/confusion_matrices/confusion_matrix_train.png')
