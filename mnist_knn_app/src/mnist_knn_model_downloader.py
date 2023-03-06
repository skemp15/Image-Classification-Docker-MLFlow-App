"""
Search in MNIST - KNN Experiment and pick the best model

"""

__date__ = "2023-01-30"
__author__ = "SamKemp"



# %% --------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import mlflow
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# %% --------------------------------------------------------------------------
# Set Random State
# -----------------------------------------------------------------------------
rng = np.random.RandomState(123)

# %% --------------------------------------------------------------------------
# Tracking uri
# -----------------------------------------------------------------------------
tracking_uri = r"file:///C:/mlflow_local/mlruns"

experiment_name = 'MNIST - KNN'

local_dir = r"C:\Users\SamKemp\Documents\JPMorgan\mnist_knn_app"

# %% --------------------------------------------------------------------------
# Set up the MLflow client
# -----------------------------------------------------------------------------
client = mlflow.tracking.MlflowClient(tracking_uri)

# %% --------------------------------------------------------------------------
# Search runs in the MNIST - KNN experiment 
# -----------------------------------------------------------------------------
# Get the experiment id using the experiment name
exp_id = client.get_experiment_by_name(experiment_name).experiment_id

# Search the runs of the experiment
runs = client.search_runs(exp_id, order_by=['metrics.accuracy DESC'])

# Get the best runs from this ordered list of runs
best_run =runs[0]
best_run_id = best_run.info.run_id

# %% --------------------------------------------------------------------------
# Download the model for the best run
# -----------------------------------------------------------------------------
# Download the model
client.download_artifacts(best_run_id, 'model', local_dir)
model_uri = "file:///" + os.path.join(local_dir, 'resources/model')
model = mlflow.sklearn.load_model(model_uri)

# Download the confusion marix 
client.download_artifacts(best_run_id, 'confusion_matrix_train.png', os.path.join(local_dir, 'resources/confusion_matrices'))

# %% --------------------------------------------------------------------------
# Evaluate model on test data
# -----------------------------------------------------------------------------
# Import test data
X_test = pd.read_csv(r'../resources/X_test.csv')
y_test = pd.read_csv(r'../resources/y_test.csv')
y_test = y_test['class'].astype('str').values

# Evaluate the model
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'{acc = }')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot(cmap='OrRd').figure_.savefig('../resources/confusion_matrices/confusion_matrix_test')


# %% --------------------------------------------------------------------------
# Convert MNIST data to images and store in file
# -----------------------------------------------------------------------------
for idx in range(15):
    fig, ax = plt.subplots(figsize=(28,28))
    img = X_test.loc[idx,:].values.reshape((28,28))
    ax.imshow(img, cmap=plt.cm.gray)
    plt.imsave(fr"C:\test_mnist_images\test_image_{idx}.png", img, cmap='gray')


# %%
