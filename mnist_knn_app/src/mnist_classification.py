"""
This is a module for performing MNIST image classification.
The main functions will take the json input from the POST
request and return the response
"""

__date__ = "2023-01-31"
__author__ = "SamKemp"



# %% --------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from joblib import load
import logging
import json
import pickle
import glob
import os
from PIL import Image
import numpy as np

# %% --------------------------------------------------------------------------
# Set up logger
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# %% --------------------------------------------------------------------------
# Load model
# -----------------------------------------------------------------------------
with open(r"../resources/model/model.pkl", 'rb') as file:
    model = pickle.load(file)

# %% --------------------------------------------------------------------------
# Define build_json function
# -----------------------------------------------------------------------------
def build_json(n):
    '''
    This function builds the skeleton json to be returned as the 
    response for the POST request
    '''
    data_dict = {}
    data_dict['results'] = {}
    data_dict['results']['count'] = n
    data_dict['results']['predictions'] = []
    for i in range(n):
        pred_dict = {}
        pred_dict["image_name"] = None
        pred_dict["category"] = None
        data_dict["results"]["predictions"].append(pred_dict)
    return data_dict


# %% --------------------------------------------------------------------------
# Define classify_images function
# -----------------------------------------------------------------------------
def classify_images(json_data):
    '''
    This is the main function that takes the json_data, performs classification, 
    and returns the result as json.
    '''
    logger.info('Trying to run classify_images')
    
    in_folder = json_data['input_folder']
    logger.info(f'Input folder: {in_folder}')

    img_list = os.listdir(in_folder)
    n_img = len(img_list)
    logger.info(f'Processing a total of {n_img} files in input folder')

    logger.info('Creating the skeleton json for output')
    data_dict = build_json(n_img)
    data_dict["results"]["count"] = n_img

    for n, img in enumerate(img_list):
        logger.info(f'Processing file {n+1}/{n_img}')
        logger.info(f'{img}: File open')
        data_dict["results"]["predictions"][n]["image_name"] = img
        
        img_open = Image.open(in_folder + '\\' + img).convert('L')
        logger.info(f"{img}: Pre-processing")
        img_array = np.array(img_open)
        img_array_flat = img_array.flatten().reshape(1,-1) 

        # Make predictions
        logger.info(f'{img}: Prediction')
        pred = model.predict(img_array_flat)[0]
        data_dict["results"]["predictions"][n]["category"] = pred

    return json.dumps(data_dict)


