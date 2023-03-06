"""
Wordcount Function

Module to count the number of occurences of sentiment words
in the 10-k reports for each compan
"""

__date__ = "2022-12-31"
__author__ = "SamKemp"



# %% --------------------------------------------------------------------------
# Imports
import yahoofinancials
from yahoofinancials import YahooFinancials
import requests
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import os
import numpy as np
import ref_data as rf
import datetime as dt
import matplotlib.pyplot as plt
import scipy.stats as ss
# -----------------------------------------------------------------------------

# %%

# %% --------------------------------------------------------------------------
# Define the function

def write_document_sentiments(input_folder, output_file):

    # Create an empty dataframe
    df_columns = ['Symbol', 'ReportType', 'FilingDate', 'Negative', 'Positive', 'Uncertainty', 'Litigious']
    word_count_df = pd.DataFrame(columns=df_columns)

    # Get folder path of 10-k files
    cwd = os.getcwd()
    input_folder_path = os.path.join(cwd, input_folder)

    # Get file path of output file
    output_file_path = os.path.join(cwd, output_file)

    # Get file path of dictionary
    dict_name = 'Loughran-McDonald_MasterDictionary_1993-2021.csv'
    dict_file_path = os.path.join(cwd, dict_name)

    # Get sentiment words
    sent_word_dict = rf.get_sentiment_word_dict()
    sent_word_list = list(sent_word_dict.keys())

    # Cycle throuh 10-k files
    os.chdir(input_folder_path)
    for file in os.listdir():

        # Create a dictionary
        file_info_dict = {}
        info_parts = file.split('_')
        file_info_dict['Symbol'] = info_parts[0]
        file_info_dict['ReportType'] = info_parts[1]
        date = info_parts[2].split('.')[0]
        file_info_dict['FilingDate'] = date

        # Read file
        file_path = os.path.join(os.getcwd(), file)
        with open(file_path, 'r', encoding="utf8") as f:
            txt_data = f.read()
            
            # Count sentiment words
            for word in sent_word_list:
                word_count = txt_data.count(word)
                file_info_dict[word] = word_count

        # Add file info dictionary to dateframe
        word_count_df = pd.concat([word_count_df, pd.DataFrame([file_info_dict])])
        print(f'File {file} added to dataframe')

    # Change to parent dictionary and save file
    os.chdir(os.path.dirname(os.getcwd()))
    word_count_df.to_csv(output_file, index=False)    
# -----------------------------------------------------------------------------

# %% --------------------------------------------------------------------------
# Function to add returns to word_count_df
# -----------------------------------------------------------------------------

def add_returns_to_df(count_file):

    # Get file path
    cwd = os.getcwd()
    file_path = os.path.join(cwd, count_file)

    # Open file as dataframe
    word_count_df = pd.read_csv(file_path)

    # Create a new empty datame
    word_count_cols = list(word_count_df.columns)
    return_cols = ['1daily_return', '2daily_return', '3daily_return', '5daily_return', '10daily_return']
    count_returns_cols = word_count_cols + return_cols
    returns_df = pd.DataFrame(columns=return_cols)
    count_returns_df = pd.DataFrame(columns = count_returns_cols)

    for index, row in word_count_df.iterrows():
        
        # Get input arguments for Yahoo function
        ticker = row['Symbol']
        start_date = row['FilingDate']
        start_date_dt = pd.to_datetime(start_date)
        end_date_dt = start_date_dt + dt.timedelta(days=31)
        end_date = str(end_date_dt.date()) 

        # Call Yahoo function
        yahoo_data = rf.get_yahoo_data(start_date, end_date, [ticker]).iloc[:1,5:-1]
        
        # Add data to returns_df
        returns_df = pd.concat([returns_df,yahoo_data], ignore_index=True)
                
        print(f'Added data for {ticker},{start_date}')

    # Combine dataframes   
    count_returns_df = pd.concat([word_count_df, returns_df], axis=1)

    print('Dataframe created')

    # Save dataframe as csv file
    count_returns_df.to_csv('count_returns.csv', index=False)

    return count_returns_df


# %% --------------------------------------------------------------------------
# Function for plotting data
# -----------------------------------------------------------------------------

# Define the function
def plot_data(word_count_file, sent_word, return_len):

    # Read file
    count_file_df = pd.read_csv(word_count_file)

    # Create axis
    fig, ax = plt.subplots()

    # Define x-data
    x = count_file_df[sent_word] 

    # Define y-data
    y = count_file_df[return_len]

    # Axis and title 
    ax.set_title('Count of sentiment word vs. returns')
    ax.set_xlabel(sent_word)
    ax.set_ylabel(return_len)

    # Add dashed line at zero
    ax.axhline(y=0, color='gray', linestyle='--')

    ax.scatter(x, y)

    plt.show()

    


# %%
