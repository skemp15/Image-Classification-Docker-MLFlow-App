"""
Module for EDGAR Analysis Functions 

This module will contain functions that include analysing 
Yahoo financial data on the 10-k report filing dates, etc. 
"""

__date__ = "2022-12-30"
__author__ = "SamKemp"



# %% --------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
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


# %% --------------------------------------------------------------------------
# Function to analyse Yahoo financial data
# -----------------------------------------------------------------------------

def get_yahoo_data(start_date, end_date,
                   tickers, time_interval = 'daily'):
    
    # Create empty dataframe
    financial_data  = pd.DataFrame(columns = ['date', 'high', 'low', 'price', 'volume', '1daily_return',
       '2daily_return', '3daily_return', '5daily_return', '10daily_return',
       'Symbol'])


    # Cycle through list of tickers
    for ticker in tickers:

        # Get Yahoo financial data 
        yahoo_data = YahooFinancials(ticker).get_historical_price_data(start_date, end_date, time_interval)
        
        # Extract financial data from Yahoo data
        prices = pd.DataFrame(yahoo_data[ticker]['prices'])

        # Convert date column to datetime
        prices['date'] = pd.to_datetime(prices['formatted_date'])

        # Drop unneccessary columns and rename adjclose 
        prices.rename(columns = {'adjclose' : 'price'}, inplace = True)
        prices = prices[['date', 'high', 'low', 'price', 'volume']]

        # Add returns columns
        for x in [1,2,3,5,10]:
            prices[str(x)+'daily_return'] = (prices['price'].shift(-x) - prices['price'])/prices['price']

        # Add ticker column
        prices['Symbol'] = ticker

        # Add prices to financial_data dataframe
        financial_data.columns = prices.columns 
        financial_data = pd.concat([financial_data, prices])
        financial_data.reset_index(drop=True, inplace=True)
    
    return financial_data


# %% --------------------------------------------------------------------------
# Function to generate dictionary of sentiment words
# -----------------------------------------------------------------------------

# Define the function
def get_sentiment_word_dict():

    # Download the dictionary

    # Define user agent
    user_agent = r'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
    
    # Make a request to get the URL
    url = r'https://sraf.nd.edu/loughranmcdonald-master-dictionary/'
    response = requests.get(url, headers={'User-Agent':user_agent})
    
    # Create the soup object and find the link
    dictionary_name = 'Loughran-McDonald_MasterDictionary_1993-2021.csv'
    soup = BeautifulSoup(response.text, 'html.parser')
    link = soup.find('a', text = dictionary_name)

    # Check if dictionary exists and if 
    # not download the dictionary
    download_response = requests.get(link['href'])
    cwd = os.getcwd()
    file_path = os.path.join(cwd, dictionary_name)
    if not os.path.exists(file_path):
        with open(file_path, 'wb') as f:
            f.write(download_response.content)
            f.close()
            print(f'Dictionary downloaded')
    else:
        print(f'Dictionary already downloaded')

    # Create the pyhton dictionary 

    # Create an empty dictionary
    sentiment_words = ['Negative', 'Positive', 'Uncertainty', 'Litigious']
    sentiment_word_dict = {}
    for sent_word in sentiment_words:
        sentiment_word_dict[sent_word] = None

    # Convert the dictionary to a dataframe
    dict_df = pd.read_csv(file_path)
    #dict_df.set_index('Word', inplace = True)

    # Cycle through sentiment words and words and
    # add word if any value in sentiment column 
    for sent_word in sentiment_words:
        sent_word_list = []
        for word in dict_df.index:        
            if dict_df.loc[word, sent_word] != 0:
                sent_word_list.append(word)
        sentiment_word_dict[sent_word] = sent_word_list
        print(f'Sentiment {sent_word} added')

    print('Dictionary created')

    return sentiment_word_dict 
                
# %%
