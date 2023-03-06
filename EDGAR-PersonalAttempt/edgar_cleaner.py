"""
Part 2 - Data Preparation and Cleaning

Functions to clean HMTL files of tags and 
special characters and convert to text files"""

__date__ = "2022-12-29"
__author__ = "SamKemp"

# %% --------------------------------------------------------------------------
# Imports
import requests
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import os
# -----------------------------------------------------------------------------

# %% --------------------------------------------------------------------------
# Function to remove tags and special characters
# -----------------------------------------------------------------------------

# Define the function
def clean_html_text(html_text):

    # Remove tags
    soup = BeautifulSoup(html_text, 'html.parser')
    untagged_text = soup.get_text()

    # Remove special characters
    clean_text = ''
    for letter in untagged_text:
        if letter.isalnum() == True or letter == ' ':
            clean_text += letter
        
    return clean_text


# %% --------------------------------------------------------------------------
# Function to take all HTML files in a specified folder,
# cleans them and writes them to a destination folder
# -----------------------------------------------------------------------------

def write_clean_html_text_files(input_folder, dest_folder):

    # Define input and destination paths
    # and change to input folder 
    cwd = os.getcwd()
    print(cwd)
    if cwd.endswith(input_folder) or cwd.endswith(dest_folder):
        cwd = os.path.dirname(os.getcwd())
    input_path = os.path.join(cwd, input_folder)
    dest_path = os.path.join(cwd, dest_folder)
    
    # Make a dest folder if it doesn't exist
    cwd = os.getcwd()
    print(cwd)
    dest_folder_path = os.path.join(cwd, dest_folder)
    print(dest_folder_path)
    if not os.path.exists(dest_folder_path):
        os.mkdir(dest_folder)

        
    os.chdir(input_path) 
    for file in os.listdir():

        # Skip if file exists
        dest_file_path = os.path.join(dest_path, file)
        dest_file_path = dest_file_path.replace('.html', '.txt')
        if not os.path.exists(dest_file_path):

            # Clean file in input folder
            file_path = os.path.join(input_path, file)
            with open(file_path, 'r') as f: 
                html_data = f.read()
                clean_data = clean_html_text(html_data)
                f.close()
                
        
            # Save file to destination folder
            with open(dest_file_path, 'w', encoding="utf-8") as f:
                f.writelines(clean_data)
                f.close()
                print(f'File {file} cleaned')
            
            
        else: 
            print(f'File {file} skipped')






# %%
