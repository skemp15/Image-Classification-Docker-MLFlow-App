"""
Data Ingestion 

Python code to download EDGAR 10-K files and save to HTML file format
"""

__date__ = "2022-12-28"
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
# Write page function
# -----------------------------------------------------------------------------
def write_page(url, file_path):
    # Define user agent
    user_agent = r'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
    # Make a request to get the URL
    response = requests.get(url, headers={'User-Agent':user_agent})
    # Write the contents of the page to the specified file path
    with open(file_path, 'w', encoding="utf-8") as f:
        f.write(response.text)

# %%

# %% --------------------------------------------------------------------------
# File downloader function
# -----------------------------------------------------------------------------

def download_files_10k(ticker, dest_folder):
    # Create the driver object
    driver = webdriver.Chrome()

    # Load the webpage and wait three seconds
    driver.get(r'https://www.sec.gov/edgar/searchedgar/companysearch')
    time.sleep(2)

    # Use xpath to search for the ticker names
    xpath_search_bar = r'//*[@id="edgar-company-person"]'
    driver.find_element("xpath", xpath_search_bar).send_keys(ticker,Keys.ENTER)
    time.sleep(2)

    # Use xpath to show 10-k reports
    xpath_10k = r'//*[@id="filingsStart"]/div[2]/div[3]/h5'
    driver.find_element("xpath", xpath_10k).click()
    time.sleep(2)

    # Use xpath to show all 10-k reports
    xpath_all_10k = r'//*[@id="filingsStart"]/div[2]/div[3]/div/button[1]'
    driver.find_element("xpath", xpath_all_10k).click()
    time.sleep(2)

    # Use xpath to filter for 10-k reports
    xpath_filter_10k = r'//*[@id="searchbox"]'
    driver.find_element("xpath", xpath_filter_10k).send_keys('10-k',Keys.ENTER)
    time.sleep(2)

    # Create a soup obkect 
    global soup
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # Find all the tables on the webpage
    global table
    table = soup.find('table', id='filingsTable')
    
    # Get a list of rows from the table
    rows = table.find_all('tr')

    # Find all links within the second column of table
    links = []
    for row in rows:
        cells = row.find_all('td')
        if len(cells) > 0:
            cell = cells[1]
            link = cell.find('a')
            links.append(link)

    # Extract the link URLs from the links
    link_urls = [link.get('href') for link in links]

    # Retrieve HTML links
    html_link_urls = []
    for link in link_urls:
        if link[1] == 'i':
            new_link = link[8:]
            html_link_urls.append(new_link)
        else:
            html_link_urls.append(link)

    # Retrieve filing dates
    dates = []
    for row in rows:
        cells = row.find_all('td')
        if len(cells) > 0:
            cell = cells[2]
            dates.append(cell.text)

    # Make folder if it doesn't exist
    cwd = os.getcwd()
    folder_path = os.path.join(cwd, dest_folder)
    if not os.path.exists(folder_path):
        os.mkdir(dest_folder)
  
    # Download links as HTML files
    for url, date in zip(html_link_urls, dates):
        new_url = 'https://www.sec.gov' + url
        file_name = ticker + '_10-k_' + date + '.html'
        file_paths = os.path.join(cwd, dest_folder, file_name)
        write_page(new_url, file_paths)
                
    # Close driver
    driver.close()

def getsp100():

        # Get a dataframe of tickers from Wikipedia
        snp = pd.read_html(r'https://en.wikipedia.org/wiki/S%26P_100', match = 'Symbol')
        df_snp = snp[0]

        # Drop duplicate of Google
        df_snp.drop([43],axis = 0, inplace = True)
        df_snp = df_snp.filter(regex='Symbol')

        # Create and return list
        list_snp = df_snp['Symbol'].values.tolist()
        
        # Change BRK.P to BRK-P
        list_snp[18] = 'BRK-B' 

        list_snp = list_snp[list_snp == 'AIG']

        return list_snp




    



# %%
