"""
Pipeline for cleaning input data 
"""

__date__ = "2023-07-27"
__author__ = "SamKemp"


# %% --------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.impute import KNNImputer

# %% --------------------------------------------------------------------------
# Drop empty columns function
# -----------------------------------------------------------------------------

def drop_empty_columns(df):
    # Get the columns before dropping empty columns
    cols_before = df.columns

    # Use the dropna() function to drop columns with all NaN (empty) values
    df.dropna(axis=1, how='all', inplace=True)

    # Get the columns after dropping empty columns
    cols_after = df.columns

    # Find the removed columns
    removed_columns = set(cols_before) - set(cols_after)
    return df, removed_columns

# %% --------------------------------------------------------------------------
# Drop empty rows
# -----------------------------------------------------------------------------
def drop_empty_rows(df):
    # Get the number of rows before dropping empty rows
    rows_before = df.shape[0]

    # Use the dropna() function to drop rows with all NaN (empty) values
    df.dropna(axis=0, how='all', inplace=True)

    # Get the number of rows after dropping empty rows
    rows_after = df.shape[0]

    # Calculate the number of rows removed
    rows_removed = rows_before - rows_after
    return df, rows_removed

# %% --------------------------------------------------------------------------
# Function to replace emails with most common email
# -----------------------------------------------------------------------------
def replace_with_most_popular_email(df, name, name_col, email_col):

    # Get most common email
    most_common_email = df[df[name_col]==name][[email_col]].value_counts().index[0][0]

    # We can update the email to keep it consistent
    df.loc[df[name_col]== name, email_col] = most_common_email

    return df

# %% --------------------------------------------------------------------------
# Convert date format
# -----------------------------------------------------------------------------

def convert_date_format(input_date):
    if pd.isna(input_date):
        return input_date  # Return NaN as is
    else:
        # Define the input and output date formats
        input_format = "%A, %d %B %Y"
        output_format = "%d/%m/%Y"

        # Parse the input date using the input format
        date_obj = datetime.strptime(input_date, input_format)

        # Convert the date object to the desired output format
        output_date = date_obj.strftime(output_format)
        return output_date
    
# %% --------------------------------------------------------------------------
# Impute missing emails and phones from ID
# -----------------------------------------------------------------------------

def replace_missing_email_phone(df):
    # Get null counts before replacing missing values
    email_null_count_1 = df["consumer_1_email"].isnull().sum()
    email_null_count_2 = df["consumer_2_email"].isnull().sum()
    phone_null_count_1 = df["consumer_1_phone_number"].isnull().sum()
    phone_null_count_2 = df["consumer_2_phone_number"].isnull().sum()

    # Forward-fill missing values for email and phone number columns within each group
    df["consumer_1_email"] = df.groupby("consumer_1_ID")["consumer_1_email"].transform(lambda x: x.fillna(method="ffill"))
    df["consumer_2_email"] = df.groupby("consumer_2_ID")["consumer_2_email"].transform(lambda x: x.fillna(method="ffill"))
    df["consumer_1_phone_number"] = df.groupby("consumer_1_ID")["consumer_1_phone_number"].transform(lambda x: x.fillna(method="ffill"))
    df["consumer_2_phone_number"] = df.groupby("consumer_2_ID")["consumer_2_phone_number"].transform(lambda x: x.fillna(method="ffill"))

    # Get null counts after replacing missing values
    print(email_null_count_1 - df["consumer_1_email"].isnull().sum(), 'missing emails replaced (consumer 1)')
    print(email_null_count_2 - df["consumer_2_email"].isnull().sum(), 'missing emails replaced (consumer 2)')
    print(phone_null_count_1 - df["consumer_1_phone_number"].isnull().sum(), 'missing phone numbers replaced (consumer 1)')
    print(phone_null_count_2 - df["consumer_2_phone_number"].isnull().sum(), 'missing phone numbers replaced (consumer 2)')

    return df

# %% --------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------

def clean_data(input_file_name, output_file_name):

    # Load data 
    df = pd.read_csv(input_file_name)

    # Remove empty columns
    df = drop_empty_columns(df)[0]

    # Remove empty rows
    df = drop_empty_rows(df)[0]

    # Drop mortgage_charge
    df = df.drop('mortgage_charge', axis=1)   

    # Drop duplicates
    df = df.drop_duplicates()

    # Update Olivia Joshua email (Note: with more time I would alter this part to cover any new mortgage advisors with multiple emails)
    df = replace_with_most_popular_email(df, 'Olivia Joshua', 'mortgage_advisor_name', 'mortgage_advisor_email')

    # Convert consumer 2 DOB
    df['consumer_2_dob'] = df['consumer_2_dob'].apply(convert_date_format)

    # Create consumer IDs 
    df['consumer_1_ID'] = df['consumer_1_name'] + '_' + df['consumer_1_dob'].astype(str) 
    df['consumer_2_ID'] = df['consumer_2_name'] + '_' + df['consumer_2_dob'].astype(str)

    # Update Rochelle Imran email and phone Note: with more time I would alter this part to cover any new consumers with multiple emails/phone numbers)
    df.loc[df['consumer_2_ID'] == 'Rochelle Imran_28/01/1968', 'consumer_2_email'] = 'clear_back_best@example.com'
    df.loc[df['consumer_2_ID'] == 'Rochelle Imran_28/01/1968', 'consumer_2_phone_number'] = '7582672214'

    # Impute missing emails and phone numbers from ID
    df = replace_missing_email_phone(df)

    # Fill nan emails and phone numbers with Unknown
    df['consumer_1_email'].fillna('Unknown', inplace=True)
    df['consumer_2_email'].fillna('Unknown', inplace=True)
    df['consumer_1_phone_number'].fillna('Unknown', inplace=True)
    df['consumer_2_phone_number'].fillna('Unknown', inplace=True)

    # Update validity columns
    df.loc[df['consumer_1_email']=='Unknown', 'consumer_1_email_validity'] = 'No email'
    df.loc[df['consumer_2_email']=='Unknown', 'consumer_2_email_validity'] = 'No email'
    df.loc[df['consumer_1_phone_number']=='Unknown', 'consumer_1_phone_number_validity'] = 'No phone number'

    # Fill NaN columns
    df['consumer_2_name'].fillna('No second consumer', inplace=True)

    # Update other columns
    df.loc[df['consumer_2_name']=='No second consumer', 'consumer_2_dob'] = 'No second consumer'
    df.loc[df['consumer_2_name']=='No second consumer', 'consumer_2_email'] = 'No second consumer'
    df.loc[df['consumer_2_name']=='No second consumer', 'consumer_2_phone_number'] = 'No second consumer'
    df.loc[df['consumer_2_name']=='No second consumer', 'consumer_2_ID'] = 'No second consumer'

    # Fill NaN columns
    df[['consumer_2_dob', 'property_address', 'mortgage_start_date']] = df[['consumer_2_dob', 'property_address', 'mortgage_start_date']].fillna('Unknown')

    # Drop mortgage_pct_owned
    df.drop('mortgage_pct_owned', axis=1, inplace=True)

    # Fill missing values in categorical columns with the mode of each column
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    for feature in categorical_features:
        df[feature] = df[feature].fillna(df[feature].mode().iloc[0])

    # Impute missing numerical values
    numerical_features = df.select_dtypes(include=['number']).columns.tolist()
    imputer = KNNImputer(n_neighbors=5)
    df[numerical_features] = imputer.fit_transform(df[numerical_features])

    # Print missing values
    print('Total missing values:', df.isna().sum().sum())

    # Save to file
    df.to_csv(output_file_name, index=False)
