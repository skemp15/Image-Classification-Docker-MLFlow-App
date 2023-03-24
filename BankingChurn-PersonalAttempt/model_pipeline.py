"""
Second attempt at the banking churn model
"""

__date__ = "2023-03-03"
__author__ = "SamKemp"



# %% --------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% --------------------------------------------------------------------------
# Set Random State
# -----------------------------------------------------------------------------
rng = np.random.RandomState(123)


# %% --------------------------------------------------------------------------
# Load Data
# -----------------------------------------------------------------------------
import pandas as pd

# read in customer dataset
customer_df = pd.read_csv('customers_tm1_e.csv')

# Filter out unwanted states
unwanted_states = ['Australia', '-999', 'UNK']
customer_df = customer_df.loc[~customer_df['state'].isin(unwanted_states)]

# Replace state abbreviations with full state names
state_mapping = {'MASS': 'Massachusetts', 'NY': 'New York', 'TX': 'Texas', 'CALIFORNIA': 'California'}
customer_df['state'] = customer_df['state'].replace(state_mapping)

customer_df['state_group'] = np.where(customer_df['state'].isin(['California', 'Texas']), customer_df['state'], 'Other')

# Create dummy variables for the state_group column
state_dummies = pd.get_dummies(customer_df['state_group'], prefix='state')

# Concatenate original dataframe and dummy variable dataframe
customer_df = pd.concat([customer_df, state_dummies], axis=1)

# Remove original state and state_group columns
customer_df.drop(['state', 'state_group'], axis=1, inplace=True)

# read in transaction dataset
transaction_df = pd.read_csv('transactions_tm1_e.csv')

# Remove outliers
transaction_df = transaction_df[transaction_df['amount'] < 100000]
transaction_df = transaction_df[transaction_df['amount'] > -100000]

customer_df = customer_df[customer_df['start_balance'] > 0]
customer_df = customer_df[customer_df['start_balance'] < 1000000 ]

# Create transaction_year_month column
transaction_df['transaction_date'] = pd.to_datetime(transaction_df['transaction_date'])
transaction_df['transaction_year_month'] = transaction_df['transaction_date'].dt.strftime('%Y-%m')

transaction_df['transaction_date'] = pd.to_datetime(transaction_df['transaction_date'])
transaction_df['transaction_year_month'] = transaction_df['transaction_date'].dt.strftime('%Y-%m')

# group by customer_id and year_month, and sum the amount
grouped_df = transaction_df.groupby(['customer_id', 'account_id', 'transaction_year_month']).agg({'amount': 'sum', 'deposit': 'sum', 'withdrawal': 'sum'}).reset_index()

# merge pivoted dataframe back with customer dataframe
monthly_df = pd.merge(grouped_df, customer_df, on='customer_id')

print(monthly_df.shape)

monthly_df.to_csv('monthly_transactions.csv')


# %% --------------------------------------------------------------------------
# New features
# -----------------------------------------------------------------------------
import datetime

# Load data
monthly_df = pd.read_csv('monthly_transactions.csv')
monthly_df = monthly_df.drop(columns='Unnamed: 0')

# Calculate age
monthly_df['age'] = (pd.to_datetime(monthly_df['transaction_year_month']).dt.year - pd.to_datetime(monthly_df['dob']).dt.year)
print(monthly_df.shape)

# Calculate account age
monthly_df['account_age_days'] = pd.to_datetime(monthly_df['transaction_year_month']) - pd.to_datetime(monthly_df['creation_date'])
monthly_df['account_age_days'] = monthly_df['account_age_days'].apply(lambda x: x.days)
monthly_df['account_age_days'] = monthly_df['account_age_days'].clip(lower=0)

print(monthly_df.shape)
monthly_df.to_csv('monthly_transactions.csv')


# %% --------------------------------------------------------------------------
# Add rolling values
# -----------------------------------------------------------------------------
monthly_df['prev_month_amount'] = monthly_df.groupby('customer_id')['amount'].shift(1)
monthly_df['prev_2month_amount'] = monthly_df.groupby('customer_id')['amount'].shift(2)
monthly_df['prev_3month_amount'] = monthly_df.groupby('customer_id')['amount'].shift(3)
print(monthly_df.shape)

monthly_df['prev_month_deposit'] = monthly_df.groupby('customer_id')['deposit'].shift(1)
monthly_df['prev_2month_deposit'] = monthly_df.groupby('customer_id')['deposit'].shift(2)
monthly_df['prev_3month_deposit'] = monthly_df.groupby('customer_id')['deposit'].shift(3)
print(monthly_df.shape)

monthly_df['prev_month_withdrawal'] = monthly_df.groupby('customer_id')['withdrawal'].shift(1)
monthly_df['prev_2month_withdrawal'] = monthly_df.groupby('customer_id')['withdrawal'].shift(2)
monthly_df['prev_3month_withdrawal'] = monthly_df.groupby('customer_id')['withdrawal'].shift(3)
print(monthly_df.shape)

print(monthly_df.shape)
monthly_df.to_csv('monthly_transactions.csv')

# %% --------------------------------------------------------------------------
# Add % previous months net positive
# -----------------------------------------------------------------------------
from dateutil.relativedelta import relativedelta

# Months net positive
monthly_df['months_since_positive'] = monthly_df.groupby('customer_id')['amount'].apply(
    lambda x: x.rolling(window=len(x), min_periods=1).apply(lambda y: (y > 0).sum(), raw=False)
).astype(int)
print(monthly_df.shape)

# Create account_age_months column
monthly_df['creation_date'] = pd.to_datetime(monthly_df['creation_date'])
monthly_df['transaction_year_month'] = pd.to_datetime(monthly_df['transaction_year_month'])
monthly_df['account_age_months'] = monthly_df.apply(lambda row: relativedelta(row['transaction_year_month'], row['creation_date']).months, axis=1)
print(monthly_df.shape)

# Create percent_positive_months column
monthly_df['percent_positive_months'] = monthly_df['months_since_positive'] / monthly_df['account_age_months']
print(monthly_df.shape)
monthly_df.to_csv('monthly_transactions.csv')


# %% --------------------------------------------------------------------------
# FRED Data
# -----------------------------------------------------------------------------
import fredapi

# set up the FRED API connection
fred = fredapi.Fred(api_key='0633660eb9365979c5c087c9ee9739f3 ')

# download monthly GDP data
gdp_data = fred.get_series('GDP')

# convert the GDP data Series to a DataFrame with a 'gdp' column
gdp_data = gdp_data.to_frame('gdp')
gdp_data = gdp_data.reset_index()
gdp_data['index'] = gdp_data['index'].dt.strftime('%Y-%m')
gdp_data['index'] = pd.to_datetime(gdp_data['index'])
gdp_data.rename(columns={'index' : 'transaction_year_month'}, inplace=True)

# Merge with monthly_df
monthly_df = pd.merge(monthly_df, gdp_data, on='transaction_year_month', how='left')

# download monthly FEDFUNDS data
fedfunds_data = fred.get_series('FEDFUNDS')

# convert the FEDFUNDS data Series to a DataFrame with a 'fedfunds' column
fedfunds_data = fedfunds_data.to_frame('fedfunds')
fedfunds_data = fedfunds_data.reset_index()
fedfunds_data['index'] = fedfunds_data['index'].dt.strftime('%Y-%m')
fedfunds_data['index'] = pd.to_datetime(fedfunds_data['index'])
fedfunds_data.rename(columns={'index': 'transaction_year_month'}, inplace=True)

# Merge with monthly_df
monthly_df = pd.merge(monthly_df, fedfunds_data, on='transaction_year_month', how='left')
print(monthly_df.shape)

monthly_df.to_csv('monthly_transactions.csv')

# %% --------------------------------------------------------------------------
# Add SP500 data
# -----------------------------------------------------------------------------
import pandas_datareader as pdr
import datetime
import yfinance as yf

monthly_df = pd.read_csv('monthly_transactions.csv')

start_date = datetime.datetime(2007, 1, 1)
end_date = datetime.datetime(2021, 1, 1)

sp500_df = yf.download('^GSPC', start=start_date, end=end_date).reset_index()
sp500_df = sp500_df[['Date', 'Adj Close']]
sp500_df['Date'] = pd.to_datetime(sp500_df['Date'])
sp500_df = sp500_df.set_index('Date')
sp500_df = sp500_df.resample('MS').first()
sp500_df = sp500_df.reset_index()
monthly_df['transaction_year_month'] = pd.to_datetime(monthly_df['transaction_year_month'])
sp500_df = sp500_df.rename(columns={'Date': 'transaction_year_month', 'Adj Close': 'sp500_close'})
monthly_df = pd.merge(monthly_df, sp500_df, on='transaction_year_month', how='left')

monthly_df.to_csv('monthly_transactions2.csv')


# %% --------------------------------------------------------------------------
# % Change for sp500 and GDP
# -----------------------------------------------------------------------------
monthly_df['gdp_pct_change'] = monthly_df['gdp'].pct_change(periods=3)
monthly_df['sp500_close_pct_change'] = monthly_df['sp500_close'].pct_change(periods=3)
print(monthly_df.shape)

monthly_df.to_csv('monthly_transactions2.csv')

# %% --------------------------------------------------------------------------
# Fill in NAN values with 0
# -----------------------------------------------------------------------------


monthly_df = monthly_df.fillna(0)
monthly_df.to_csv('monthly_transactions.csv')

# %% --------------------------------------------------------------------------
# Calculate churn
# -----------------------------------------------------------------------------
# find the last transaction year month for each customer
last_transactions = monthly_df.groupby('customer_id')['transaction_year_month'].max().reset_index()

# merge last transaction year month with monthly_df
monthly_df = pd.merge(monthly_df, last_transactions, on='customer_id', suffixes=('', '_last'))

# convert transaction_year_month column to datetime type
monthly_df['transaction_year_month'] = pd.to_datetime(monthly_df['transaction_year_month'])

# set churn values to 1 if it is the final transaction_year_month for each customer and 0 if not
monthly_df['churn'] = monthly_df['transaction_year_month'] == monthly_df['transaction_year_month_last']
monthly_df['churn'] = monthly_df['churn'].astype(int)

# set churn values to 0 for entries with transaction_year_month 2020-3 and 2020-04 
monthly_df.loc[monthly_df['transaction_year_month'].isin(['2020-03', '2020-05']), 'churn'] = 0

monthly_df.to_csv('monthly_transactions.csv')

# %% --------------------------------------------------------------------------
# Train/Test Split
# -----------------------------------------------------------------------------
from sklearn.model_selection import train_test_split

# Drop columns
drop_columns =['churn', 'customer_id', 'account_id', 'transaction_year_month', 'dob', 'creation_date', 'sp500_close', 'percent_positive_months', 'account_age_months', 'months_since_positive', 'transaction_year_month_last', 'gdp', 'account_age_days'] 

# select the relevant features
features = monthly_df.drop(columns=drop_columns).columns

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(monthly_df[features], monthly_df['churn'], test_size=0.2, random_state=rng, stratify=monthly_df['churn'])



# %% --------------------------------------------------------------------------
# Make model 
# -----------------------------------------------------------------------------
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, roc_auc_score

# Split the data into training and validation sets
X_train2, X_valid, y_train2, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0)


def objective(trial):
    # Define the hyperparameters to tune
    n_estimators = trial.suggest_int('n_estimators', 50, 100)
    max_depth = trial.suggest_int('max_depth', 3, 7)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 6)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
    max_features = trial.suggest_uniform('max_features', 0.1, 0.9)

    # Define the model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=rng
    )

    # Fit the model on the training data
    model.fit(X_train2, y_train2)

    # Get the predicted labels for the validation data
    y_pred = model.predict(X_valid)

    # Calculate the accuracy score for the validation data
    score = recall_score(y_valid, y_pred)

    return score

# Define the study and optimize the objective function
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Print the best hyperparameters and the corresponding accuracy score
print(f"Best parameters: {study.best_params}")
print(f"Best accuracy: {study.best_value:.2f}")


# %% --------------------------------------------------------------------------
# Run model on test data 
# -----------------------------------------------------------------------------
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


# Fit the model and make predictions
model = RandomForestClassifier(
    n_estimators = study.best_params['n_estimators'],
    max_depth = study.best_params['max_depth'],
    min_samples_split = study.best_params['min_samples_split'],
    min_samples_leaf = study.best_params['min_samples_leaf'],
    max_features = study.best_params['max_features']
)
model.fit(X_train, y_train)


# %% --------------------------------------------------------------------------
#  Save model
# -----------------------------------------------------------------------------
import pickle

with open('RFmodel.pickle', 'wb') as file:
    pickle.dump(model, file)

# %% --------------------------------------------------------------------------
# Evaluate model
# -----------------------------------------------------------------------------

# Load model
with open('RFmodel.pickle', 'rb') as file:
    model = pickle.load(file)

# Load test data 
X_test.read_csv('X_test.csv')
y_test.read_csv('X_test.csv')

# Make predictions
y_pred = model.predict(X_test)

# Print classification report and auc score
print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test, y_pred))

# Create a confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(cm, display_labels=['No', 'Yes'])
fig, ax = plt.subplots(figsize=(8, 8))
cm_display.plot(cmap='OrRd', ax=ax)
plt.title('Confusion Matrix')
plt.show()


# %%
