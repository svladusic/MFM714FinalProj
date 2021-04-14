"""
Created on Wed Apr  7 11:49:09 2021

@author: Teci
"""

import os
import pandas as pd
import numpy as np

# I set the working directory to be the parent folder of the current script.
# This is to ensure that there are no issues when importing the data, which
# I put in separate folder.
os.chdir('..')

# I will primarily be using pandas, as it easy to manipulate data with the
# package, and because pandas dfs can easily be transformed into numpy arrays,
# should the need arise.

# Read the issuer data stored in the relevant .xls file.
issuers_df = pd.read_excel(".\\Data\\Issuers_for_project_2021.xls")
# No further manipulation needs to be done; the df is already clean.


def fixBondName(inp):
    """
    Replace the str 'bond' with 'Bond'.

    Parameters
    ----------
    inp : str
        Any arbitrary string.

    Returns
    -------
    out : str
        A string which is equivalent to inp if inp != 'bond' and 'Bond' if
        inp == 'bond'.

    """
    if inp == 'bond':
        out = 'Bond'
    else:
        out = inp
    return out


portfolio_df = pd.read_excel(".\\Data\\Portfolio_for_project_2021.xls")
# The df is clean, but one column has a particularly long title. I'm renaming
# it for convenience's sake.
portfolio_df.rename(columns={'Instrument type (Bond or CDS)': 'Instrument'},
                    inplace=True)
inst = portfolio_df[['Instrument']]
portfolio_df[['Instrument']] = inst.applymap(fixBondName)

yields_df = pd.read_excel(".\\Data\\Yield curves of 20210209.xlsx")
# The loaded df is not clean. In particular the column names are loaded as the
# second row of entries, and the first row of entries is just a row of zeros.

# Set the column names to those given by the 2nd row's entries.
yields_df.columns = yields_df.iloc[1]
# Remove the first two rows, now that the column names have been changed.
yields_df = yields_df[2:]
# Removing the first two rows makes the df index start at 2, so we will reset
# the indices.
yields_df = yields_df.reset_index()
# For whatever reason, resetting the indices just puts the old indices as a
# new column. So we remove the column that corresponds to the former indices.
yields_df = yields_df.drop(columns='index')

# Read the transition matrix data stored in the relevant .csv file.
transition_mat_df = pd.read_excel(".\\Data\\Final Transition Matrix.xlsx")
# I want to set the indices/column names to correspond to intial/migration
# ratings, respectively. The column names already correspond to the migration
# rating, so I just need to change the index. Fortunately, the column 'TEMP' is
# just a column corresponding to the initial credit ratings, so we simply set
# this column to be the df index.
transition_mat_df = transition_mat_df.set_index('From/to')
# Create a numpy array representation of the transition matrix (I suspect some
# actual matrix multiplication is coming up soon).
# transition_mat_np = transition_mat_df.to_numpy()

# The number of issuers determines the number of rows/columns in the
# correlation matrix. So we save the number of issuers as an integer and pass
# it through various numpy functions to generate the appropriate correlation
# matrix.
no_issuers = issuers_df.shape[0]

# Get correlation matrix according to specifications in Part 2 document.


def getCorrMat(issuers):
    """
    Given a set of issuer names, return credit migration correlation matrix.

    The credit metrics approach relies on drawing correlated normal return
    draws to determine the new credit rating of a name over some time. As such,
    one must specify how names are correlated with one another. As noted in
    part 2 of the project instructions, we assume that two names/firms in the
    same industry have a correlation coefficient of 0.65, and a correlation
    coefficient of 0.28 if they are different. Thus, this function takes in a
    pandas array of issuers, and returns an nxn correlation matrix, where n is
    the number of names/firms in the portfolio.

    Parameters
    ----------
    issuers : pd.df
        A pandas DataFrame version of the "Issuers_for_project_2021.xls" excel
        sheet.

    Returns
    -------
    corrMat : pd.df
        A pandas DataFrame representing the correlation matrix rho for the
        firms in a portfolio (correlations given bby instructions).
    """
    corrMat = pd.DataFrame(index=issuers_df['Name'],
                           columns=issuers_df['Name'])
    corrMat = corrMat.fillna(0)
    m, n = corrMat.shape
    for i in range(m):
        for j in range(n):
            if issuers_df['Name'].iloc[i] == issuers_df['Name'].iloc[j]:
                corrMat.iloc[i, j] = 1
            elif (issuers_df['Industry'].iloc[i]
                  == issuers_df['Industry'].iloc[j]):
                corrMat.iloc[i, j] = 0.65
            else:
                corrMat.iloc[i, j] = 0.28

    return corrMat


# Get correlation matrix for portfolio, and convert to np.array to easily
# compute Cholesky decomposition.
correlation_mat_df = getCorrMat(issuers_df)
correlation_mat_np = correlation_mat_df.to_numpy()
cholesky_np = np.linalg.cholesky(correlation_mat_np)
# Get DataFrame version of Cholesky decomposition should it be required.
cholesky_df = pd.DataFrame(cholesky_np)
cholesky_df.columns = issuers_df['Name']
cholesky_df.index = issuers_df['Name']

# Change portfolio DataFrame so it is easier to mark assets to market after a
# credit simulation. In particular, the aaset prices/M2M depend on the time
# until maturity, rather than the specific maturity date.
maturities = pd.to_datetime(portfolio_df['Maturity Date'])
# Convert excel dates to the datetime format. This will make is striaghtforward
# to compute the required times to maturity.
# Per Prof. Lozinski's email, assume that "today" is Feb 9th 2021.
initial_dates = pd.to_datetime(['2021-02-09']*42)
remaining_time = maturities - initial_dates
# The step two instructions specify that we treat all assets with a time to
# maturity (henceforth ttm) <1y as if they had ttm = 1y.
remaining_time = remaining_time.apply(lambda x: float(x.days)/365)
remaining_time = remaining_time.apply(lambda x: x if x > 1 else 1)
# Include ttm and ttm after a year in portfolio DataFrame to make pricing
# simpler when that becomes a concern.
portfolio_df[['Current Time Remaining']] = remaining_time
portfolio_df[['Time Remaining t=1']] = remaining_time.apply(lambda x: x-1)
# In order to price bonds correctly, get tenor closest to actual time
# remaining.
tenorVals = yields_df['In years'].to_numpy()


def getTenors(time_remaining, tenorVals):
    """
    Get appropriate tenor given asset ttm.

    For our asset pricing, we will assume that all assets are priced as if they
    had a maturity given by the yields_df table. 
    Parameters
    ----------
    time_remaining : TYPE
        DESCRIPTION.
    tenorVals : TYPE
        DESCRIPTION.

    Returns
    -------
    tenor : TYPE
        DESCRIPTION.

    """
    index = np.argmin(np.abs(tenorVals-time_remaining))
    tenor = tenorVals[index]
    return tenor


curTimeRem = portfolio_df['Current Time Remaining']
laterTimeRem = portfolio_df['Time Remaining t=1']
portfolio_df[['Current Tenor']] = curTimeRem.apply(
    lambda x: (getTenors(x, tenorVals)))
portfolio_df[['Tenor t=1']] = laterTimeRem .apply(
    lambda x: (getTenors(x, tenorVals)))

# Get yields for each credit migration. This can be done using the yield
# multiplier.

def getYM(asset, yields_df, issuers_df):
    assetYield = asset['Yield']
    issuer = asset['Name']
    assetRating = issuers_df[issuers_df['Name'] == issuer]
    assetRating = assetRating['Issuer Rating'].tolist()
    assetRating = assetRating[0]
    assetTenor = asset['Current Tenor']
    credYield = yields_df[yields_df['In years'] == assetTenor][assetRating]
    credYield = credYield.to_list()
    credYield = credYield[0]
    yieldMult = assetYield/credYield
    return yieldMult


portfolio_df[['YM']] = portfolio_df.apply(
    lambda row: getYM(row, yields_df, issuers_df), axis=1)

yields_df = yields_df.set_index(['In years'])


def getFutureYields(row, yields_df):
    tenor = row['Tenor t=1']
    yields = yields_df.loc[tenor].drop(index=['Tenor', 'Government'])
    yields = row['YM'] * yields
    row = row.append(yields)
    return row


ultimate_pricing_df = portfolio_df.apply(
    lambda row: getFutureYields(row, yields_df), axis=1)
ultimate_pricing_df[['D']] = np.random.beta(1.62, 1.86, 42)
reduced_pricing_df = ultimate_pricing_df.drop(
    columns=['CUSIP', 'Price', 'Theoretical Clean Price',
             'Expected Recovery rate'])
reduced_pricing_df = reduced_pricing_df.set_index(['Name'])
# change cwd back to code file
os.chdir('./Code')
