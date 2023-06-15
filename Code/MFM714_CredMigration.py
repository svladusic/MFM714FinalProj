"""Module implements part 2 of the MFM 714 Final Project."""

# import os
import pandas as pd
import numpy as np
import scipy.stats as sps
import MFM714_datacleaner as dc

# Import relevant arrays from data cleaning module.
transition_mat_df = dc.transition_mat_df
cholesky_np = dc.cholesky_np
issuers_df = dc.issuers_df

# Create correspondance between issuer rating and integer indices. This will
# be useful when switching between pd DataFrames and np arrays.
rating_to_indices = {'AAA': 0, 'AA': 1, 'A': 2, 'BBB': 3, 'BB': 4, 'B': 5,
                     'CCC': 6, 'D': 7}
indices_to_ratings = {0: 'AAA', 1: 'AA', 2: 'A', 3: 'BBB', 4: 'BB', 5: 'B',
                      6: 'CCC', 7: 'D'}


def getRowSums(row):
    """
    Return row of cumulative transition probabilities given transition matrix.

    In order to implement the credit metrics approach, we need to consider the
    cumulative transition probabilities. For instance, how likely is it that a
    name has a credit rating of BBB or worse after a 1 year time horizon? This
    function takes a row of the transition probability matrix and returns the
    related row of the cumulative transition probability matrix. Then using the
    apply method, we can convert the whole transition matrix to its cumulative
    counterpart.

    Parameters
    ----------
    row : Pd.df
        Pandas DataFrame which represents row of transition probability matrix.

    Returns
    -------
    pd.Series
        Pandas Series which represents row of cumulative transition probability
        matrix.

    """
    return pd.Series([sum(row[i:]) for i in range(row.size)])


cum_tp_df = transition_mat_df.apply(getRowSums, axis=1)
norm_cum_tp_df = cum_tp_df.applymap(sps.norm.ppf)
norm_cum_tp_np = norm_cum_tp_df.to_numpy()

# return the inverse standard normal CDF values of the cumulative transition
# probabilities. We can use these values to simulate where the issuer's credit
# rating lands via correlated draws.


def getNewRating(name, draw):
    """
    Return a name's new credit rating given draw and previous credit rating.

    This function takes a name's correlated random draw and its previous credit
    rating in order to find its new credit rating. In particular, we can find
    the name's inverted normal cumulative transition probabilities from the
    "nom_cum_tp_np" array. The function then finds the "bucket" the draw is in
    and returns the corresponding credit rating.

    Parameters
    ----------
    name : dt.Series
        Series that will ultimately represent a row in the "issuers_df" data
        frame.
    draw : np.float
        float that represents the correlated normal draw determining the name's
        new credit rating. this parameter is a np.float because the inverted
        normal cumulative transition probabilities are sometimes the "inf"
        np.float value. Thus the draw should be a np.float so that there are no
        issues when using booleans over the draw and the np.inf values.

    Returns
    -------
    new_rating : str
        The new credit rating.

    """
    name_rating_index = rating_to_indices[name['Issuer Rating']]
    new_index = np.max(np.where(draw < norm_cum_tp_np[name_rating_index]))
    new_rating = indices_to_ratings[new_index]
    return new_rating


def getDraws(cholesky):
    """
    Get random draws that simulate credit rating of name after one year.

    Given a numpy array representing the cholesky decomposition of the
    correlation matrix between various firms, return a vector of correlated
    normal draws. These draws will be used to simulate the name's credit rating
    after one year.

    Parameters
    ----------
    cholesky : np.array
        np.array that represents the Cholesky decomposition of the name's asset
        return correlations.

    Returns
    -------
    np.array
        np.array that represents one correlated normal draw given the cholesky
        matrix taken as input.
    """
    n = cholesky.shape[0]
    uncorr_normal_draw = np.random.normal(size=n)
    corr_normal_draw = cholesky@uncorr_normal_draw
    return corr_normal_draw


def simulateNewRatings(cholesky, issuers, n=1):
    draw = getDraws(cholesky)
    no_issuers = issuers_df.shape[0]
    simRatings = []
    for i in range(n):
        if i*20/n == float(i*20//n):
            print(str(i*100/n) + '% of migration simulations complete')
        draw = getDraws(cholesky)
        migDraw = [getNewRating(issuers_df.iloc[j], draw[j])
                   for j in range(no_issuers)]
        simRatings.append(migDraw)
    simRatings = np.array(simRatings).T
    simRatings = pd.DataFrame(simRatings)
    simRatings.index = issuers_df['Name']
    simRatings.columns = ["Sim " + str(i) for i in range(1, n+1)]
    return simRatings

if __name__ == "__main__":
  newRatings = simulateNewRatings(cholesky_np, issuers_df, 2000)
