# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 12:42:14 2021

@author: Teci
"""
import numpy as np
import pandas as pd
import MFM714_datacleaner as dc
import MFM714_credmigration as cm

# Import relevant data structures and functions from other files
portfolio_df = dc.portfolio_df
ultimate_pricing_df = dc.ultimate_pricing_df
reduced_pricing_df = dc.reduced_pricing_df
yields_df = dc.yields_df
issuers_df = dc.issuers_df
cholesky_np = dc.cholesky_np
simulateNewRatings = cm.simulateNewRatings

# Because we ignore the accrued interest and coupons, comparisons between
# The price given in original portfolio is misleading. For instance, a 20y
# coupon will be worth much less using the simplified pricing function
# described in the assignment instructions. Therefore we price the assets with
# a crude pricer from the beginning.


def getCurrentBondPrice(asset):
    disc = np.exp(-asset['Yield']*asset['Current Time Remaining'])
    price = asset['Notional']*disc
    return price


def getCurrentPrices(asset):
    if asset['Instrument'] == 'Bond':
        price = getCurrentBondPrice(asset)

    else:
        price = 0

    price_df = pd.Series(price, index=['Current Price'])
    row = asset.append(price_df)
    return row


reduced_pricing_df = reduced_pricing_df.apply(
    lambda row: getCurrentPrices(row), axis=1)

def getMigPrice(N, coupon, instr, prevYield, currentYield, Mat):
    if instr == 'Bond':
        disc = np.exp(-currentYield*Mat)
        price = N * disc
    else:
        currAnnuity = (coupon/currentYield)*(1-np.exp(-currentYield*Mat))
        fixedAnnuity = (coupon/prevYield)*(1-np.exp(-prevYield*Mat))
        price = currAnnuity - fixedAnnuity
    return price


def getSimPortVal(simMig, simNo, portfolio):
    newPort = portfolio.apply(
        lambda row: getMigPrice(row['Notional'],
                                row['Coupon'],
                                row['Instrument'],
                                row['Yield'],
                                row[simMig.loc[row.name][simNo]],
                                row['Tenor t=1']),
        axis=1)
    return newPort

simMig = simulateNewRatings(cholesky_np, issuers_df)
x = getSimPortVal(simMig, 'Sim 1', reduced_pricing_df)


def getSimulationValues(creditSims, portfolio):
    n = creditSims.shape[1]
    simPorts = []
    print('')

    for i in range(n):
        if i*20/n == float(i*20//n):
            print(str(i*100/n) + '% of pricing simulations complete')
        index = 'Sim ' + str(i+1)
        simMig = creditSims[[index]]
        simPort = getSimPortVal(simMig, index, portfolio)
        simPort = simPort.tolist()
        simPorts.append(simPort)

    simPorts = np.array(simPorts).T
    simPorts = pd.DataFrame(simPorts)
    simPorts.columns = ['Port ' + str(i) for i in range(1, n+1)]
    simPorts = simPorts.set_index(reduced_pricing_df.index)
    return simPorts


creditSims = simulateNewRatings(cholesky_np, issuers_df, 2000)
simNewPortfolios = getSimulationValues(creditSims, reduced_pricing_df)


def getVaR(series, q):
    """
    Return VaR given a series of portfolio value changes/returns.

    Parameters
    ----------
    series : Series
        A pandas series of returns to compute VaR with.
    q : float
        A float between 0 and 1 which represents the VaR threshold/quantile.

    Returns
    -------
    VaR : float
        The estimated VaR of the portfolio returns at threshold/quantile q.
    """
    series = series.sort_values(ascending=True)  # Sort list for easy indexing.
    n = series.size
    index = int(n*(1-q))
    # normally the index corresponding to VaR is n*q+1. But this form assumes
    # both that the list is descending and that indexing begins with one.
    # Neither condition holds here, hence the somewhat unusual form.
    VaR = series.iloc[index]
    # If we assume finite scenarios are uniformly distributed, this value
    # corresponds to the usual value of VaR.
    return VaR


def getCVaR(series, q):
    """
    Return CVaR given a series of portfolio value changes/returns.

    Parameters
    ----------
    series : Series
        A pandas series of returns to compute VaR with.
    q : float
        A float between 0 and 1 which represents the VaR threshold/quantile.

    Returns
    -------
    VaR : float
        The estimated CVaR of the portfolio returns at threshold/quantile q.
    """
    series = series.sort_values(ascending=True)
    n = series.size
    indices = np.arange(int(n*(1-q)) + 1)
    # normally the index corresponding to VaR is n*q+1. But this form assumes
    # both that the list is descending and that indexing begins with one.
    # Neither condition holds here, hence the somewhat unusual form.
    CVaR = np.mean([series.iloc[i] for i in indices])
    # As noted in the "getVaR" function, series.iloc[i] corresponds to the VaR
    # at threshold q should n*(1-q) = i. The CVaR is defined as the average
    # over VaR, hence the "np.mean" function.
    return CVaR


oldPortVal = reduced_pricing_df['Current Price'].sum()
newPortVals = simNewPortfolios.sum()
changeInPortVal = newPortVals - oldPortVal

mean_port_change = changeInPortVal.mean()
port_std = changeInPortVal.std()
port_VaR_99 = getVaR(changeInPortVal, 0.99)
port_VaR_95 = getVaR(changeInPortVal, 0.95)
port_CVaR_98 = getCVaR(changeInPortVal, 0.98)

# print relevant values.
print("Average of change in portfolio value: {0}".format(mean_port_change))
print("StDev of change in portfolio value: {0}".format(port_std))
print("99% VaR of change in portfolio value: {0}".format(port_VaR_99))
print("95% VaR of change in portfolio value: {0}".format(port_VaR_95))
print("98% CVaR of change in portfolio value: {0}".format(port_CVaR_98))
